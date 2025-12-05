import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.init import xavier_uniform_, xavier_normal_, constant_


class SpectrumEncoder(nn.Module):
    """
    Augmented Spectrum Encoding

    Description:
    Maps continuous eigenvalues (frequencies) into a high-dimensional vector space
    using sinusoidal functions. This acts as a "Positional Encoding" for the
    graph spectrum, allowing the neural network to distinguish between low-frequency
    (global) and high-frequency (local) signals effectively.
    """

    def __init__(self, hidden_dim=128):
        super(SpectrumEncoder, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)

    def forward(self, e):
        # input:  [N] (Eigenvalues / Frequencies)
        # output: [N, d] (Encoded Spectral Embeddings)

        ee = e * self.constant
        div = torch.exp(
            torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)
        ).to(e.device)
        pe = ee.unsqueeze(1) * div
        eeig = torch.cat((e.unsqueeze(1), torch.sin(pe), torch.cos(pe)), dim=1)

        return self.eig_w(eeig)


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network for Spectral Proxies

    Description:
    A standard MLP used within the Hierarchical Filtering Mechanism to process
    spectral representations and generate filter coefficients.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class HierarchicalLayer(nn.Module):
    """
    Iterative Structural Propagation Layer

    Logic: H^(l) = sigma( M_hier * H^(l-1) )
    Where M_hier = M_inter + sum(M_intra)

    This layer learns the combination weights for the operator components and
    performs the actual propagation step.
    """

    def __init__(self, nbases, ncombines, prop_dropout=0.0, norm="none", metis_dim=0):
        super(HierarchicalLayer, self).__init__()
        self.prop_dropout = nn.Dropout(prop_dropout)
        self.metis_dim = int(metis_dim)
        self.base_dim = nbases - self.metis_dim

        # The learnable weights to combine different basis components
        if norm == "none":
            self.weight = nn.Parameter(torch.ones((1, nbases, ncombines)))
        else:
            self.weight = nn.Parameter(torch.empty((1, nbases, ncombines)))
            nn.init.normal_(self.weight, mean=0.0, std=0.01)

        if norm == "layer":
            self.norm = nn.LayerNorm(ncombines)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(ncombines)
        else:
            self.norm = None

    def forward(self, features, metis_payload=None):
        if isinstance(features, torch.Tensor):
            feats = torch.unbind(features, dim=1)
        else:
            feats = list(features)

        if not feats:
            raise ValueError("HierarchicalLayer requires at least one feature map.")

        weight = self.weight
        combine_dim = weight.size(-1)
        out = torch.zeros_like(feats[0])

        for idx, feat in enumerate(feats):
            feat = self.prop_dropout(feat)
            out = out + feat * weight[:, idx]

        if self.metis_dim > 0 and metis_payload is not None:
            metis_basis, metis_proj = metis_payload
            if metis_proj.size(0) != self.metis_dim:
                raise ValueError(
                    f"Expected {self.metis_dim} METIS components, got {metis_proj.size(0)}"
                )
            weighted_proj = metis_proj * weight[
                :, self.base_dim : self.base_dim + self.metis_dim
            ].squeeze(0)
            metis_out = torch.matmul(metis_basis, weighted_proj)
            metis_out = self.prop_dropout(metis_out)
            out = out + metis_out

        # x = self.prop_dropout(x) * self.weight  # [N, m, d] * [1, m, d]
        # x = torch.sum(x, dim=1)

        if self.norm is not None:
            # x = self.norm(x)
            # x = F.relu(x)
            out = self.norm(out)
            out = F.relu(out)

        # return x
        return out


class SPHierarchicalModel(nn.Module):

    def __init__(
        self,
        nclass,
        nfeat,
        nlayer=1,
        hidden_dim=128,
        nheads=1,
        tran_dropout=0.0,
        feat_dropout=0.0,
        prop_dropout=0.0,
        norm="none",
        metis_dim=0,
    ):
        super(SPHierarchicalModel, self).__init__()

        self.norm = norm
        self.nfeat = nfeat
        self.nlayer = nlayer
        self.nheads = nheads
        self.hidden_dim = hidden_dim

        self.metis_dim = int(metis_dim) if metis_dim is not None else 0
        self.intra = self.metis_dim > 0
        self.combine_dim = nclass if norm == "none" else hidden_dim

        self.feat_encoder = nn.Sequential(
            nn.Linear(nfeat, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nclass),
        )

        # for arxiv & penn
        self.linear_encoder = nn.Linear(nfeat, hidden_dim)
        self.classify = nn.Linear(hidden_dim, nclass)

        self.eig_encoder = SpectrumEncoder(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, nheads)

        self.mha_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.mha_dropout = nn.Dropout(tran_dropout)
        self.ffn_dropout = nn.Dropout(tran_dropout)
        self.mha = nn.MultiheadAttention(hidden_dim, nheads, tran_dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, hidden_dim, hidden_dim)

        self.feat_dp1 = nn.Dropout(feat_dropout)
        self.feat_dp2 = nn.Dropout(feat_dropout)

        nbases = nheads + 1 + self.metis_dim

        layer_dim = nclass if norm == "none" else hidden_dim
        self.layers = nn.ModuleList(
            [
                HierarchicalLayer(
                    nbases, layer_dim, prop_dropout, norm=norm, metis_dim=self.metis_dim
                )
                for _ in range(nlayer)
            ]
        )

        if self.intra:
            self.metis_filter = nn.Linear(1, self.combine_dim)
            nn.init.zeros_(self.metis_filter.weight)
            nn.init.ones_(self.metis_filter.bias)
        else:
            self.metis_filter = None

    def forward(self, e, u, x, metis_payload=None):
        N = e.size(0)
        ut = u.permute(1, 0)

        if self.norm == "none":
            h = self.feat_dp1(x)
            h = self.feat_encoder(h)
            h = self.feat_dp2(h)
        else:
            h = self.feat_dp1(x)
            h = self.linear_encoder(h)

        eig = self.eig_encoder(e)  # [N, d]

        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)

        new_e = self.decoder(eig)  # [N, m]

        if self.intra:
            if metis_payload is None:
                raise ValueError(
                    "SPHierarchicalModel configured with METIS support but no payload provided."
                )
            if not isinstance(metis_payload, tuple) or len(metis_payload) != 2:
                raise ValueError("METIS payload must be a (basis, eigenvalues) tuple.")
            metis_basis, metis_eigs = metis_payload
            metis_basis = metis_basis.to(h.device, dtype=h.dtype)
            metis_eigs = (
                metis_eigs.to(h.device, dtype=h.dtype)
                if metis_eigs is not None
                else None
            )
        else:
            metis_basis = None
            metis_eigs = None

        for conv in self.layers:
            base_feats = [h]
            utx = ut @ h

            for i in range(self.nheads):
                head_feat = u @ (new_e[:, i].unsqueeze(1) * utx)  # [N, d]
                base_feats.append(head_feat)

            if self.intra:
                metis_proj = metis_basis.t() @ h
                if metis_eigs is not None and self.metis_filter is not None:
                    filters = self.metis_filter(metis_eigs.unsqueeze(-1))
                    metis_proj = metis_proj * filters
                conv_payload = (metis_basis, metis_proj)
            else:
                conv_payload = None

            h = conv(base_feats, conv_payload)

        if self.norm == "none":
            return h
        else:
            h = self.feat_dp2(h)
            h = self.classify(h)
            return h
