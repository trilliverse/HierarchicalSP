import os
import sys
import pickle as pkl
import dgl
import numpy as np
import pandas as pd
import scipy as sp
import torch
import networkx as nx
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from scipy import io
from sklearn.preprocessing import label_binarize
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.datasets import Flickr, Reddit2, Yelp


def feature_normalize(x: np.ndarray):
    """Row-normalize feature matrix."""
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def load_planetoid_data(dataset: str):
    """Load Planetoid datasets: Cora, Citeseer, Pubmed."""

    def _parse_index_file(filename: str):
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index

    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objects = []
    for name in names:
        with open(f"node_raw_data/{dataset}/ind.{dataset}.{name}", "rb") as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding="latin1"))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = _parse_index_file(
        f"node_raw_data/{dataset}/ind.{dataset}.test.index"
    )
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    y = np.vstack((ally, ty))
    y[test_idx_reorder, :] = y[test_idx_range, :]

    adj = adj.todense()
    x = features.todense()
    x = feature_normalize(x)

    return adj, x, y


def load_amazon_data(dataset: str):
    """Load Amazon datasets: Computers, Photo."""

    path = f"node_raw_data/amazon_electronics_{dataset}.npz"
    data = np.load(path, allow_pickle=True)

    adj = sp.sparse.csr_matrix(
        (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
        shape=data["adj_shape"],
    ).toarray()
    x = sp.sparse.csr_matrix(
        (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
        shape=data["attr_shape"],
    ).toarray()
    x = feature_normalize(x)
    y = data["labels"]

    return adj, x, y


def load_heterophilous_data(dataset: str):
    """Load Heterophilous datasets: Chameleon, Squirrel, Actor."""

    edge_df = pd.read_csv(f"node_raw_data/{dataset}/out1_graph_edges.txt", sep="\t")
    node_df = pd.read_csv(
        f"node_raw_data/{dataset}/out1_node_feature_label.txt", sep="\t"
    )
    feature = node_df[node_df.columns[1]]
    y = node_df[node_df.columns[2]].to_numpy()

    num_nodes = len(y)
    adj = np.zeros((num_nodes, num_nodes))

    source = edge_df[edge_df.columns[0]].to_numpy()
    target = edge_df[edge_df.columns[1]].to_numpy()

    adj[source, target] = 1.0
    adj[target, source] = 1.0

    if dataset == "actor":
        nfeat = 932
        x = np.zeros((len(y), nfeat))

        feature = feature.astype(str).str.split(",")
        for ind, feat in enumerate(feature):
            for ff in feat:
                x[ind, int(ff)] = 1.0

    else:
        feature = feature.astype(str).str.split(",")
        new_feat = []
        for feat in feature:
            new_feat.append([int(f) for f in feat])
        x = np.array(new_feat)

    x = feature_normalize(x)

    return adj, x, y


def load_fb100_data(dataset: str = "penn"):
    """Load FB100 dataset: Penn94."""
    mat = io.loadmat("node_raw_data/Penn94.mat")
    A = mat["A"]
    metadata = mat["local_info"]

    edge_index = A.nonzero()
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1

    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    label = torch.LongTensor(label)

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)

    return g, node_feat, label


def load_ogbn_data(dataset: str):
    """Load OGBN datasets: ogbn-arxiv, ogbn-products."""

    data = DglNodePropPredDataset(name=f"ogbn-{dataset}", root="node_raw_data/")
    g = data[0][0]
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)
    x = g.ndata["feat"]
    y = data[0][1]
    return g, x, y


def load_pyg_data(dataset: str):
    """Load PyG datasets: Flickr, Reddit2, Yelp."""

    dataset_map = {
        "flickr": Flickr,
        "reddit": Reddit2,
        "yelp": Yelp,
    }

    root = f"node_raw_data/{dataset}"
    dataset = dataset_map[dataset](root=root)
    data = dataset[0]

    x = data.x.float()
    y = data.y

    if y.dtype != torch.long:
        if y.dim() == 2 and y.size(1) == 1:
            y = y.view(-1).long()
        else:
            y = y.to(torch.long)

    g = dgl.graph((data.edge_index[0], data.edge_index[1]), num_nodes=data.num_nodes)
    g = dgl.add_reverse_edges(g)
    g = dgl.to_simple(g)

    return g, x, y


__all__ = [
    "load_planetoid_data",  # cora, citeseer, pubmed
    "load_amazon_data",  # computers, photo
    "load_heterophilous_data",  # chameleon, squirrel, actor
    "load_fb100_data",  # penn
    "load_ogbn_data",  # ogbn-arxiv, ogbn-products
    "load_pyg_data",  # flickr, reddit, yelp
]
