import os
import time
import math
import random
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from torch_geometric.datasets import Flickr, Reddit2, Yelp


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
        if module.bias is not None:
            module.bias.data.zero_()


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False


def get_split(dataset, y, nclass, seed=0):

    if dataset == "arxiv":
        dataset = DglNodePropPredDataset("ogbn-arxiv", root="node_raw_data")
        split = dataset.get_idx_split()
        train, valid, test = split["train"], split["valid"], split["test"]
        return train, valid, test

    elif dataset == "penn":
        split = np.load("node_raw_data/fb100-Penn94-splits.npy", allow_pickle=True)[0]
        train, valid, test = split["train"], split["valid"], split["test"]
        return train, valid, test

    elif dataset in ["flickr", "reddit", "yelp"]:
        dataset_map = {
            "flickr": Flickr,
            "reddit": Reddit2,
            "yelp": Yelp,
        }
        root = os.path.join("node_raw_data", dataset)

        data = dataset_map[dataset](root=root)[0]

        def mask_to_indices(mask):
            if mask is None:
                return np.array([], dtype=np.int64)
            mask = mask if mask.dim() == 1 else mask.view(-1)
            return mask.nonzero(as_tuple=False).view(-1).cpu().numpy()

        train = mask_to_indices(getattr(data, "train_mask", None))
        valid_mask = getattr(data, "val_mask", getattr(data, "valid_mask", None))
        valid = mask_to_indices(valid_mask)
        test = mask_to_indices(getattr(data, "test_mask", None))

        return train, valid, test

    else:
        y = y.cpu()

        percls_trn = int(round(0.6 * len(y) / nclass))
        val_lb = int(round(0.2 * len(y)))

        indices = []
        for i in range(nclass):
            index = (y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0), device=index.device)]
            indices.append(index)

        train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        valid_index = rest_index[:val_lb]
        test_index = rest_index[val_lb:]

        return train_index, valid_index, test_index


def get_data_path(dataset, config):
    """
    Unified data path generator for both small/large graphs,
    covering inter/intra, inter_rank, METIS partitions, and spectral depth (sm/lm).
    """

    inter = config.get("inter", True)
    inter_rank = config.get("inter_rank", None)
    sm = config.get("sm", 0)
    lm = config.get("lm", 0)

    intra = config.get("intra", False)
    num_parts = config.get("num_parts", None)

    filename = [dataset]

    if inter:
        if sm > 0 or lm > 0:
            filename.append(f"inter_sm{sm}_lm{lm}")
        else:
            if inter_rank is None:
                filename.append("interN")
            else:
                filename.append(f"inter{inter_rank}")
    else:
        filename.append("inter0")

    if intra:
        assert (
            num_parts is not None
        ), "num_parts must be specified for intra-partitioned."
        filename.append(f"intra{num_parts}")
    else:
        filename.append("intra0")

    filename = "_".join(filename) + ".pt"

    # Check for the environment and set the base directory accordingly
    base_dir = "processed"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, filename)
