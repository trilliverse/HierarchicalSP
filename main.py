import time
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from model import SPHierarchicalModel
from utils import (
    count_parameters,
    init_params,
    seed_everything,
    get_split,
    get_data_path,
)


def main_worker(args, config):
    print(args, config)
    seed_everything(args.seed)
    device = "cuda:{}".format(args.cuda)
    torch.cuda.set_device(device)

    epoch = config["epoch"]
    patience = config.get("patience", 200)
    lr = config["lr"]
    weight_decay = config["weight_decay"]
    nclass = config["nclass"]
    nlayer = config["nlayer"]
    hidden_dim = config["hidden_dim"]
    num_heads = config["num_heads"]
    tran_dropout = config["tran_dropout"]
    feat_dropout = config["feat_dropout"]
    prop_dropout = config["prop_dropout"]
    norm = config["norm"]
    intra = config.get("intra", False)
    num_parts = config.get("num_parts", None)

    metis_basis = None
    metis_eigs = None
    metis_dim = 0

    path = get_data_path(args.dataset, config)
    loaded = torch.load(path)
    print(f"[load] {path}")

    if intra:
        if len(loaded) == 6:
            e, u, x, y, metis_basis, metis_eigs = loaded
        else:
            e, u, x, y, metis_basis = loaded
            metis_eigs = None
    else:
        e, u, x, y = loaded
        metis_basis = None
        metis_eigs = None

    e, u, x, y = e.cuda(), u.cuda(), x.cuda(), y.cuda()
    if intra and metis_basis is not None:
        metis_basis = metis_basis.cuda()
        metis_eigs = metis_eigs.cuda() if metis_eigs is not None else None
        metis_dim = metis_basis.size(1)

    if len(y.size()) > 1:
        if y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        else:
            y = y.view(-1)

    train, valid, test = get_split(args.dataset, y, nclass, args.seed)
    train, valid, test = map(torch.LongTensor, (train, valid, test))
    train, valid, test = train.cuda(), valid.cuda(), test.cuda()

    nfeat = x.size(1)
    net = SPHierarchicalModel(
        nclass,
        nfeat,
        nlayer,
        hidden_dim,
        num_heads,
        tran_dropout,
        feat_dropout,
        prop_dropout,
        norm,
        metis_dim=metis_dim,
    ).cuda()
    net.apply(init_params)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"[model] Total parameters: {count_parameters(net)}")

    res = []
    min_loss = 100.0
    counter = 0
    evaluation = torchmetrics.Accuracy(task="multiclass", num_classes=nclass)

    metis_payload = None
    if metis_basis is not None:
        metis_payload = (metis_basis, metis_eigs)

    epoch_times = []
    allocated_memory_vals = []
    peak_memory_vals = []

    for idx in range(epoch):

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()

        net.train()
        optimizer.zero_grad()
        logits = net(e, u, x, metis_payload)

        loss = F.cross_entropy(logits[train], y[train])

        loss.backward()
        optimizer.step()

        net.eval()
        logits = net(e, u, x, metis_payload)

        val_loss = F.cross_entropy(logits[valid], y[valid]).item()

        val_acc = evaluation(logits[valid].cpu(), y[valid].cpu()).item()
        test_acc = evaluation(logits[test].cpu(), y[test].cpu()).item()
        res.append([val_loss, val_acc, test_acc])

        # epoch time in milliseconds
        epoch_time = (time.time() - epoch_start) * 1000.0

        # allocated memory in GB
        allocated_memory = (
            torch.cuda.memory_allocated() / 1024.0 / 1024.0 / 1024.0
            if torch.cuda.is_available()
            else 0
        )

        # peak memory in GB
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
            if torch.cuda.is_available()
            else 0
        )

        epoch_times.append(epoch_time)
        allocated_memory_vals.append(allocated_memory)
        peak_memory_vals.append(peak_memory)

        print(
            f"[{idx:04d}] "
            f"ValLoss: {val_loss:.6f} "
            f"ValAcc: {val_acc:.6f} "
            f"TestAcc: {test_acc:.6f} "
            f"Time: {epoch_time:.4f}ms "
            f"AllocMem: {allocated_memory:.2f}GB "
            f"PeakMem: {peak_memory:.2f}GB"
        )

        if val_loss < min_loss:
            min_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print(f"[early-stop] patience {patience} reached.")
            break

    max_acc1 = sorted(res, key=lambda x: x[0], reverse=False)[0][-1]
    max_acc2 = sorted(res, key=lambda x: x[1], reverse=True)[0][-1]

    avg_epoch_time = np.mean(epoch_times)
    allocated_memory_usage = np.max(allocated_memory_vals)
    peak_memory_usage = np.max(peak_memory_vals)

    return max_acc1, max_acc2, avg_epoch_time, allocated_memory_usage, peak_memory_usage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--dataset", default="cora")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    config_path = f"config.yaml"
    config = yaml.safe_load(open(config_path))[dataset]
    print(f"[config] config file: {config_path})")
    print(f"[config] {dataset} config: {config}")

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config["seeds"] = seeds

    acc1_list, acc2_list = [], []
    time_list = []
    alloc_mem_list, peak_mem_list = [], []

    for seed in seeds:
        args.seed = seed
        metrics = main_worker(args, config)
        acc1, acc2, avg_time, alloc_mem, peak_mem = metrics
        acc1, acc2 = acc1 * 100.0, acc2 * 100.0
        acc1_list.append(acc1)
        acc2_list.append(acc2)
        time_list.append(avg_time)
        alloc_mem_list.append(alloc_mem)
        peak_mem_list.append(peak_mem)

        print(
            f"[Seed {seed}] "
            f"Acc1: {acc1:.2f}%, Acc2: {acc2:.2f}% "
            f"AvgEpochTime: {avg_time:.4f}ms "
            f"AllocMem: {alloc_mem:.2f}GB "
            f"PeakMem: {peak_mem:.2f}GB"
        )

    print(config)

    print(f"Summary Results over {len(seeds)} runs of dataset: {args.dataset}")
    print(
        f"Acc1: {np.mean(acc1_list):.2f} ± {np.std(acc1_list):.2f}"
        f" (min: {np.min(acc1_list):.2f}, max: {np.max(acc1_list):.2f})"
    )
    print(
        f"Acc2: {np.mean(acc2_list):.2f} ± {np.std(acc2_list):.2f}"
        f" (min: {np.min(acc2_list):.2f}, max: {np.max(acc2_list):.2f})"
    )
    print(
        f"AvgEpochTime: {np.mean(time_list):.4f} ± {np.std(time_list):.4f}ms"
        f" (min: {np.min(time_list):.4f}ms, max: {np.max(time_list):.4f}ms)"
    )
    print(
        f"AllocMem: {np.mean(alloc_mem_list):.2f} ± {np.std(alloc_mem_list):.2f}GB"
        f" (min: {np.min(alloc_mem_list):.2f}GB, max: {np.max(alloc_mem_list):.2f}GB)"
    )
    print(
        f"PeakMem: {np.mean(peak_mem_list):.2f} ± {np.std(peak_mem_list):.2f}GB"
        f" (min: {np.min(peak_mem_list):.2f}GB, max: {np.max(peak_mem_list):.2f}GB)"
    )
