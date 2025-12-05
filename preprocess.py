import os
import sys
import math
import time
import yaml
import metis
import nxmetis
import scipy as sp
import numpy as np
import networkx as nx
import torch
from contextlib import contextmanager
from numpy.linalg import eig, eigh
import argparse
from datasets import *
from utils import get_data_path


@contextmanager
def record_stage_time(stage_name: str, collector=None):
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        print(f"[preprocess] {stage_name} took {duration:.2f} seconds")
        if collector is not None:
            collector.append((stage_name, duration))


def build_metis_membership(a, num_parts=None):
    """Return a METIS-based membership matrix of shape [N, num_parts]."""

    if sp.sparse.issparse(a):
        try:
            graph = nx.from_scipy_sparse_array(a)
        except AttributeError:
            graph = nx.from_scipy_sparse_matrix(a)
    else:
        graph = nx.from_numpy_array(np.asarray(a))

    num_nodes = graph.number_of_nodes()

    if num_parts is None:
        raise ValueError("Please provide num_parts in config for metis partitioning")

    num_parts = max(1, min(num_nodes, num_parts))

    if num_parts == 1:
        membership = torch.ones((num_nodes, 1), dtype=torch.float32)
        membership /= math.sqrt(num_nodes)
        print(f"[preprocess] METIS part sizes: [{num_nodes}]")
        return membership, [list(range(num_nodes))]

    _, partitions = nxmetis.partition(G=graph, nparts=num_parts)
    print(f"[preprocess] METIS part sizes: {list(map(len, partitions))}")
    avg_size = sum(len(p) for p in partitions) / len(partitions)
    print(f"[preprocess] METIS average part size: {avg_size:.2f}")
    # _, partitions = metis.part_graph(graph=graph, nparts=num_parts)

    membership = torch.zeros((num_nodes, len(partitions)), dtype=torch.float32)

    for pid, nodes in enumerate(partitions):
        if not nodes:
            continue
        scale = 1.0 / math.sqrt(len(nodes))
        membership[list(nodes), pid] = scale

    return membership, partitions


def compute_block_spectral_basis(
    a,
    partitions,
    max_block_rank=4,
    drop_constant=True,
    tol=1e-5,
    return_eigenvalues=False,
):
    """Construct a block-diagonal spectral basis from METIS partitions.

    Each block receives a small number (``max_block_rank``) of eigenvectors of
    its normalized Laplacian. The constant component is optionally removed to
    avoid duplicating the global low-pass behaviour that already exists in the
    full-graph spectrum.
    """

    if max_block_rank <= 0:
        basis = torch.zeros((a.shape[0], 0), dtype=torch.float32)
        if return_eigenvalues:
            return basis, torch.zeros((0,), dtype=torch.float32)
        return basis

    if sp.sparse.issparse(a):
        adj_csr = a.tocsr()
    else:
        adj_csr = sp.sparse.csr_matrix(np.asarray(a))

    num_nodes = adj_csr.shape[0]
    block_vectors = []
    block_nodes = []
    block_eigs = []

    for nodes in partitions:
        block_size = len(nodes)
        if block_size == 0:
            continue

        target_rank = min(max_block_rank, block_size)
        if target_rank == 0:
            continue

        compute_rank = (
            target_rank + 1 if drop_constant and block_size > 1 else target_rank
        )

        sub_adj = adj_csr[nodes, :][:, nodes]
        deg = np.array(sub_adj.sum(axis=1)).flatten()
        deg[deg == 0.0] = 1.0
        deg_inv_sqrt = sp.sparse.diags(np.power(deg, -0.5))
        lap = sp.sparse.eye(block_size) - deg_inv_sqrt @ sub_adj @ deg_inv_sqrt

        try:
            if block_size <= compute_rank:
                eigvals, eigvecs = np.linalg.eigh(lap.toarray())
            else:
                eigvals, eigvecs = sp.sparse.linalg.eigsh(
                    lap, k=compute_rank, which="SM", tol=tol
                )
                order = np.argsort(eigvals)
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
        except Exception:
            eigvals, eigvecs = np.linalg.eigh(lap.toarray())

        eigvals = np.asarray(eigvals, dtype=np.float32)

        if drop_constant and block_size > 1:
            eigvecs = eigvecs[:, 1:]
            eigvals = eigvals[1:]

        eigvecs = eigvecs[:, :target_rank]
        eigvals = eigvals[:target_rank]

        if eigvecs.size == 0:
            continue

        eigvecs = torch.from_numpy(eigvecs.astype(np.float32))
        eigvecs -= eigvecs.mean(dim=0, keepdim=True)
        norms = torch.norm(eigvecs, dim=0, keepdim=True)
        valid = (norms > 1e-6).squeeze(0)
        if valid.ndim == 0:
            valid = valid.unsqueeze(0)
        if not torch.any(valid):
            continue
        eigvecs = eigvecs[:, valid]
        eigvecs = eigvecs / norms[:, valid]

        eigvals = torch.from_numpy(eigvals)[valid]

        block_vectors.append(eigvecs)
        block_nodes.append(nodes)
        block_eigs.append(eigvals)

    total_dim = sum(vec.shape[1] for vec in block_vectors)
    if total_dim == 0:
        basis = torch.zeros((num_nodes, 0), dtype=torch.float32)
        if return_eigenvalues:
            return basis, torch.zeros((0,), dtype=torch.float32)
        return basis

    basis = torch.zeros((num_nodes, total_dim), dtype=torch.float32)
    eigs = (
        torch.zeros((total_dim,), dtype=torch.float32) if return_eigenvalues else None
    )

    offset = 0
    for nodes, vecs, evals in zip(block_nodes, block_vectors, block_eigs):
        width = vecs.shape[1]
        if width == 0:
            continue
        basis[nodes, offset : offset + width] = vecs
        if eigs is not None:
            eigs[offset : offset + width] = evals
        offset += width

    if return_eigenvalues:
        return basis, eigs
    return basis


def subgraph_decomposition(
    adj,
    partitions,
    return_eigenvalues=True,
):
    """Construct a block-diagonal spectral basis from METIS partitions."""

    if sp.sparse.issparse(adj):
        adj_dense = adj.toarray()
    else:
        adj_dense = np.asarray(adj)

    num_nodes = adj_dense.shape[0]

    block_nodes = []
    block_vectors = []
    block_eigs = []
    total_dim = 0

    for nodes in partitions:
        block_size = len(nodes)
        if block_size == 0:
            continue

        # 取子图的邻接矩阵（dense）
        sub_adj = adj_dense[np.ix_(nodes, nodes)]
        e_block, u_block = eigen_decompositon(sub_adj)  # 返回 np.ndarray

        # 转成 torch
        e_block = torch.from_numpy(e_block.astype(np.float32))
        u_block = torch.from_numpy(u_block.astype(np.float32))

        block_nodes.append(nodes)
        block_eigs.append(e_block)
        block_vectors.append(u_block)
        total_dim += u_block.shape[1]

    if total_dim == 0:
        basis = torch.zeros((num_nodes, 0), dtype=torch.float32)
        if return_eigenvalues:
            return basis, torch.zeros((0,), dtype=torch.float32)
        return basis

    # 把各个 block 的谱基拼成一个 [N, total_dim] 矩阵
    basis = torch.zeros((num_nodes, total_dim), dtype=torch.float32)
    eigs = (
        torch.zeros((total_dim,), dtype=torch.float32) if return_eigenvalues else None
    )

    offset = 0
    for nodes, vecs, evals in zip(block_nodes, block_vectors, block_eigs):
        width = vecs.shape[1]
        if width == 0:
            continue
        basis[nodes, offset : offset + width] = vecs
        if eigs is not None:
            eigs[offset : offset + width] = evals
        offset += width

    if return_eigenvalues:
        return basis, eigs
    return basis


def subgraph_decomposition_large(
    adj,
    partitions,
    return_eigenvalues=True,
):
    """Construct block-diagonal spectral bases without densifying the full graph.

    Instead of returning a single gigantic dense matrix, we store the per-partition
    eigenvectors/eigenvalues so that downstream consumers can materialize the
    necessary slices lazily on the desired device. This dramatically reduces the
    immediate GPU memory requirements when each block keeps its full spectrum.
    """

    if sp.sparse.issparse(adj):
        adj_csr = adj.tocsr()
        num_nodes = adj_csr.shape[0]
    else:
        adj_dense = np.asarray(adj)
        num_nodes = adj_dense.shape[0]

    block_entries = []
    block_eigs = []
    total_dim = 0

    for nodes in partitions:
        block_size = len(nodes)
        if block_size == 0:
            continue

        # sub_adj = adj_dense[np.ix_(nodes, nodes)]
        if sp.sparse.issparse(adj):
            sub_adj = adj_csr[nodes, :][:, nodes].toarray()
        else:
            sub_adj = adj_dense[np.ix_(nodes, nodes)]

        e_block, u_block = eigen_decompositon(sub_adj)

        e_block = torch.from_numpy(e_block.astype(np.float32))
        u_block = torch.from_numpy(u_block.astype(np.float32))

        block_entries.append(
            {
                "nodes": torch.as_tensor(nodes, dtype=torch.long),
                "vectors": u_block,
            }
        )
        block_eigs.append(e_block)
        total_dim += u_block.shape[1]

    if total_dim == 0:
        empty_basis = {
            "type": "block",
            "num_nodes": num_nodes,
            "total_dim": 0,
            "blocks": [],
        }
        empty_eigs = torch.zeros((0,), dtype=torch.float32)
        if return_eigenvalues:
            return empty_basis, empty_eigs
        return empty_basis

    basis = {
        "type": "block",
        "num_nodes": num_nodes,
        "total_dim": total_dim,
        "blocks": block_entries,
    }

    eigs = torch.cat(block_eigs, dim=0) if return_eigenvalues else None

    if return_eigenvalues:
        return basis, eigs
    return basis


def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.0] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.0] = 1.0
    deg = np.diag(deg**-0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L


def eigen_decompositon(g, max_rank=None, which="SM", tol=1e-5, strategy="balanced"):
    """Return (optionally truncated) eigenpairs of the normalized Laplacian."""
    g = normalize_graph(g)

    # dataset = "pubmed"
    if max_rank is None or max_rank <= 0 or max_rank >= g.shape[0]:
        e, u = eigh(g)
        # visualize_eigenvalues(e, dataset=dataset)
        # sys.exit(0)
        return e, u

    # ``eigsh`` expects a sparse matrix input; convert once to avoid repeated
    # densification in upstream callers that already store dense adjacencies.
    g = sp.sparse.csr_matrix(g)
    max_valid_rank = max(0, g.shape[0] - 1)
    rank = min(int(max_rank), max_valid_rank)
    if rank <= 0:
        e, u = eigh(g)
        return e, u

    e, u = sp.sparse.linalg.eigsh(g, k=rank, which=which, tol=tol)
    order = np.argsort(e)
    e = e[order]
    u = u[:, order]
    return e, u


def empty_spectrum(num_nodes):
    """Create placeholder tensors when global spectra are disabled."""

    num_nodes = int(num_nodes)
    # e = torch.zeros(num_nodes, dtype=torch.float32)
    e = torch.zeros((0,), dtype=torch.float32)
    u = torch.zeros((num_nodes, 0), dtype=torch.float32)
    return e, u


def eig_dgl_adj_sparse(g, sm=0, lm=0):
    A = g.adj_external(scipy_fmt="csr")
    deg = np.array(A.sum(axis=0)).flatten()
    # deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0.0] = 1.0
    D_ = sp.sparse.diags(deg**-0.5)

    A_ = D_.dot(A.dot(D_))
    L_ = sp.sparse.eye(g.num_nodes()) - A_

    if sm > 0:
        e1, u1 = sp.sparse.linalg.eigsh(L_, k=sm, which="SM", tol=1e-5)
        e1, u1 = map(torch.FloatTensor, (e1, u1))

    if lm > 0:
        e2, u2 = sp.sparse.linalg.eigsh(L_, k=lm, which="LM", tol=1e-5)
        e2, u2 = map(torch.FloatTensor, (e2, u2))

    if sm > 0 and lm > 0:
        return torch.cat((e1, e2), dim=0), torch.cat((u1, u2), dim=1)
    elif sm > 0:
        return e1, u1
    elif lm > 0:
        return e2, u2
    else:
        pass


def generate_node_data(dataset, config):

    os.makedirs("processed", exist_ok=True)  # processed data dir
    out_path = get_data_path(dataset, config)

    inter = config["inter"]
    intra = config["intra"]
    num_parts = config.get("num_parts", None)

    inter_rank = config.get("inter_rank", None)
    if inter_rank is not None:
        inter_rank = int(inter_rank)
        print(
            f"[preprocess] Using inter_rank = {inter_rank} for global spectrum truncation"
        )

    core_stage_times = []

    def time_stage(stage_name: str):
        return record_stage_time(stage_name, collector=core_stage_times)

    if dataset in ["cora", "citeseer", "pubmed"]:

        adj, x, y = load_planetoid_data(dataset)

        if inter:
            with time_stage(f"Global spectral decomposition"):
                e, u = eigen_decompositon(adj, max_rank=inter_rank)
            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
        else:
            e, u = empty_spectrum(adj.shape[0])
            print(f"[preprocess] Skipping global spectrum for dataset: {dataset}")

        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        if intra:
            with time_stage(f"METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                local_basis, local_eigs = subgraph_decomposition(adj, partitions)

            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")

    elif dataset in ["photo", "computers"]:

        adj, x, y = load_amazon_data(dataset)

        if inter:
            with time_stage("Global spectral decomposition"):
                e, u = eigen_decompositon(adj, max_rank=inter_rank)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
        else:
            e, u = empty_spectrum(adj.shape[0])
            print(f"[preprocess] Skipping global spectrum for dataset: {dataset}")

        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        if intra:
            with time_stage("METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                local_basis, local_eigs = subgraph_decomposition(adj, partitions)

            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")

    elif dataset in ["chameleon", "squirrel", "actor"]:

        adj, x, y = load_heterophilous_data(dataset)

        if inter:
            with time_stage("Global spectral decomposition"):
                e, u = eigen_decompositon(adj, max_rank=inter_rank)

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
        else:
            e, u = empty_spectrum(adj.shape[0])
            print(f"[preprocess] Skipping global spectrum for dataset: {dataset}")

        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        if intra:
            with time_stage("METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                local_basis, local_eigs = subgraph_decomposition(adj, partitions)
            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")

    elif dataset in ["penn"]:

        g, x, y = load_fb100_data(dataset)

        sm = config["sm"]
        lm = config["lm"]

        if inter:
            with time_stage("Global spectral decomposition"):
                e, u = eig_dgl_adj_sparse(g, sm=sm, lm=lm)
        else:
            e, u = empty_spectrum(g.num_nodes())
            print(f"[preprocess] Skipping global spectrum for dataset: {dataset}")

        if intra:
            adj = g.adj_external(scipy_fmt="csr")
            with time_stage("METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                local_basis, local_eigs = subgraph_decomposition(adj, partitions)
                # local_basis, local_eigs = compute_block_spectral_basis(
                #     adj, partitions, max_block_rank=3, return_eigenvalues=True
                # )
            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")

    elif dataset in ["arxiv", "products"]:

        g, x, y = load_ogbn_data(dataset)
        sm = config["sm"]
        lm = config["lm"]

        with time_stage("Global spectral decomposition"):
            e, u = eig_dgl_adj_sparse(g, sm=sm, lm=lm)

        if intra:
            adj = g.adj_external(scipy_fmt="csr")
            with time_stage("METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                # local_basis, local_eigs = subgraph_decomposition_large(adj, partitions)
                local_basis, local_eigs = compute_block_spectral_basis(
                    adj, partitions, max_block_rank=2, return_eigenvalues=True
                )
            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")

    elif dataset in ["flickr", "reddit", "yelp"]:

        g, x, y = load_pyg_data(dataset)
        sm = config["sm"]
        lm = config["lm"]

        if inter:
            with time_stage("Global spectral decomposition"):
                e, u = eig_dgl_adj_sparse(g, sm=sm, lm=lm)
        else:
            e, u = empty_spectrum(g.num_nodes())
            print(f"[preprocess] Skipping global spectrum for dataset: {dataset}")

        if intra:
            adj = g.adj_external(scipy_fmt="csr")
            with time_stage("METIS partitioning and local spectra"):
                _, partitions = build_metis_membership(adj, num_parts=num_parts)
                local_basis, local_eigs = subgraph_decomposition_large(adj, partitions)
                # local_basis, local_eigs = subgraph_decomposition(adj, partitions)
                # local_basis, local_eigs = compute_block_spectral_basis(
                #     adj, partitions, max_block_rank=2, return_eigenvalues=True
                # )
            torch.save([e, u, x, y, local_basis, local_eigs], out_path)
            print(f"[preprocess] saved e, u, x, y, metis -> {out_path}")
        else:
            torch.save([e, u, x, y], out_path)
            print(f"[preprocess] saved e, u, x, y -> {out_path}")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    if core_stage_times:
        total_core_time = sum(duration for _, duration in core_stage_times)
        print(
            f"[preprocess] Total core preprocessing time: {total_core_time:.2f} seconds"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    args = parser.parse_args()
    dataset = args.dataset.lower()

    config_path = "config.yaml"
    config = yaml.safe_load(open(config_path))[dataset]
    print(f"[preprocess] config file: {config_path}\n{config}")

    print(f"[preprocess] Start preprocessing for dataset: {dataset} ...")
    generate_node_data(dataset, config)
    print(f"[preprocess] Finished preprocessing for dataset: {dataset}.")
