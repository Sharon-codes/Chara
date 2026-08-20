"""Thermodynamic graph and heat-kernel utilities."""
import numpy as np

def laplacian_from_edges(edges, nodes, source="source", target="target", weight="weight"):
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for row in edges.to_dict("records"):
        a, b = row[source], row[target]
        if a in index and b in index and a != b:
            w = float(row.get(weight, 1.0))
            adjacency[index[a], index[b]] += w
            adjacency[index[b], index[a]] += w
    return np.diag(adjacency.sum(axis=1)) - adjacency

def heat_kernel(laplacian, diffusion_time=0.1):
    values, vectors = np.linalg.eigh((laplacian + laplacian.T) / 2.0)
    return (vectors * np.exp(-diffusion_time * np.clip(values, 0, None))) @ vectors.T

def exponential_chara_laplacian(adjacency, edge_variance, tau=0.5):
    mask = (adjacency > 0) & ~np.eye(adjacency.shape[0], dtype=bool)
    values = edge_variance[mask]
    z = np.zeros_like(edge_variance, dtype=float)
    if values.size:
        z[mask] = (values - values.mean()) / (values.std() + 1e-12)
    weighted = adjacency * np.exp(tau * z)
    weighted = (weighted + weighted.T) / 2.0
    np.fill_diagonal(weighted, 0.0)
    return np.diag(weighted.sum(axis=1)) - weighted
