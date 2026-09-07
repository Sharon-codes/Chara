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


def compute_dirichlet_energy(x, laplacian):
    """
    Computes Dirichlet energy E(x) = x^T L x along the manifold graph.
    
    Parameters:
    -----------
    x : array-like of shape (n_samples, n_features) or (n_features,)
        Patient transcriptomic profile(s).
    laplacian : 2D array-like of shape (n_features, n_features)
        Symmetric Graph Laplacian matrix.
        
    Returns:
    --------
    energy : float or 1D ndarray
        Dirichlet smoothness energy for each patient.
    """
    arr = np.asarray(x, dtype=np.float64)
    L = np.asarray(laplacian, dtype=np.float64)
    L = (L + L.T) / 2.0
    if arr.ndim == 1:
        return float(arr @ L @ arr)
    return np.sum((arr @ L) * arr, axis=1)

