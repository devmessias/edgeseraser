import numpy as np
import scipy.sparse as sp


def construct_sp_matrices(
    weights: np.ndarray,
    edges: np.ndarray,
    num_vertices: int,
    is_directed: bool = True,
):
    if is_directed:
        w_adj = sp.csr_matrix(
            (weights, edges.T), shape=(num_vertices, num_vertices), dtype=np.float64
        )
    else:
        w_adj = sp.csr_matrix(
            (np.concatenate([weights, weights]), np.vstack((edges, edges[:, ::-1])).T),
            shape=(num_vertices, num_vertices),
            dtype=np.float64,
        )
    adj = w_adj.copy()
    adj[adj > 0] = 1
    return w_adj, adj
