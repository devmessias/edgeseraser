import numpy as np
import scipy.sparse as sp


def construct_sp_matrices(
    weights: np.ndarray,
    edges: np.ndarray,
    num_vertices: int,
    is_directed: bool = True,
):
    if  weights.shape[0] < num_vertices * (num_vertices - 1)/1.2:
        w_adj = sp.csr_matrix(
            (weights, edges.T), shape=(num_vertices, num_vertices), dtype=np.float64
        )
    else:
        w_adj = np.zeros((num_vertices, num_vertices), dtype=np.float64)
        w_adj[edges[:, 0], edges[:, 1]] = weights
    
    if not is_directed:
        w_adj = w_adj + w_adj.T
    adj = w_adj.copy()
    adj[adj > 0] = 1
    return w_adj, adj
