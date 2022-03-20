from typing import Tuple

import numpy as np
import scipy.sparse as sparse
from edgeseraser.misc.typing import NpArrayEdges, NpArrayEdgesFloat


def construct_sp_matrices(
    weights: NpArrayEdgesFloat,
    edges: NpArrayEdges,
    num_vertices: int,
    is_directed: bool = True,
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if weights.shape[0] < num_vertices * (num_vertices - 1) / 1.2:
        w_adj = sparse.csr_matrix(
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
