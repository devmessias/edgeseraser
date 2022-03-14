import sys
import warnings
from typing import Optional

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np

from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract
from edgeseraser.misc.matrix import construct_sp_matrices

warnings.simplefilter("ignore", FutureWarning)


def stick_break_scores(
    w_degree: np.ndarray, degree: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Calculate the stick-breaking scores for each edge.

    Args:
        w_degree: np.array
            edge weighted degree
        degree: np.array
            degree of each vertex
        edges: np.array
            edges of the graph
        weights: np.array
            edge weights

    Returns:
        np.array:
            **alphas** stick-breaking scores for each edge
    """
    alphas = np.ones(edges.shape[0])
    ids_d1 = degree > 1
    st = weights[ids_d1] / w_degree[ids_d1]
    assert np.all(st <= 1)
    alphas[ids_d1] = (1 - st) ** (degree[ids_d1] - 1)
    return alphas


def cond_edges2erase(alphas: np.ndarray, thresh: float = 0.1) -> np.ndarray:
    """
    Args:
        alphas: np.array
            edge scores
        thresh: float
            Between 0 and 1.
    Returns:
        np.array:
            indices of edges to be erased

    """
    ids2erase = np.argwhere(alphas < thresh).flatten()
    return ids2erase


def scores_generic_graph(
    num_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    cond: Literal["or", "both", "out", "in"] = "or",
    is_directed: bool = False,
) -> np.ndarray:
    """
    Args:
        num_vertices: int
            number ofvertices
        edges: np.array
            edges
        weights: np.array
            edge weights
        cond: str
            "out", "in", "both", "or"
    Returns:
        np.array:
            **alphas** edge scores

    """
    w_adj, adj = construct_sp_matrices(
        weights, edges, num_vertices, is_directed=is_directed
    )
    calc_degree = lambda x, i: np.asarray(x.sum(axis=i)).flatten().astype(np.float64)
    iin = edges[:, 1]
    iout = edges[:, 0]
    w_degree_out = calc_degree(w_adj, 0)[iout]
    degree_out = calc_degree(adj, 0)[iout]
    w_degree_in = calc_degree(w_adj, 1)[iin]
    degree_in = calc_degree(adj, 1)[iin]
    if cond == "out":
        alphas = stick_break_scores(w_degree_out, degree_out, edges, weights)
    elif cond == "in":
        alphas = stick_break_scores(w_degree_in, degree_in, edges, weights)
    else:
        alphas_out = stick_break_scores(w_degree_out, degree_out, edges, weights)
        alphas_in = stick_break_scores(w_degree_in, degree_in, edges, weights)
        if cond == "both":
            alphas = np.maximum(alphas_out, alphas_in)
        elif cond == "or":
            alphas = np.minimum(alphas_out, alphas_in)

    return alphas


def filter_generic_graph(
    num_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    thresh: float = 0.8,
    cond: Literal["or", "both", "out", "in"] = "or",
    is_directed: bool = False,
):

    alphas = scores_generic_graph(
        num_vertices, edges, weights, cond=cond, is_directed=is_directed
    )

    ids2erase = cond_edges2erase(alphas, thresh=thresh)
    return ids2erase


def filter_nx_graph(
    g,
    thresh: float = 0.8,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: Optional[str] = None,
    remap_labels: bool = False,
) -> None:
    """Filter edges from a networkx graph using the disparity filter.
    (Dirichet proccess)

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
            Between 0 and 1.
        cond: str
            "out", "in", "both", "or"

    """
    assert thresh > 0.0 and thresh < 1.0, "thresh must be between 0 and 1"
    edges, weights, num_vertices, opts = nx_extract(g, remap_labels, field)
    is_directed = g.is_directed()
    ids2erase = filter_generic_graph(
        num_vertices, edges, weights, cond=cond, is_directed=is_directed, thresh=thresh
    )
    nx_erase(g, edges[ids2erase], opts)


def filter_ig_graph(
    g,
    thresh: float = 0.8,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: Optional[str] = None,
) -> None:
    """Filter edges from a igraph instance using the disparity filter.
    (Dirichet proccess)

    Parameters:
        g: igraph.Graph
            graph to be filtered
        thresh: float
            Between 0 and 1.
        cond: str
            "out", "in", "both", "or"

    """
    assert thresh > 0.0 and thresh < 1.0, "thresh must be between 0 and 1"

    edges, weights, num_vertices, opts = ig_extract(g, field)
    is_directed = g.is_directed()
    ids2erase = filter_generic_graph(
        num_vertices, edges, weights, cond=cond, is_directed=is_directed, thresh=thresh
    )
    ig_erase(g, ids2erase)
