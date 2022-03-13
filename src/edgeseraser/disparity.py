import sys
import warnings

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
import scipy.sparse as sp  # type: ignore

from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract

warnings.simplefilter("ignore", FutureWarning)


def get_disparity_integral(norm_weight, degree):
    """
    calculate the significance (alpha) for the disparity filter
    """
    disparity_integral = ((1.0 - norm_weight) ** degree) / (
        (degree - 1.0) * (norm_weight - 1.0)
    )

    return disparity_integral


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

    st = w_degree[edges[:, 0]]
    degree = degree[edges[:, 0]]
    w_degree = w_degree[edges[:, 0]]

    ids_w0 = np.argwhere(st == 0)
    ids_d1 = np.argwhere(degree < 2)
    st[ids_w0] = 1
    degree[ids_d1] = 10
    st = weights / st
    st[st == 1.0] -= 10e-4
    alphas = 1.0 - (
        (degree - 1.0)
        * (get_disparity_integral(st, degree) - get_disparity_integral(0.0, degree))
    )

    alphas[ids_d1] = 0.0
    alphas[ids_w0] = 0.0
    return alphas


def cond_stick_edges2erase(alphas: np.ndarray, thresh: float = 0.1) -> np.ndarray:
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
    ids2erase = np.argwhere(alphas > thresh).flatten()
    return ids2erase


def filter_generic_graph(
    num_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    cond: Literal["or", "both", "out", "in"] = "or",
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
    w_adj = sp.csr_matrix((weights, edges.T), shape=(num_vertices, num_vertices))
    adj = sp.csr_matrix(
        (np.ones(edges.shape[0]), edges.T), shape=(num_vertices, num_vertices)
    )

    calc_degree = lambda x, i: np.asarray(x.sum(axis=i)).flatten().astype(np.float64)
    w_degree_out = calc_degree(w_adj, 0)
    degree_out = calc_degree(adj, 0)
    w_degree_in = calc_degree(w_adj, 1)
    degree_in = calc_degree(adj, 1)
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


def filter_nx_graph(
    g,
    thresh: float = 0.5,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: str = "weight",
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
    edges, weights, num_vertices, nodelabel2index = nx_extract(g, remap_labels, field)

    alphas = filter_generic_graph(num_vertices, edges, weights, cond=cond)
    ids2erase = cond_stick_edges2erase(alphas, thresh=thresh)
    nx_erase(g, edges[ids2erase], nodelabel2index)


def filter_ig_graph(
    g,
    thresh: float = 0.5,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: str = "weight",
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

    edges, weights, num_vertices = ig_extract(g, field)

    alphas = filter_generic_graph(num_vertices, edges, weights, cond=cond)
    ids2erase = cond_stick_edges2erase(alphas, thresh=thresh)
    ig_erase(g, ids2erase)
