import sys
import warnings

import numpy as np
import networkx as nx
import igraph as ig
from typing import Optional, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract
from edgeseraser.misc.matrix import construct_sp_matrices

warnings.simplefilter("ignore", FutureWarning)


def stick_break_scores(
    wdegree: np.ndarray, degree: np.ndarray, edges: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Calculate the stick-breaking scores for each edge.

    Args:
        wdegree: np.array
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
    st = weights[ids_d1] / wdegree[ids_d1]
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
    ids2erase = np.argwhere(alphas > thresh).flatten()
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
    wdegree_out = calc_degree(w_adj, 0)[iout]
    degree_out = calc_degree(adj, 0)[iout]
    wdegree_in = calc_degree(w_adj, 1)[iin]
    degree_in = calc_degree(adj, 1)[iin]
    if cond == "out":
        alphas = stick_break_scores(wdegree_out, degree_out, edges, weights)
    elif cond == "in":
        alphas = stick_break_scores(wdegree_in, degree_in, edges, weights)
    else:
        alphas_out = stick_break_scores(wdegree_out, degree_out, edges, weights)
        alphas_in = stick_break_scores(wdegree_in, degree_in, edges, weights)
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter edges from a graph using the disparity filter.
    (Dirichet proccess)

    Args:
        g: networkx.Graph
            graph to be filtered
        thresh: float
            Between 0 and 1.
        cond: str
            "out", "in", "both", "or"
        is_directed: bool
            if True, the graph is considered as directed
    Returns:
        (np.array, np.array)
        -  indices of edges to be erased
        -  alphas scores for each edge


    """
    alphas = scores_generic_graph(
        num_vertices, edges, weights, cond=cond, is_directed=is_directed
    )

    ids2erase = cond_edges2erase(alphas, thresh=thresh)
    return ids2erase, alphas


def filter_nx_graph(
    g: Union[nx.Graph, nx.DiGraph],
    thresh: float = 0.8,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: Optional[str] = None,
    remap_labels: bool = False,
    save_scores: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter edges from a networkx graph using the disparity filter.
    (Dirichet proccess)

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
            Between 0 and 1.
        cond: str
            "out", "in", "both", "or"
        field: str
            edge weight field
        remap_labels: bool
            if True, the labels of the graph will be remapped to consecutive integers
        save_scores: bool (default: False)
            if True, the scores of the edges will be saved in the graph attribute
    Returns:
        (np.array, np.array)
        -  indices of edges erased
        -  alphas scores for each edge

    """
    assert thresh > 0.0 and thresh < 1.0, "thresh must be between 0 and 1"
    edges, weights, num_vertices, opts = nx_extract(g, remap_labels, field)
    ids2erase, alphas = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        cond=cond,
        is_directed=opts["is_directed"],
        thresh=thresh,
    )
    if save_scores:
        nx.set_edge_attributes(g, {
            (u, v): {"alpha": a}
            for u, v, a in zip(edges[:, 0], edges[:, 1], alphas)
        })
    nx_erase(g, edges[ids2erase], opts)
    return ids2erase, alphas


def filter_ig_graph(
    g: ig.Graph,
    thresh: float = 0.8,
    cond: Literal["or", "both", "out", "in"] = "or",
    field: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter edges from a igraph instance using the disparity filter.
    (Dirichet proccess)

    Parameters:
        g: igraph.Graph
            graph to be filtered
        thresh: float
            Between 0 and 1.
        cond: str
            "out", "in", "both", "or"
        field: str or None
            field to use for edge weights
    Returns:
        (np.array, np.array)
        -  indices of edges erased
        -  alphas scores for each edge

    """
    assert thresh > 0.0 and thresh < 1.0, "thresh must be between 0 and 1"
    edges, weights, num_vertices, opts = ig_extract(g, field)
    ids2erase, alphas = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        cond=cond,
        is_directed=opts["is_directed"],
        thresh=thresh,
    )
    ig_erase(g, ids2erase)
    return ids2erase, alphas
