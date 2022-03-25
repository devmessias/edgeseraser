import sys
import warnings

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Optional, Tuple, Union

import igraph as ig
import networkx as nx
import numpy as np
from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract
from edgeseraser.misc.fast_math import NbGammaLnCache
from edgeseraser.misc.matrix import construct_sp_matrices
from edgeseraser.misc.typing import NpArrayEdges, NpArrayEdgesFloat, NpArrayEdgesIds
from edgeseraser.polya_tools.numba_tools import (
    NbComputePolyaCacheDict,
    NbComputePolyaCacheSzuszik,
)
from edgeseraser.polya_tools.statistics import polya_cdf
from scipy.sparse import csr_matrix

warnings.simplefilter("ignore", FutureWarning)

OptionsForCache = Literal["lru-nb", "lru-nb-szuszik", "lru-nbf", "lru-py-nb", "nb"]


def scores_generic_graph(
    num_vertices: int,
    edges: NpArrayEdges,
    weights: NpArrayEdgesFloat,
    a: float = 1,
    apt_lvl: int = 10,
    is_directed: bool = False,
    eps: float = 1e-20,
    optimization: OptionsForCache = "lru-nb-szuszik",
) -> NpArrayEdgesFloat:
    """Compute the probability for each edge using the Pólya-based method for
    a generic weighted graph.

    Args:
        num_vertices: int
            number of vertices
        edges: np.array
            edges
        weights: np.array
            edge weights
        a: float
        apt_lvl: int
        is_directed: bool
        eps: float
        optimization: OptionsForCache

    Returns:
        np.array:
            edge scores. Probability values

    """
    w_adj, adj = construct_sp_matrices(
        weights, edges, num_vertices, is_directed=is_directed
    )

    def calc_degree(x: csr_matrix, i: int) -> NpArrayEdgesFloat:
        return np.asarray(x.sum(axis=i)).flatten()

    ids_out = edges[:, 0]
    ids_in = edges[:, 1]
    wdegree_out = calc_degree(w_adj, 1)[ids_out]
    wdegree_in = calc_degree(w_adj, 0)[ids_in]
    degree_out = calc_degree(adj, 1)[ids_out]
    degree_in = calc_degree(adj, 0)[ids_in]

    if np.mod(weights, 1).sum() > eps:
        # non integer weights
        apt_lvl = 0
    if optimization == "lru-nb":
        cache_obj = NbComputePolyaCacheDict()
    elif optimization == "lru-nb-szuszik":
        cache_obj = NbComputePolyaCacheSzuszik()
    elif optimization == "lru-nbf":
        cache_obj = NbGammaLnCache()
    else:
        cache_obj = None
    p_in = polya_cdf(
        wdegree_in,
        degree_in,
        weights,
        a,
        apt_lvl,
        optimization=optimization,
        cache_obj=cache_obj,
    )
    p_out = polya_cdf(
        wdegree_out,
        degree_out,
        weights,
        a,
        apt_lvl,
        optimization=optimization,
        cache_obj=cache_obj,
    )

    p: NpArrayEdgesFloat = np.minimum(p_in, p_out)
    return p


def cond_edges2erase(alphas: NpArrayEdgesFloat, thresh: float = 0.1) -> NpArrayEdgesIds:
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
    ids2erase: NpArrayEdgesIds = np.argwhere(alphas > thresh).flatten().astype("int64")
    return ids2erase


def filter_generic_graph(
    num_vertices: int,
    edges: NpArrayEdges,
    weights: NpArrayEdgesFloat,
    thresh: float = 0.4,
    a: float = 1,
    apt_lvl: int = 10,
    is_directed: bool = False,
    eps: float = 1e-20,
    optimization: OptionsForCache = "lru-nb-szuszik",
) -> Tuple[NpArrayEdgesIds, NpArrayEdgesFloat]:
    """Filter the graph using the Pólya-based method.

    Args:
        num_vertices: int
            number of vertices
        edges: np.array
            edges
        weights: np.array
            edge weights
        thresh: float
        a: float
        apt_lvl: int
        is_directed: bool
        eps: float

    Returns:
        (np.array, np.array)
        -  indices of edges to be erased
        -  probability for each edge

    """
    p = scores_generic_graph(
        num_vertices,
        edges,
        weights,
        a=a,
        apt_lvl=apt_lvl,
        is_directed=is_directed,
        eps=eps,
        optimization=optimization,
    )

    ids2erase = cond_edges2erase(p, thresh=thresh)
    return ids2erase, p


def filter_nx_graph(
    g: Union[nx.Graph, nx.DiGraph],
    thresh: float = 0.5,
    field: Optional[str] = None,
    a: float = 2,
    apt_lvl: int = 10,
    remap_labels: bool = False,
    save_scores: bool = False,
    optimization: OptionsForCache = "lru-nb-szuszik",
) -> Tuple[NpArrayEdgesIds, NpArrayEdgesFloat]:
    """Filter edges from a networkx graph using the Pólya-Urn filter.

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
        field: str
        a: float
            0 is the Binomial distribution,
            1 the filter will behave like the Disparity filter.
        apt_lvl: int
        remap_labels: bool
            If True, the labels of the nodes are remapped to consecutive integers.
        save_scores: bool (default: False)
            If True, the scores of the edges are saved in the graph.

    Returns:
        (np.array, np.array)
        -  indices of edges erased
        -  probability for each edge

    """
    edges, weights, num_vertices, opts = nx_extract(g, remap_labels, field)
    is_directed: bool = opts["is_directed"]
    ids2erase, probs = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        is_directed=is_directed,
        a=a,
        apt_lvl=apt_lvl,
        thresh=thresh,
        optimization=optimization,
    )
    if save_scores:
        nx.set_edge_attributes(
            g,
            {
                (u, v): {"prob": prob}
                for u, v, prob in zip(edges[:, 0], edges[:, 1], probs)
            },
        )

    nx_erase(g, edges[ids2erase], opts)
    return ids2erase, probs


def filter_ig_graph(
    g: ig.Graph,
    thresh: float = 0.5,
    field: Optional[str] = None,
    a: float = 2,
    apt_lvl: int = 10,
    optimization: OptionsForCache = "lru-nb-szuszik",
) -> Tuple[NpArrayEdgesIds, NpArrayEdgesFloat]:
    """Filter edges from a igraph using the Pólya-Urn filter.

    Parameters:
        g: ig.Graph
            graph to be filtered
        thresh: float
        field: str
        a: float
            0 is the Binomial distribution,
            1 the filter will behave like the Disparity filter.
        apt_lvl: int

     Return:
        (np.array, np.array)
        -  indices of edges erased
        -  probability for each edge
    """

    edges, weights, num_vertices, opts = ig_extract(g, field)
    is_directed: bool = opts["is_directed"]
    ids2erase, probs = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        is_directed=is_directed,
        a=a,
        apt_lvl=apt_lvl,
        thresh=thresh,
        optimization=optimization,
    )
    ig_erase(g, ids2erase)
    return ids2erase, probs
