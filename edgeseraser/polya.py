import warnings
from typing import Optional, Tuple, Union

import igraph as ig
import networkx as nx
import numpy as np
import scipy.stats as stats
from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract
from edgeseraser.misc.fast_math import nbbetaln, nbgammaln
from edgeseraser.misc.matrix import construct_sp_matrices
from edgeseraser.misc.typing import (
    NpArrayEdges,
    NpArrayEdgesBool,
    NpArrayEdgesFloat,
    NpArrayEdgesIds,
)
from numba import njit
from scipy.sparse import csr_matrix
from scipy.special import gamma

warnings.simplefilter("ignore", FutureWarning)


def compute_polya_pdf_approx(
    w: NpArrayEdgesFloat, n: NpArrayEdgesFloat, k: NpArrayEdgesFloat, a: float = 0.0
) -> NpArrayEdgesFloat:
    """

    Args:
        w: np.array
            edge weights
        n: np.array
        k: np.array
            degree
        a: float (default: 0)
    Returns:
        np.array

    """
    if a == 0.0:
        prob_success: NpArrayEdgesFloat = 1.0 / k
        p: NpArrayEdgesFloat = stats.binom.cdf(w, n, prob_success)
        return p
    a_inv = 1 / a
    b = (k - 1) * a_inv
    if a == 1.0:
        p = (1 - w / n) ** (k - 1)
    else:
        p = 1 / gamma(a_inv) * ((1 - w / n) ** (b)) * ((w * k / n * a) ** (a_inv - 1))

    return p


@njit(
    nogil=True,
)
def compute_polya_pdf(
    s, w: NpArrayEdgesFloat, n: NpArrayEdgesFloat, k: NpArrayEdgesFloat, a: float
) -> NpArrayEdgesFloat:
    """

    Args:
        w: np.array
            edge weights
        n: np.array
            number of samples
        k: np.array
            degree
        a: float (default: 0)
    Returns:
        np.array

    """
    a_inv = 1 / a
    b = (k - 1) * a_inv
    ones = np.ones(s, dtype=np.float64)
    p: NpArrayEdgesFloat = np.exp(
        nbgammaln(n + ones, s)
        + nbbetaln(w + a_inv, n - w + b, s)
        - nbgammaln(w + ones, s)
        - nbgammaln(n - w + ones, s)
        - nbbetaln(a_inv * ones, b * ones, s)
    )

    return p


@njit(fastmath=True, nogil=True, error_model="numpy")
def integer_cdf(
    n: int,
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    a: float,
    weights: NpArrayEdgesFloat,
) -> NpArrayEdgesFloat:
    """Compute the prob of the integer weight distribution using numba JIT and parallelization.

    Args:
        wdegree: np.array
            edge weighted degrees
        degree: np.array
            vertex degrees
        a: float
        weights: np.array
            edge weights
    Returns:
        np.array:
            Probability values for each edge with integer weights

    """
    p = np.zeros(n, dtype=np.float64)
    for i in range(n):
        wi = int(weights[i])
        if wi < 2:
            p[i] = 0.0
            continue
        x = np.arange(0, wi)
        polya_pdf = compute_polya_pdf(wi, x, wdegree[i], degree[i], a)
        p[i] = 1 - np.sum(polya_pdf)
    return p


def polya_cdf(
    weights: NpArrayEdgesFloat,
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    a: float,
    apt_lvl: float,
    eps: float = 1e-20,
    check_consistency: bool = False,
) -> NpArrayEdgesFloat:
    """
    Args:
        weights: np.array
            edge weights
        wdegree: np.array
            edge weighted degrees
        degree: np.array
            vertex degrees
        a: float
        apt_lvl: int
        eps: float
        check_consistency: bool (default: False)
    Returns:
        np.array:
            Probability values for each edge

    """
    k_high: NpArrayEdgesBool = degree > 1
    k = degree[k_high]
    w = weights[k_high]
    s = wdegree[k_high]
    scores = np.zeros_like(weights)
    if a == 0:
        p = compute_polya_pdf_approx(w, s, k)
        return p

    a_inv = 1.0 / a
    b = (k - 1) * a_inv
    if check_consistency:
        diff_strength = s - w
        assert np.all(diff_strength >= 0)
    idx = (
        (s - w >= apt_lvl * (b + 1))
        * (w >= apt_lvl * np.maximum(a_inv, 1))
        * (s >= apt_lvl * np.maximum(k * a_inv, 1))
        * (k >= apt_lvl * (a - 1 + eps))
    )
    p = np.zeros(w.shape[0])
    idx_true = np.argwhere(idx).flatten()
    if len(idx_true) > 0:
        p[idx_true] = compute_polya_pdf_approx(
            w[idx_true], s[idx_true], k[idx_true], a=a
        )
    non_zero_idx = np.argwhere(~idx).flatten()
    non_zero_idx = non_zero_idx[(s[~idx] > 0) & (k[~idx] > 1)]
    if len(non_zero_idx) > 0:
        n = len(non_zero_idx)
        p[non_zero_idx] = integer_cdf(
            n, s[non_zero_idx], k[non_zero_idx], a, w[non_zero_idx]
        )

    scores[k_high] = p
    return scores


def scores_generic_graph(
    num_vertices: int,
    edges: NpArrayEdges,
    weights: NpArrayEdgesFloat,
    a: float = 1,
    apt_lvl: int = 10,
    is_directed: bool = False,
    eps: float = 1e-20,
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
    p_in = polya_cdf(weights, wdegree_in, degree_in, a, apt_lvl)
    p_out = polya_cdf(weights, wdegree_out, degree_out, a, apt_lvl)
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
) -> Tuple[NpArrayEdgesIds, NpArrayEdgesFloat]:
    """Filter edges from a networkx graph using the Pólya filter.

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
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
) -> Tuple[NpArrayEdgesIds, NpArrayEdgesFloat]:
    """Filter edges from a networkx graph using the Pólya filter.

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
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
    )
    ig_erase(g, ids2erase)
    return ids2erase, probs
