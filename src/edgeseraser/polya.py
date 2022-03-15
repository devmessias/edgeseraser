import warnings
from typing import Optional

import numpy as np
import scipy.stats as stats
from scipy.special import betaln, gamma, gammaln

from edgeseraser.misc.backend import ig_erase, ig_extract, nx_erase, nx_extract
from edgeseraser.misc.matrix import construct_sp_matrices

warnings.simplefilter("ignore", FutureWarning)


def compute_polya_pdf(
    w: np.ndarray, n: np.ndarray, k, a: float = 0.0, approx: bool = False
) -> np.ndarray:
    """

    Args:
        w: np.array
            edge weights
        n: np.array
            number of samples
        k: np.array
        a: float (default: 0)
        approx: bool (default: False)
            if True, use approximation
    Returns:
        np.array

    """
    if a == 0.0:
        prob_success = 1.0 / k
        p = stats.binom.cdf(w, n, prob_success)
        return p

    a_inv = 1 / a
    b = (k - 1) * a_inv
    if approx:
        if a == 1.0:
            p = (1 - w / n) ** (k - 1)
        else:
            p = (
                1
                / gamma(a_inv)
                * ((1 - w / n) ** (b))
                * ((w * k / n * a) ** (a_inv - 1))
            )
    else:
        p = np.exp(
            gammaln(n + 1)
            + betaln(w + a_inv, n - w + b)
            - gammaln(w + 1)
            - gammaln(n - w + 1)
            - betaln(a_inv, b)
        )

    return p


def polya_cdf(weights, w_degree, degree, a, apt_lvl, eps=1e-20):
    """
    Args:
        weights: np.array
            edge weights
        w_degree: np.array
            edge weighted degrees
        degree: np.array
            vertex degrees
        a: float
        apt_lvl: int
        eps: float
    Returns:
        np.array:
            Probability values

    """
    k_high = degree > 1
    k = degree[k_high]
    w = weights[k_high]
    s = w_degree[k_high]
    scores = np.zeros_like(weights)
    if a == 0:
        p = compute_polya_pdf(w, s, k)
        return p

    a_inv = 1.0 / a
    b = (k - 1) * a_inv
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
        p[idx_true] = compute_polya_pdf(
            w[idx_true], s[idx_true], k[idx_true], a=a, approx=True
        )
    non_zero_idx = np.argwhere(~idx).flatten()
    non_zero_idx = [i for i in non_zero_idx if s[i] > 0 and k[i] > 1]
    if len(non_zero_idx) > 0:
        for i, i_idx in enumerate(non_zero_idx):
            si = s[i_idx]
            ki = k[i_idx]
            x = np.arange(0, int(w[i]))
            polya_pdf = compute_polya_pdf(x, si, ki, a)
            p[i_idx] = 1 - np.sum(polya_pdf)
    scores[k_high] = p
    return scores


def scores_generic_graph(
    num_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    a: float = 1,
    apt_lvl: int = 10,
    is_directed: bool = False,
    eps: float = 1e-20,
) -> np.ndarray:
    """Compute the probability for each edge using the Pólya-based method.

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

    calc_degree = lambda x, i: np.asarray(x.sum(axis=i)).flatten().astype(np.float64)
    ids_out = edges[:, 0]
    ids_in = edges[:, 1]
    s_out = calc_degree(w_adj, 1)[ids_out]
    s_in = calc_degree(w_adj, 0)[ids_in]
    k_out = calc_degree(adj, 1)[ids_out]
    k_in = calc_degree(adj, 0)[ids_in]

    if np.mod(weights, 1).sum() > eps:
        # non integer weights
        apt_lvl = 0
    p_in = polya_cdf(weights, s_in, k_in, a, apt_lvl)
    p_out = polya_cdf(weights, s_out, k_out, a, apt_lvl)
    p = np.minimum(p_in, p_out)
    return p


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


def filter_generic_graph(
    num_vertices: int,
    edges: np.ndarray,
    weights: np.ndarray,
    thresh: float = 0.4,
    a: float = 1,
    apt_lvl: int = 10,
    is_directed: bool = False,
    eps: float = 1e-20,
) -> np.ndarray:
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
        np.array:
            indices of edges to be erased

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
    return ids2erase


def filter_nx_graph(
    g,
    thresh: float = 0.5,
    field: Optional[str] = None,
    a: float = 2,
    apt_lvl: int = 10,
    remap_labels: bool = False,
) -> None:
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
    """
    edges, weights, num_vertices, opts = nx_extract(g, remap_labels, field)
    is_directed: bool = opts["is_directed"]
    ids2erase = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        is_directed=is_directed,
        a=a,
        apt_lvl=apt_lvl,
        thresh=thresh,
    )

    nx_erase(g, edges[ids2erase], opts)


def filter_ig_graph(
    g,
    thresh: float = 0.5,
    field: Optional[str] = None,
    a: float = 2,
    apt_lvl: int = 10,
    remap_labels: bool = False,
) -> None:
    """Filter edges from a networkx graph using the Pólya filter.

    Parameters:
        g: networkx.Graph
            graph to be filtered
        thresh: float
        a: float
            0 is the Binomial distribution,
            1 the filter will behave like the Disparity filter.
        apt_lvl: int

    """

    edges, weights, num_vertices, opts = ig_extract(g, field)
    is_directed: bool = opts["is_directed"]
    alphas = filter_generic_graph(
        num_vertices,
        edges,
        weights,
        is_directed=is_directed,
        a=a,
        apt_lvl=apt_lvl,
        thresh=thresh,
    )
    ids2erase = cond_edges2erase(alphas, thresh=thresh)
    ig_erase(g, ids2erase)
