import sys
import warnings
from functools import lru_cache

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import numpy as np
import scipy.stats as stats
from edgeseraser.misc.typing import NpArrayEdgesBool, NpArrayEdgesFloat
from edgeseraser.polya_tools.numba_tools import (
    compute_polya_pdf,
    integer_cdf_lru_nb,
    integer_cdf_lru_nb_f,
    integer_cdf_nb,
)
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


@lru_cache(None)
def polya_cdf_lru_py_native(size, w, k, a):
    """Compute the Pólya-Urn for integer weights

    The results are cached using python's lru_cache.

    Args:
        size: int
            size of the distribution
        w: float
            weighted degree
        k: float
            degree
        a: float
    Returns:
        float:
            between 0 and 1

    """
    x = np.arange(0, size, dtype=np.float64)
    polya_pdf = compute_polya_pdf(x, w, k, a)
    return 1 - np.sum(polya_pdf)


def integer_cdf_lru_py_native(
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    weights: NpArrayEdgesFloat,
    a: float,
) -> NpArrayEdgesFloat:
    """Compute the prob of the integer weight distribution using numba JIT and parallelization.

    Args:
        wdegree: np.array
            edge weighted degrees
        degree: np.array
            vertex degrees
        weights: np.array
            edge weights
        a: float
    Returns:
        np.array:
            Probability values for each edge with integer weights

    """
    n = wdegree.shape[0]
    p = np.zeros(n, dtype=np.float64)
    for i in range(n):
        wi = int(weights[i])
        if wi < 2:
            p[i] = 0.0
            continue
        wd = wdegree[i]
        d = degree[i]

        p[i] = polya_cdf_lru_py_native(wi, wd, d, a)
    return p


def polya_cdf(
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    weights: NpArrayEdgesFloat,
    a: float,
    apt_lvl: float,
    eps: float = 1e-20,
    check_consistency: bool = False,
    optimization: Literal[
        "lru-nb", "lru-nb-szuszik", "lru-nbf", "lru-py-nb", "nb"
    ] = "lru-nb",
    cache_obj=None,
) -> NpArrayEdgesFloat:
    """
    Args:
        wdegree: np.array
            edge weighted degrees
        degree: np.array
            vertex degrees
        weights: np.array
            edge weights
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
        if optimization == "lru-nb" or optimization == "lru-nb-szuszik":
            if cache_obj is None:
                raise ValueError("Cache object is None")
            vals = integer_cdf_lru_nb(
                s[non_zero_idx],
                k[non_zero_idx],
                w[non_zero_idx],
                a=a,
                cache_obj=cache_obj,
            )
        elif optimization == "lru-nbf":
            vals = integer_cdf_lru_nb_f(
                s[non_zero_idx],
                k[non_zero_idx],
                w[non_zero_idx],
                a=a,
                cache_obj=cache_obj,
            )
        elif optimization == "lru-py-nb":
            vals = integer_cdf_lru_py_native(
                s[non_zero_idx], k[non_zero_idx], w[non_zero_idx], a
            )
        elif optimization == "nb":
            vals = integer_cdf_nb(s[non_zero_idx], k[non_zero_idx], w[non_zero_idx], a)
        p[non_zero_idx] = vals

    scores[k_high] = p
    return scores
