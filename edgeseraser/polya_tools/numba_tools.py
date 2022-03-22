import numpy as np
from edgeseraser.misc.fast_math import nbbetaln_parallel, nbgammaln_parallel
from edgeseraser.misc.typing import NpArrayEdgesFloat
from numba import njit
from numba.experimental import jitclass
from numba.typed import Dict as nb_Dict
from numba.types import DictType as nb_DictType
from numba.types import float64 as nb_float64
from numba.types import int64 as nb_int64


@njit(
    nb_float64[:](nb_float64[:], nb_float64, nb_float64, nb_float64),
    nogil=True,
)
def compute_polya_pdf(
    x: NpArrayEdgesFloat, w: NpArrayEdgesFloat, k: NpArrayEdgesFloat, a: float
) -> NpArrayEdgesFloat:
    """

    Args:
        x: np.array
            array of integer weights
        w: float
            weighted degree
        k: float
            degree
        a: float
    Returns:
        np.array

    """
    a_inv = 1.0 / a
    b = (k - 1.0) * a_inv
    ones = np.ones(x.shape[0], dtype=np.float64)
    p: NpArrayEdgesFloat = np.exp(
        nbgammaln_parallel(w + ones)
        + nbbetaln_parallel(x + a_inv, w - x + b)
        - nbgammaln_parallel(x + ones)
        - nbgammaln_parallel(w - x + ones)
        - nbbetaln_parallel(a_inv * ones, b * ones)
    )

    return p


@njit(nb_int64(nb_int64, nb_int64))
def szuszik(a, b):
    """
    Szuszik's pairing algorithm is a map from a pair of natural numbers to a unique
    natural number.
        $\\mathcal{N}\times\\mathcal{N}\\mapsto N$.

    Args:
        a: int
            first integer
        b: int
            second integer
    Returns:
        int
            Szuszik's algorithm result

    """
    if a != max(a, b):
        map_key = pow(b, 2) + a
    else:
        map_key = pow(a, 2) + a + b

    return map_key


@jitclass(
    {
        "lw": nb_DictType(nb_int64, nb_float64),
    }
)
class NbComputePolyaCacheSzuszik:
    def __init__(self):
        self.lw = nb_Dict.empty(nb_int64, nb_float64)

    def call(self, size, w, k, a):

        ki = int(k)
        wi = int(w)
        map_key = szuszik(wi, ki)
        map_key = szuszik(map_key, size)
        if map_key not in self.lw:
            x = np.arange(0, size, dtype=np.float64)
            polya_pdf = 1 - np.sum(compute_polya_pdf(x, w, k, a))
            self.lw[map_key] = polya_pdf

        return self.lw[map_key]


@jitclass(
    {
        "lw": nb_DictType(
            nb_int64, nb_DictType(nb_int64, nb_DictType(nb_int64, nb_float64))
        ),
    }
)
class NbComputePolyaCacheDict:
    def __init__(self):
        self.lw = nb_Dict.empty(
            nb_int64, nb_Dict.empty(nb_int64, nb_Dict.empty(nb_int64, nb_float64))
        )

    def call(self, size, w, k, a):
        ki = int(k)
        wi = int(w)
        size = int(size)
        compute = True
        if ki not in self.lw:
            self.lw[ki] = {wi: {size: 0.0}}
        else:
            if wi not in self.lw[ki]:
                self.lw[ki][wi] = {size: 0.0}
            else:
                if size in self.lw[ki][wi]:
                    compute = False

        if compute:
            x = np.arange(0, size, dtype=np.float64)
            polya_pdf = 1 - np.sum(compute_polya_pdf(x, w, k, a))
            self.lw[ki][wi][size] = polya_pdf

        return self.lw[ki][wi][size]


@njit(fastmath=True, nogil=True, error_model="numpy")
def integer_cdf_nb(
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
        a: float
        weights: np.array
            edge weights
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
        x = np.arange(0, wi, dtype=np.float64)
        polya_pdf = compute_polya_pdf(x, wd, d, a)
        p[i] = 1 - np.sum(polya_pdf)
    return p


@njit(fastmath=True, nogil=True, error_model="numpy")
def integer_cdf_lru_nb(
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    weights: NpArrayEdgesFloat,
    a: float,
    cache_obj,
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
    n = wdegree.shape[0]
    p = np.zeros(n, dtype=np.float64)
    for i in range(n):
        wi = int(weights[i])
        if wi < 2:
            p[i] = 0.0
            continue
        wd = wdegree[i]
        d = degree[i]
        p[i] = cache_obj.call(wi, wd, d, a)
    return p


@njit(fastmath=True, nogil=True, error_model="numpy")
def integer_cdf_lru_nb_f(
    wdegree: NpArrayEdgesFloat,
    degree: NpArrayEdgesFloat,
    weights: NpArrayEdgesFloat,
    a: float,
    cache_obj,
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
    a_inv = 1 / a
    n = wdegree.shape[0]
    p = np.zeros(n, dtype=np.float64)

    for i in range(n):
        wi = int(weights[i])
        if wi < 2:
            p[i] = 0.0
            continue
        wd = wdegree[i]
        d = degree[i]
        b = (d - 1.0) * a_inv
        betaln2 = (
            cache_obj.callp(a_inv) + cache_obj.callp(b) - cache_obj.callp(a_inv + b)
        )
        gammaln1 = cache_obj.callp(wd + 1.0)
        gammalnbetaln1 = cache_obj.callp(wd + a_inv + b)
        r = 0.0
        x = 0.0
        for _ in range(0, wi):
            z1 = x + a_inv
            z2 = wd - x + b
            z3 = x + 1.0
            z4 = wd - x + 1.0
            v1 = cache_obj.callp(z1)
            v2 = cache_obj.callp(z2)
            v3 = cache_obj.callp(z3)
            v4 = cache_obj.callp(z4)

            betaln1 = v1 + v2 - gammalnbetaln1
            r += np.exp(gammaln1 - v3 - v4 + betaln1 - betaln2)
            x += 1
        p[i] = 1 - r
    return p
