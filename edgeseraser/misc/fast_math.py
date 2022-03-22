"""Contains some math functions that are faster than the standard library."""

import numpy as np
from numba import njit, prange
from numba.experimental import jitclass
from numba.typed import Dict as nb_Dict
from numba.types import DictType as nb_DictType
from numba.types import float64 as nb_float64


@jitclass({"lru": nb_DictType(nb_float64, nb_float64)})
class NbGammaLnCache:
    def __init__(self):
        self.lru = nb_Dict.empty(nb_float64, nb_float64)

    def put(self, z, v):
        self.lru[z] = v

    def callp(self, z):
        if z not in self.lru:
            r = nbgammaln(z)
            self.lru[z] = r
        return self.lru[z]

    def call(self, z):
        if z not in self.lru:
            r = nbgammaln(z)
            return r
        return self.lru[z]


@njit(fastmath=True, nogil=True, error_model="numpy")
def nbgammaln(z: float) -> float:
    """Compute the log of the gamma function.

    Algorithm extracted from **Numerical Recipes in C**[1] chapter 6.1


    [1]: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/

    Args:
        z: np.float64
    Returns:
        np.float64
    """
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )
    y = z
    f = 2.5066282746310005 / z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = 0.999999999999997092
    for j in range(14):
        y = y + 1.0
        ser = ser + coefs[j] / y

    out = tmp + np.log(f * ser)
    return out


@njit(fastmath=True, nogil=True, error_model="numpy", parallel=True)
def nbgammaln_parallel(z: np.ndarray) -> np.ndarray:
    """Compute the log of the gamma function.

    Algorithm extracted from **Numerical Recipes in C**[1] chapter 6.1


    [1]: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/

    Args:
        z: np.array
    Returns:
        np.array

    """
    coefs = np.array(
        [
            57.1562356658629235,
            -59.5979603554754912,
            14.1360979747417471,
            -0.491913816097620199,
            0.339946499848118887e-4,
            0.465236289270485756e-4,
            -0.983744753048795646e-4,
            0.158088703224912494e-3,
            -0.210264441724104883e-3,
            0.217439618115212643e-3,
            -0.164318106536763890e-3,
            0.844182239838527433e-4,
            -0.261908384015814087e-4,
            0.368991826595316234e-5,
        ]
    )
    n = z.shape[0]
    out = np.zeros(n)
    for i in prange(n):
        y = z[i]
        tmp = z[i] + 5.24218750000000000
        tmp = (z[i] + 0.5) * np.log(tmp) - tmp
        ser = 0.999999999999997092
        for j in range(14):
            y = y + 1.0
            ser = ser + coefs[j] / y

        out[i] = tmp + np.log(2.5066282746310005 * ser / z[i])
    return out


@njit(fastmath=True, error_model="numpy")
def nbbetaln_parallel(z, w):
    """Compute the  LogBeta function.

    Algorithm extracted from **Numerical Recipes in C**[1] chapter 6.1


    [1]: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/

    Args:
        z: np.array
        w: np.array
    Returns:
        np.array

    $\\beta = \\Gamma(z) + \\Gamma(w) - \\Gamma(z + w)$
    $\\ln(\\abs \\beta)$
    """

    arg = nbgammaln_parallel(z) + nbgammaln_parallel(w) - nbgammaln_parallel(z + w)
    return arg
