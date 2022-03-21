"""Contains some math functions that are faster than the standard library."""

import numpy as np
from numba import njit, prange


@njit(fastmath=True, nogil=True, error_model="numpy", parallel=True)
def nbgammaln(z: np.ndarray, n: int) -> np.ndarray:
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


@njit(fastmath=True, nogil=False, error_model="numpy", parallel=True)
def nbgammaln_np(z: np.ndarray, n: int) -> np.ndarray:
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
    ones = np.ones(n, dtype=np.float64)
    y = ones * z
    tmp = z + 5.24218750000000000
    tmp = (z + 0.5) * np.log(tmp) - tmp
    ser = ones * 0.999999999999997092

    for j in prange(14):
        y = y + 1.0
        cc = coefs[j]
        ser = ser + cc / y

    out = tmp + np.log(2.5066282746310005 * ser / z)
    return out


@njit(fastmath=True, error_model="numpy")
def nbbeta(z, w, n):
    """Compute the beta function.

    Algorithm extracted from **Numerical Recipes in C**[1] chapter 6.1


    [1]: http://www.it.uom.gr/teaching/linearalgebra/NumericalRecipiesInC/

    Args:
        z: np.array
        w: np.array
    Returns:
        np.array

    """
    arg = nbgammaln(z, n) + nbgammaln(w, n) - nbgammaln(z + w, n)
    beta = np.exp(arg)
    return beta


@njit(fastmath=True, error_model="numpy")
def nbbetaln(z, w, n):
    """Compute the log of the beta function.

    We know
    $$ \\ln\\beta = \\ln\\abs(\\beta)$$



    Args:
        z: np.array
    Returns:
        np.array

    """
    beta = nbbeta(z, w, n)
    out = np.log(np.abs(beta))

    return out
