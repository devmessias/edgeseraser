import numpy as np
import pytest
from edgeseraser.misc.fast_math import (
    NbGammaLnCache,
    nbbetaln_parallel,
    nbgammaln_parallel,
)
from scipy.special import betaln, gammaln


def test_gammaln():
    z0 = np.arange(0, 100).astype("float64")
    assert np.allclose(gammaln(z0), nbgammaln_parallel(z0))
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i).astype("float64")
        arr_scipy = gammaln(zs)
        arr_fast = nbgammaln_parallel(zs)
        assert np.allclose(arr_scipy, arr_fast)


def test_lru_gammaln():
    z0 = np.arange(0, 100).astype("float64")
    z0 = np.array([1, 2.0, 4.0])
    s = z0.shape[0]
    gamaln_lru = NbGammaLnCache()
    scipy_r = gammaln(z0)
    fast_r = np.zeros_like(z0)
    for i in range(s):
        fast_r[i] = gamaln_lru.callp(z0[i])
    assert np.allclose(scipy_r, fast_r)


def test_betaln():
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i)
        ws = np.linspace(0.001, 100, 10**i)
        arr_scipy = betaln(zs, ws)
        arr_fast = nbbetaln_parallel(zs, ws)
        assert np.allclose(arr_scipy, arr_fast)
    z0 = np.arange(0, 100)
    w0 = np.arange(1000, 1100)
    assert np.allclose(betaln(z0, w0), nbbetaln_parallel(z0, w0))


@pytest.mark.parametrize("what", ["nb", "scipy"])
@pytest.mark.benchmark(group="GammaLn numba vs scipy")
def test_gammaln_nb_vs_scipy_perf(benchmark, what):
    benchmark.extra_info["size array"] = 1000000
    zs = np.linspace(0.001, 1000, 1000000)
    if what == "scipy":
        benchmark(gammaln, zs)
    elif what == "nb":
        benchmark(nbgammaln_parallel, zs)
