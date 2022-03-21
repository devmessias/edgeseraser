import numpy as np
import pytest
from edgeseraser.misc.fast_math import nbbetaln, nbgammaln
from scipy.special import betaln, gammaln

SIZE_BENCHMARK = 1000000


def test_gammaln():
    z0 = np.arange(0, 100).astype("float64")
    s = z0.shape[0]
    assert np.allclose(gammaln(z0), nbgammaln(z0, s))
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i).astype("float64")
        s = zs.shape[0]
        arr_scipy = gammaln(zs)
        arr_fast = nbgammaln(zs, s)
        assert np.allclose(arr_scipy, arr_fast)


def test_betaln():
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i)
        ws = np.linspace(0.001, 100, 10**i)
        s = zs.shape[0]
        arr_scipy = betaln(zs, ws)
        arr_fast = nbbetaln(zs, ws, s)
        assert np.allclose(arr_scipy, arr_fast)
    z0 = np.arange(0, 100)
    w0 = np.arange(1000, 1100)
    s = z0.shape[0]
    assert np.allclose(betaln(z0, w0), nbbetaln(z0, w0, s))


@pytest.mark.benchmark(group="GammaLn numba vs scipy")
def test_gammaln_scipy_perf(benchmark):
    benchmark.extra_info["size array"] = SIZE_BENCHMARK
    zs = np.linspace(0.001, 100, SIZE_BENCHMARK)
    benchmark(gammaln, zs)


@pytest.mark.benchmark(group="GammaLn numba vs scipy")
def test_gammaln_numba_perf(benchmark):
    benchmark.extra_info["size array"] = SIZE_BENCHMARK
    zs = np.linspace(0.001, 100, SIZE_BENCHMARK)
    s = zs.shape[0]
    benchmark(nbgammaln, zs, s)


@pytest.mark.benchmark(group="BetaLn numba vs scipy")
def test_betaln_scipy_perf(benchmark):
    benchmark.extra_info["size array"] = SIZE_BENCHMARK
    zs = np.linspace(0.001, 100, SIZE_BENCHMARK)
    ws = np.linspace(0.001, 100, SIZE_BENCHMARK)
    benchmark(betaln, zs, ws)


@pytest.mark.benchmark(group="BetaLn numba vs scipy")
def test_betaln_numba_perf(benchmark):
    benchmark.extra_info["size array"] = SIZE_BENCHMARK
    zs = np.linspace(0.001, 100, SIZE_BENCHMARK)
    ws = np.linspace(0.001, 100, SIZE_BENCHMARK)
    benchmark(nbbetaln, zs, ws, SIZE_BENCHMARK)
