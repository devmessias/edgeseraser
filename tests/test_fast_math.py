import numpy as np
from edgeseraser.misc.fast_math import nbbetaln, nbgammaln
from scipy.special import betaln, gammaln


def test_gammaln():
    z0 = np.arange(0, 100)
    assert np.allclose(gammaln(z0), nbgammaln(z0))
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i)
        arr_scipy = gammaln(zs)
        arr_fast = nbgammaln(zs)
        assert np.allclose(arr_scipy, arr_fast)


def test_betaln():
    for i in range(1, 3):
        zs = np.linspace(0.001, 100, 10**i)
        ws = np.linspace(0.001, 100, 10**i)
        arr_scipy = betaln(zs, ws)
        arr_fast = nbbetaln(zs, ws)
        assert np.allclose(arr_scipy, arr_fast)
    z0 = np.arange(0, 100)
    w0 = np.arange(1000, 1100)
    assert np.allclose(betaln(z0, w0), nbbetaln(z0, w0))