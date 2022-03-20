from scipy.sparse.linalg.interface import LinearOperator
from typing import Any

SNAUPD_ERRORS = DNAUPD_ERRORS
CNAUPD_ERRORS = ZNAUPD_ERRORS
SSAUPD_ERRORS = DSAUPD_ERRORS

class ArpackError(RuntimeError):
    def __init__(self, info, infodict=...) -> None: ...

class ArpackNoConvergence(ArpackError):
    eigenvalues: Any
    eigenvectors: Any
    def __init__(self, msg, eigenvalues, eigenvectors) -> None: ...

class _ArpackParams:
    resid: Any
    sigma: int
    v: Any
    iparam: Any
    mode: Any
    n: Any
    tol: Any
    k: Any
    maxiter: Any
    ncv: Any
    which: Any
    tp: Any
    info: Any
    converged: bool
    ido: int
    def __init__(self, n, k, tp, mode: int = ..., sigma: Any | None = ..., ncv: Any | None = ..., v0: Any | None = ..., maxiter: Any | None = ..., which: str = ..., tol: int = ...) -> None: ...

class _SymmetricArpackParams(_ArpackParams):
    OP: Any
    B: Any
    bmat: str
    OPa: Any
    OPb: Any
    A_matvec: Any
    workd: Any
    workl: Any
    iterate_infodict: Any
    extract_infodict: Any
    ipntr: Any
    def __init__(self, n, k, tp, matvec, mode: int = ..., M_matvec: Any | None = ..., Minv_matvec: Any | None = ..., sigma: Any | None = ..., ncv: Any | None = ..., v0: Any | None = ..., maxiter: Any | None = ..., which: str = ..., tol: int = ...): ...
    converged: bool
    def iterate(self) -> None: ...
    def extract(self, return_eigenvectors): ...

class _UnsymmetricArpackParams(_ArpackParams):
    OP: Any
    B: Any
    bmat: str
    OPa: Any
    OPb: Any
    matvec: Any
    workd: Any
    workl: Any
    iterate_infodict: Any
    extract_infodict: Any
    ipntr: Any
    rwork: Any
    def __init__(self, n, k, tp, matvec, mode: int = ..., M_matvec: Any | None = ..., Minv_matvec: Any | None = ..., sigma: Any | None = ..., ncv: Any | None = ..., v0: Any | None = ..., maxiter: Any | None = ..., which: str = ..., tol: int = ...): ...
    converged: bool
    def iterate(self) -> None: ...
    def extract(self, return_eigenvectors): ...

class SpLuInv(LinearOperator):
    M_lu: Any
    shape: Any
    dtype: Any
    isreal: Any
    def __init__(self, M) -> None: ...

class LuInv(LinearOperator):
    M_lu: Any
    shape: Any
    dtype: Any
    def __init__(self, M) -> None: ...

class IterInv(LinearOperator):
    M: Any
    dtype: Any
    shape: Any
    ifunc: Any
    tol: Any
    def __init__(self, M, ifunc=..., tol: int = ...) -> None: ...

class IterOpInv(LinearOperator):
    A: Any
    M: Any
    sigma: Any
    OP: Any
    shape: Any
    ifunc: Any
    tol: Any
    def __init__(self, A, M, sigma, ifunc=..., tol: int = ...): ...
    @property
    def dtype(self): ...

def eigs(A, k: int = ..., M: Any | None = ..., sigma: Any | None = ..., which: str = ..., v0: Any | None = ..., ncv: Any | None = ..., maxiter: Any | None = ..., tol: int = ..., return_eigenvectors: bool = ..., Minv: Any | None = ..., OPinv: Any | None = ..., OPpart: Any | None = ...): ...
def eigsh(A, k: int = ..., M: Any | None = ..., sigma: Any | None = ..., which: str = ..., v0: Any | None = ..., ncv: Any | None = ..., maxiter: Any | None = ..., tol: int = ..., return_eigenvectors: bool = ..., Minv: Any | None = ..., OPinv: Any | None = ..., mode: str = ...): ...
def svds(A, k: int = ..., ncv: Any | None = ..., tol: int = ..., which: str = ..., v0: Any | None = ..., maxiter: Any | None = ..., return_singular_vectors: bool = ..., solver: str = ...): ...
