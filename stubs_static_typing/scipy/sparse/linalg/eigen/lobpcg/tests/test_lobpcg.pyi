from scipy.linalg import eig as eig, eigh as eigh, orth as orth, toeplitz as toeplitz
from scipy.sparse import diags as diags, eye as eye, spdiags as spdiags
from scipy.sparse.linalg import LinearOperator as LinearOperator, eigs as eigs
from scipy.sparse.linalg.eigen.lobpcg import lobpcg as lobpcg

def ElasticRod(n): ...
def MikotaPair(n): ...
def compare_solutions(A, B, m) -> None: ...
def test_Small() -> None: ...
def test_ElasticRod() -> None: ...
def test_MikotaPair() -> None: ...
def test_regression() -> None: ...
def test_diagonal() -> None: ...
def test_fiedler_small_8() -> None: ...
def test_fiedler_large_12() -> None: ...
def test_hermitian() -> None: ...
def test_eigs_consistency(n, atol) -> None: ...
def test_verbosity(tmpdir) -> None: ...
def test_tolerance_float32() -> None: ...
def test_random_initial_float32() -> None: ...
def test_maxit_None() -> None: ...
def test_diagonal_data_types(): ...