from scipy.sparse.linalg import interface as interface
from scipy.sparse.sputils import matrix as matrix
from typing import Any

class TestLinearOperator:
    A: Any
    B: Any
    C: Any
    def setup_method(self) -> None: ...
    def test_matvec(self): ...
    def test_matmul(self): ...

class TestAsLinearOperator:
    cases: Any
    dtype: Any
    shape: Any
    def setup_method(self): ...
    def test_basic(self) -> None: ...
    def test_dot(self) -> None: ...

def test_repr(): ...
def test_identity() -> None: ...
def test_attributes(): ...
def matvec(x): ...
def test_pickle() -> None: ...
def test_inheritance(): ...
def test_dtypes_of_operator_sum() -> None: ...
def test_no_double_init(): ...
def test_adjoint_conjugate() -> None: ...
def test_ndim() -> None: ...
def test_transpose_noconjugate() -> None: ...
