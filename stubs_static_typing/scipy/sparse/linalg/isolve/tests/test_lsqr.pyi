from scipy.sparse.linalg import lsqr as lsqr
from typing import Any

n: int
G: Any
normal: Any
norm: Any
gg: Any
hh: Any
b: Any
tol: float
show: bool
maxit: Any

def test_basic() -> None: ...
def test_gh_2466() -> None: ...
def test_well_conditioned_problems() -> None: ...
def test_b_shapes() -> None: ...
def test_initialization() -> None: ...
