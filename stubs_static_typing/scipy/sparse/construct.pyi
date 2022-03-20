from typing import Any

def spdiags(data, diags, m, n, format: Any | None = ...): ...
def diags(diagonals, offsets: int = ..., shape: Any | None = ..., format: Any | None = ..., dtype: Any | None = ...): ...
def identity(n, dtype: str = ..., format: Any | None = ...): ...
def eye(m, n: Any | None = ..., k: int = ..., dtype=..., format: Any | None = ...): ...
def kron(A, B, format: Any | None = ...): ...
def kronsum(A, B, format: Any | None = ...): ...
def hstack(blocks, format: Any | None = ..., dtype: Any | None = ...): ...
def vstack(blocks, format: Any | None = ..., dtype: Any | None = ...): ...
def bmat(blocks, format: Any | None = ..., dtype: Any | None = ...): ...
def block_diag(mats, format: Any | None = ..., dtype: Any | None = ...): ...
def random(m, n, density: float = ..., format: str = ..., dtype: Any | None = ..., random_state: Any | None = ..., data_rvs: Any | None = ...): ...
def rand(m, n, density: float = ..., format: str = ..., dtype: Any | None = ..., random_state: Any | None = ...): ...
