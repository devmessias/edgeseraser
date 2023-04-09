from .data import _data_matrix, _minmax_mixin
from typing import Any

class coo_matrix(_data_matrix, _minmax_mixin):
    format: str
    row: Any
    col: Any
    data: Any
    has_canonical_format: bool
    def __init__(self, arg1, shape: Any | None = ..., dtype: Any | None = ..., copy: bool = ...) -> None: ...
    def reshape(self, *args, **kwargs): ...
    def getnnz(self, axis: Any | None = ...): ...
    def transpose(self, axes: Any | None = ..., copy: bool = ...): ...
    def resize(self, *shape) -> None: ...
    def toarray(self, order: Any | None = ..., out: Any | None = ...): ...
    def tocsc(self, copy: bool = ...): ...
    def tocsr(self, copy: bool = ...): ...
    def tocoo(self, copy: bool = ...): ...
    def todia(self, copy: bool = ...): ...
    def todok(self, copy: bool = ...): ...
    def diagonal(self, k: int = ...): ...
    def sum_duplicates(self) -> None: ...
    def eliminate_zeros(self) -> None: ...

def isspmatrix_coo(x): ...