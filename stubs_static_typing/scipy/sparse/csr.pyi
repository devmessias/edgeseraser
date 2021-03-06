from .compressed import _cs_matrix
from typing import Any

class csr_matrix(_cs_matrix):
    format: str
    def transpose(self, axes: Any | None = ..., copy: bool = ...): ...
    def tolil(self, copy: bool = ...): ...
    def tocsr(self, copy: bool = ...): ...
    def tocsc(self, copy: bool = ...): ...
    def tobsr(self, blocksize: Any | None = ..., copy: bool = ...): ...
    def __iter__(self): ...
    def getrow(self, i): ...
    def getcol(self, i): ...

def isspmatrix_csr(x): ...
