from typing import Any

def upcast(*args): ...
def getdtype(dtype, a: Any | None = ..., default: Any | None = ...): ...
def getdata(obj, dtype: Any | None = ..., copy: bool = ...): ...
def get_sum_dtype(dtype): ...
def isscalarlike(x): ...
def isintlike(x): ...
def isshape(x, nonneg: bool = ...): ...
def issequence(t): ...
def ismatrix(t): ...
def isdense(x): ...
