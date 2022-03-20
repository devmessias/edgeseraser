import sys
from typing import NewType, Tuple

import numpy as np

if sys.version_info >= (3, 8):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict


class NxOpts(TypedDict):
    is_directed: bool
    nodelabel2index: dict[str, int]


class IgOpts(TypedDict):
    is_directed: bool


AnyLen = NewType("AnyLen", int)
if sys.version_info >= (3, 9):
    NpArrayEdges = np.ndarray[Tuple[AnyLen, Literal[2]], np.dtype[np.float_]]
    NpArrayEdgesFloat = np.ndarray[AnyLen, np.dtype[np.float_]]
    NpArrayEdgesIds = np.ndarray[AnyLen, np.dtype[np.int_]]
else:
    NpArrayEdges = np.ndarray
    NpArrayEdgesFloat = np.ndarray
    NpArrayEdgesDataInt = np.ndarray
    NpArrayEdgesIds = np.ndarray
