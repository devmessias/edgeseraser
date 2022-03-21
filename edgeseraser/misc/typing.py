import sys
from typing import Dict

import numpy as np
import numpy.typing as npt

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class NxOpts(TypedDict):
    is_directed: bool
    nodelabel2index: Dict[str, int]


class IgOpts(TypedDict):
    is_directed: bool


NpArrayEdges = npt.NDArray[np.int_]
NpArrayEdgesFloat = npt.NDArray[np.float_]
NpArrayEdgesBool = npt.NDArray[np.bool_]
NpArrayEdgesIds = npt.NDArray[np.int_]
NpArrayFloat = npt.NDArray[np.float_]
