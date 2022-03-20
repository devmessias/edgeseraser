import sys
from typing import Dict

import numpy as np

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class NxOpts(TypedDict):
    is_directed: bool
    nodelabel2index: Dict[str, int]


class IgOpts(TypedDict):
    is_directed: bool


NpArrayEdges = np.ndarray
NpArrayEdgesFloat = np.ndarray
NpArrayEdgesDataInt = np.ndarray
NpArrayEdgesIds = np.ndarray
