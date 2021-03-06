from scipy.sparse.csgraph import NegativeCycleError as NegativeCycleError, bellman_ford as bellman_ford, construct_dist_matrix as construct_dist_matrix, dijkstra as dijkstra, johnson as johnson, shortest_path as shortest_path
from typing import Any

directed_G: Any
undirected_G: Any
unweighted_G: Any
directed_SP: Any
directed_sparse_zero_G: Any
directed_sparse_zero_SP: Any
undirected_sparse_zero_G: Any
undirected_sparse_zero_SP: Any
directed_pred: Any
undirected_SP: Any
undirected_SP_limit_2: Any
undirected_SP_limit_0: Any
undirected_pred: Any
methods: Any

def test_dijkstra_limit() -> None: ...
def test_directed() -> None: ...
def test_undirected() -> None: ...
def test_directed_sparse_zero() -> None: ...
def test_undirected_sparse_zero() -> None: ...
def test_dijkstra_indices_min_only(directed, SP_ans, indices) -> None: ...
def test_shortest_path_min_only_random(n) -> None: ...
def test_shortest_path_indices() -> None: ...
def test_predecessors() -> None: ...
def test_construct_shortest_path() -> None: ...
def test_unweighted_path() -> None: ...
def test_negative_cycles() -> None: ...
def test_masked_input() -> None: ...
def test_overwrite() -> None: ...
def test_buffer(method) -> None: ...
def test_NaN_warnings() -> None: ...
def test_sparse_matrices() -> None: ...
