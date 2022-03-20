from scipy.sparse import coo_matrix as coo_matrix, csr_matrix as csr_matrix, diags as diags
from scipy.sparse.csgraph import maximum_bipartite_matching as maximum_bipartite_matching, min_weight_full_bipartite_matching as min_weight_full_bipartite_matching
from typing import Any

def test_maximum_bipartite_matching_raises_on_dense_input() -> None: ...
def test_maximum_bipartite_matching_empty_graph() -> None: ...
def test_maximum_bipartite_matching_empty_left_partition() -> None: ...
def test_maximum_bipartite_matching_empty_right_partition() -> None: ...
def test_maximum_bipartite_matching_graph_with_no_edges() -> None: ...
def test_maximum_bipartite_matching_graph_that_causes_augmentation() -> None: ...
def test_maximum_bipartite_matching_graph_with_more_rows_than_columns() -> None: ...
def test_maximum_bipartite_matching_graph_with_more_columns_than_rows() -> None: ...
def test_maximum_bipartite_matching_explicit_zeros_count_as_edges() -> None: ...
def test_maximum_bipartite_matching_feasibility_of_result() -> None: ...
def test_matching_large_random_graph_with_one_edge_incident_to_each_vertex() -> None: ...
def test_min_weight_full_matching_trivial_graph(num_rows, num_cols) -> None: ...
def test_min_weight_full_matching_infeasible_problems(biadjacency_matrix) -> None: ...
def test_explicit_zero_causes_warning() -> None: ...
def linear_sum_assignment_assertions(solver, array_type, sign, test_case) -> None: ...

linear_sum_assignment_test_cases: Any

def test_min_weight_full_matching_small_inputs(sign, test_case) -> None: ...