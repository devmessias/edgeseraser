from scipy.sparse import csc_matrix as csc_matrix, csr_matrix as csr_matrix, lil_matrix as lil_matrix

def test_csc_getrow() -> None: ...
def test_csc_getcol() -> None: ...
def test_csc_empty_slices(matrix_input, axis, expected_shape) -> None: ...
def test_argmax_overflow(ax) -> None: ...
