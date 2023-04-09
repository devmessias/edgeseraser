from scipy._lib._testutils import check_free_memory as check_free_memory
from scipy.linalg import inv as inv, norm as norm
from scipy.sparse import SparseEfficiencyWarning as SparseEfficiencyWarning, bsr_matrix as bsr_matrix, csc_matrix as csc_matrix, csr_matrix as csr_matrix, dok_matrix as dok_matrix, identity as identity, isspmatrix as isspmatrix, lil_matrix as lil_matrix, spdiags as spdiags
from scipy.sparse.linalg import SuperLU as SuperLU
from scipy.sparse.linalg.dsolve import MatrixRankWarning as MatrixRankWarning, factorized as factorized, spilu as spilu, splu as splu, spsolve as spsolve, spsolve_triangular as spsolve_triangular, use_solver as use_solver
from typing import Any

sup_sparse_efficiency: Any
has_umfpack: bool

def toarray(a): ...
def setup_bug_8278(): ...

class TestFactorized:
    n: Any
    A: Any
    def setup_method(self) -> None: ...
    def test_singular_without_umfpack(self) -> None: ...
    def test_singular_with_umfpack(self) -> None: ...
    def test_non_singular_without_umfpack(self) -> None: ...
    def test_non_singular_with_umfpack(self) -> None: ...
    def test_cannot_factorize_nonsquare_matrix_without_umfpack(self) -> None: ...
    def test_factorizes_nonsquare_matrix_with_umfpack(self) -> None: ...
    def test_call_with_incorrectly_sized_matrix_without_umfpack(self) -> None: ...
    def test_call_with_incorrectly_sized_matrix_with_umfpack(self) -> None: ...
    def test_call_with_cast_to_complex_without_umfpack(self) -> None: ...
    def test_call_with_cast_to_complex_with_umfpack(self) -> None: ...
    def test_assume_sorted_indices_flag(self) -> None: ...
    def test_bug_8278(self) -> None: ...

class TestLinsolve:
    def setup_method(self) -> None: ...
    def test_singular(self) -> None: ...
    def test_singular_gh_3312(self) -> None: ...
    def test_twodiags(self) -> None: ...
    def test_bvector_smoketest(self) -> None: ...
    def test_bmatrix_smoketest(self) -> None: ...
    def test_non_square(self) -> None: ...
    def test_example_comparison(self) -> None: ...
    def test_shape_compatibility(self) -> None: ...
    def test_ndarray_support(self) -> None: ...
    def test_gssv_badinput(self): ...
    def test_sparsity_preservation(self) -> None: ...
    def test_dtype_cast(self) -> None: ...
    def test_bug_8278(self) -> None: ...

class TestSplu:
    n: Any
    A: Any
    def setup_method(self) -> None: ...
    def test_splu_smoketest(self) -> None: ...
    def test_spilu_smoketest(self) -> None: ...
    def test_spilu_drop_rule(self) -> None: ...
    def test_splu_nnz0(self) -> None: ...
    def test_spilu_nnz0(self) -> None: ...
    def test_splu_basic(self) -> None: ...
    def test_splu_perm(self) -> None: ...
    def test_natural_permc(self, splu_fun, rtol) -> None: ...
    def test_lu_refcount(self) -> None: ...
    def test_bad_inputs(self) -> None: ...
    def test_superlu_dlamch_i386_nan(self) -> None: ...
    def test_lu_attr(self) -> None: ...
    def test_threads_parallel(self) -> None: ...

class TestSpsolveTriangular:
    def setup_method(self) -> None: ...
    def test_singular(self) -> None: ...
    def test_bad_shape(self) -> None: ...
    def test_input_types(self) -> None: ...
    def test_random(self): ...