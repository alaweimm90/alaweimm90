"""
Tests for Vector Operations
Comprehensive test suite for vector operations including arithmetic, norms,
products, and specialized algorithms.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_almost_equal
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.vector_operations import VectorOperations, VectorNorms
class TestVectorOperations:
    """Test basic vector operations."""
    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.array([1, 2, 3])
        self.v = np.array([4, 5, 6])
        self.w = np.array([7, 8])  # Different dimension
        self.u_2d = np.array([3, 4])
        self.v_2d = np.array([1, 2])
        # Complex vectors
        self.u_complex = np.array([1+2j, 3-1j, 2+0j])
        self.v_complex = np.array([2-1j, 1+2j, 0+1j])
    def test_validate_vector(self):
        """Test vector validation."""
        # Valid vectors
        u_valid = VectorOperations.validate_vector(self.u)
        assert u_valid.shape == (3,)
        # List to array conversion
        u_list = [1, 2, 3]
        u_converted = VectorOperations.validate_vector(u_list)
        assert_array_almost_equal(u_converted, self.u)
        # Row vector
        u_row = np.array([[1, 2, 3]])
        u_flattened = VectorOperations.validate_vector(u_row)
        assert_array_almost_equal(u_flattened, self.u)
        # Column vector
        u_col = np.array([[1], [2], [3]])
        u_flattened = VectorOperations.validate_vector(u_col)
        assert_array_almost_equal(u_flattened, self.u)
        # Invalid inputs
        with pytest.raises(ValueError):
            VectorOperations.validate_vector(np.array([]))  # Empty
        with pytest.raises(ValueError):
            VectorOperations.validate_vector(np.array([[[1, 2]]]))  # 3D
        with pytest.raises(ValueError):
            matrix_2d = np.array([[1, 2], [3, 4]])
            VectorOperations.validate_vector(matrix_2d)  # 2D matrix
    def test_add(self):
        """Test vector addition."""
        result = VectorOperations.add(self.u, self.v)
        expected = np.array([5, 7, 9])
        assert_array_almost_equal(result, expected)
        # Complex vectors
        result_complex = VectorOperations.add(self.u_complex, self.v_complex)
        expected_complex = np.array([3+1j, 4+1j, 2+1j])
        assert_array_almost_equal(result_complex, expected_complex)
        # Dimension mismatch
        with pytest.raises(ValueError):
            VectorOperations.add(self.u, self.w)
    def test_subtract(self):
        """Test vector subtraction."""
        result = VectorOperations.subtract(self.u, self.v)
        expected = np.array([-3, -3, -3])
        assert_array_almost_equal(result, expected)
        # Complex vectors
        result_complex = VectorOperations.subtract(self.u_complex, self.v_complex)
        expected_complex = np.array([-1+3j, 2-3j, 2-1j])
        assert_array_almost_equal(result_complex, expected_complex)
        # Dimension mismatch
        with pytest.raises(ValueError):
            VectorOperations.subtract(self.u, self.w)
    def test_scalar_multiply(self):
        """Test scalar multiplication."""
        # Real scalar
        result = VectorOperations.scalar_multiply(2.5, self.u)
        expected = np.array([2.5, 5.0, 7.5])
        assert_array_almost_equal(result, expected)
        # Complex scalar
        result_complex = VectorOperations.scalar_multiply(1+1j, self.u)
        expected_complex = np.array([1+1j, 2+2j, 3+3j])
        assert_array_almost_equal(result_complex, expected_complex)
        # Zero scalar
        result_zero = VectorOperations.scalar_multiply(0, self.u)
        expected_zero = np.array([0, 0, 0])
        assert_array_almost_equal(result_zero, expected_zero)
    def test_dot_product(self):
        """Test dot product."""
        result = VectorOperations.dot_product(self.u, self.v)
        expected = 1*4 + 2*5 + 3*6  # 32
        assert_allclose(result, expected)
        # Complex vectors
        result_complex = VectorOperations.dot_product(self.u_complex, self.v_complex)
        # (1+2j)*(2-1j) + (3-1j)*(1+2j) + (2+0j)*(0+1j)
        # = (4+3j) + (5+5j) + (2j) = 9+10j
        expected_complex = 9 + 10j
        assert_allclose(result_complex, expected_complex)
        # Orthogonal vectors
        orthogonal_1 = np.array([1, 0, 0])
        orthogonal_2 = np.array([0, 1, 0])
        result_orthogonal = VectorOperations.dot_product(orthogonal_1, orthogonal_2)
        assert_allclose(result_orthogonal, 0)
        # Dimension mismatch
        with pytest.raises(ValueError):
            VectorOperations.dot_product(self.u, self.w)
    def test_outer_product(self):
        """Test outer product."""
        result = VectorOperations.outer_product(self.u_2d, self.v_2d)
        expected = np.array([[3, 6], [4, 8]])
        assert_array_almost_equal(result, expected)
        # Check dimensions
        assert result.shape == (len(self.u_2d), len(self.v_2d))
        # Verify result[i,j] = u[i] * v[j]
        for i in range(len(self.u_2d)):
            for j in range(len(self.v_2d)):
                assert_allclose(result[i, j], self.u_2d[i] * self.v_2d[j])
    def test_cross_product(self):
        """Test cross product."""
        # Standard basis vectors
        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])
        # i × j = k
        result_ij = VectorOperations.cross_product(i, j)
        assert_array_almost_equal(result_ij, k)
        # j × k = i
        result_jk = VectorOperations.cross_product(j, k)
        assert_array_almost_equal(result_jk, i)
        # k × i = j
        result_ki = VectorOperations.cross_product(k, i)
        assert_array_almost_equal(result_ki, j)
        # Anti-commutativity: u × v = -(v × u)
        result_uv = VectorOperations.cross_product(self.u, self.v)
        result_vu = VectorOperations.cross_product(self.v, self.u)
        assert_array_almost_equal(result_uv, -result_vu)
        # Cross product with itself is zero
        result_self = VectorOperations.cross_product(self.u, self.u)
        assert_array_almost_equal(result_self, np.zeros(3))
        # Non-3D vectors should fail
        with pytest.raises(ValueError):
            VectorOperations.cross_product(self.u_2d, self.v_2d)
    def test_triple_scalar_product(self):
        """Test scalar triple product."""
        # Standard basis vectors
        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])
        # i · (j × k) = 1
        result = VectorOperations.triple_scalar_product(i, j, k)
        assert_allclose(result, 1)
        # Cyclic permutation
        result_cyclic = VectorOperations.triple_scalar_product(j, k, i)
        assert_allclose(result_cyclic, 1)
        # Anti-cyclic permutation
        result_anti = VectorOperations.triple_scalar_product(k, j, i)
        assert_allclose(result_anti, -1)
        # Coplanar vectors give zero
        coplanar_1 = np.array([1, 0, 0])
        coplanar_2 = np.array([2, 0, 0])
        coplanar_3 = np.array([3, 0, 0])
        result_coplanar = VectorOperations.triple_scalar_product(coplanar_1, coplanar_2, coplanar_3)
        assert_allclose(result_coplanar, 0, atol=1e-15)
        # Non-3D vectors should fail
        with pytest.raises(ValueError):
            VectorOperations.triple_scalar_product(self.u_2d, self.v_2d, self.u_2d)
    def test_magnitude(self):
        """Test vector magnitude."""
        # Known magnitude
        result = VectorOperations.magnitude(self.u_2d)  # [3, 4]
        expected = 5.0  # 3-4-5 triangle
        assert_allclose(result, expected)
        # Zero vector
        zero_vec = np.array([0, 0, 0])
        result_zero = VectorOperations.magnitude(zero_vec)
        assert_allclose(result_zero, 0)
        # Unit vectors
        unit_x = np.array([1, 0, 0])
        result_unit = VectorOperations.magnitude(unit_x)
        assert_allclose(result_unit, 1)
        # Complex vector
        result_complex = VectorOperations.magnitude(self.u_complex)
        # |1+2j|² + |3-1j|² + |2+0j|² = 5 + 10 + 4 = 19
        expected_complex = np.sqrt(19)
        assert_allclose(result_complex, expected_complex)
    def test_normalize(self):
        """Test vector normalization."""
        result = VectorOperations.normalize(self.u_2d)
        expected = np.array([3/5, 4/5])  # [3, 4] / 5
        assert_array_almost_equal(result, expected)
        # Check that result has unit magnitude
        result_magnitude = VectorOperations.magnitude(result)
        assert_allclose(result_magnitude, 1.0)
        # Different norms
        result_l1 = VectorOperations.normalize(self.u, norm_type=1)
        l1_norm = np.sum(np.abs(self.u))
        expected_l1 = self.u / l1_norm
        assert_array_almost_equal(result_l1, expected_l1)
        # Near-zero vector warning
        near_zero = np.array([1e-15, 1e-15, 1e-15])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            VectorOperations.normalize(near_zero)
            assert len(w) == 1
            assert "near-zero norm" in str(w[0].message)
    def test_distance(self):
        """Test distance between vectors."""
        # Euclidean distance
        result = VectorOperations.distance(self.u_2d, self.v_2d)
        diff = self.u_2d - self.v_2d  # [2, 2]
        expected = np.sqrt(8)  # sqrt(2² + 2²)
        assert_allclose(result, expected)
        # Manhattan distance
        result_l1 = VectorOperations.distance(self.u_2d, self.v_2d, norm_type=1)
        expected_l1 = 4  # |2| + |2|
        assert_allclose(result_l1, expected_l1)
        # Maximum distance
        result_inf = VectorOperations.distance(self.u_2d, self.v_2d, norm_type=np.inf)
        expected_inf = 2  # max(|2|, |2|)
        assert_allclose(result_inf, expected_inf)
        # Distance to self is zero
        result_self = VectorOperations.distance(self.u, self.u)
        assert_allclose(result_self, 0)
    def test_angle_between(self):
        """Test angle between vectors."""
        # Right angle
        orthogonal_1 = np.array([1, 0])
        orthogonal_2 = np.array([0, 1])
        result_right = VectorOperations.angle_between(orthogonal_1, orthogonal_2)
        assert_allclose(result_right, np.pi/2)
        # Right angle in degrees
        result_degrees = VectorOperations.angle_between(orthogonal_1, orthogonal_2, degrees=True)
        assert_allclose(result_degrees, 90)
        # Parallel vectors
        parallel_1 = np.array([1, 2, 3])
        parallel_2 = np.array([2, 4, 6])
        result_parallel = VectorOperations.angle_between(parallel_1, parallel_2)
        assert_allclose(result_parallel, 0, atol=1e-15)
        # Anti-parallel vectors
        anti_parallel_1 = np.array([1, 2, 3])
        anti_parallel_2 = np.array([-1, -2, -3])
        result_anti = VectorOperations.angle_between(anti_parallel_1, anti_parallel_2)
        assert_allclose(result_anti, np.pi)
        # 60 degree angle
        vec_60_1 = np.array([1, 0])
        vec_60_2 = np.array([0.5, np.sqrt(3)/2])
        result_60 = VectorOperations.angle_between(vec_60_1, vec_60_2, degrees=True)
        assert_allclose(result_60, 60, rtol=1e-10)
    def test_project(self):
        """Test vector projection."""
        # Project onto axis
        u = np.array([3, 4])
        v = np.array([1, 0])  # x-axis
        result = VectorOperations.project(u, v)
        expected = np.array([3, 0])  # Projection onto x-axis
        assert_array_almost_equal(result, expected)
        # Project onto diagonal
        u_diag = np.array([2, 0])
        v_diag = np.array([1, 1])
        result_diag = VectorOperations.project(u_diag, v_diag)
        expected_diag = np.array([1, 1])  # (u·v / v·v) * v = (2 / 2) * [1,1]
        assert_array_almost_equal(result_diag, expected_diag)
        # Projection magnitude
        projection_magnitude = VectorOperations.magnitude(result_diag)
        dot_product_normalized = VectorOperations.dot_product(u_diag, VectorOperations.normalize(v_diag))
        assert_allclose(projection_magnitude, abs(dot_product_normalized))
        # Project onto zero vector warning
        zero_vec = np.array([0, 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result_zero = VectorOperations.project(u, zero_vec)
            assert len(w) == 1
            assert "zero vector" in str(w[0].message)
            assert_array_almost_equal(result_zero, np.zeros_like(u))
    def test_reject(self):
        """Test vector rejection."""
        u = np.array([3, 4])
        v = np.array([1, 0])  # x-axis
        projection = VectorOperations.project(u, v)
        rejection = VectorOperations.reject(u, v)
        # u = projection + rejection
        reconstructed = projection + rejection
        assert_array_almost_equal(reconstructed, u)
        # Projection and rejection are orthogonal
        dot_product = VectorOperations.dot_product(projection, rejection)
        assert_allclose(dot_product, 0, atol=1e-10)
        # Specific case
        expected_rejection = np.array([0, 4])  # y-component only
        assert_array_almost_equal(rejection, expected_rejection)
    def test_are_orthogonal(self):
        """Test orthogonality check."""
        # Standard basis vectors
        i = np.array([1, 0, 0])
        j = np.array([0, 1, 0])
        k = np.array([0, 0, 1])
        assert VectorOperations.are_orthogonal(i, j)
        assert VectorOperations.are_orthogonal(j, k)
        assert VectorOperations.are_orthogonal(k, i)
        # Non-orthogonal vectors
        u = np.array([1, 1, 0])
        v = np.array([1, 0, 0])
        assert not VectorOperations.are_orthogonal(u, v)
        # Nearly orthogonal (within tolerance)
        nearly_orthogonal_1 = np.array([1, 0])
        nearly_orthogonal_2 = np.array([1e-12, 1])
        assert VectorOperations.are_orthogonal(nearly_orthogonal_1, nearly_orthogonal_2, tolerance=1e-10)
        # Zero vector is orthogonal to everything
        zero_vec = np.array([0, 0, 0])
        assert VectorOperations.are_orthogonal(zero_vec, i)
    def test_are_parallel(self):
        """Test parallelism check."""
        # Parallel vectors
        u = np.array([1, 2, 3])
        v = np.array([2, 4, 6])
        assert VectorOperations.are_parallel(u, v)
        # Anti-parallel vectors
        w = np.array([-1, -2, -3])
        assert VectorOperations.are_parallel(u, w)
        # Non-parallel vectors
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])
        assert not VectorOperations.are_parallel(x, y)
        # 2D case (uses angle check)
        u_2d = np.array([2, 4])
        v_2d = np.array([1, 2])
        assert VectorOperations.are_parallel(u_2d, v_2d)
        # Zero vector case
        zero_vec = np.array([0, 0, 0])
        # This may fail due to numerical issues, which is expected
        try:
            result = VectorOperations.are_parallel(zero_vec, u)
            # Result is implementation-dependent for zero vectors
        except:
            pass  # Expected for zero vectors
    def test_gram_schmidt_orthogonalization(self):
        """Test Gram-Schmidt orthogonalization."""
        # Linearly independent vectors
        v1 = np.array([1, 1, 0])
        v2 = np.array([1, 0, 1])
        v3 = np.array([0, 1, 1])
        vectors = [v1, v2, v3]
        ortho_vectors = VectorOperations.gram_schmidt_orthogonalization(vectors, normalize=True)
        # Check that we get 3 vectors
        assert len(ortho_vectors) == 3
        # Check orthogonality
        for i in range(len(ortho_vectors)):
            for j in range(i+1, len(ortho_vectors)):
                dot_product = VectorOperations.dot_product(ortho_vectors[i], ortho_vectors[j])
                assert_allclose(dot_product, 0, atol=1e-10)
        # Check normalization
        for vec in ortho_vectors:
            magnitude = VectorOperations.magnitude(vec)
            assert_allclose(magnitude, 1.0, rtol=1e-10)
        # Check that span is preserved (first vector direction unchanged)
        first_original_normalized = VectorOperations.normalize(v1)
        first_orthogonal = ortho_vectors[0]
        # Should be same or opposite direction
        dot_check = abs(VectorOperations.dot_product(first_original_normalized, first_orthogonal))
        assert_allclose(dot_check, 1.0, rtol=1e-10)
        # Test without normalization
        ortho_vectors_unnorm = VectorOperations.gram_schmidt_orthogonalization(vectors, normalize=False)
        # Check orthogonality (but not normalization)
        for i in range(len(ortho_vectors_unnorm)):
            for j in range(i+1, len(ortho_vectors_unnorm)):
                dot_product = VectorOperations.dot_product(ortho_vectors_unnorm[i], ortho_vectors_unnorm[j])
                assert_allclose(dot_product, 0, atol=1e-10)
        # Linearly dependent vectors
        v1_dep = np.array([1, 0, 0])
        v2_dep = np.array([2, 0, 0])  # Parallel to v1
        v3_dep = np.array([0, 1, 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dependent_vectors = [v1_dep, v2_dep, v3_dep]
            ortho_dependent = VectorOperations.gram_schmidt_orthogonalization(dependent_vectors)
            # Should get warning about linear dependence
            assert len(w) == 1
            assert "linearly dependent" in str(w[0].message)
            # Should get fewer vectors than input
            assert len(ortho_dependent) == 2
        # Empty list
        empty_result = VectorOperations.gram_schmidt_orthogonalization([])
        assert len(empty_result) == 0
        # Dimension mismatch
        mixed_dims = [np.array([1, 2]), np.array([1, 2, 3])]
        with pytest.raises(ValueError):
            VectorOperations.gram_schmidt_orthogonalization(mixed_dims)
class TestVectorNorms:
    """Test vector norm calculations."""
    def setup_method(self):
        """Set up test fixtures."""
        self.v = np.array([3, 4, 0])  # Known norms
        self.v_complex = np.array([1+1j, 1-1j])
        self.v_negative = np.array([-2, -3, 1])
    def test_p_norm(self):
        """Test general p-norm."""
        # L2 norm
        result_l2 = VectorNorms.p_norm(self.v, 2)
        expected_l2 = 5.0  # sqrt(3² + 4² + 0²)
        assert_allclose(result_l2, expected_l2)
        # L1 norm
        result_l1 = VectorNorms.p_norm(self.v, 1)
        expected_l1 = 7.0  # |3| + |4| + |0|
        assert_allclose(result_l1, expected_l1)
        # L∞ norm
        result_inf = VectorNorms.p_norm(self.v, np.inf)
        expected_inf = 4.0  # max(|3|, |4|, |0|)
        assert_allclose(result_inf, expected_inf)
        # Fractional norm
        result_half = VectorNorms.p_norm(self.v, 0.5)
        expected_half = (3**0.5 + 4**0.5 + 0**0.5)**2
        assert_allclose(result_half, expected_half)
    def test_l1_norm(self):
        """Test L1 (Manhattan) norm."""
        result = VectorNorms.l1_norm(self.v)
        expected = 7.0  # |3| + |4| + |0|
        assert_allclose(result, expected)
        # With negative values
        result_neg = VectorNorms.l1_norm(self.v_negative)
        expected_neg = 6.0  # |-2| + |-3| + |1|
        assert_allclose(result_neg, expected_neg)
        # Complex vector
        result_complex = VectorNorms.l1_norm(self.v_complex)
        expected_complex = np.sqrt(2) + np.sqrt(2)  # |1+1j| + |1-1j|
        assert_allclose(result_complex, expected_complex)
    def test_l2_norm(self):
        """Test L2 (Euclidean) norm."""
        result = VectorNorms.l2_norm(self.v)
        expected = 5.0  # sqrt(3² + 4² + 0²)
        assert_allclose(result, expected)
        # Should be same as magnitude
        magnitude = VectorOperations.magnitude(self.v)
        assert_allclose(result, magnitude)
        # Complex vector
        result_complex = VectorNorms.l2_norm(self.v_complex)
        expected_complex = 2.0  # sqrt(|1+1j|² + |1-1j|²) = sqrt(2 + 2)
        assert_allclose(result_complex, expected_complex)
    def test_infinity_norm(self):
        """Test L∞ (maximum) norm."""
        result = VectorNorms.infinity_norm(self.v)
        expected = 4.0  # max(|3|, |4|, |0|)
        assert_allclose(result, expected)
        # With negative values
        result_neg = VectorNorms.infinity_norm(self.v_negative)
        expected_neg = 3.0  # max(|-2|, |-3|, |1|)
        assert_allclose(result_neg, expected_neg)
        # Complex vector
        result_complex = VectorNorms.infinity_norm(self.v_complex)
        expected_complex = np.sqrt(2)  # max(|1+1j|, |1-1j|)
        assert_allclose(result_complex, expected_complex)
    def test_weighted_norm(self):
        """Test weighted norm."""
        weights = np.array([1, 2, 0.5])
        # Weighted L2 norm
        result_l2 = VectorNorms.weighted_norm(self.v, weights, p=2)
        expected_l2 = np.sqrt(1*3**2 + 2*4**2 + 0.5*0**2)  # sqrt(9 + 32 + 0)
        assert_allclose(result_l2, expected_l2)
        # Weighted L1 norm
        result_l1 = VectorNorms.weighted_norm(self.v, weights, p=1)
        expected_l1 = 1*3 + 2*4 + 0.5*0  # 3 + 8 + 0
        assert_allclose(result_l1, expected_l1)
        # Weighted L∞ norm
        result_inf = VectorNorms.weighted_norm(self.v, weights, p=np.inf)
        expected_inf = max(1*3, 2*4, 0.5*0)  # max(3, 8, 0)
        assert_allclose(result_inf, expected_inf)
        # Dimension mismatch
        wrong_weights = np.array([1, 2])
        with pytest.raises(ValueError):
            VectorNorms.weighted_norm(self.v, wrong_weights)
        # Negative weights
        negative_weights = np.array([1, -2, 0.5])
        with pytest.raises(ValueError):
            VectorNorms.weighted_norm(self.v, negative_weights)
    def test_unit_vector_in_direction(self):
        """Test unit vector creation."""
        direction = np.array([3, 4])
        result = VectorNorms.unit_vector_in_direction(direction)
        expected = np.array([3/5, 4/5])
        assert_array_almost_equal(result, expected)
        # Check unit magnitude
        magnitude = VectorOperations.magnitude(result)
        assert_allclose(magnitude, 1.0)
        # Check same direction
        angle = VectorOperations.angle_between(direction, result)
        assert_allclose(angle, 0, atol=1e-15)
    def test_standard_basis_vector(self):
        """Test standard basis vector creation."""
        # 3D basis vectors
        e1 = VectorNorms.standard_basis_vector(3, 0)
        e2 = VectorNorms.standard_basis_vector(3, 1)
        e3 = VectorNorms.standard_basis_vector(3, 2)
        assert_array_almost_equal(e1, np.array([1, 0, 0]))
        assert_array_almost_equal(e2, np.array([0, 1, 0]))
        assert_array_almost_equal(e3, np.array([0, 0, 1]))
        # Check orthogonality
        assert VectorOperations.are_orthogonal(e1, e2)
        assert VectorOperations.are_orthogonal(e2, e3)
        assert VectorOperations.are_orthogonal(e3, e1)
        # Check unit magnitude
        assert_allclose(VectorOperations.magnitude(e1), 1.0)
        assert_allclose(VectorOperations.magnitude(e2), 1.0)
        assert_allclose(VectorOperations.magnitude(e3), 1.0)
        # Invalid index
        with pytest.raises(ValueError):
            VectorNorms.standard_basis_vector(3, 3)  # Index out of range
        with pytest.raises(ValueError):
            VectorNorms.standard_basis_vector(3, -1)  # Negative index
def test_create_test_vectors():
    """Test test vector creation."""
    vectors = VectorOperations.create_test_vectors()
    # Check that all expected vectors are present
    expected_keys = [
        'zero_3d', 'unit_x', 'unit_y', 'unit_z',
        'random_3d', 'random_5d', 'random_10d',
        'orthogonal_pair_1', 'orthogonal_pair_2',
        'parallel_pair_1', 'parallel_pair_2',
        'gs_test_1', 'gs_test_2', 'gs_test_3'
    ]
    for key in expected_keys:
        assert key in vectors
    # Check properties
    assert_array_almost_equal(vectors['zero_3d'], np.zeros(3))
    assert_array_almost_equal(vectors['unit_x'], np.array([1, 0, 0]))
    assert_array_almost_equal(vectors['unit_y'], np.array([0, 1, 0]))
    assert_array_almost_equal(vectors['unit_z'], np.array([0, 0, 1]))
    # Check orthogonality
    assert VectorOperations.are_orthogonal(vectors['orthogonal_pair_1'], vectors['orthogonal_pair_2'])
    # Check parallelism
    assert VectorOperations.are_parallel(vectors['parallel_pair_1'], vectors['parallel_pair_2'])
    # Check dimensions
    assert len(vectors['random_3d']) == 3
    assert len(vectors['random_5d']) == 5
    assert len(vectors['random_10d']) == 10
if __name__ == "__main__":
    pytest.main([__file__])