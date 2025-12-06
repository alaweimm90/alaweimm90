"""
Vector Operations for Scientific Computing
Comprehensive vector operations including arithmetic, norms, products,
and specialized algorithms for scientific applications.
"""
import numpy as np
from typing import Union, Tuple, List, Optional
import warnings
class VectorOperations:
    """
    Core vector operations for scientific computing.
    Features:
    - Vector arithmetic with dimension checking
    - Vector norms and metrics
    - Inner and outer products
    - Cross products and vector geometry
    - Vector projections and orthogonalization
    """
    @staticmethod
    def validate_vector(vector: np.ndarray, name: str = "vector") -> np.ndarray:
        """Validate vector input and convert to numpy array."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        # Handle row/column vectors
        if vector.ndim == 2:
            if vector.shape[0] == 1:
                vector = vector.flatten()
            elif vector.shape[1] == 1:
                vector = vector.flatten()
            else:
                raise ValueError(f"{name} must be 1-dimensional or a row/column vector")
        elif vector.ndim != 1:
            raise ValueError(f"{name} must be 1-dimensional")
        if vector.size == 0:
            raise ValueError(f"{name} cannot be empty")
        return vector
    @staticmethod
    def add(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vector addition with dimension checking.
        Parameters:
            u, v: Input vectors
        Returns:
            u + v
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        return u + v
    @staticmethod
    def subtract(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vector subtraction with dimension checking.
        Parameters:
            u, v: Input vectors
        Returns:
            u - v
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        return u - v
    @staticmethod
    def scalar_multiply(scalar: Union[int, float, complex], v: np.ndarray) -> np.ndarray:
        """
        Scalar multiplication of vector.
        Parameters:
            scalar: Scalar value
            v: Input vector
        Returns:
            scalar * v
        """
        v = VectorOperations.validate_vector(v, "v")
        return scalar * v
    @staticmethod
    def dot_product(u: np.ndarray, v: np.ndarray) -> Union[float, complex]:
        """
        Dot product (inner product) of two vectors.
        Parameters:
            u, v: Input vectors
        Returns:
            u · v
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        return np.dot(u, v)
    @staticmethod
    def outer_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Outer product of two vectors.
        Parameters:
            u, v: Input vectors
        Returns:
            u ⊗ v (matrix where result[i,j] = u[i] * v[j])
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        return np.outer(u, v)
    @staticmethod
    def cross_product(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Cross product of two 3D vectors.
        Parameters:
            u, v: 3D vectors
        Returns:
            u × v
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != 3 or len(v) != 3:
            raise ValueError("Cross product requires 3D vectors")
        return np.cross(u, v)
    @staticmethod
    def triple_scalar_product(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> Union[float, complex]:
        """
        Scalar triple product: u · (v × w).
        Parameters:
            u, v, w: 3D vectors
        Returns:
            u · (v × w)
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        w = VectorOperations.validate_vector(w, "w")
        if len(u) != 3 or len(v) != 3 or len(w) != 3:
            raise ValueError("Triple scalar product requires 3D vectors")
        cross_vw = VectorOperations.cross_product(v, w)
        return VectorOperations.dot_product(u, cross_vw)
    @staticmethod
    def magnitude(v: np.ndarray) -> float:
        """
        Vector magnitude (Euclidean norm).
        Parameters:
            v: Input vector
        Returns:
            ||v||_2
        """
        v = VectorOperations.validate_vector(v, "v")
        return np.linalg.norm(v)
    @staticmethod
    def normalize(v: np.ndarray, norm_type: Union[int, float, str] = 2) -> np.ndarray:
        """
        Normalize vector to unit length.
        Parameters:
            v: Input vector
            norm_type: Type of norm (1, 2, inf, etc.)
        Returns:
            v / ||v||
        """
        v = VectorOperations.validate_vector(v, "v")
        norm_v = np.linalg.norm(v, ord=norm_type)
        if norm_v < 1e-12:
            warnings.warn("Vector has near-zero norm, normalization may be unstable")
            return v
        return v / norm_v
    @staticmethod
    def distance(u: np.ndarray, v: np.ndarray, norm_type: Union[int, float, str] = 2) -> float:
        """
        Distance between two vectors.
        Parameters:
            u, v: Input vectors
            norm_type: Type of norm for distance calculation
        Returns:
            ||u - v||
        """
        diff = VectorOperations.subtract(u, v)
        return np.linalg.norm(diff, ord=norm_type)
    @staticmethod
    def angle_between(u: np.ndarray, v: np.ndarray, degrees: bool = False) -> float:
        """
        Angle between two vectors.
        Parameters:
            u, v: Input vectors
            degrees: Whether to return angle in degrees
        Returns:
            Angle between vectors in radians (or degrees)
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        # Normalize vectors
        u_norm = VectorOperations.normalize(u)
        v_norm = VectorOperations.normalize(v)
        # Compute angle using dot product
        cos_angle = np.clip(np.dot(u_norm, v_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if degrees:
            angle = np.degrees(angle)
        return angle
    @staticmethod
    def project(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Project vector u onto vector v.
        Parameters:
            u: Vector to project
            v: Vector to project onto
        Returns:
            Projection of u onto v
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        v_dot_v = np.dot(v, v)
        if v_dot_v < 1e-12:
            warnings.warn("Cannot project onto zero vector")
            return np.zeros_like(u)
        return (np.dot(u, v) / v_dot_v) * v
    @staticmethod
    def reject(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vector rejection: component of u orthogonal to v.
        Parameters:
            u: Vector to reject
            v: Vector to reject from
        Returns:
            Rejection of u from v (u - proj_v(u))
        """
        projection = VectorOperations.project(u, v)
        return VectorOperations.subtract(u, projection)
    @staticmethod
    def are_orthogonal(u: np.ndarray, v: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if two vectors are orthogonal.
        Parameters:
            u, v: Input vectors
            tolerance: Tolerance for orthogonality check
        Returns:
            True if vectors are orthogonal
        """
        dot_product = VectorOperations.dot_product(u, v)
        return abs(dot_product) < tolerance
    @staticmethod
    def are_parallel(u: np.ndarray, v: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if two vectors are parallel.
        Parameters:
            u, v: Input vectors
            tolerance: Tolerance for parallelism check
        Returns:
            True if vectors are parallel
        """
        u = VectorOperations.validate_vector(u, "u")
        v = VectorOperations.validate_vector(v, "v")
        if len(u) != len(v):
            raise ValueError(f"Vector dimensions must match: {len(u)} != {len(v)}")
        # Check if cross product is zero (for 3D vectors)
        if len(u) == 3:
            cross = VectorOperations.cross_product(u, v)
            return np.linalg.norm(cross) < tolerance
        # For general case, check if angle is 0 or π
        try:
            angle = VectorOperations.angle_between(u, v)
            return abs(angle) < tolerance or abs(angle - np.pi) < tolerance
        except:
            return False
    @staticmethod
    def gram_schmidt_orthogonalization(vectors: List[np.ndarray],
                                     normalize: bool = True) -> List[np.ndarray]:
        """
        Gram-Schmidt orthogonalization of vector list.
        Parameters:
            vectors: List of vectors to orthogonalize
            normalize: Whether to normalize the orthogonal vectors
        Returns:
            List of orthogonal (or orthonormal) vectors
        """
        if not vectors:
            return []
        # Validate all vectors
        vectors = [VectorOperations.validate_vector(v, f"vector_{i}") for i, v in enumerate(vectors)]
        # Check dimensions
        dim = len(vectors[0])
        for i, v in enumerate(vectors[1:], 1):
            if len(v) != dim:
                raise ValueError(f"All vectors must have same dimension. Vector {i} has dimension {len(v)}, expected {dim}")
        orthogonal_vectors = []
        for v in vectors:
            # Start with the original vector
            orthogonal_v = v.copy()
            # Subtract projections onto all previous orthogonal vectors
            for ortho_v in orthogonal_vectors:
                projection = VectorOperations.project(orthogonal_v, ortho_v)
                orthogonal_v = VectorOperations.subtract(orthogonal_v, projection)
            # Check if vector is linearly independent
            if np.linalg.norm(orthogonal_v) < 1e-12:
                warnings.warn(f"Vector is linearly dependent and will be skipped")
                continue
            # Normalize if requested
            if normalize:
                orthogonal_v = VectorOperations.normalize(orthogonal_v)
            orthogonal_vectors.append(orthogonal_v)
        return orthogonal_vectors
class VectorNorms:
    """
    Vector norm calculations and related operations.
    Features:
    - p-norms (including L1, L2, L∞)
    - Weighted norms
    - Norm comparisons and relations
    """
    @staticmethod
    def p_norm(v: np.ndarray, p: Union[int, float, str]) -> float:
        """
        Compute p-norm of vector.
        Parameters:
            v: Input vector
            p: Norm order (1, 2, inf, -inf, etc.)
        Returns:
            ||v||_p
        """
        v = VectorOperations.validate_vector(v, "v")
        return np.linalg.norm(v, ord=p)
    @staticmethod
    def l1_norm(v: np.ndarray) -> float:
        """L1 norm (Manhattan norm): sum of absolute values."""
        return VectorNorms.p_norm(v, 1)
    @staticmethod
    def l2_norm(v: np.ndarray) -> float:
        """L2 norm (Euclidean norm): square root of sum of squares."""
        return VectorNorms.p_norm(v, 2)
    @staticmethod
    def infinity_norm(v: np.ndarray) -> float:
        """L∞ norm (maximum norm): maximum absolute value."""
        return VectorNorms.p_norm(v, np.inf)
    @staticmethod
    def weighted_norm(v: np.ndarray, weights: np.ndarray, p: Union[int, float] = 2) -> float:
        """
        Weighted p-norm of vector.
        Parameters:
            v: Input vector
            weights: Weight vector (positive values)
            p: Norm order
        Returns:
            Weighted p-norm
        """
        v = VectorOperations.validate_vector(v, "v")
        weights = VectorOperations.validate_vector(weights, "weights")
        if len(v) != len(weights):
            raise ValueError(f"Vector and weights must have same length: {len(v)} != {len(weights)}")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        if p == np.inf:
            return np.max(weights * np.abs(v))
        elif p == 1:
            return np.sum(weights * np.abs(v))
        elif p == 2:
            return np.sqrt(np.sum(weights * v**2))
        else:
            return np.power(np.sum(weights * np.power(np.abs(v), p)), 1/p)
    @staticmethod
    def unit_vector_in_direction(direction: np.ndarray) -> np.ndarray:
        """
        Create unit vector in given direction.
        Parameters:
            direction: Direction vector
        Returns:
            Unit vector in direction
        """
        return VectorOperations.normalize(direction)
    @staticmethod
    def standard_basis_vector(dimension: int, index: int) -> np.ndarray:
        """
        Create standard basis vector (e_i).
        Parameters:
            dimension: Vector dimension
            index: Index of non-zero element (0-based)
        Returns:
            Standard basis vector
        """
        if index < 0 or index >= dimension:
            raise ValueError(f"Index {index} out of range for dimension {dimension}")
        e = np.zeros(dimension)
        e[index] = 1.0
        return e
def create_test_vectors() -> dict:
    """Create a set of test vectors for validation."""
    vectors = {}
    # Standard vectors
    vectors['zero_3d'] = np.array([0.0, 0.0, 0.0])
    vectors['unit_x'] = np.array([1.0, 0.0, 0.0])
    vectors['unit_y'] = np.array([0.0, 1.0, 0.0])
    vectors['unit_z'] = np.array([0.0, 0.0, 1.0])
    # Random vectors
    np.random.seed(42)
    vectors['random_3d'] = np.random.randn(3)
    vectors['random_5d'] = np.random.randn(5)
    vectors['random_10d'] = np.random.randn(10)
    # Specific test cases
    vectors['orthogonal_pair_1'] = np.array([1.0, 0.0, 0.0])
    vectors['orthogonal_pair_2'] = np.array([0.0, 1.0, 0.0])
    vectors['parallel_pair_1'] = np.array([1.0, 2.0, 3.0])
    vectors['parallel_pair_2'] = np.array([2.0, 4.0, 6.0])
    # Vectors for Gram-Schmidt test
    vectors['gs_test_1'] = np.array([1.0, 1.0, 0.0])
    vectors['gs_test_2'] = np.array([1.0, 0.0, 1.0])
    vectors['gs_test_3'] = np.array([0.0, 1.0, 1.0])
    return vectors