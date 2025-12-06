# vector_operations
**Module:** `Python/Linear_Algebra/core/vector_operations.py`
## Overview
Vector Operations for Scientific Computing
Comprehensive vector operations including arithmetic, norms, products,
and specialized algorithms for scientific applications.
## Functions
### `create_test_vectors()`
Create a set of test vectors for validation.
**Source:** [Line 497](Python/Linear_Algebra/core/vector_operations.py#L497)
## Classes
### `VectorOperations`
Core vector operations for scientific computing.
Features:
- Vector arithmetic with dimension checking
- Vector norms and metrics
- Inner and outer products
- Cross products and vector geometry
- Vector projections and orthogonalization
#### Methods
##### `validate_vector(vector, name)`
Validate vector input and convert to numpy array.
**Source:** [Line 26](Python/Linear_Algebra/core/vector_operations.py#L26)
##### `add(u, v)`
Vector addition with dimension checking.
Parameters:
u, v: Input vectors
Returns:
u + v
**Source:** [Line 48](Python/Linear_Algebra/core/vector_operations.py#L48)
##### `subtract(u, v)`
Vector subtraction with dimension checking.
Parameters:
u, v: Input vectors
Returns:
u - v
**Source:** [Line 67](Python/Linear_Algebra/core/vector_operations.py#L67)
##### `scalar_multiply(scalar, v)`
Scalar multiplication of vector.
Parameters:
scalar: Scalar value
v: Input vector
Returns:
scalar * v
**Source:** [Line 86](Python/Linear_Algebra/core/vector_operations.py#L86)
##### `dot_product(u, v)`
Dot product (inner product) of two vectors.
Parameters:
u, v: Input vectors
Returns:
u · v
**Source:** [Line 101](Python/Linear_Algebra/core/vector_operations.py#L101)
##### `outer_product(u, v)`
Outer product of two vectors.
Parameters:
u, v: Input vectors
Returns:
u ⊗ v (matrix where result[i,j] = u[i] * v[j])
**Source:** [Line 120](Python/Linear_Algebra/core/vector_operations.py#L120)
##### `cross_product(u, v)`
Cross product of two 3D vectors.
Parameters:
u, v: 3D vectors
Returns:
u × v
**Source:** [Line 136](Python/Linear_Algebra/core/vector_operations.py#L136)
##### `triple_scalar_product(u, v, w)`
Scalar triple product: u · (v × w).
Parameters:
u, v, w: 3D vectors
Returns:
u · (v × w)
**Source:** [Line 155](Python/Linear_Algebra/core/vector_operations.py#L155)
##### `magnitude(v)`
Vector magnitude (Euclidean norm).
Parameters:
v: Input vector
Returns:
||v||_2
**Source:** [Line 176](Python/Linear_Algebra/core/vector_operations.py#L176)
##### `normalize(v, norm_type)`
Normalize vector to unit length.
Parameters:
v: Input vector
norm_type: Type of norm (1, 2, inf, etc.)
Returns:
v / ||v||
**Source:** [Line 190](Python/Linear_Algebra/core/vector_operations.py#L190)
##### `distance(u, v, norm_type)`
Distance between two vectors.
Parameters:
u, v: Input vectors
norm_type: Type of norm for distance calculation
Returns:
||u - v||
**Source:** [Line 212](Python/Linear_Algebra/core/vector_operations.py#L212)
##### `angle_between(u, v, degrees)`
Angle between two vectors.
Parameters:
u, v: Input vectors
degrees: Whether to return angle in degrees
Returns:
Angle between vectors in radians (or degrees)
**Source:** [Line 227](Python/Linear_Algebra/core/vector_operations.py#L227)
##### `project(u, v)`
Project vector u onto vector v.
Parameters:
u: Vector to project
v: Vector to project onto
Returns:
Projection of u onto v
**Source:** [Line 258](Python/Linear_Algebra/core/vector_operations.py#L258)
##### `reject(u, v)`
Vector rejection: component of u orthogonal to v.
Parameters:
u: Vector to reject
v: Vector to reject from
Returns:
Rejection of u from v (u - proj_v(u))
**Source:** [Line 284](Python/Linear_Algebra/core/vector_operations.py#L284)
##### `are_orthogonal(u, v, tolerance)`
Check if two vectors are orthogonal.
Parameters:
u, v: Input vectors
tolerance: Tolerance for orthogonality check
Returns:
True if vectors are orthogonal
**Source:** [Line 299](Python/Linear_Algebra/core/vector_operations.py#L299)
##### `are_parallel(u, v, tolerance)`
Check if two vectors are parallel.
Parameters:
u, v: Input vectors
tolerance: Tolerance for parallelism check
Returns:
True if vectors are parallel
**Source:** [Line 314](Python/Linear_Algebra/core/vector_operations.py#L314)
##### `gram_schmidt_orthogonalization(vectors, normalize)`
Gram-Schmidt orthogonalization of vector list.
Parameters:
vectors: List of vectors to orthogonalize
normalize: Whether to normalize the orthogonal vectors
Returns:
List of orthogonal (or orthonormal) vectors
**Source:** [Line 344](Python/Linear_Algebra/core/vector_operations.py#L344)
**Class Source:** [Line 13](Python/Linear_Algebra/core/vector_operations.py#L13)
### `VectorNorms`
Vector norm calculations and related operations.
Features:
- p-norms (including L1, L2, L∞)
- Weighted norms
- Norm comparisons and relations
#### Methods
##### `p_norm(v, p)`
Compute p-norm of vector.
Parameters:
v: Input vector
p: Norm order (1, 2, inf, -inf, etc.)
Returns:
||v||_p
**Source:** [Line 404](Python/Linear_Algebra/core/vector_operations.py#L404)
##### `l1_norm(v)`
L1 norm (Manhattan norm): sum of absolute values.
**Source:** [Line 419](Python/Linear_Algebra/core/vector_operations.py#L419)
##### `l2_norm(v)`
L2 norm (Euclidean norm): square root of sum of squares.
**Source:** [Line 424](Python/Linear_Algebra/core/vector_operations.py#L424)
##### `infinity_norm(v)`
L∞ norm (maximum norm): maximum absolute value.
**Source:** [Line 429](Python/Linear_Algebra/core/vector_operations.py#L429)
##### `weighted_norm(v, weights, p)`
Weighted p-norm of vector.
Parameters:
v: Input vector
weights: Weight vector (positive values)
p: Norm order
Returns:
Weighted p-norm
**Source:** [Line 434](Python/Linear_Algebra/core/vector_operations.py#L434)
##### `unit_vector_in_direction(direction)`
Create unit vector in given direction.
Parameters:
direction: Direction vector
Returns:
Unit vector in direction
**Source:** [Line 465](Python/Linear_Algebra/core/vector_operations.py#L465)
##### `standard_basis_vector(dimension, index)`
Create standard basis vector (e_i).
Parameters:
dimension: Vector dimension
index: Index of non-zero element (0-based)
Returns:
Standard basis vector
**Source:** [Line 478](Python/Linear_Algebra/core/vector_operations.py#L478)
**Class Source:** [Line 393](Python/Linear_Algebra/core/vector_operations.py#L393)