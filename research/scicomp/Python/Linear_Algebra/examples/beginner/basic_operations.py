"""
Basic Linear Algebra Operations - Beginner Example
This example demonstrates fundamental linear algebra operations including
vector operations, matrix arithmetic, and basic problem solving.
Learning Objectives:
- Understand vector and matrix representations
- Perform basic arithmetic operations
- Compute norms, dot products, and cross products
- Solve simple linear systems
- Visualize geometric interpretations
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
# Add Linear_Algebra package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.vector_operations import VectorOperations, VectorNorms
from core.matrix_operations import MatrixOperations
from core.linear_systems import DirectSolvers
def main():
    """Run basic linear algebra operations example."""
    print("Basic Linear Algebra Operations - Beginner Example")
    print("=" * 55)
    print("This example covers fundamental vector and matrix operations")
    print("Learning: Basic operations, geometric interpretation, simple systems\n")
    # Set up Berkeley color scheme
    berkeley_blue = '#003262'
    california_gold = '#FDB515'
    # Vector operations demonstration
    demonstrate_vector_operations()
    # Matrix operations demonstration
    demonstrate_matrix_operations()
    # Geometric interpretation
    demonstrate_geometric_concepts()
    # Simple linear systems
    demonstrate_linear_systems()
    # Real-world application
    physics_application_example()
    print("\n" + "=" * 55)
    print("Basic Linear Algebra Complete!")
    print("Key Learning Points:")
    print("• Vectors represent quantities with magnitude and direction")
    print("• Matrices can represent transformations and systems of equations")
    print("• Dot products measure alignment, cross products measure perpendicularity")
    print("• Linear systems appear everywhere in science and engineering")
def demonstrate_vector_operations():
    """Demonstrate basic vector operations."""
    print("Vector Operations")
    print("=" * 20)
    # Create example vectors
    print("1. Creating Vectors")
    print("-" * 18)
    u = np.array([3, 4])
    v = np.array([1, 2])
    w = np.array([2, -1, 3])  # 3D vector
    print(f"2D Vector u: {u}")
    print(f"2D Vector v: {v}")
    print(f"3D Vector w: {w}")
    # Basic arithmetic
    print("\n2. Vector Arithmetic")
    print("-" * 18)
    # Addition and subtraction
    sum_uv = VectorOperations.add(u, v)
    diff_uv = VectorOperations.subtract(u, v)
    print(f"u + v = {sum_uv}")
    print(f"u - v = {diff_uv}")
    # Scalar multiplication
    scaled_u = VectorOperations.scalar_multiply(2.5, u)
    print(f"2.5 * u = {scaled_u}")
    # Vector magnitude and normalization
    print("\n3. Vector Magnitude and Direction")
    print("-" * 32)
    mag_u = VectorOperations.magnitude(u)
    mag_v = VectorOperations.magnitude(v)
    print(f"Magnitude of u: {mag_u:.3f}")
    print(f"Magnitude of v: {mag_v:.3f}")
    # Unit vectors
    u_unit = VectorOperations.normalize(u)
    v_unit = VectorOperations.normalize(v)
    print(f"Unit vector in direction of u: {u_unit}")
    print(f"Unit vector in direction of v: {v_unit}")
    print(f"Magnitude of normalized u: {VectorOperations.magnitude(u_unit):.3f}")
    # Dot product
    print("\n4. Dot Product (measures alignment)")
    print("-" * 35)
    dot_uv = VectorOperations.dot_product(u, v)
    angle_uv = VectorOperations.angle_between(u, v, degrees=True)
    print(f"u · v = {dot_uv}")
    print(f"Angle between u and v: {angle_uv:.1f} degrees")
    # Demonstrate orthogonal vectors
    orthogonal_to_u = np.array([-u[1], u[0]])  # Rotate 90 degrees
    dot_orthogonal = VectorOperations.dot_product(u, orthogonal_to_u)
    print(f"Vector orthogonal to u: {orthogonal_to_u}")
    print(f"u · orthogonal_vector = {dot_orthogonal} (should be 0)")
    # Cross product (3D)
    print("\n5. Cross Product (3D vectors)")
    print("-" * 28)
    a_3d = np.array([1, 0, 0])  # x-axis
    b_3d = np.array([0, 1, 0])  # y-axis
    cross_ab = VectorOperations.cross_product(a_3d, b_3d)
    print(f"x-axis × y-axis = {cross_ab} (should be z-axis)")
    # Verify orthogonality
    print(f"Cross product is orthogonal to both inputs:")
    print(f"  (x × y) · x = {VectorOperations.dot_product(cross_ab, a_3d)}")
    print(f"  (x × y) · y = {VectorOperations.dot_product(cross_ab, b_3d)}")
    # Vector norms
    print("\n6. Different Vector Norms")
    print("-" * 24)
    test_vector = np.array([3, 4, 0])
    l1_norm = VectorNorms.l1_norm(test_vector)
    l2_norm = VectorNorms.l2_norm(test_vector)
    inf_norm = VectorNorms.infinity_norm(test_vector)
    print(f"Vector: {test_vector}")
    print(f"L1 norm (Manhattan): {l1_norm}")
    print(f"L2 norm (Euclidean): {l2_norm}")
    print(f"L∞ norm (Maximum): {inf_norm}")
def demonstrate_matrix_operations():
    """Demonstrate basic matrix operations."""
    print("\n\nMatrix Operations")
    print("=" * 20)
    # Create example matrices
    print("1. Creating Matrices")
    print("-" * 18)
    A = np.array([[2, 1], [1, 3]])
    B = np.array([[1, 2], [3, 1]])
    C = np.array([[1, 2, 1], [0, 1, 2]])  # Non-square
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C (2×3):")
    print(C)
    # Basic arithmetic
    print("\n2. Matrix Arithmetic")
    print("-" * 18)
    # Addition and subtraction
    sum_AB = MatrixOperations.matrix_add(A, B)
    diff_AB = MatrixOperations.matrix_subtract(A, B)
    print("A + B:")
    print(sum_AB)
    print("\nA - B:")
    print(diff_AB)
    # Matrix multiplication
    product_AB = MatrixOperations.matrix_multiply(A, B)
    product_AC = MatrixOperations.matrix_multiply(A, C)
    print("\nA × B:")
    print(product_AB)
    print("\nA × C:")
    print(product_AC)
    # Matrix properties
    print("\n3. Matrix Properties")
    print("-" * 18)
    trace_A = MatrixOperations.trace(A)
    det_A = MatrixOperations.determinant(A)
    rank_A = MatrixOperations.rank(A)
    print(f"Trace of A: {trace_A}")
    print(f"Determinant of A: {det_A}")
    print(f"Rank of A: {rank_A}")
    # Transpose
    A_T = MatrixOperations.transpose(A)
    print("\nA transpose:")
    print(A_T)
    # Matrix norms
    frobenius_norm = MatrixOperations.frobenius_norm(A)
    spectral_norm = MatrixOperations.spectral_norm(A)
    condition_number = MatrixOperations.condition_number(A)
    print(f"\nFrobenius norm of A: {frobenius_norm:.3f}")
    print(f"Spectral norm of A: {spectral_norm:.3f}")
    print(f"Condition number of A: {condition_number:.3f}")
    # Matrix powers
    print("\n4. Matrix Powers")
    print("-" * 15)
    A_squared = MatrixOperations.matrix_power(A, 2)
    A_cubed = MatrixOperations.matrix_power(A, 3)
    print("A²:")
    print(A_squared)
    print("\nA³:")
    print(A_cubed)
def demonstrate_geometric_concepts():
    """Demonstrate geometric interpretation of linear algebra."""
    print("\n\nGeometric Concepts")
    print("=" * 20)
    # Vector projections
    print("1. Vector Projections")
    print("-" * 19)
    u = np.array([4, 2])
    v = np.array([3, 0])  # Along x-axis
    proj_u_on_v = VectorOperations.project(u, v)
    rejection = VectorOperations.reject(u, v)
    print(f"Vector u: {u}")
    print(f"Vector v: {v}")
    print(f"Projection of u onto v: {proj_u_on_v}")
    print(f"Rejection (orthogonal component): {rejection}")
    # Verify orthogonality
    dot_check = VectorOperations.dot_product(proj_u_on_v, rejection)
    print(f"Projection · Rejection = {dot_check:.10f} (should be 0)")
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    # Plot vectors
    plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='u')
    plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='v')
    plt.quiver(0, 0, proj_u_on_v[0], proj_u_on_v[1], angles='xy', scale_units='xy', scale=1, color='green', width=0.005, label='proj_v(u)')
    # Draw projection lines
    plt.plot([u[0], proj_u_on_v[0]], [u[1], proj_u_on_v[1]], 'k--', alpha=0.7, label='rejection')
    plt.xlim(-1, 5)
    plt.ylim(-1, 3)
    plt.grid(True, alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vector Projection')
    plt.legend()
    plt.axis('equal')
    # 3D cross product visualization
    plt.subplot(1, 2, 2, projection='3d')
    ax = plt.gca()
    # 3D vectors
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    c = VectorOperations.cross_product(a, b)
    # Plot vectors
    ax.quiver(0, 0, 0, a[0], a[1], a[2], color='red', arrow_length_ratio=0.1, label='a')
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='blue', arrow_length_ratio=0.1, label='b')
    ax.quiver(0, 0, 0, c[0], c[1], c[2], color='green', arrow_length_ratio=0.1, label='a × b')
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.set_zlim([0, 1.2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cross Product in 3D')
    ax.legend()
    plt.tight_layout()
    plt.show()
    # Gram-Schmidt orthogonalization
    print("\n2. Gram-Schmidt Orthogonalization")
    print("-" * 34)
    # Start with linearly independent but not orthogonal vectors
    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 1])
    v3 = np.array([0, 1, 1])
    vectors = [v1, v2, v3]
    orthogonal_vectors = VectorOperations.gram_schmidt_orthogonalization(vectors)
    print("Original vectors:")
    for i, v in enumerate(vectors):
        print(f"  v{i+1}: {v}")
    print("\nOrthogonal vectors:")
    for i, v in enumerate(orthogonal_vectors):
        print(f"  u{i+1}: {v}")
        print(f"  |u{i+1}|: {VectorOperations.magnitude(v):.3f}")
    # Verify orthogonality
    print("\nOrthogonality check:")
    for i in range(len(orthogonal_vectors)):
        for j in range(i+1, len(orthogonal_vectors)):
            dot = VectorOperations.dot_product(orthogonal_vectors[i], orthogonal_vectors[j])
            print(f"  u{i+1} · u{j+1} = {dot:.10f}")
def demonstrate_linear_systems():
    """Demonstrate solving linear systems."""
    print("\n\nLinear Systems")
    print("=" * 15)
    print("1. Simple 2×2 System")
    print("-" * 19)
    # System: 2x + y = 5
    #         x + 3y = 6
    A = np.array([[2, 1], [1, 3]])
    b = np.array([5, 6])
    print("System of equations:")
    print("  2x + y = 5")
    print("  x + 3y = 6")
    print("\nMatrix form: A x = b")
    print("A =")
    print(A)
    print(f"b = {b}")
    # Solve using LU decomposition
    result = DirectSolvers.lu_solve(A, b)
    if result.success:
        x, y = result.solution
        print(f"\nSolution: x = {x:.3f}, y = {y:.3f}")
        # Verify solution
        verification = A @ result.solution
        print(f"Verification: A × solution = {verification}")
        print(f"Original b = {b}")
        print(f"Residual = {result.residual_norm:.2e}")
    else:
        print("Failed to solve system")
    print("\n2. Overdetermined System (Least Squares)")
    print("-" * 40)
    # More equations than unknowns - find best fit
    A_over = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
    b_over = np.array([2.1, 2.9, 4.2, 4.8])  # Noisy line data
    print("Overdetermined system (4 equations, 2 unknowns):")
    print("Finding line y = mx + c that best fits points")
    print("Points: (1, 2.1), (2, 2.9), (3, 4.2), (4, 4.8)")
    # Solve using QR (least squares)
    result = DirectSolvers.qr_solve(A_over, b_over)
    if result.success:
        c, m = result.solution  # y = mx + c, so [c, m] = [intercept, slope]
        print(f"\nBest fit line: y = {m:.3f}x + {c:.3f}")
        print(f"Residual norm: {result.residual_norm:.3f}")
        # Plot the fit
        plt.figure(figsize=(8, 6))
        x_data = np.array([1, 2, 3, 4])
        y_data = b_over
        plt.scatter(x_data, y_data, color='#003262', s=50, label='Data points', zorder=5)
        x_line = np.linspace(0.5, 4.5, 100)
        y_line = m * x_line + c
        plt.plot(x_line, y_line, color='#FDB515', linewidth=2, label=f'Best fit: y = {m:.3f}x + {c:.3f}')
        # Show residuals
        y_predicted = A_over @ result.solution
        for i in range(len(x_data)):
            plt.plot([x_data[i], x_data[i]], [y_data[i], y_predicted[i]], 'r--', alpha=0.7)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Least Squares Line Fitting')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
def physics_application_example():
    """Demonstrate linear algebra in physics applications."""
    print("\n\nPhysics Application: Force Balance")
    print("=" * 35)
    print("Problem: Find tension forces in cables supporting a mass")
    print("Three cables support a 100 kg mass at equilibrium")
    print("Cable 1: 30° above horizontal")
    print("Cable 2: 45° above horizontal")
    print("Cable 3: Vertical")
    # Force balance equations:
    # Sum of forces in x-direction: T1*cos(30°) - T2*cos(45°) = 0
    # Sum of forces in y-direction: T1*sin(30°) + T2*sin(45°) + T3 = mg
    # Cable 3 is vertical, so T3 provides no horizontal force
    mass = 100  # kg
    g = 9.81    # m/s²
    weight = mass * g
    # Angles in radians
    angle1 = np.radians(30)
    angle2 = np.radians(45)
    # Set up system of equations
    # [cos(30°)  -cos(45°)   0] [T1]   [0]
    # [sin(30°)   sin(45°)   1] [T2] = [mg]
    #                              [T3]
    A = np.array([
        [np.cos(angle1), -np.cos(angle2), 0],
        [np.sin(angle1),  np.sin(angle2), 1]
    ])
    b = np.array([0, weight])
    print(f"\nWeight = {weight:.1f} N")
    print("\nForce balance equations:")
    print("Horizontal: T₁cos(30°) - T₂cos(45°) = 0")
    print("Vertical:   T₁sin(30°) + T₂sin(45°) + T₃ = mg")
    # This is an underdetermined system (2 equations, 3 unknowns)
    # Add constraint that T3 = 0 (no vertical cable) to make it solvable
    print("\nAssuming no vertical cable (T₃ = 0):")
    A_reduced = A[:, :2]  # Remove T3 column
    result = DirectSolvers.lu_solve(A_reduced, b)
    if result.success:
        T1, T2 = result.solution
        print(f"\nTension forces:")
        print(f"Cable 1 (30°): T₁ = {T1:.1f} N")
        print(f"Cable 2 (45°): T₂ = {T2:.1f} N")
        # Verify force balance
        F_x = T1 * np.cos(angle1) - T2 * np.cos(angle2)
        F_y = T1 * np.sin(angle1) + T2 * np.sin(angle2)
        print(f"\nVerification:")
        print(f"Net horizontal force: {F_x:.2e} N (should be 0)")
        print(f"Net vertical force: {F_y:.1f} N (should equal weight)")
        print(f"Force balance error: {abs(F_y - weight):.2e} N")
        # Visualization
        plt.figure(figsize=(10, 6))
        # Plot force diagram
        plt.subplot(1, 2, 1)
        # Mass at origin
        plt.scatter(0, 0, s=200, c='black', marker='s', label='Mass (100 kg)')
        # Cable forces
        scale = 0.002  # Scale for visualization
        # Cable 1
        F1_x = T1 * np.cos(angle1) * scale
        F1_y = T1 * np.sin(angle1) * scale
        plt.arrow(0, 0, F1_x, F1_y, head_width=0.02, head_length=0.01, fc='blue', ec='blue', label=f'T₁ = {T1:.0f} N')
        # Cable 2
        F2_x = -T2 * np.cos(angle2) * scale
        F2_y = T2 * np.sin(angle2) * scale
        plt.arrow(0, 0, F2_x, F2_y, head_width=0.02, head_length=0.01, fc='red', ec='red', label=f'T₂ = {T2:.0f} N')
        # Weight
        plt.arrow(0, 0, 0, -weight * scale, head_width=0.02, head_length=0.01, fc='green', ec='green', label=f'Weight = {weight:.0f} N')
        plt.xlim(-0.4, 0.4)
        plt.ylim(-0.3, 0.3)
        plt.xlabel('Horizontal Position')
        plt.ylabel('Vertical Position')
        plt.title('Force Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        # Plot angle diagram
        plt.subplot(1, 2, 2)
        # Draw cables
        cable_length = 1.0
        x1 = cable_length * np.cos(angle1)
        y1 = cable_length * np.sin(angle1)
        x2 = -cable_length * np.cos(angle2)
        y2 = cable_length * np.sin(angle2)
        plt.plot([0, x1], [0, y1], 'b-', linewidth=3, label=f'Cable 1 (30°)')
        plt.plot([0, x2], [0, y2], 'r-', linewidth=3, label=f'Cable 2 (45°)')
        # Mass
        plt.scatter(0, 0, s=200, c='black', marker='s', label='Mass')
        # Angle annotations
        plt.annotate(f'{np.degrees(angle1):.0f}°', xy=(0.3, 0.1), fontsize=12)
        plt.annotate(f'{np.degrees(angle2):.0f}°', xy=(-0.3, 0.1), fontsize=12)
        plt.xlim(-1.2, 1.2)
        plt.ylim(-0.2, 1.0)
        plt.xlabel('Horizontal Position')
        plt.ylabel('Vertical Position')
        plt.title('Cable Configuration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    main()