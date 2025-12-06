# symbolic_computation
**Module:** `Python/Symbolic_Algebra/core/symbolic_computation.py`
## Overview
Symbolic computation and algebraic manipulation.
This module provides symbolic algebra tools including:
- Expression manipulation
- Symbolic differentiation and integration
- Equation solving
- Series expansions
## Classes
### `SymbolicExpression`
Symbolic expression representation and manipulation.
#### Methods
##### `__init__(self, expression)`
Initialize symbolic expression.
Args:
expression: String or SymPy expression
**Source:** [Line 22](Python/Symbolic_Algebra/core/symbolic_computation.py#L22)
##### `__str__(self)`
String representation.
**Source:** [Line 38](Python/Symbolic_Algebra/core/symbolic_computation.py#L38)
##### `__repr__(self)`
String representation for debugging.
**Source:** [Line 42](Python/Symbolic_Algebra/core/symbolic_computation.py#L42)
##### `__add__(self, other)`
Addition operator.
**Source:** [Line 46](Python/Symbolic_Algebra/core/symbolic_computation.py#L46)
##### `__sub__(self, other)`
Subtraction operator.
**Source:** [Line 53](Python/Symbolic_Algebra/core/symbolic_computation.py#L53)
##### `__mul__(self, other)`
Multiplication operator.
**Source:** [Line 60](Python/Symbolic_Algebra/core/symbolic_computation.py#L60)
##### `__truediv__(self, other)`
Division operator.
**Source:** [Line 67](Python/Symbolic_Algebra/core/symbolic_computation.py#L67)
##### `__pow__(self, other)`
Power operator.
**Source:** [Line 74](Python/Symbolic_Algebra/core/symbolic_computation.py#L74)
##### `simplify(self)`
Simplify the expression.
**Source:** [Line 81](Python/Symbolic_Algebra/core/symbolic_computation.py#L81)
##### `expand(self)`
Expand the expression.
**Source:** [Line 85](Python/Symbolic_Algebra/core/symbolic_computation.py#L85)
##### `factor(self)`
Factor the expression.
**Source:** [Line 89](Python/Symbolic_Algebra/core/symbolic_computation.py#L89)
##### `collect(self, variable)`
Collect terms with respect to variable.
**Source:** [Line 93](Python/Symbolic_Algebra/core/symbolic_computation.py#L93)
##### `substitute(self, substitutions)`
Substitute variables with values or expressions.
**Source:** [Line 98](Python/Symbolic_Algebra/core/symbolic_computation.py#L98)
##### `evaluate(self, values)`
Evaluate expression numerically.
**Source:** [Line 106](Python/Symbolic_Algebra/core/symbolic_computation.py#L106)
##### `differentiate(self, variable, order)`
Compute derivative with respect to variable.
**Source:** [Line 115](Python/Symbolic_Algebra/core/symbolic_computation.py#L115)
##### `integrate(self, variable, limits)`
Integrate expression with respect to variable.
**Source:** [Line 120](Python/Symbolic_Algebra/core/symbolic_computation.py#L120)
##### `series(self, variable, point, order)`
Taylor series expansion.
**Source:** [Line 131](Python/Symbolic_Algebra/core/symbolic_computation.py#L131)
##### `limit(self, variable, point, direction)`
Compute limit.
**Source:** [Line 137](Python/Symbolic_Algebra/core/symbolic_computation.py#L137)
##### `solve(self, variable)`
Solve equation for variable.
**Source:** [Line 143](Python/Symbolic_Algebra/core/symbolic_computation.py#L143)
##### `to_latex(self)`
Convert to LaTeX representation.
**Source:** [Line 149](Python/Symbolic_Algebra/core/symbolic_computation.py#L149)
##### `to_numpy_function(self, variables)`
Convert to NumPy function.
**Source:** [Line 153](Python/Symbolic_Algebra/core/symbolic_computation.py#L153)
**Class Source:** [Line 19](Python/Symbolic_Algebra/core/symbolic_computation.py#L19)
### `SymbolicMatrix`
Symbolic matrix operations.
#### Methods
##### `__init__(self, matrix)`
Initialize symbolic matrix.
Args:
matrix: Matrix data
**Source:** [Line 165](Python/Symbolic_Algebra/core/symbolic_computation.py#L165)
##### `__str__(self)`
String representation.
**Source:** [Line 181](Python/Symbolic_Algebra/core/symbolic_computation.py#L181)
##### `__add__(self, other)`
Matrix addition.
**Source:** [Line 185](Python/Symbolic_Algebra/core/symbolic_computation.py#L185)
##### `__sub__(self, other)`
Matrix subtraction.
**Source:** [Line 192](Python/Symbolic_Algebra/core/symbolic_computation.py#L192)
##### `__mul__(self, other)`
Matrix multiplication.
**Source:** [Line 199](Python/Symbolic_Algebra/core/symbolic_computation.py#L199)
##### `transpose(self)`
Matrix transpose.
**Source:** [Line 206](Python/Symbolic_Algebra/core/symbolic_computation.py#L206)
##### `determinant(self)`
Matrix determinant.
**Source:** [Line 210](Python/Symbolic_Algebra/core/symbolic_computation.py#L210)
##### `inverse(self)`
Matrix inverse.
**Source:** [Line 217](Python/Symbolic_Algebra/core/symbolic_computation.py#L217)
##### `eigenvalues(self)`
Matrix eigenvalues.
**Source:** [Line 224](Python/Symbolic_Algebra/core/symbolic_computation.py#L224)
##### `eigenvectors(self)`
Matrix eigenvectors.
**Source:** [Line 232](Python/Symbolic_Algebra/core/symbolic_computation.py#L232)
##### `trace(self)`
Matrix trace.
**Source:** [Line 246](Python/Symbolic_Algebra/core/symbolic_computation.py#L246)
##### `rank(self)`
Matrix rank.
**Source:** [Line 253](Python/Symbolic_Algebra/core/symbolic_computation.py#L253)
##### `nullspace(self)`
Matrix nullspace.
**Source:** [Line 257](Python/Symbolic_Algebra/core/symbolic_computation.py#L257)
##### `row_echelon_form(self)`
Row echelon form.
**Source:** [Line 262](Python/Symbolic_Algebra/core/symbolic_computation.py#L262)
##### `simplify(self)`
Simplify all matrix elements.
**Source:** [Line 267](Python/Symbolic_Algebra/core/symbolic_computation.py#L267)
##### `substitute(self, substitutions)`
Substitute variables in matrix.
**Source:** [Line 272](Python/Symbolic_Algebra/core/symbolic_computation.py#L272)
**Class Source:** [Line 162](Python/Symbolic_Algebra/core/symbolic_computation.py#L162)
### `EquationSolver`
Symbolic equation solver.
#### Methods
##### `solve_linear_system(equations, variables)`
Solve system of linear equations.
Args:
equations: List of equation strings
variables: List of variable names
Returns:
Dictionary of solutions
**Source:** [Line 286](Python/Symbolic_Algebra/core/symbolic_computation.py#L286)
##### `solve_polynomial(polynomial, variable)`
Solve polynomial equation.
Args:
polynomial: Polynomial string
variable: Variable to solve for
Returns:
List of roots
**Source:** [Line 326](Python/Symbolic_Algebra/core/symbolic_computation.py#L326)
##### `solve_differential_equation(equation, function, variable)`
Solve ordinary differential equation.
Args:
equation: ODE string
function: Function name (e.g., 'y')
variable: Independent variable (e.g., 'x')
Returns:
General solution
**Source:** [Line 344](Python/Symbolic_Algebra/core/symbolic_computation.py#L344)
**Class Source:** [Line 282](Python/Symbolic_Algebra/core/symbolic_computation.py#L282)
### `SeriesExpansion`
Series expansion utilities.
#### Methods
##### `taylor_series(expression, variable, point, order)`
Taylor series expansion.
Args:
expression: Expression string
variable: Expansion variable
point: Expansion point
order: Order of expansion
Returns:
Series expansion
**Source:** [Line 374](Python/Symbolic_Algebra/core/symbolic_computation.py#L374)
##### `laurent_series(expression, variable, point, order)`
Laurent series expansion.
Args:
expression: Expression string
variable: Expansion variable
point: Expansion point
order: Order of expansion
Returns:
Laurent series
**Source:** [Line 395](Python/Symbolic_Algebra/core/symbolic_computation.py#L395)
##### `fourier_series(expression, variable, period, n_terms)`
Fourier series expansion.
Args:
expression: Expression string
variable: Variable name
period: Period of function
n_terms: Number of terms
Returns:
Fourier series
**Source:** [Line 417](Python/Symbolic_Algebra/core/symbolic_computation.py#L417)
**Class Source:** [Line 370](Python/Symbolic_Algebra/core/symbolic_computation.py#L370)
### `SymbolicIntegration`
Symbolic integration utilities.
#### Methods
##### `indefinite_integral(expression, variable)`
Compute indefinite integral.
Args:
expression: Expression to integrate
variable: Integration variable
Returns:
Indefinite integral
**Source:** [Line 456](Python/Symbolic_Algebra/core/symbolic_computation.py#L456)
##### `definite_integral(expression, variable, limits)`
Compute definite integral.
Args:
expression: Expression to integrate
variable: Integration variable
limits: Integration limits (a, b)
Returns:
Definite integral value
**Source:** [Line 474](Python/Symbolic_Algebra/core/symbolic_computation.py#L474)
##### `multiple_integral(expression, variables, limits)`
Compute multiple integral.
Args:
expression: Expression to integrate
variables: List of integration variables
limits: List of limits for each variable
Returns:
Multiple integral value
**Source:** [Line 494](Python/Symbolic_Algebra/core/symbolic_computation.py#L494)
##### `line_integral(vector_field, path, parameter, limits)`
Compute line integral of vector field.
Args:
vector_field: Vector field components [Fx, Fy, Fz]
path: Parametric path [x(t), y(t), z(t)]
parameter: Path parameter
limits: Parameter limits
Returns:
Line integral value
**Source:** [Line 518](Python/Symbolic_Algebra/core/symbolic_computation.py#L518)
**Class Source:** [Line 452](Python/Symbolic_Algebra/core/symbolic_computation.py#L452)
### `SymbolicTransforms`
Symbolic transform operations.
#### Methods
##### `laplace_transform(expression, variable, transform_var)`
Compute Laplace transform.
Args:
expression: Expression to transform
variable: Original variable (usually t)
transform_var: Transform variable (usually s)
Returns:
Laplace transform
**Source:** [Line 558](Python/Symbolic_Algebra/core/symbolic_computation.py#L558)
##### `inverse_laplace_transform(expression, transform_var, variable)`
Compute inverse Laplace transform.
Args:
expression: Expression to transform
transform_var: Transform variable (usually s)
variable: Target variable (usually t)
Returns:
Inverse Laplace transform
**Source:** [Line 580](Python/Symbolic_Algebra/core/symbolic_computation.py#L580)
##### `fourier_transform(expression, variable, transform_var)`
Compute Fourier transform.
Args:
expression: Expression to transform
variable: Original variable
transform_var: Transform variable
Returns:
Fourier transform
**Source:** [Line 602](Python/Symbolic_Algebra/core/symbolic_computation.py#L602)
##### `inverse_fourier_transform(expression, transform_var, variable)`
Compute inverse Fourier transform.
Args:
expression: Expression to transform
transform_var: Transform variable
variable: Target variable
Returns:
Inverse Fourier transform
**Source:** [Line 624](Python/Symbolic_Algebra/core/symbolic_computation.py#L624)
**Class Source:** [Line 554](Python/Symbolic_Algebra/core/symbolic_computation.py#L554)