"""
Symbolic computation and algebraic manipulation.
This module provides symbolic algebra tools including:
- Expression manipulation
- Symbolic differentiation and integration
- Equation solving
- Series expansions
"""
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any, Callable
import re
from fractions import Fraction
from collections import defaultdict
import sympy as sp
class SymbolicExpression:
    """Symbolic expression representation and manipulation."""
    def __init__(self, expression: Union[str, sp.Expr]):
        """
        Initialize symbolic expression.
        Args:
            expression: String or SymPy expression
        """
        if isinstance(expression, str):
            self.expr = sp.sympify(expression)
        elif isinstance(expression, sp.Expr):
            self.expr = expression
        else:
            raise ValueError("Expression must be string or SymPy expression")
        self.variables = list(self.expr.free_symbols)
    def __str__(self) -> str:
        """String representation."""
        return str(self.expr)
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"SymbolicExpression({self.expr})"
    def __add__(self, other):
        """Addition operator."""
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self.expr + other.expr)
        else:
            return SymbolicExpression(self.expr + other)
    def __sub__(self, other):
        """Subtraction operator."""
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self.expr - other.expr)
        else:
            return SymbolicExpression(self.expr - other)
    def __mul__(self, other):
        """Multiplication operator."""
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self.expr * other.expr)
        else:
            return SymbolicExpression(self.expr * other)
    def __truediv__(self, other):
        """Division operator."""
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self.expr / other.expr)
        else:
            return SymbolicExpression(self.expr / other)
    def __pow__(self, other):
        """Power operator."""
        if isinstance(other, SymbolicExpression):
            return SymbolicExpression(self.expr ** other.expr)
        else:
            return SymbolicExpression(self.expr ** other)
    def simplify(self) -> 'SymbolicExpression':
        """Simplify the expression."""
        return SymbolicExpression(sp.simplify(self.expr))
    def expand(self) -> 'SymbolicExpression':
        """Expand the expression."""
        return SymbolicExpression(sp.expand(self.expr))
    def factor(self) -> 'SymbolicExpression':
        """Factor the expression."""
        return SymbolicExpression(sp.factor(self.expr))
    def collect(self, variable: str) -> 'SymbolicExpression':
        """Collect terms with respect to variable."""
        var = sp.Symbol(variable)
        return SymbolicExpression(sp.collect(self.expr, var))
    def substitute(self, substitutions: Dict[str, Union[float, str]]) -> 'SymbolicExpression':
        """Substitute variables with values or expressions."""
        subs_dict = {}
        for var, val in substitutions.items():
            subs_dict[sp.Symbol(var)] = val
        return SymbolicExpression(self.expr.subs(subs_dict))
    def evaluate(self, values: Dict[str, float]) -> float:
        """Evaluate expression numerically."""
        subs_dict = {}
        for var, val in values.items():
            subs_dict[sp.Symbol(var)] = val
        result = self.expr.subs(subs_dict)
        return float(result.evalf())
    def differentiate(self, variable: str, order: int = 1) -> 'SymbolicExpression':
        """Compute derivative with respect to variable."""
        var = sp.Symbol(variable)
        return SymbolicExpression(sp.diff(self.expr, var, order))
    def integrate(self, variable: str, limits: Optional[Tuple] = None) -> 'SymbolicExpression':
        """Integrate expression with respect to variable."""
        var = sp.Symbol(variable)
        if limits is None:
            result = sp.integrate(self.expr, var)
        else:
            result = sp.integrate(self.expr, (var, limits[0], limits[1]))
        return SymbolicExpression(result)
    def series(self, variable: str, point: float = 0, order: int = 6) -> 'SymbolicExpression':
        """Taylor series expansion."""
        var = sp.Symbol(variable)
        series_expr = sp.series(self.expr, var, point, order).removeO()
        return SymbolicExpression(series_expr)
    def limit(self, variable: str, point: float, direction: str = '+-') -> 'SymbolicExpression':
        """Compute limit."""
        var = sp.Symbol(variable)
        limit_expr = sp.limit(self.expr, var, point, direction)
        return SymbolicExpression(limit_expr)
    def solve(self, variable: str) -> List:
        """Solve equation for variable."""
        var = sp.Symbol(variable)
        solutions = sp.solve(self.expr, var)
        return [float(sol.evalf()) if sol.is_real else complex(sol.evalf()) for sol in solutions]
    def to_latex(self) -> str:
        """Convert to LaTeX representation."""
        return sp.latex(self.expr)
    def to_numpy_function(self, variables: Optional[List[str]] = None) -> Callable:
        """Convert to NumPy function."""
        if variables is None:
            variables = [str(var) for var in self.variables]
        var_symbols = [sp.Symbol(var) for var in variables]
        return sp.lambdify(var_symbols, self.expr, 'numpy')
class SymbolicMatrix:
    """Symbolic matrix operations."""
    def __init__(self, matrix: Union[List[List], np.ndarray, sp.Matrix]):
        """
        Initialize symbolic matrix.
        Args:
            matrix: Matrix data
        """
        if isinstance(matrix, (list, np.ndarray)):
            self.matrix = sp.Matrix(matrix)
        elif isinstance(matrix, sp.Matrix):
            self.matrix = matrix
        else:
            raise ValueError("Matrix must be list, numpy array, or SymPy Matrix")
        self.shape = self.matrix.shape
    def __str__(self) -> str:
        """String representation."""
        return str(self.matrix)
    def __add__(self, other):
        """Matrix addition."""
        if isinstance(other, SymbolicMatrix):
            return SymbolicMatrix(self.matrix + other.matrix)
        else:
            return SymbolicMatrix(self.matrix + other)
    def __sub__(self, other):
        """Matrix subtraction."""
        if isinstance(other, SymbolicMatrix):
            return SymbolicMatrix(self.matrix - other.matrix)
        else:
            return SymbolicMatrix(self.matrix - other)
    def __mul__(self, other):
        """Matrix multiplication."""
        if isinstance(other, SymbolicMatrix):
            return SymbolicMatrix(self.matrix * other.matrix)
        else:
            return SymbolicMatrix(self.matrix * other)
    def transpose(self) -> 'SymbolicMatrix':
        """Matrix transpose."""
        return SymbolicMatrix(self.matrix.T)
    def determinant(self) -> SymbolicExpression:
        """Matrix determinant."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("Determinant only defined for square matrices")
        return SymbolicExpression(self.matrix.det())
    def inverse(self) -> 'SymbolicMatrix':
        """Matrix inverse."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("Inverse only defined for square matrices")
        return SymbolicMatrix(self.matrix.inv())
    def eigenvalues(self) -> List[SymbolicExpression]:
        """Matrix eigenvalues."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("Eigenvalues only defined for square matrices")
        eigvals = self.matrix.eigenvals()
        return [SymbolicExpression(val) for val in eigvals.keys()]
    def eigenvectors(self) -> List[Tuple[SymbolicExpression, int, List]]:
        """Matrix eigenvectors."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("Eigenvectors only defined for square matrices")
        eigvects = self.matrix.eigenvects()
        result = []
        for eigenval, multiplicity, eigenvectors in eigvects:
            eigvec_list = [SymbolicMatrix(vec) for vec in eigenvectors]
            result.append((SymbolicExpression(eigenval), multiplicity, eigvec_list))
        return result
    def trace(self) -> SymbolicExpression:
        """Matrix trace."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("Trace only defined for square matrices")
        return SymbolicExpression(self.matrix.trace())
    def rank(self) -> int:
        """Matrix rank."""
        return self.matrix.rank()
    def nullspace(self) -> List['SymbolicMatrix']:
        """Matrix nullspace."""
        null_vecs = self.matrix.nullspace()
        return [SymbolicMatrix(vec) for vec in null_vecs]
    def row_echelon_form(self) -> 'SymbolicMatrix':
        """Row echelon form."""
        rref_matrix, _ = self.matrix.rref()
        return SymbolicMatrix(rref_matrix)
    def simplify(self) -> 'SymbolicMatrix':
        """Simplify all matrix elements."""
        simplified = self.matrix.applyfunc(sp.simplify)
        return SymbolicMatrix(simplified)
    def substitute(self, substitutions: Dict[str, Union[float, str]]) -> 'SymbolicMatrix':
        """Substitute variables in matrix."""
        subs_dict = {}
        for var, val in substitutions.items():
            subs_dict[sp.Symbol(var)] = val
        substituted = self.matrix.subs(subs_dict)
        return SymbolicMatrix(substituted)
class EquationSolver:
    """Symbolic equation solver."""
    @staticmethod
    def solve_linear_system(equations: List[str], variables: List[str]) -> Dict[str, SymbolicExpression]:
        """
        Solve system of linear equations.
        Args:
            equations: List of equation strings
            variables: List of variable names
        Returns:
            Dictionary of solutions
        """
        # Convert to SymPy equations
        sp_equations = []
        for eq in equations:
            if '=' in eq:
                left, right = eq.split('=')
                sp_equations.append(sp.Eq(sp.sympify(left), sp.sympify(right)))
            else:
                sp_equations.append(sp.sympify(eq))
        # Convert variables to symbols
        sp_variables = [sp.Symbol(var) for var in variables]
        # Solve system
        solutions = sp.solve(sp_equations, sp_variables)
        # Convert back to SymbolicExpression objects
        result = {}
        if isinstance(solutions, dict):
            for var, sol in solutions.items():
                result[str(var)] = SymbolicExpression(sol)
        elif isinstance(solutions, list):
            for i, sol in enumerate(solutions):
                if isinstance(sol, dict):
                    for var, val in sol.items():
                        result[f"{var}_{i}"] = SymbolicExpression(val)
        return result
    @staticmethod
    def solve_polynomial(polynomial: str, variable: str) -> List[SymbolicExpression]:
        """
        Solve polynomial equation.
        Args:
            polynomial: Polynomial string
            variable: Variable to solve for
        Returns:
            List of roots
        """
        poly_expr = sp.sympify(polynomial)
        var = sp.Symbol(variable)
        roots = sp.solve(poly_expr, var)
        return [SymbolicExpression(root) for root in roots]
    @staticmethod
    def solve_differential_equation(equation: str, function: str,
                                  variable: str) -> SymbolicExpression:
        """
        Solve ordinary differential equation.
        Args:
            equation: ODE string
            function: Function name (e.g., 'y')
            variable: Independent variable (e.g., 'x')
        Returns:
            General solution
        """
        x = sp.Symbol(variable)
        y = sp.Function(function)
        # Parse equation (simplified parsing)
        eq_expr = sp.sympify(equation.replace(f"{function}'", f"Derivative({function}(x), x)"))
        eq_expr = eq_expr.replace(sp.Symbol(function), y(x))
        # Solve ODE
        solution = sp.dsolve(eq_expr, y(x))
        return SymbolicExpression(solution.rhs)
class SeriesExpansion:
    """Series expansion utilities."""
    @staticmethod
    def taylor_series(expression: str, variable: str, point: float = 0,
                     order: int = 6) -> SymbolicExpression:
        """
        Taylor series expansion.
        Args:
            expression: Expression string
            variable: Expansion variable
            point: Expansion point
            order: Order of expansion
        Returns:
            Series expansion
        """
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        series = sp.series(expr, var, point, order).removeO()
        return SymbolicExpression(series)
    @staticmethod
    def laurent_series(expression: str, variable: str, point: float = 0,
                      order: int = 6) -> SymbolicExpression:
        """
        Laurent series expansion.
        Args:
            expression: Expression string
            variable: Expansion variable
            point: Expansion point
            order: Order of expansion
        Returns:
            Laurent series
        """
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        # Laurent series includes negative powers
        series = sp.series(expr, var, point, order, dir='+').removeO()
        return SymbolicExpression(series)
    @staticmethod
    def fourier_series(expression: str, variable: str, period: float = 2*np.pi,
                      n_terms: int = 5) -> SymbolicExpression:
        """
        Fourier series expansion.
        Args:
            expression: Expression string
            variable: Variable name
            period: Period of function
            n_terms: Number of terms
        Returns:
            Fourier series
        """
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        # Calculate Fourier coefficients
        a0 = (2/period) * sp.integrate(expr, (var, 0, period))
        series = a0 / 2
        for n in range(1, n_terms + 1):
            # Cosine coefficient
            an = (2/period) * sp.integrate(expr * sp.cos(2*np.pi*n*var/period),
                                         (var, 0, period))
            # Sine coefficient
            bn = (2/period) * sp.integrate(expr * sp.sin(2*np.pi*n*var/period),
                                         (var, 0, period))
            series += an * sp.cos(2*np.pi*n*var/period) + bn * sp.sin(2*np.pi*n*var/period)
        return SymbolicExpression(series)
class SymbolicIntegration:
    """Symbolic integration utilities."""
    @staticmethod
    def indefinite_integral(expression: str, variable: str) -> SymbolicExpression:
        """
        Compute indefinite integral.
        Args:
            expression: Expression to integrate
            variable: Integration variable
        Returns:
            Indefinite integral
        """
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        integral = sp.integrate(expr, var)
        return SymbolicExpression(integral)
    @staticmethod
    def definite_integral(expression: str, variable: str,
                         limits: Tuple[float, float]) -> SymbolicExpression:
        """
        Compute definite integral.
        Args:
            expression: Expression to integrate
            variable: Integration variable
            limits: Integration limits (a, b)
        Returns:
            Definite integral value
        """
        expr = sp.sympify(expression)
        var = sp.Symbol(variable)
        integral = sp.integrate(expr, (var, limits[0], limits[1]))
        return SymbolicExpression(integral)
    @staticmethod
    def multiple_integral(expression: str, variables: List[str],
                         limits: List[Tuple[float, float]]) -> SymbolicExpression:
        """
        Compute multiple integral.
        Args:
            expression: Expression to integrate
            variables: List of integration variables
            limits: List of limits for each variable
        Returns:
            Multiple integral value
        """
        expr = sp.sympify(expression)
        # Build integration specification
        integration_vars = []
        for var, (a, b) in zip(variables, limits):
            integration_vars.append((sp.Symbol(var), a, b))
        integral = sp.integrate(expr, *integration_vars)
        return SymbolicExpression(integral)
    @staticmethod
    def line_integral(vector_field: List[str], path: List[str],
                     parameter: str, limits: Tuple[float, float]) -> SymbolicExpression:
        """
        Compute line integral of vector field.
        Args:
            vector_field: Vector field components [Fx, Fy, Fz]
            path: Parametric path [x(t), y(t), z(t)]
            parameter: Path parameter
            limits: Parameter limits
        Returns:
            Line integral value
        """
        t = sp.Symbol(parameter)
        # Convert to SymPy expressions
        F = [sp.sympify(component) for component in vector_field]
        r = [sp.sympify(component) for component in path]
        # Calculate dr/dt
        dr = [sp.diff(component, t) for component in r]
        # Dot product F · dr
        integrand = sum(F[i] * dr[i] for i in range(len(F)))
        # Substitute path into vector field
        variables = ['x', 'y', 'z']
        for i, var in enumerate(variables[:len(r)]):
            integrand = integrand.subs(sp.Symbol(var), r[i])
        # Integrate
        integral = sp.integrate(integrand, (t, limits[0], limits[1]))
        return SymbolicExpression(integral)
class SymbolicTransforms:
    """Symbolic transform operations."""
    @staticmethod
    def laplace_transform(expression: str, variable: str = 't',
                         transform_var: str = 's') -> SymbolicExpression:
        """
        Compute Laplace transform.
        Args:
            expression: Expression to transform
            variable: Original variable (usually t)
            transform_var: Transform variable (usually s)
        Returns:
            Laplace transform
        """
        expr = sp.sympify(expression)
        t = sp.Symbol(variable)
        s = sp.Symbol(transform_var)
        # Compute Laplace transform
        L_transform = sp.laplace_transform(expr, t, s)
        return SymbolicExpression(L_transform[0])
    @staticmethod
    def inverse_laplace_transform(expression: str, transform_var: str = 's',
                                 variable: str = 't') -> SymbolicExpression:
        """
        Compute inverse Laplace transform.
        Args:
            expression: Expression to transform
            transform_var: Transform variable (usually s)
            variable: Target variable (usually t)
        Returns:
            Inverse Laplace transform
        """
        expr = sp.sympify(expression)
        s = sp.Symbol(transform_var)
        t = sp.Symbol(variable)
        # Compute inverse Laplace transform
        inv_transform = sp.inverse_laplace_transform(expr, s, t)
        return SymbolicExpression(inv_transform)
    @staticmethod
    def fourier_transform(expression: str, variable: str = 'x',
                         transform_var: str = 'k') -> SymbolicExpression:
        """
        Compute Fourier transform.
        Args:
            expression: Expression to transform
            variable: Original variable
            transform_var: Transform variable
        Returns:
            Fourier transform
        """
        expr = sp.sympify(expression)
        x = sp.Symbol(variable)
        k = sp.Symbol(transform_var)
        # Compute Fourier transform
        F_transform = sp.fourier_transform(expr, x, k)
        return SymbolicExpression(F_transform)
    @staticmethod
    def inverse_fourier_transform(expression: str, transform_var: str = 'k',
                                 variable: str = 'x') -> SymbolicExpression:
        """
        Compute inverse Fourier transform.
        Args:
            expression: Expression to transform
            transform_var: Transform variable
            variable: Target variable
        Returns:
            Inverse Fourier transform
        """
        expr = sp.sympify(expression)
        k = sp.Symbol(transform_var)
        x = sp.Symbol(variable)
        # Compute inverse Fourier transform
        inv_transform = sp.inverse_fourier_transform(expr, k, x)
        return SymbolicExpression(inv_transform)