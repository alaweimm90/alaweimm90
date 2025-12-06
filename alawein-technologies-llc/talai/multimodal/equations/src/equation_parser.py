"""
Core Equation Parser for Mathematical Content

Handles parsing of various mathematical notation formats and
converts them to structured representations.
"""

import re
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.mathematica import parse_mathematica
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from lark import Lark, Transformer
import mathml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class EquationType(Enum):
    """Types of mathematical equations"""
    ALGEBRAIC = "algebraic"
    DIFFERENTIAL = "differential"
    INTEGRAL = "integral"
    MATRIX = "matrix"
    STATISTICAL = "statistical"
    LOGICAL = "logical"
    SET_THEORY = "set_theory"
    NUMBER_THEORY = "number_theory"
    GEOMETRIC = "geometric"
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    UNKNOWN = "unknown"


class NotationFormat(Enum):
    """Mathematical notation formats"""
    LATEX = "latex"
    MATHML = "mathml"
    ASCII = "ascii"
    UNICODE = "unicode"
    WOLFRAM = "wolfram"
    PYTHON = "python"
    MATLAB = "matlab"


@dataclass
class ParsedEquation:
    """Structured representation of a parsed equation"""
    raw_text: str
    parsed_expr: Optional[sp.Expr]
    equation_type: EquationType
    variables: List[str]
    constants: List[str]
    functions: List[str]
    operators: List[str]
    complexity_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    confidence: float = 1.0


class EquationParser:
    """
    Advanced equation parser supporting multiple notation formats
    and comprehensive mathematical understanding
    """

    def __init__(self, notation_format: NotationFormat = NotationFormat.LATEX):
        """
        Initialize the equation parser

        Args:
            notation_format: Primary notation format to use
        """
        self.notation_format = notation_format

        # Initialize SymPy
        self.init_sympy_symbols()

        # Setup custom parsers
        self.latex_grammar = self._setup_latex_grammar()
        self.ascii_grammar = self._setup_ascii_grammar()

        # Common mathematical patterns
        self.patterns = {
            'derivative': re.compile(r'\\frac\{d(.+?)\}\{d(.+?)\}|\\partial|\\nabla'),
            'integral': re.compile(r'\\int|\\iint|\\iiint|\\oint'),
            'matrix': re.compile(r'\\begin\{[bp]?matrix\}|\\matrix'),
            'summation': re.compile(r'\\sum|\\prod'),
            'limit': re.compile(r'\\lim'),
            'statistics': re.compile(r'\\bar|\\hat|\\tilde|\\mathbb\{E\}|\\text\{Var\}'),
            'set_theory': re.compile(r'\\in|\\subset|\\cup|\\cap|\\emptyset'),
            'logic': re.compile(r'\\land|\\lor|\\neg|\\implies|\\iff')
        }

        # Physics and chemistry specific patterns
        self.domain_patterns = {
            'physics': re.compile(r'\\hbar|\\vec|\\dot|\\ddot|mc\^2|\\psi|\\phi'),
            'chemistry': re.compile(r'\\ce\{|\\rightarrow|\\rightleftharpoons|H_2O|CO_2')
        }

        logger.info(f"EquationParser initialized with format: {notation_format}")

    def init_sympy_symbols(self):
        """Initialize common mathematical symbols in SymPy"""
        # Common variables
        self.symbols = {
            'x': sp.Symbol('x'),
            'y': sp.Symbol('y'),
            'z': sp.Symbol('z'),
            't': sp.Symbol('t'),
            'theta': sp.Symbol('theta'),
            'phi': sp.Symbol('phi'),
            'alpha': sp.Symbol('alpha'),
            'beta': sp.Symbol('beta'),
            'gamma': sp.Symbol('gamma'),
            'lambda': sp.Symbol('lambda'),
            'mu': sp.Symbol('mu'),
            'sigma': sp.Symbol('sigma')
        }

        # Physical constants
        self.constants = {
            'pi': sp.pi,
            'e': sp.E,
            'c': sp.Symbol('c'),  # Speed of light
            'h': sp.Symbol('h'),  # Planck's constant
            'hbar': sp.Symbol('hbar'),  # Reduced Planck's constant
            'k': sp.Symbol('k'),  # Boltzmann constant
            'G': sp.Symbol('G'),  # Gravitational constant
            'epsilon_0': sp.Symbol('epsilon_0'),  # Vacuum permittivity
            'mu_0': sp.Symbol('mu_0')  # Vacuum permeability
        }

    def parse(
        self,
        equation_text: str,
        format_hint: Optional[NotationFormat] = None,
        domain: Optional[str] = None
    ) -> ParsedEquation:
        """
        Parse mathematical equation from text

        Args:
            equation_text: Raw equation text
            format_hint: Optional hint about notation format
            domain: Optional domain (physics, chemistry, etc.)

        Returns:
            ParsedEquation object with structured representation
        """
        format_to_use = format_hint or self.notation_format

        try:
            # Parse based on format
            if format_to_use == NotationFormat.LATEX:
                parsed = self._parse_latex(equation_text)
            elif format_to_use == NotationFormat.MATHML:
                parsed = self._parse_mathml(equation_text)
            elif format_to_use == NotationFormat.ASCII:
                parsed = self._parse_ascii(equation_text)
            elif format_to_use == NotationFormat.UNICODE:
                parsed = self._parse_unicode(equation_text)
            elif format_to_use == NotationFormat.WOLFRAM:
                parsed = self._parse_wolfram(equation_text)
            else:
                parsed = self._parse_generic(equation_text)

            # Detect equation type
            equation_type = self._detect_equation_type(equation_text, parsed)

            # Extract components
            variables = self._extract_variables(parsed)
            constants = self._extract_constants(parsed)
            functions = self._extract_functions(parsed)
            operators = self._extract_operators(parsed)

            # Calculate complexity
            complexity = self._calculate_complexity(parsed)

            # Add domain-specific metadata
            metadata = self._extract_metadata(equation_text, domain)

            return ParsedEquation(
                raw_text=equation_text,
                parsed_expr=parsed,
                equation_type=equation_type,
                variables=variables,
                constants=constants,
                functions=functions,
                operators=operators,
                complexity_score=complexity,
                metadata=metadata,
                errors=[],
                confidence=0.95
            )

        except Exception as e:
            logger.error(f"Failed to parse equation: {e}")
            return ParsedEquation(
                raw_text=equation_text,
                parsed_expr=None,
                equation_type=EquationType.UNKNOWN,
                variables=[],
                constants=[],
                functions=[],
                operators=[],
                complexity_score=0,
                metadata={},
                errors=[str(e)],
                confidence=0.0
            )

    def _parse_latex(self, latex_str: str) -> sp.Expr:
        """Parse LaTeX equation"""
        try:
            # Clean LaTeX string
            cleaned = self._clean_latex(latex_str)

            # Use SymPy's LaTeX parser
            expr = parse_latex(cleaned)

            # Post-process for common issues
            expr = self._post_process_sympy(expr)

            return expr

        except Exception as e:
            logger.warning(f"SymPy LaTeX parsing failed, trying custom parser: {e}")
            return self._custom_latex_parse(latex_str)

    def _clean_latex(self, latex_str: str) -> str:
        """Clean and normalize LaTeX string"""
        # Remove display math delimiters
        cleaned = latex_str.strip()
        cleaned = re.sub(r'^\$+|\$+$', '', cleaned)
        cleaned = re.sub(r'^\\[|\]$', '', cleaned)
        cleaned = re.sub(r'^\\{|\\}$', '', cleaned)

        # Replace common LaTeX commands
        replacements = {
            '\\cdot': '*',
            '\\times': '*',
            '\\div': '/',
            '\\pm': '+-',
            '\\mp': '-+',
            '\\infty': 'oo',
            '\\alpha': 'alpha',
            '\\beta': 'beta',
            '\\gamma': 'gamma',
            '\\theta': 'theta',
            '\\lambda': 'lambda',
            '\\mu': 'mu',
            '\\sigma': 'sigma',
            '\\pi': 'pi'
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned

    def _custom_latex_parse(self, latex_str: str) -> sp.Expr:
        """Custom LaTeX parser for complex expressions"""
        # Implement custom parsing logic for edge cases
        # This would handle special LaTeX constructs not supported by SymPy

        # For now, return a placeholder
        return sp.Symbol('unparsed_latex')

    def _parse_mathml(self, mathml_str: str) -> sp.Expr:
        """Parse MathML equation"""
        try:
            # Parse MathML using BeautifulSoup
            soup = BeautifulSoup(mathml_str, 'xml')

            # Find math element
            math_elem = soup.find('math')

            if math_elem:
                # Convert MathML to SymPy expression
                return self._mathml_to_sympy(math_elem)

        except Exception as e:
            logger.error(f"MathML parsing failed: {e}")

        return sp.Symbol('unparsed_mathml')

    def _mathml_to_sympy(self, math_elem) -> sp.Expr:
        """Convert MathML element to SymPy expression"""
        # Simplified MathML to SymPy conversion
        # Full implementation would handle all MathML elements

        # Handle identifier
        mi = math_elem.find('mi')
        if mi:
            return sp.Symbol(mi.text)

        # Handle number
        mn = math_elem.find('mn')
        if mn:
            return sp.Number(mn.text)

        # Handle operator
        mo = math_elem.find('mo')
        if mo:
            op_map = {
                '+': sp.Add,
                '-': sp.Mul,
                '*': sp.Mul,
                '/': sp.Mul,
                '=': sp.Eq
            }
            # Would need to handle operands
            return sp.Symbol(mo.text)

        return sp.Symbol('mathml_expr')

    def _parse_ascii(self, ascii_str: str) -> sp.Expr:
        """Parse ASCII math notation"""
        try:
            # Convert ASCII math to SymPy
            # Handle common ASCII math patterns

            # Replace ASCII patterns with SymPy-friendly format
            processed = ascii_str
            processed = re.sub(r'sqrt\((.*?)\)', r'sqrt(\1)', processed)
            processed = re.sub(r'\^', '**', processed)

            # Use SymPy's string parser
            return sp.sympify(processed)

        except Exception as e:
            logger.error(f"ASCII parsing failed: {e}")
            return sp.Symbol('unparsed_ascii')

    def _parse_unicode(self, unicode_str: str) -> sp.Expr:
        """Parse Unicode mathematical notation"""
        # Map Unicode symbols to SymPy
        unicode_map = {
            '∫': 'Integral',
            '∑': 'Sum',
            '∏': 'Product',
            '√': 'sqrt',
            '∞': 'oo',
            'π': 'pi',
            '∂': 'Derivative',
            '∇': 'grad',
            '×': '*',
            '÷': '/',
            '±': '+-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '≈': '~',
            '∈': 'in',
            '∉': 'not in',
            '⊂': 'subset',
            '∪': 'union',
            '∩': 'intersection'
        }

        processed = unicode_str
        for unicode_char, sympy_equiv in unicode_map.items():
            processed = processed.replace(unicode_char, sympy_equiv)

        try:
            return sp.sympify(processed)
        except:
            return sp.Symbol('unparsed_unicode')

    def _parse_wolfram(self, wolfram_str: str) -> sp.Expr:
        """Parse Wolfram Language/Mathematica notation"""
        try:
            # Use SymPy's Mathematica parser
            return parse_mathematica(wolfram_str)
        except Exception as e:
            logger.error(f"Wolfram parsing failed: {e}")
            return sp.Symbol('unparsed_wolfram')

    def _parse_generic(self, text: str) -> sp.Expr:
        """Generic parser that tries multiple formats"""
        # Try each parser in order
        parsers = [
            self._parse_latex,
            self._parse_ascii,
            self._parse_unicode,
            lambda x: sp.sympify(x)
        ]

        for parser in parsers:
            try:
                result = parser(text)
                if result and result != sp.Symbol('unparsed'):
                    return result
            except:
                continue

        return sp.Symbol('unparsed')

    def _post_process_sympy(self, expr: sp.Expr) -> sp.Expr:
        """Post-process SymPy expression for consistency"""
        if expr is None:
            return sp.Symbol('undefined')

        # Simplify if possible
        try:
            expr = sp.simplify(expr)
        except:
            pass

        return expr

    def _detect_equation_type(self, text: str, expr: Optional[sp.Expr]) -> EquationType:
        """Detect the type of equation"""
        # Text-based detection
        for eq_type, pattern in self.patterns.items():
            if pattern.search(text):
                if eq_type == 'derivative':
                    return EquationType.DIFFERENTIAL
                elif eq_type == 'integral':
                    return EquationType.INTEGRAL
                elif eq_type == 'matrix':
                    return EquationType.MATRIX
                elif eq_type == 'statistics':
                    return EquationType.STATISTICAL
                elif eq_type == 'set_theory':
                    return EquationType.SET_THEORY
                elif eq_type == 'logic':
                    return EquationType.LOGICAL

        # Expression-based detection
        if expr:
            if expr.has(sp.Derivative):
                return EquationType.DIFFERENTIAL
            elif expr.has(sp.Integral):
                return EquationType.INTEGRAL
            elif isinstance(expr, (sp.Matrix, sp.MatrixBase)):
                return EquationType.MATRIX

        # Domain-specific detection
        for domain, pattern in self.domain_patterns.items():
            if pattern.search(text):
                if domain == 'physics':
                    return EquationType.PHYSICS
                elif domain == 'chemistry':
                    return EquationType.CHEMISTRY

        return EquationType.ALGEBRAIC

    def _extract_variables(self, expr: Optional[sp.Expr]) -> List[str]:
        """Extract variables from expression"""
        if not expr:
            return []

        try:
            return sorted([str(s) for s in expr.free_symbols])
        except:
            return []

    def _extract_constants(self, expr: Optional[sp.Expr]) -> List[str]:
        """Extract constants from expression"""
        if not expr:
            return []

        constants = []
        try:
            # Find numerical constants
            for atom in sp.preorder_traversal(expr):
                if atom.is_number and not atom.is_integer:
                    if atom == sp.pi:
                        constants.append('π')
                    elif atom == sp.E:
                        constants.append('e')
                    else:
                        constants.append(str(atom))
        except:
            pass

        return constants

    def _extract_functions(self, expr: Optional[sp.Expr]) -> List[str]:
        """Extract functions from expression"""
        if not expr:
            return []

        functions = []
        try:
            for atom in sp.preorder_traversal(expr):
                if isinstance(atom, sp.Function):
                    functions.append(str(atom.func))
                elif isinstance(atom, (sp.sin, sp.cos, sp.tan, sp.exp, sp.log)):
                    functions.append(str(type(atom).__name__))
        except:
            pass

        return list(set(functions))

    def _extract_operators(self, expr: Optional[sp.Expr]) -> List[str]:
        """Extract operators from expression"""
        if not expr:
            return []

        operators = []
        try:
            for atom in sp.preorder_traversal(expr):
                if isinstance(atom, sp.Add):
                    operators.append('+')
                elif isinstance(atom, sp.Mul):
                    operators.append('*')
                elif isinstance(atom, sp.Pow):
                    operators.append('^')
                elif isinstance(atom, sp.Eq):
                    operators.append('=')
        except:
            pass

        return list(set(operators))

    def _calculate_complexity(self, expr: Optional[sp.Expr]) -> float:
        """Calculate equation complexity score"""
        if not expr:
            return 0.0

        complexity = 0.0

        try:
            # Count operations
            complexity += len(list(sp.preorder_traversal(expr))) * 0.1

            # Add complexity for special functions
            if expr.has(sp.Integral):
                complexity += 2.0
            if expr.has(sp.Derivative):
                complexity += 1.5
            if expr.has(sp.Sum):
                complexity += 1.0

            # Add for transcendental functions
            for func in [sp.sin, sp.cos, sp.exp, sp.log]:
                if expr.has(func):
                    complexity += 0.5

        except:
            pass

        return min(complexity, 10.0)  # Cap at 10

    def _extract_metadata(self, text: str, domain: Optional[str]) -> Dict[str, Any]:
        """Extract additional metadata from equation"""
        metadata = {}

        # Domain information
        if domain:
            metadata['domain'] = domain

        # Check for specific patterns
        if '\\begin{cases}' in text:
            metadata['piecewise'] = True

        if '\\lim' in text:
            metadata['has_limits'] = True

        if '\\sum' in text or '\\prod' in text:
            metadata['has_series'] = True

        # Count terms
        metadata['term_count'] = text.count('+') + text.count('-') + 1

        return metadata

    def _setup_latex_grammar(self):
        """Setup Lark grammar for LaTeX parsing"""
        # Simplified LaTeX grammar
        grammar = r"""
        ?start: expression

        ?expression: term
                   | expression "+" term  -> add
                   | expression "-" term  -> sub

        ?term: factor
             | term "*" factor  -> mul
             | term "/" factor  -> div

        ?factor: power
               | factor "^" power  -> pow

        ?power: atom
              | "(" expression ")"

        ?atom: NUMBER  -> number
             | WORD    -> variable
             | "\\" WORD  -> command

        %import common.NUMBER
        %import common.WORD
        %import common.WS
        %ignore WS
        """

        return Lark(grammar, parser='lalr')

    def _setup_ascii_grammar(self):
        """Setup grammar for ASCII math parsing"""
        # Similar to LaTeX but for ASCII notation
        return None  # Placeholder

    def batch_parse(
        self,
        equations: List[str],
        format_hint: Optional[NotationFormat] = None
    ) -> List[ParsedEquation]:
        """
        Parse multiple equations in batch

        Args:
            equations: List of equation strings
            format_hint: Optional format hint for all equations

        Returns:
            List of ParsedEquation objects
        """
        results = []

        for eq in equations:
            parsed = self.parse(eq, format_hint)
            results.append(parsed)

        return results

    def validate_equation(self, equation: ParsedEquation) -> Dict[str, Any]:
        """
        Validate parsed equation for correctness

        Args:
            equation: ParsedEquation to validate

        Returns:
            Validation results with issues if any
        """
        issues = []
        warnings = []

        # Check if parsing succeeded
        if equation.parsed_expr is None:
            issues.append("Failed to parse equation")

        # Check for undefined variables
        if equation.parsed_expr:
            try:
                undefined = []
                for sym in equation.parsed_expr.free_symbols:
                    if str(sym) not in self.symbols and str(sym) not in self.constants:
                        undefined.append(str(sym))

                if undefined:
                    warnings.append(f"Undefined symbols: {', '.join(undefined)}")
            except:
                pass

        # Check dimensional consistency (simplified)
        if equation.equation_type == EquationType.PHYSICS:
            # Would implement dimensional analysis
            pass

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'confidence': equation.confidence
        }

    def export_to_format(
        self,
        equation: ParsedEquation,
        target_format: NotationFormat
    ) -> str:
        """
        Export parsed equation to specified format

        Args:
            equation: ParsedEquation to export
            target_format: Target notation format

        Returns:
            Equation in target format
        """
        if not equation.parsed_expr:
            return equation.raw_text

        try:
            if target_format == NotationFormat.LATEX:
                return sp.latex(equation.parsed_expr)
            elif target_format == NotationFormat.MATHML:
                return sp.mathml(equation.parsed_expr)
            elif target_format == NotationFormat.PYTHON:
                return str(equation.parsed_expr)
            elif target_format == NotationFormat.MATLAB:
                return self._to_matlab(equation.parsed_expr)
            else:
                return str(equation.parsed_expr)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return equation.raw_text

    def _to_matlab(self, expr: sp.Expr) -> str:
        """Convert SymPy expression to MATLAB syntax"""
        matlab_str = str(expr)

        # Replace SymPy syntax with MATLAB
        replacements = {
            '**': '^',
            'pi': 'pi',
            'exp': 'exp',
            'log': 'log',
            'sqrt': 'sqrt',
            'sin': 'sin',
            'cos': 'cos',
            'tan': 'tan'
        }

        for old, new in replacements.items():
            matlab_str = matlab_str.replace(old, new)

        return matlab_str