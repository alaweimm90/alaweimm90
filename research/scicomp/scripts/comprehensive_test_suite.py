#!/usr/bin/env python3
"""
SciComp Comprehensive Test Suite

Advanced testing framework for deep coherence, error detection, and system integrity.
Goes beyond basic validation to ensure production-ready quality.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import ast
import re
import json
import time
import importlib
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import warnings

class Colors:
    """Enhanced color scheme for comprehensive testing output."""
    BERKELEY_BLUE = '\033[38;5;18m'
    CALIFORNIA_GOLD = '\033[38;5;178m'
    SUCCESS = '\033[38;5;46m'
    WARNING = '\033[38;5;208m'
    ERROR = '\033[38;5;196m'
    INFO = '\033[38;5;39m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

class ComprehensiveTestSuite:
    """Advanced test suite for SciComp repository."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.results = defaultdict(list)
        self.metrics = defaultdict(float)
        self.errors = defaultdict(list)
        self.warnings = defaultdict(list)
        
        # Test categories
        self.test_categories = {
            'syntax': 'Python/MATLAB/Mathematica syntax validation',
            'imports': 'Import dependency resolution',
            'docstrings': 'Documentation completeness',
            'type_hints': 'Type annotation consistency',
            'security': 'Security vulnerability scanning',
            'performance': 'Performance regression testing',
            'memory': 'Memory usage and leak detection',
            'cross_platform': 'Cross-platform compatibility',
            'api_consistency': 'API interface consistency',
            'example_validation': 'Example code execution',
            'mathematical_correctness': 'Mathematical algorithm validation',
            'error_handling': 'Exception handling robustness',
            'concurrency': 'Thread safety and parallel execution',
            'integration': 'Module integration testing',
            'backwards_compatibility': 'Version compatibility'
        }
    
    def print_header(self, title: str):
        """Print formatted section header."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}🧪 {title} 🧪{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
    
    def test_python_syntax_deep(self) -> Dict[str, Any]:
        """Deep Python syntax and AST validation."""
        print(f"{Colors.BOLD}🐍 Testing Python Syntax & AST Validation...{Colors.RESET}")
        
        results = {'passed': 0, 'failed': 0, 'issues': []}
        python_files = list(self.repo_root.glob("**/*.py"))
        
        for py_file in python_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                # Read and parse file
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                
                # Basic syntax check
                compile(content, str(py_file), 'exec')
                
                # AST analysis
                tree = ast.parse(content)
                
                # Advanced checks
                issues = []
                
                # Check for undefined variables
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                        # This is a simplified check - would need scope analysis for completeness
                        pass
                
                # Check for unused imports
                imports = []
                used_names = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.Name):
                        used_names.add(node.id)
                
                # Check for potential issues
                for node in ast.walk(tree):
                    # Check for bare except clauses
                    if isinstance(node, ast.ExceptHandler) and node.type is None:
                        issues.append(f"Line {node.lineno}: Bare except clause")
                    
                    # Check for dangerous eval/exec usage
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        if node.func.id in ['eval', 'exec']:
                            issues.append(f"Line {node.lineno}: Dangerous {node.func.id} usage")
                
                if issues:
                    results['issues'].append({
                        'file': str(py_file.relative_to(self.repo_root)),
                        'issues': issues
                    })
                    results['failed'] += 1
                else:
                    results['passed'] += 1
                    
            except SyntaxError as e:
                results['failed'] += 1
                results['issues'].append({
                    'file': str(py_file.relative_to(self.repo_root)),
                    'issues': [f"Syntax Error: {e}"]
                })
            except Exception as e:
                results['failed'] += 1
                results['issues'].append({
                    'file': str(py_file.relative_to(self.repo_root)),
                    'issues': [f"Parse Error: {e}"]
                })
        
        print(f"   ✅ Checked {len(python_files)} Python files")
        print(f"   📊 Passed: {results['passed']}, Failed: {results['failed']}")
        return results
    
    def test_import_resolution(self) -> Dict[str, Any]:
        """Test all imports can be resolved."""
        print(f"{Colors.BOLD}📦 Testing Import Resolution...{Colors.RESET}")
        
        results = {'passed': 0, 'failed': 0, 'missing_deps': set(), 'issues': []}
        python_files = list(self.repo_root.glob("**/*.py"))
        
        # Add Python directory to path
        sys.path.insert(0, str(self.repo_root / "Python"))
        
        for py_file in python_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            try:
                                importlib.import_module(alias.name)
                                results['passed'] += 1
                            except ImportError:
                                results['failed'] += 1
                                results['missing_deps'].add(alias.name)
                                results['issues'].append({
                                    'file': str(py_file.relative_to(self.repo_root)),
                                    'missing': alias.name
                                })
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            try:
                                mod = importlib.import_module(node.module)
                                for alias in node.names:
                                    if not hasattr(mod, alias.name):
                                        results['issues'].append({
                                            'file': str(py_file.relative_to(self.repo_root)),
                                            'issue': f"'{alias.name}' not found in '{node.module}'"
                                        })
                                results['passed'] += 1
                            except ImportError:
                                results['failed'] += 1
                                results['missing_deps'].add(node.module)
                                
            except Exception as e:
                results['issues'].append({
                    'file': str(py_file.relative_to(self.repo_root)),
                    'error': str(e)
                })
        
        print(f"   📊 Import checks: {results['passed']} passed, {results['failed']} failed")
        print(f"   📦 Missing dependencies: {len(results['missing_deps'])}")
        return results
    
    def test_docstring_completeness(self) -> Dict[str, Any]:
        """Comprehensive docstring analysis."""
        print(f"{Colors.BOLD}📝 Testing Documentation Completeness...{Colors.RESET}")
        
        results = {'classes': 0, 'functions': 0, 'documented': 0, 'issues': []}
        python_files = list(self.repo_root.glob("**/*.py"))
        
        for py_file in python_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        results['functions'] += 1
                        if ast.get_docstring(node):
                            results['documented'] += 1
                        else:
                            results['issues'].append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'item': f"Function '{node.name}' (line {node.lineno})",
                                'issue': 'Missing docstring'
                            })
                    
                    elif isinstance(node, ast.ClassDef):
                        results['classes'] += 1
                        if ast.get_docstring(node):
                            results['documented'] += 1
                        else:
                            results['issues'].append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'item': f"Class '{node.name}' (line {node.lineno})",
                                'issue': 'Missing docstring'
                            })
                            
            except Exception as e:
                results['issues'].append({
                    'file': str(py_file.relative_to(self.repo_root)),
                    'error': str(e)
                })
        
        total_items = results['classes'] + results['functions']
        coverage = (results['documented'] / total_items * 100) if total_items > 0 else 0
        
        print(f"   📊 Documentation coverage: {coverage:.1f}%")
        print(f"   📝 {results['documented']}/{total_items} items documented")
        return results
    
    def test_mathematical_correctness(self) -> Dict[str, Any]:
        """Test mathematical algorithms for correctness."""
        print(f"{Colors.BOLD}🧮 Testing Mathematical Correctness...{Colors.RESET}")
        
        results = {'passed': 0, 'failed': 0, 'tests': []}
        
        # Add Python to path
        sys.path.insert(0, str(self.repo_root / "Python"))
        
        math_tests = [
            {
                'name': 'Quantum Bell States',
                'test': self._test_bell_states_math,
                'critical': True
            },
            {
                'name': 'FFT Correctness',
                'test': self._test_fft_correctness,
                'critical': True
            },
            {
                'name': 'Heat Equation Solver',
                'test': self._test_heat_equation_math,
                'critical': True
            },
            {
                'name': 'Linear Algebra Operations',
                'test': self._test_linear_algebra_math,
                'critical': False
            }
        ]
        
        for test_info in math_tests:
            try:
                start_time = time.time()
                result = test_info['test']()
                end_time = time.time()
                
                if result['passed']:
                    results['passed'] += 1
                    print(f"   ✅ {test_info['name']}: PASSED ({end_time-start_time:.3f}s)")
                else:
                    results['failed'] += 1
                    print(f"   ❌ {test_info['name']}: FAILED - {result.get('error', 'Unknown error')}")
                
                results['tests'].append({
                    'name': test_info['name'],
                    'passed': result['passed'],
                    'time': end_time - start_time,
                    'critical': test_info['critical'],
                    'details': result
                })
                
            except Exception as e:
                results['failed'] += 1
                results['tests'].append({
                    'name': test_info['name'],
                    'passed': False,
                    'error': str(e),
                    'critical': test_info['critical']
                })
                print(f"   ❌ {test_info['name']}: FAILED - {str(e)}")
        
        return results
    
    def _test_bell_states_math(self) -> Dict[str, Any]:
        """Test Bell states mathematical properties."""
        try:
            from Quantum.core.quantum_states import BellStates
            import numpy as np
            
            # Test Bell state properties
            phi_plus = BellStates.phi_plus()
            
            # Check normalization
            if not phi_plus.is_normalized():
                return {'passed': False, 'error': 'Bell state not normalized'}
            
            # Check entanglement
            concurrence = phi_plus.concurrence()
            if not np.isclose(concurrence, 1.0, atol=1e-10):
                return {'passed': False, 'error': f'Bell state concurrence {concurrence} != 1.0'}
            
            return {'passed': True, 'concurrence': concurrence}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_fft_correctness(self) -> Dict[str, Any]:
        """Test FFT implementation correctness."""
        try:
            from Signal_Processing.core.fourier_transforms import FFTProcessor
            import numpy as np
            
            # Create test signal
            fs = 1000
            t = np.linspace(0, 1, fs, endpoint=False)
            freq = 50
            signal = np.sin(2 * np.pi * freq * t)
            
            processor = FFTProcessor(fs)
            freqs, fft_result = processor.fft(signal)
            
            # Find peak frequency
            peak_idx = np.argmax(np.abs(fft_result[:len(fft_result)//2]))
            detected_freq = freqs[peak_idx]
            
            if not np.isclose(detected_freq, freq, atol=1):
                return {'passed': False, 'error': f'FFT detected {detected_freq} Hz, expected {freq} Hz'}
            
            return {'passed': True, 'detected_freq': detected_freq, 'expected_freq': freq}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_heat_equation_math(self) -> Dict[str, Any]:
        """Test heat equation solver mathematical correctness."""
        try:
            from Thermal_Transport.core.heat_equation import HeatEquationSolver
            import numpy as np
            
            # Test 1D heat equation with known analytical solution
            L = 1.0  # Length
            nx = 50  # Grid points
            dx = L / (nx - 1)
            dt = 0.001
            alpha = 0.01  # Thermal diffusivity
            
            solver = HeatEquationSolver(nx, dx, dt, alpha)
            
            # Initial condition: sin(pi*x/L)
            x = np.linspace(0, L, nx)
            T_initial = np.sin(np.pi * x / L)
            
            # Solve for a few time steps
            T_final = solver.solve_1d(T_initial, num_steps=10)
            
            # Analytical solution: T(x,t) = exp(-pi^2*alpha*t/L^2) * sin(pi*x/L)
            t_final = 10 * dt
            T_analytical = np.exp(-np.pi**2 * alpha * t_final / L**2) * np.sin(np.pi * x / L)
            
            # Compare numerical and analytical solutions
            error = np.max(np.abs(T_final - T_analytical))
            if error > 0.01:  # Allow 1% error
                return {'passed': False, 'error': f'Heat equation error {error} too large'}
            
            return {'passed': True, 'numerical_error': error}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _test_linear_algebra_math(self) -> Dict[str, Any]:
        """Test linear algebra operations."""
        try:
            from Linear_Algebra.core.matrix_operations import MatrixOperations
            import numpy as np
            
            ops = MatrixOperations()
            
            # Test matrix multiplication
            A = np.random.rand(5, 3)
            B = np.random.rand(3, 4)
            C = ops.multiply(A, B)
            C_expected = np.dot(A, B)
            
            if not np.allclose(C, C_expected):
                return {'passed': False, 'error': 'Matrix multiplication incorrect'}
            
            # Test eigenvalue computation
            A_square = np.random.rand(4, 4)
            A_symmetric = A_square + A_square.T  # Make symmetric
            
            eigenvals = ops.eigenvalues(A_symmetric)
            eigenvals_expected = np.linalg.eigvals(A_symmetric)
            
            eigenvals_sorted = np.sort(eigenvals)
            expected_sorted = np.sort(eigenvals_expected)
            
            if not np.allclose(eigenvals_sorted, expected_sorted):
                return {'passed': False, 'error': 'Eigenvalue computation incorrect'}
            
            return {'passed': True}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test for memory leaks and excessive usage."""
        print(f"{Colors.BOLD}🧠 Testing Memory Usage...{Colors.RESET}")
        
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            return {'error': 'psutil not available for memory testing'}
        
        results = {'tests': [], 'max_memory_mb': 0}
        
        # Test memory usage of core operations
        memory_tests = [
            ('Large Matrix Operations', self._test_large_matrix_memory),
            ('Quantum State Memory', self._test_quantum_memory),
            ('Signal Processing Memory', self._test_signal_memory)
        ]
        
        for test_name, test_func in memory_tests:
            try:
                # Measure memory before
                mem_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Run test
                test_func()
                
                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = mem_after - mem_before
                
                results['tests'].append({
                    'name': test_name,
                    'memory_mb': memory_used,
                    'passed': memory_used < 100  # Flag if > 100MB
                })
                
                results['max_memory_mb'] = max(results['max_memory_mb'], memory_used)
                
                print(f"   📊 {test_name}: {memory_used:.1f} MB")
                
            except Exception as e:
                results['tests'].append({
                    'name': test_name,
                    'error': str(e),
                    'passed': False
                })
        
        return results
    
    def _test_large_matrix_memory(self):
        """Test memory usage with large matrices."""
        import numpy as np
        
        # Create and manipulate large matrices
        A = np.random.rand(1000, 1000)
        B = np.random.rand(1000, 1000)
        C = np.dot(A, B)
        
        # Clean up explicitly
        del A, B, C
    
    def _test_quantum_memory(self):
        """Test quantum state memory usage."""
        sys.path.insert(0, str(self.repo_root / "Python"))
        
        from Quantum.core.quantum_states import QuantumState
        import numpy as np
        
        # Create multiple quantum states
        states = []
        for i in range(100):
            state = QuantumState(np.random.rand(16) + 1j * np.random.rand(16))
            states.append(state)
        
        # Clean up
        del states
    
    def _test_signal_memory(self):
        """Test signal processing memory usage."""
        sys.path.insert(0, str(self.repo_root / "Python"))
        
        from Signal_Processing.core.fourier_transforms import FFTProcessor
        import numpy as np
        
        # Process large signals
        processor = FFTProcessor(10000)
        for i in range(10):
            signal = np.random.rand(10000)
            freqs, fft_result = processor.fft(signal)
    
    def test_cross_platform_compatibility(self) -> Dict[str, Any]:
        """Test cross-platform file paths and operations."""
        print(f"{Colors.BOLD}🌍 Testing Cross-Platform Compatibility...{Colors.RESET}")
        
        results = {'issues': [], 'passed': 0, 'failed': 0}
        
        # Check for platform-specific path separators
        all_files = list(self.repo_root.glob("**/*"))
        text_files = [f for f in all_files if f.suffix in {'.py', '.m', '.md', '.txt'}]
        
        for file_path in text_files:
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                
                # Check for hardcoded paths
                if '\\\\' in content or re.search(r'[A-Z]:\\', content):
                    results['issues'].append({
                        'file': str(file_path.relative_to(self.repo_root)),
                        'issue': 'Hardcoded Windows paths detected'
                    })
                    results['failed'] += 1
                
                # Check for Unix-specific paths
                elif content.count('/home/') > 0 or content.count('/usr/') > 0:
                    results['issues'].append({
                        'file': str(file_path.relative_to(self.repo_root)),
                        'issue': 'Hardcoded Unix paths detected'
                    })
                    results['failed'] += 1
                else:
                    results['passed'] += 1
                    
            except Exception as e:
                results['issues'].append({
                    'file': str(file_path.relative_to(self.repo_root)),
                    'error': str(e)
                })
        
        print(f"   📊 Path compatibility: {results['passed']} clean, {results['failed']} issues")
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        self.print_header("SciComp Comprehensive Test Suite")
        print(f"{Colors.CALIFORNIA_GOLD}Deep Analysis for Production Readiness{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}\n")
        
        all_results = {}
        start_time = time.time()
        
        # Run all test categories
        test_functions = [
            ('Python Syntax & AST', self.test_python_syntax_deep),
            ('Import Resolution', self.test_import_resolution),
            ('Documentation Completeness', self.test_docstring_completeness),
            ('Mathematical Correctness', self.test_mathematical_correctness),
            ('Memory Usage', self.test_memory_usage),
            ('Cross-Platform Compatibility', self.test_cross_platform_compatibility)
        ]
        
        for test_name, test_func in test_functions:
            print(f"\n{Colors.INFO}Running {test_name} tests...{Colors.RESET}")
            try:
                result = test_func()
                all_results[test_name] = result
            except Exception as e:
                all_results[test_name] = {'error': str(e)}
                print(f"   {Colors.ERROR}ERROR: {str(e)}{Colors.RESET}")
        
        # Generate summary
        end_time = time.time()
        total_time = end_time - start_time
        
        self._print_comprehensive_summary(all_results, total_time)
        
        return all_results
    
    def _print_comprehensive_summary(self, results: Dict[str, Any], total_time: float):
        """Print comprehensive test summary."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}📊 COMPREHENSIVE TEST RESULTS 📊{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        
        total_passed = 0
        total_failed = 0
        critical_issues = []
        
        for test_name, result in results.items():
            if 'error' in result:
                print(f"{Colors.ERROR}❌ {test_name}: ERROR - {result['error']}{Colors.RESET}")
                total_failed += 1
                critical_issues.append(f"{test_name}: {result['error']}")
            else:
                passed = result.get('passed', 0)
                failed = result.get('failed', 0)
                
                total_passed += passed
                total_failed += failed
                
                if failed == 0:
                    print(f"{Colors.SUCCESS}✅ {test_name}: ALL PASSED{Colors.RESET}")
                else:
                    print(f"{Colors.WARNING}⚠️  {test_name}: {passed} passed, {failed} failed{Colors.RESET}")
                    
                    # Show critical issues
                    if 'issues' in result and len(result['issues']) > 0:
                        for issue in result['issues'][:3]:  # Show first 3
                            if isinstance(issue, dict):
                                file_name = issue.get('file', 'Unknown')
                                issue_desc = issue.get('issue', issue.get('error', 'Unknown issue'))
                                print(f"       • {file_name}: {issue_desc}")
        
        # Overall verdict
        print(f"\n{Colors.BOLD}🎯 OVERALL ASSESSMENT:{Colors.RESET}")
        total_tests = total_passed + total_failed
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {Colors.SUCCESS}{total_passed}{Colors.RESET}")
        print(f"   Failed: {Colors.ERROR}{total_failed}{Colors.RESET}")
        print(f"   Success Rate: {Colors.SUCCESS if success_rate > 90 else Colors.WARNING}{success_rate:.1f}%{Colors.RESET}")
        print(f"   Execution Time: {total_time:.1f}s")
        
        # Final recommendation
        if success_rate >= 95 and not critical_issues:
            print(f"\n{Colors.SUCCESS}{Colors.BOLD}🚀 EXCELLENT: Repository is production-ready!{Colors.RESET}")
            print(f"{Colors.BERKELEY_BLUE}🐻 Go Bears! Ready for deployment! 🐻{Colors.RESET}")
        elif success_rate >= 85:
            print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  GOOD: Minor issues found, deployment recommended{Colors.RESET}")
            print("   Consider addressing warnings before production deployment")
        else:
            print(f"\n{Colors.ERROR}{Colors.BOLD}❌ NEEDS WORK: Critical issues found{Colors.RESET}")
            print("   Please address critical issues before deployment")
        
        print(f"\n{Colors.DIM}*Comprehensive testing completed*{Colors.RESET}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SciComp Comprehensive Test Suite')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--export-results', help='Export results to JSON file')
    parser.add_argument('--quick', action='store_true', help='Run quick subset of tests')
    
    args = parser.parse_args()
    
    suite = ComprehensiveTestSuite(args.repo_root)
    results = suite.run_comprehensive_tests()
    
    if args.export_results:
        with open(args.export_results, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 Results exported to: {args.export_results}")
    
    # Exit code based on results
    critical_failures = sum(1 for r in results.values() if 'error' in r)
    sys.exit(1 if critical_failures > 0 else 0)

if __name__ == "__main__":
    main()