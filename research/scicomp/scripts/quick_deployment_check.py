#!/usr/bin/env python3
"""
SciComp Quick Deployment Check

Fast verification script to ensure deployment readiness.
Runs essential checks in under 30 seconds.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import sys
import time
import os
from pathlib import Path

# Add Python modules to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Python"))

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header():
    """Print deployment check header."""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}🚀 SciComp Quick Deployment Check 🚀{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.YELLOW}University of California, Berkeley{Colors.RESET}\n")

def check_core_imports():
    """Check if core modules can be imported."""
    print(f"{Colors.BOLD}1. Checking Core Module Imports...{Colors.RESET}")
    
    modules_to_check = [
        ('Quantum.core.quantum_states', 'BellStates'),
        ('Linear_Algebra.core.matrix_operations', 'MatrixOperations'),
        ('Thermal_Transport.core.heat_equation', 'HeatEquation1D'),
        ('Signal_Processing.signal_analysis', 'SignalAnalyzer'),
        ('Machine_Learning.supervised', 'LinearRegressionSolver'),
        ('Optimization.unconstrained', 'BFGSOptimizer'),
    ]
    
    passed = 0
    failed = 0
    
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                print(f"   {Colors.GREEN}✓{Colors.RESET} {module_name}")
                passed += 1
            else:
                print(f"   {Colors.RED}✗{Colors.RESET} {module_name} - {class_name} not found")
                failed += 1
        except ImportError as e:
            print(f"   {Colors.RED}✗{Colors.RESET} {module_name} - Import error")
            failed += 1
    
    return passed, failed

def check_quantum_functionality():
    """Quick test of quantum functionality."""
    print(f"\n{Colors.BOLD}2. Testing Quantum Functionality...{Colors.RESET}")
    
    try:
        from Quantum.core.quantum_states import BellStates
        import numpy as np
        
        # Create Bell state
        phi_plus = BellStates.phi_plus()
        
        # Check normalization
        if phi_plus.is_normalized():
            print(f"   {Colors.GREEN}✓{Colors.RESET} Bell state normalized")
            
            # Check entanglement
            concurrence = phi_plus.concurrence()
            if np.isclose(concurrence, 1.0, atol=1e-10):
                print(f"   {Colors.GREEN}✓{Colors.RESET} Bell state entanglement verified (concurrence={concurrence:.6f})")
                return True
            else:
                print(f"   {Colors.YELLOW}⚠{Colors.RESET} Bell state concurrence={concurrence:.6f} (expected 1.0)")
                return False
        else:
            print(f"   {Colors.RED}✗{Colors.RESET} Bell state not normalized")
            return False
            
    except Exception as e:
        print(f"   {Colors.RED}✗{Colors.RESET} Quantum test failed: {str(e)[:50]}")
        return False

def check_numerical_computation():
    """Quick test of numerical computations."""
    print(f"\n{Colors.BOLD}3. Testing Numerical Computations...{Colors.RESET}")
    
    try:
        from Linear_Algebra.core.matrix_operations import MatrixOperations
        import numpy as np
        
        ops = MatrixOperations()
        
        # Test matrix multiplication
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        C = ops.multiply(A, B)
        
        expected = np.array([[19, 22], [43, 50]])
        if np.allclose(C, expected):
            print(f"   {Colors.GREEN}✓{Colors.RESET} Matrix multiplication correct")
        else:
            print(f"   {Colors.RED}✗{Colors.RESET} Matrix multiplication incorrect")
            return False
        
        # Test eigenvalues
        eigenvals = ops.eigenvalues(A)
        expected_eigenvals = np.linalg.eigvals(A)
        
        if np.allclose(sorted(eigenvals), sorted(expected_eigenvals)):
            print(f"   {Colors.GREEN}✓{Colors.RESET} Eigenvalue computation correct")
            return True
        else:
            print(f"   {Colors.RED}✗{Colors.RESET} Eigenvalue computation incorrect")
            return False
            
    except Exception as e:
        print(f"   {Colors.RED}✗{Colors.RESET} Numerical test failed: {str(e)[:50]}")
        return False

def check_file_structure():
    """Check essential files and directories exist."""
    print(f"\n{Colors.BOLD}4. Checking File Structure...{Colors.RESET}")
    
    essential_items = [
        ('README.md', 'file'),
        ('requirements.txt', 'file'),
        ('setup.py', 'file'),
        ('LICENSE', 'file'),
        ('Python', 'dir'),
        ('MATLAB', 'dir'),
        ('Mathematica', 'dir'),
        ('examples', 'dir'),
        ('docs', 'dir'),
        ('scripts', 'dir'),
    ]
    
    repo_root = Path(__file__).parent.parent
    passed = 0
    failed = 0
    
    for item_name, item_type in essential_items:
        item_path = repo_root / item_name
        
        if item_type == 'file':
            if item_path.is_file():
                print(f"   {Colors.GREEN}✓{Colors.RESET} {item_name}")
                passed += 1
            else:
                print(f"   {Colors.RED}✗{Colors.RESET} {item_name} (missing)")
                failed += 1
        else:  # directory
            if item_path.is_dir():
                print(f"   {Colors.GREEN}✓{Colors.RESET} {item_name}/")
                passed += 1
            else:
                print(f"   {Colors.RED}✗{Colors.RESET} {item_name}/ (missing)")
                failed += 1
    
    return passed, failed

def check_dependencies():
    """Check if key dependencies are available."""
    print(f"\n{Colors.BOLD}5. Checking Dependencies...{Colors.RESET}")
    
    dependencies = [
        'numpy',
        'scipy',
        'matplotlib',
        'sympy',
    ]
    
    passed = 0
    failed = 0
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   {Colors.GREEN}✓{Colors.RESET} {dep}")
            passed += 1
        except ImportError:
            print(f"   {Colors.RED}✗{Colors.RESET} {dep} (not installed)")
            failed += 1
    
    return passed, failed

def calculate_readiness_score(results):
    """Calculate deployment readiness based on checks."""
    total_passed = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    
    if total_passed + total_failed == 0:
        return 0
    
    pass_rate = total_passed / (total_passed + total_failed)
    
    # Calculate score (60-100 scale)
    score = int(60 + (pass_rate * 40))
    
    # Bonus for critical functionality
    if results[1] and results[2]:  # Quantum and numerical tests passed
        score += 5
    
    return min(100, score)

def main():
    """Run quick deployment check."""
    print_header()
    
    start_time = time.time()
    
    # Run all checks
    results = []
    
    # 1. Core imports
    import_passed, import_failed = check_core_imports()
    results.append((import_passed, import_failed))
    
    # 2. Quantum functionality
    quantum_passed = check_quantum_functionality()
    results.append((1 if quantum_passed else 0, 0 if quantum_passed else 1))
    
    # 3. Numerical computation
    numerical_passed = check_numerical_computation()
    results.append((1 if numerical_passed else 0, 0 if numerical_passed else 1))
    
    # 4. File structure
    files_passed, files_failed = check_file_structure()
    results.append((files_passed, files_failed))
    
    # 5. Dependencies
    deps_passed, deps_failed = check_dependencies()
    results.append((deps_passed, deps_failed))
    
    # Calculate totals
    total_passed = sum(r[0] for r in results)
    total_failed = sum(r[1] for r in results)
    
    # Calculate deployment readiness
    readiness_score = calculate_readiness_score(results)
    
    # Print summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}📊 DEPLOYMENT CHECK SUMMARY{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*60}{Colors.RESET}")
    
    print(f"\nTotal Checks Passed: {Colors.GREEN}{total_passed}{Colors.RESET}")
    print(f"Total Checks Failed: {Colors.RED}{total_failed}{Colors.RESET}")
    print(f"Execution Time: {time.time() - start_time:.2f}s")
    
    print(f"\n{Colors.BOLD}🏆 DEPLOYMENT READINESS SCORE: {readiness_score}/100{Colors.RESET}")
    
    if readiness_score >= 90:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ EXCELLENT: Ready for production deployment!{Colors.RESET}")
        print(f"{Colors.BLUE}🐻 Go Bears! All systems operational! 🐻{Colors.RESET}")
        exit_code = 0
    elif readiness_score >= 75:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ GOOD: Ready with minor issues{Colors.RESET}")
        print("Consider addressing failed checks for optimal deployment")
        exit_code = 0
    elif readiness_score >= 60:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ ACCEPTABLE: Basic functionality working{Colors.RESET}")
        print("Address critical issues before production deployment")
        exit_code = 1
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ NOT READY: Critical issues detected{Colors.RESET}")
        print("Must fix failures before deployment")
        exit_code = 1
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"*SciComp Quick Deployment Check Complete*")
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())