#!/usr/bin/env python3
"""
SciComp Performance Regression Tests

Comprehensive performance testing and regression detection for SciComp framework.
Establishes performance baselines and detects regressions across releases.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import time
import json
import statistics
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
import warnings

class PerformanceRegression:
    """Performance regression testing framework."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.results = {}
        self.baselines = {}
        self.regressions = []
        
        # Performance thresholds
        self.regression_threshold = 1.5  # 50% slower is a regression
        self.improvement_threshold = 0.8  # 20% faster is an improvement
        
        # Load existing baselines if available
        self.baseline_file = self.repo_root / "performance_baselines.json"
        self.load_baselines()
        
        # Add Python path
        sys.path.insert(0, str(self.repo_root / "Python"))
    
    def load_baselines(self):
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")
                self.baselines = {}
    
    def save_baselines(self):
        """Save current results as new baselines."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"✅ Baselines saved to {self.baseline_file}")
        except Exception as e:
            print(f"Error saving baselines: {e}")
    
    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Tuple[float, Any]:
        """Measure function execution time with multiple runs."""
        times = []
        result = None
        
        # Warm-up run
        try:
            func(*args, **kwargs)
        except:
            pass
        
        # Multiple measurement runs
        for _ in range(3):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                return float('inf'), f"Error: {str(e)}"
        
        # Return median time and result
        return statistics.median(times), result
    
    def test_quantum_performance(self) -> Dict[str, Any]:
        """Test quantum computation performance."""
        print("🔬 Testing quantum computation performance...")
        
        results = {}
        
        try:
            from Quantum.core.quantum_states import BellStates, QuantumState
            import numpy as np
            
            # Test 1: Bell state creation
            exec_time, _ = self.measure_execution_time(BellStates.phi_plus)
            results['bell_state_creation'] = exec_time
            
            # Test 2: Quantum state evolution (small system)
            def test_evolution():
                state = QuantumState(np.array([1, 0, 0, 0], dtype=complex))
                H = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
                H = H + H.conj().T  # Make Hermitian
                return state.evolve(H, 0.1)
            
            exec_time, _ = self.measure_execution_time(test_evolution)
            results['quantum_evolution_4d'] = exec_time
            
            # Test 3: Large quantum state operations
            def test_large_state():
                state_vector = np.random.rand(256) + 1j * np.random.rand(256)
                state_vector = state_vector / np.linalg.norm(state_vector)
                state = QuantumState(state_vector)
                return state.is_normalized()
            
            exec_time, _ = self.measure_execution_time(test_large_state)
            results['large_quantum_state_256d'] = exec_time
            
            # Test 4: Concurrence calculation
            phi_plus = BellStates.phi_plus()
            exec_time, _ = self.measure_execution_time(phi_plus.concurrence)
            results['concurrence_calculation'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_linear_algebra_performance(self) -> Dict[str, Any]:
        """Test linear algebra performance."""
        print("🔢 Testing linear algebra performance...")
        
        results = {}
        
        try:
            from Linear_Algebra.core.matrix_operations import MatrixOperations
            import numpy as np
            
            ops = MatrixOperations()
            
            # Test 1: Matrix multiplication (100x100)
            def test_matmul_100():
                A = np.random.rand(100, 100)
                B = np.random.rand(100, 100)
                return ops.multiply(A, B)
            
            exec_time, _ = self.measure_execution_time(test_matmul_100)
            results['matrix_multiply_100x100'] = exec_time
            
            # Test 2: Matrix multiplication (500x500)
            def test_matmul_500():
                A = np.random.rand(500, 500)
                B = np.random.rand(500, 500)
                return ops.multiply(A, B)
            
            exec_time, _ = self.measure_execution_time(test_matmul_500)
            results['matrix_multiply_500x500'] = exec_time
            
            # Test 3: Eigenvalue computation
            def test_eigenvals():
                A = np.random.rand(100, 100)
                A = A + A.T  # Make symmetric
                return ops.eigenvalues(A)
            
            exec_time, _ = self.measure_execution_time(test_eigenvals)
            results['eigenvalues_100x100'] = exec_time
            
            # Test 4: Matrix inversion
            def test_inverse():
                A = np.random.rand(50, 50) + 0.1 * np.eye(50)  # Ensure invertible
                return ops.inverse(A)
            
            exec_time, _ = self.measure_execution_time(test_inverse)
            results['matrix_inverse_50x50'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_signal_processing_performance(self) -> Dict[str, Any]:
        """Test signal processing performance."""
        print("📡 Testing signal processing performance...")
        
        results = {}
        
        try:
            from Signal_Processing.core.fourier_transforms import FFTProcessor
            import numpy as np
            
            # Test 1: FFT of 1k samples
            def test_fft_1k():
                processor = FFTProcessor(1000)
                signal = np.random.rand(1000)
                return processor.fft(signal)
            
            exec_time, _ = self.measure_execution_time(test_fft_1k)
            results['fft_1k_samples'] = exec_time
            
            # Test 2: FFT of 10k samples
            def test_fft_10k():
                processor = FFTProcessor(10000)
                signal = np.random.rand(10000)
                return processor.fft(signal)
            
            exec_time, _ = self.measure_execution_time(test_fft_10k)
            results['fft_10k_samples'] = exec_time
            
            # Test 3: Spectrogram computation
            def test_spectrogram():
                from Signal_Processing.spectral_analysis import SpectralAnalyzer
                analyzer = SpectralAnalyzer()
                signal = np.random.rand(5000)
                return analyzer.spectrogram(signal, 1000)
            
            exec_time, _ = self.measure_execution_time(test_spectrogram)
            results['spectrogram_5k_samples'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_thermal_performance(self) -> Dict[str, Any]:
        """Test thermal transport performance."""
        print("🌡️ Testing thermal transport performance...")
        
        results = {}
        
        try:
            from Thermal_Transport.core.heat_equation import HeatEquationSolver
            import numpy as np
            
            # Test 1: 1D Heat equation (small grid)
            def test_heat_1d_small():
                nx = 100
                dx = 0.01
                dt = 0.001
                alpha = 0.01
                solver = HeatEquationSolver(nx, dx, dt, alpha)
                T_initial = np.sin(np.pi * np.linspace(0, 1, nx))
                return solver.solve_1d(T_initial, num_steps=50)
            
            exec_time, _ = self.measure_execution_time(test_heat_1d_small)
            results['heat_equation_1d_100pts_50steps'] = exec_time
            
            # Test 2: 1D Heat equation (large grid)
            def test_heat_1d_large():
                nx = 1000
                dx = 0.001
                dt = 0.0001
                alpha = 0.01
                solver = HeatEquationSolver(nx, dx, dt, alpha)
                T_initial = np.sin(np.pi * np.linspace(0, 1, nx))
                return solver.solve_1d(T_initial, num_steps=100)
            
            exec_time, _ = self.measure_execution_time(test_heat_1d_large)
            results['heat_equation_1d_1000pts_100steps'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_ml_performance(self) -> Dict[str, Any]:
        """Test machine learning performance."""
        print("🧠 Testing ML performance...")
        
        results = {}
        
        try:
            from Machine_Learning.neural_networks import SimpleNeuralNetwork
            from Machine_Learning.supervised import LinearRegressionSolver
            import numpy as np
            
            # Test 1: Linear regression training
            def test_linear_regression():
                X = np.random.rand(1000, 10)
                y = np.random.rand(1000)
                solver = LinearRegressionSolver()
                return solver.fit(X, y)
            
            exec_time, _ = self.measure_execution_time(test_linear_regression)
            results['linear_regression_1000x10'] = exec_time
            
            # Test 2: Neural network forward pass
            def test_nn_forward():
                nn = SimpleNeuralNetwork([50, 25, 10, 1])
                X = np.random.rand(100, 50)
                return nn.forward(X)
            
            exec_time, _ = self.measure_execution_time(test_nn_forward)
            results['neural_network_forward_100x50'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def test_memory_performance(self) -> Dict[str, Any]:
        """Test memory-intensive operations."""
        print("🧠 Testing memory performance...")
        
        results = {}
        
        try:
            import numpy as np
            import gc
            
            # Test 1: Large array allocation and deallocation
            def test_large_array():
                arrays = []
                for _ in range(10):
                    arr = np.random.rand(1000, 1000)
                    arrays.append(arr)
                del arrays
                gc.collect()
                return True
            
            exec_time, _ = self.measure_execution_time(test_large_array)
            results['large_array_allocation'] = exec_time
            
            # Test 2: Memory-intensive quantum operations
            def test_memory_quantum():
                from Quantum.core.quantum_states import QuantumState
                states = []
                for _ in range(50):
                    state = QuantumState(np.random.rand(64) + 1j * np.random.rand(64))
                    states.append(state.density_matrix)
                del states
                gc.collect()
                return True
            
            exec_time, _ = self.measure_execution_time(test_memory_quantum)
            results['memory_quantum_operations'] = exec_time
            
        except Exception as e:
            results['error'] = str(e)
        
        return results
    
    def analyze_regression(self, test_name: str, current_time: float, baseline_time: float) -> str:
        """Analyze if there's a performance regression."""
        if baseline_time == 0:
            return "NO_BASELINE"
        
        ratio = current_time / baseline_time
        
        if ratio >= self.regression_threshold:
            self.regressions.append({
                'test': test_name,
                'current': current_time,
                'baseline': baseline_time,
                'slowdown': ratio
            })
            return "REGRESSION"
        elif ratio <= self.improvement_threshold:
            return "IMPROVEMENT"
        else:
            return "STABLE"
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        print("🚀 SciComp Performance Regression Tests")
        print("=" * 80)
        print("University of California, Berkeley")
        print("Performance Analysis & Regression Detection\n")
        
        # Define test suites
        test_suites = [
            ("Quantum Computations", self.test_quantum_performance),
            ("Linear Algebra", self.test_linear_algebra_performance),
            ("Signal Processing", self.test_signal_processing_performance),
            ("Thermal Transport", self.test_thermal_performance),
            ("Machine Learning", self.test_ml_performance),
            ("Memory Operations", self.test_memory_performance),
        ]
        
        overall_results = {}
        
        for suite_name, test_func in test_suites:
            print(f"\n🧪 Running {suite_name} tests...")
            try:
                suite_results = test_func()
                overall_results[suite_name] = suite_results
                
                # Analyze each test in the suite
                for test_name, exec_time in suite_results.items():
                    if test_name != 'error' and isinstance(exec_time, (int, float)):
                        baseline_key = f"{suite_name}.{test_name}"
                        baseline_time = self.baselines.get(baseline_key, 0)
                        
                        status = self.analyze_regression(test_name, exec_time, baseline_time)
                        
                        if status == "REGRESSION":
                            print(f"   ⚠️  {test_name}: {exec_time:.4f}s (REGRESSION: {exec_time/baseline_time:.1f}x slower)")
                        elif status == "IMPROVEMENT":
                            print(f"   ✅ {test_name}: {exec_time:.4f}s (IMPROVED: {baseline_time/exec_time:.1f}x faster)")
                        elif status == "STABLE":
                            print(f"   ✅ {test_name}: {exec_time:.4f}s (stable)")
                        else:
                            print(f"   📊 {test_name}: {exec_time:.4f}s (no baseline)")
                
                if 'error' in suite_results:
                    print(f"   ❌ Suite error: {suite_results['error']}")
                    
            except Exception as e:
                print(f"   ❌ Suite failed: {str(e)}")
                overall_results[suite_name] = {'error': str(e)}
        
        # Store results for future baseline comparison
        self.results = {}
        for suite_name, suite_results in overall_results.items():
            for test_name, result in suite_results.items():
                if test_name != 'error' and isinstance(result, (int, float)):
                    self.results[f"{suite_name}.{test_name}"] = result
        
        return overall_results
    
    def print_performance_summary(self, results: Dict[str, Any]):
        """Print performance test summary."""
        print("\n" + "="*80)
        print("📊 PERFORMANCE TEST SUMMARY")
        print("="*80)
        
        total_tests = 0
        successful_tests = 0
        failed_suites = 0
        
        for suite_name, suite_results in results.items():
            if 'error' in suite_results:
                failed_suites += 1
            else:
                for test_name, result in suite_results.items():
                    total_tests += 1
                    if isinstance(result, (int, float)) and result != float('inf'):
                        successful_tests += 1
        
        print(f"Total Test Suites: {len(results)}")
        print(f"Successful Tests: {successful_tests}/{total_tests}")
        print(f"Failed Suites: {failed_suites}")
        print(f"Performance Regressions: {len(self.regressions)}")
        
        # Show regressions
        if self.regressions:
            print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            for reg in self.regressions:
                print(f"   • {reg['test']}: {reg['slowdown']:.1f}x slower ({reg['current']:.4f}s vs {reg['baseline']:.4f}s)")
        
        # Overall assessment
        print("\n🎯 PERFORMANCE ASSESSMENT:")
        if len(self.regressions) == 0:
            print("✅ EXCELLENT: No performance regressions detected")
        elif len(self.regressions) <= 2:
            print("⚠️  GOOD: Minor performance regressions detected")
        else:
            print("❌ POOR: Multiple performance regressions detected")
        
        print("\n*Performance testing completed*")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SciComp Performance Regression Tests')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--save-baseline', action='store_true', help='Save current results as new baseline')
    parser.add_argument('--export-results', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    tester = PerformanceRegression(args.repo_root)
    results = tester.run_performance_tests()
    
    tester.print_performance_summary(results)
    
    if args.save_baseline:
        tester.save_baselines()
    
    if args.export_results:
        with open(args.export_results, 'w') as f:
            json.dump({
                'results': results,
                'regressions': tester.regressions,
                'baselines': tester.baselines
            }, f, indent=2)
        print(f"\n📄 Results exported to: {args.export_results}")
    
    # Exit with error if regressions found
    sys.exit(1 if len(tester.regressions) > 0 else 0)

if __name__ == "__main__":
    main()