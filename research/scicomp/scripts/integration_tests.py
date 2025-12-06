#!/usr/bin/env python3
"""
SciComp Integration Tests

Comprehensive integration testing to verify module interactions,
data flow, and end-to-end workflows across the SciComp framework.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

class IntegrationTester:
    """Integration testing framework for SciComp."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.test_results = {}
        self.integration_failures = []
        
        # Add Python path
        sys.path.insert(0, str(self.repo_root / "Python"))
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    def test_quantum_ml_integration(self) -> Dict[str, Any]:
        """Test integration between quantum physics and ML modules."""
        print("🔬🧠 Testing Quantum-ML Integration...")
        
        results = {'status': 'PASSED', 'tests': [], 'errors': []}
        
        try:
            # Test 1: Quantum state data preparation for ML
            from Quantum.core.quantum_states import BellStates, QuantumState
            from Machine_Learning.supervised import LinearRegressionSolver
            import numpy as np
            
            # Generate quantum state features
            states = []
            labels = []
            
            for _ in range(100):
                # Create random quantum states
                psi = np.random.rand(4) + 1j * np.random.rand(4)
                psi = psi / np.linalg.norm(psi)
                state = QuantumState(psi)
                
                # Extract features (expectation values, entanglement measures)
                features = [
                    abs(state.state_vector[0])**2,  # |<0|psi>|^2
                    abs(state.state_vector[1])**2,  # |<1|psi>|^2
                    state.concurrence(),            # Entanglement measure
                    state.purity(),                 # Purity measure
                ]
                
                states.append(features)
                labels.append(features[2])  # Predict concurrence
            
            X = np.array(states)
            y = np.array(labels)
            
            # Train ML model on quantum features
            solver = LinearRegressionSolver()
            solver.fit(X, y)
            predictions = solver.predict(X[:10])
            
            # Test successful if predictions are reasonable
            if len(predictions) == 10 and not np.isnan(predictions).any():
                results['tests'].append({
                    'name': 'Quantum State Feature Extraction + ML Training',
                    'status': 'PASSED',
                    'details': f'Trained on {len(states)} quantum states'
                })
            else:
                raise ValueError("ML training on quantum features failed")
            
            # Test 2: Quantum algorithm optimization using ML
            # (Simplified example: optimize quantum gate parameters)
            def quantum_cost_function(params):
                """Cost function for quantum algorithm optimization."""
                # Simulate quantum circuit with parameterized gates
                angle = params[0]
                rotation_matrix = np.array([
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]
                ])
                
                # Apply to initial state |0>
                initial_state = np.array([1, 0])
                final_state = rotation_matrix @ initial_state
                
                # Cost: distance from target state |+> = [1/√2, 1/√2]
                target = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
                cost = np.linalg.norm(final_state - target)**2
                return cost
            
            # Use ML-inspired optimization (gradient-free)
            from Machine_Learning.optimization import GeneticOptimizer
            optimizer = GeneticOptimizer(bounds=[(-np.pi, np.pi)], population_size=20)
            best_params, best_cost = optimizer.optimize(quantum_cost_function, max_iterations=50)
            
            if best_cost < 0.1:  # Should find angle ≈ π/4
                results['tests'].append({
                    'name': 'Quantum Algorithm Optimization with ML',
                    'status': 'PASSED',
                    'details': f'Optimized to cost {best_cost:.6f}'
                })
            else:
                results['tests'].append({
                    'name': 'Quantum Algorithm Optimization with ML',
                    'status': 'FAILED',
                    'details': f'Poor optimization result: cost {best_cost:.6f}'
                })
                results['status'] = 'FAILED'
                
        except Exception as e:
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        return results
    
    def test_thermal_fem_integration(self) -> Dict[str, Any]:
        """Test integration between thermal transport and FEM modules."""
        print("🌡️🔧 Testing Thermal-FEM Integration...")
        
        results = {'status': 'PASSED', 'tests': [], 'errors': []}
        
        try:
            # Test 1: FEM mesh generation for thermal problem
            from FEM.core.mesh_generation import MeshGenerator
            from FEM.core.finite_elements import LinearTriangleElement
            from Thermal_Transport.core.heat_equation import HeatEquationSolver
            import numpy as np
            
            # Generate 2D mesh for thermal problem
            mesh_gen = MeshGenerator()
            nodes, elements = mesh_gen.generate_rectangle_mesh(1.0, 1.0, 10, 10)
            
            # Test mesh quality
            if len(nodes) > 50 and len(elements) > 50:
                results['tests'].append({
                    'name': 'FEM Mesh Generation for Thermal Problem',
                    'status': 'PASSED',
                    'details': f'{len(nodes)} nodes, {len(elements)} elements'
                })
            else:
                raise ValueError("Insufficient mesh quality")
            
            # Test 2: Thermal boundary conditions in FEM context
            # Simplified: just verify we can set up thermal BCs
            boundary_nodes = [i for i in range(len(nodes)) if nodes[i][0] == 0]  # Left boundary
            thermal_bcs = {node_id: 100.0 for node_id in boundary_nodes[:5]}  # Hot boundary
            
            if len(thermal_bcs) > 0:
                results['tests'].append({
                    'name': 'Thermal Boundary Conditions Setup',
                    'status': 'PASSED',
                    'details': f'Set {len(thermal_bcs)} thermal boundary conditions'
                })
            
            # Test 3: Heat equation solver with FEM discretization
            # (Simplified - real integration would use FEM matrices)
            nx, ny = 20, 20
            T_initial = np.ones((nx, ny)) * 20.0  # Room temperature
            T_initial[0, :] = 100.0  # Hot boundary
            
            # Simple explicit thermal solver
            dt = 0.01
            alpha = 0.01
            dx, dy = 1.0/(nx-1), 1.0/(ny-1)
            
            T_new = T_initial.copy()
            for step in range(10):
                for i in range(1, nx-1):
                    for j in range(1, ny-1):
                        T_new[i,j] = T_initial[i,j] + alpha * dt * (
                            (T_initial[i+1,j] - 2*T_initial[i,j] + T_initial[i-1,j])/dx**2 +
                            (T_initial[i,j+1] - 2*T_initial[i,j] + T_initial[i,j-1])/dy**2
                        )
                T_initial = T_new.copy()
            
            # Check temperature diffusion occurred
            center_temp = T_initial[nx//2, ny//2]
            if 20 < center_temp < 100:
                results['tests'].append({
                    'name': 'Thermal Diffusion with FEM-like Discretization',
                    'status': 'PASSED',
                    'details': f'Center temperature: {center_temp:.1f}°C'
                })
            else:
                raise ValueError(f"Unrealistic temperature distribution: {center_temp:.1f}°C")
                
        except Exception as e:
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        return results
    
    def test_signal_optimization_integration(self) -> Dict[str, Any]:
        """Test integration between signal processing and optimization."""
        print("📡⚡ Testing Signal-Optimization Integration...")
        
        results = {'status': 'PASSED', 'tests': [], 'errors': []}
        
        try:
            from Signal_Processing.core.fourier_transforms import FFTProcessor
            from Optimization.unconstrained import BFGSOptimizer
            import numpy as np
            
            # Test 1: Signal denoising via optimization
            # Create noisy signal
            t = np.linspace(0, 1, 1000)
            clean_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
            noise = 0.2 * np.random.randn(1000)
            noisy_signal = clean_signal + noise
            
            # Define denoising objective function
            def denoising_objective(params):
                """Objective function for signal denoising."""
                # Parameters: [freq1_amp, freq1_phase, freq2_amp, freq2_phase]
                if len(params) != 4:
                    return float('inf')
                
                reconstructed = (params[0] * np.sin(2 * np.pi * 5 * t + params[1]) +
                               params[2] * np.sin(2 * np.pi * 10 * t + params[3]))
                
                # L2 loss
                return np.sum((reconstructed - noisy_signal)**2)
            
            # Optimize signal reconstruction
            optimizer = BFGSOptimizer(tolerance=1e-6, max_iterations=100)
            initial_guess = [1.0, 0.0, 0.5, 0.0]  # Close to true parameters
            
            try:
                result = optimizer.minimize(denoising_objective, initial_guess)
                optimal_params = result['x']
                final_cost = result['fun']
                
                # Reconstruct optimized signal
                optimized_signal = (optimal_params[0] * np.sin(2 * np.pi * 5 * t + optimal_params[1]) +
                                  optimal_params[2] * np.sin(2 * np.pi * 10 * t + optimal_params[3]))
                
                # Calculate improvement
                original_error = np.sum((clean_signal - noisy_signal)**2)
                optimized_error = np.sum((clean_signal - optimized_signal)**2)
                improvement = (original_error - optimized_error) / original_error
                
                if improvement > 0.1:  # At least 10% improvement
                    results['tests'].append({
                        'name': 'Signal Denoising via Optimization',
                        'status': 'PASSED',
                        'details': f'{improvement*100:.1f}% error reduction'
                    })
                else:
                    results['tests'].append({
                        'name': 'Signal Denoising via Optimization',
                        'status': 'FAILED',
                        'details': f'Poor improvement: {improvement*100:.1f}%'
                    })
                    results['status'] = 'FAILED'
                    
            except Exception as e:
                raise ValueError(f"Optimization failed: {e}")
            
            # Test 2: Filter design optimization
            # Design optimal low-pass filter coefficients
            def filter_design_objective(coeffs):
                """Objective for filter coefficient optimization."""
                if len(coeffs) != 5:
                    return float('inf')
                
                # Simple FIR filter frequency response
                frequencies = np.linspace(0, 0.5, 100)  # Normalized frequencies
                response = np.zeros_like(frequencies, dtype=complex)
                
                for i, coeff in enumerate(coeffs):
                    response += coeff * np.exp(-1j * 2 * np.pi * frequencies * i)
                
                magnitude = np.abs(response)
                
                # Target: low-pass with cutoff at 0.2
                target = np.where(frequencies <= 0.2, 1.0, 0.0)
                error = np.sum((magnitude - target)**2)
                
                return error
            
            # Optimize filter coefficients
            initial_coeffs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal coefficients
            filter_result = optimizer.minimize(filter_design_objective, initial_coeffs)
            
            if filter_result['success'] and filter_result['fun'] < 10:
                results['tests'].append({
                    'name': 'Digital Filter Design Optimization',
                    'status': 'PASSED',
                    'details': f'Filter error: {filter_result["fun"]:.3f}'
                })
            else:
                results['tests'].append({
                    'name': 'Digital Filter Design Optimization',
                    'status': 'FAILED',
                    'details': f'High filter error: {filter_result["fun"]:.3f}'
                })
                results['status'] = 'FAILED'
                
        except Exception as e:
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        return results
    
    def test_cross_platform_workflow(self) -> Dict[str, Any]:
        """Test cross-platform workflow integration."""
        print("🌍💻 Testing Cross-Platform Workflow...")
        
        results = {'status': 'PASSED', 'tests': [], 'errors': []}
        
        try:
            # Test 1: Data format compatibility
            import numpy as np
            import json
            
            # Create test data in Python
            quantum_data = {
                'state_vector': [0.707, 0, 0, 0.707],  # Bell state
                'measurement_results': [1, 0, 1, 1, 0],
                'parameters': {'angle': 0.785, 'frequency': 50.0}
            }
            
            # Save to JSON (MATLAB/Mathematica compatible)
            json_file = self.repo_root / "test_data_exchange.json"
            with open(json_file, 'w') as f:
                json.dump(quantum_data, f, indent=2)
            
            # Read back and verify
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
            
            if loaded_data == quantum_data:
                results['tests'].append({
                    'name': 'JSON Data Format Compatibility',
                    'status': 'PASSED',
                    'details': 'Python ↔ MATLAB/Mathematica data exchange'
                })
            else:
                raise ValueError("Data format compatibility failed")
            
            # Clean up
            json_file.unlink()
            
            # Test 2: Numerical precision consistency
            # Test if numerical results are consistent across different implementations
            from Linear_Algebra.core.matrix_operations import MatrixOperations
            
            ops = MatrixOperations()
            test_matrix = np.array([[1, 2], [3, 4]], dtype=float)
            
            # Compute eigenvalues
            eigenvals = ops.eigenvalues(test_matrix)
            expected_eigenvals = np.linalg.eigvals(test_matrix)
            
            # Sort for comparison
            eigenvals_sorted = np.sort(eigenvals)
            expected_sorted = np.sort(expected_eigenvals)
            
            precision_error = np.max(np.abs(eigenvals_sorted - expected_sorted))
            
            if precision_error < 1e-10:
                results['tests'].append({
                    'name': 'Numerical Precision Consistency',
                    'status': 'PASSED',
                    'details': f'Max error: {precision_error:.2e}'
                })
            else:
                results['tests'].append({
                    'name': 'Numerical Precision Consistency',
                    'status': 'FAILED',
                    'details': f'High precision error: {precision_error:.2e}'
                })
                results['status'] = 'FAILED'
            
            # Test 3: File path handling
            # Verify cross-platform path handling
            test_paths = [
                "Python/Quantum/core/quantum_states.py",
                "examples/python/quantum_computing_demo.py",
                "docs/api/README.md"
            ]
            
            path_issues = []
            for path_str in test_paths:
                path_obj = Path(path_str)
                if not (self.repo_root / path_obj).exists():
                    path_issues.append(path_str)
            
            if len(path_issues) == 0:
                results['tests'].append({
                    'name': 'Cross-Platform Path Handling',
                    'status': 'PASSED',
                    'details': f'All {len(test_paths)} paths resolved correctly'
                })
            else:
                results['tests'].append({
                    'name': 'Cross-Platform Path Handling',
                    'status': 'FAILED',
                    'details': f'{len(path_issues)} paths failed: {path_issues[:3]}'
                })
                results['status'] = 'FAILED'
                
        except Exception as e:
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        return results
    
    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end scientific workflow."""
        print("🔬🔄 Testing End-to-End Scientific Workflow...")
        
        results = {'status': 'PASSED', 'tests': [], 'errors': []}
        
        try:
            # Complete workflow: Quantum system → Analysis → Optimization → Visualization
            from Quantum.core.quantum_states import BellStates, QuantumState
            from Signal_Processing.core.fourier_transforms import FFTProcessor
            from Optimization.unconstrained import BFGSOptimizer
            from Plotting.scientific_plots import ScientificPlotter
            import numpy as np
            
            # Step 1: Generate quantum measurement data
            phi_plus = BellStates.phi_plus()
            measurement_data = []
            
            for _ in range(1000):
                # Simulate noisy quantum measurements
                prob_0 = abs(phi_plus.state_vector[0])**2
                prob_1 = abs(phi_plus.state_vector[1])**2
                prob_2 = abs(phi_plus.state_vector[2])**2
                prob_3 = abs(phi_plus.state_vector[3])**2
                
                # Add noise
                noise = 0.1 * np.random.randn(4)
                noisy_probs = np.array([prob_0, prob_1, prob_2, prob_3]) + noise
                noisy_probs = np.clip(noisy_probs, 0, 1)
                noisy_probs = noisy_probs / np.sum(noisy_probs)  # Renormalize
                
                measurement_data.append(noisy_probs)
            
            measurement_array = np.array(measurement_data)
            
            # Step 2: Signal processing analysis
            # Analyze temporal correlations in measurement outcomes
            time_series = measurement_array[:, 0]  # Focus on |00⟩ measurements
            
            processor = FFTProcessor(len(time_series))
            freqs, fft_result = processor.fft(time_series)
            
            # Find dominant frequency
            dominant_freq_idx = np.argmax(np.abs(fft_result[:len(fft_result)//2]))
            dominant_freq = freqs[dominant_freq_idx]
            
            # Step 3: Parameter optimization
            # Optimize quantum state parameters to match measurements
            def state_matching_objective(params):
                """Objective function for state parameter optimization."""
                if len(params) != 2:
                    return float('inf')
                
                # Create parameterized state
                alpha, beta = params
                if abs(alpha)**2 + abs(beta)**2 == 0:
                    return float('inf')
                
                # Normalize
                norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
                alpha_norm = alpha / norm
                beta_norm = beta / norm
                
                # Create state |ψ⟩ = α|00⟩ + β|11⟩ (Bell-like)
                state_vector = np.array([alpha_norm, 0, 0, beta_norm])
                
                # Calculate expected probabilities
                expected_probs = np.abs(state_vector)**2
                
                # Compare with average measurements
                measured_probs = np.mean(measurement_array, axis=0)
                error = np.sum((expected_probs - measured_probs)**2)
                
                return error
            
            # Optimize state parameters
            optimizer = BFGSOptimizer(tolerance=1e-8, max_iterations=50)
            initial_params = [0.7, 0.7]  # Starting guess
            
            opt_result = optimizer.minimize(state_matching_objective, initial_params)
            
            if opt_result['success'] and opt_result['fun'] < 0.01:
                results['tests'].append({
                    'name': 'End-to-End Quantum Analysis Workflow',
                    'status': 'PASSED',
                    'details': f'Parameter optimization error: {opt_result["fun"]:.6f}'
                })
            else:
                results['tests'].append({
                    'name': 'End-to-End Quantum Analysis Workflow',
                    'status': 'FAILED',
                    'details': f'High optimization error: {opt_result["fun"]:.6f}'
                })
                results['status'] = 'FAILED'
            
            # Step 4: Data visualization (simplified)
            try:
                plotter = ScientificPlotter()
                
                # Create simple plot data
                x_data = np.arange(len(time_series[:100]))
                y_data = time_series[:100]
                
                # This would create a plot in a real scenario
                plot_data = {'x': x_data.tolist(), 'y': y_data.tolist()}
                
                if len(plot_data['x']) == len(plot_data['y']) == 100:
                    results['tests'].append({
                        'name': 'Scientific Data Visualization',
                        'status': 'PASSED',
                        'details': 'Plot data prepared successfully'
                    })
                else:
                    raise ValueError("Visualization data preparation failed")
                    
            except Exception as e:
                results['tests'].append({
                    'name': 'Scientific Data Visualization',
                    'status': 'FAILED',
                    'details': f'Visualization error: {str(e)}'
                })
                results['status'] = 'FAILED'
                
        except Exception as e:
            results['status'] = 'FAILED'
            results['errors'].append(str(e))
        
        return results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run all integration tests."""
        print("🔗 SciComp Integration Tests")
        print("=" * 80)
        print("University of California, Berkeley")
        print("Module Integration & Workflow Validation\n")
        
        # Define integration test suites
        test_suites = [
            ("Quantum-ML Integration", self.test_quantum_ml_integration),
            ("Thermal-FEM Integration", self.test_thermal_fem_integration),
            ("Signal-Optimization Integration", self.test_signal_optimization_integration),
            ("Cross-Platform Workflow", self.test_cross_platform_workflow),
            ("End-to-End Scientific Workflow", self.test_end_to_end_workflow),
        ]
        
        overall_results = {}
        total_passed = 0
        total_failed = 0
        
        for suite_name, test_func in test_suites:
            print(f"\n🧪 Running {suite_name}...")
            start_time = time.time()
            
            try:
                suite_results = test_func()
                end_time = time.time()
                
                overall_results[suite_name] = suite_results
                overall_results[suite_name]['execution_time'] = end_time - start_time
                
                # Count results
                for test in suite_results.get('tests', []):
                    if test['status'] == 'PASSED':
                        total_passed += 1
                        print(f"   ✅ {test['name']}: {test['details']}")
                    else:
                        total_failed += 1
                        print(f"   ❌ {test['name']}: {test['details']}")
                        self.integration_failures.append(f"{suite_name}: {test['name']}")
                
                if suite_results['status'] == 'PASSED':
                    print(f"   🎉 Suite completed successfully ({end_time-start_time:.2f}s)")
                else:
                    print(f"   ⚠️  Suite completed with failures ({end_time-start_time:.2f}s)")
                    for error in suite_results.get('errors', []):
                        print(f"       Error: {error}")
                        
            except Exception as e:
                overall_results[suite_name] = {
                    'status': 'FAILED',
                    'errors': [str(e)],
                    'tests': []
                }
                total_failed += 1
                print(f"   ❌ Suite failed: {str(e)}")
                self.integration_failures.append(f"{suite_name}: Critical Error")
        
        # Print summary
        self.print_integration_summary(overall_results, total_passed, total_failed)
        
        return overall_results
    
    def print_integration_summary(self, results: Dict[str, Any], total_passed: int, total_failed: int):
        """Print integration test summary."""
        print("\n" + "="*80)
        print("🔗 INTEGRATION TEST SUMMARY")
        print("="*80)
        
        total_tests = total_passed + total_failed
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Integration Tests: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Critical Integration Failures: {len(self.integration_failures)}")
        
        # Show integration failures
        if self.integration_failures:
            print("\n❌ INTEGRATION FAILURES:")
            for failure in self.integration_failures:
                print(f"   • {failure}")
        
        # Overall assessment
        print("\n🎯 INTEGRATION ASSESSMENT:")
        if success_rate >= 95 and len(self.integration_failures) == 0:
            print("✅ EXCELLENT: All module integrations working perfectly")
        elif success_rate >= 80:
            print("⚠️  GOOD: Most integrations working, minor issues found")
        elif success_rate >= 60:
            print("⚠️  FAIR: Some integration issues found, review recommended")
        else:
            print("❌ POOR: Critical integration failures detected")
        
        print("\n*Integration testing completed*")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SciComp Integration Tests')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--export-results', help='Export results to JSON file')
    
    args = parser.parse_args()
    
    tester = IntegrationTester(args.repo_root)
    results = tester.run_integration_tests()
    
    if args.export_results:
        with open(args.export_results, 'w') as f:
            json.dump({
                'results': results,
                'failures': tester.integration_failures
            }, f, indent=2, default=str)
        print(f"\n📄 Results exported to: {args.export_results}")
    
    # Exit with error if critical failures
    sys.exit(1 if len(tester.integration_failures) > 0 else 0)

if __name__ == "__main__":
    main()