"""
Preconditioned QAP Solver with FFT-Laplace Acceleration

This module integrates all FFT-Laplace preconditioning components
to provide an enhanced QAP solver with significant performance improvements
for structured problems.

Key Features:
1. Automatic structure detection
2. FFT-accelerated operations for grid graphs
3. Spectral initialization
4. Adaptive preconditioning strategies
5. Integration with existing optimization methods
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .fft_ops import FFTOperations
from .graph_utils import GraphUtilities
from .laplace_detector import GraphType, LaplacianDetector
from .spectral_ordering import SpectralOrdering

logger = logging.getLogger(__name__)


class PreconditionedQAPSolver:
    """
    Enhanced QAP solver with FFT-Laplace preconditioning.

    This solver automatically detects structure in QAP instances
    and applies appropriate preconditioning techniques for
    improved performance.
    """

    def __init__(self, enable_fft: bool = True, verbose: bool = False):
        """
        Initialize preconditioned QAP solver.

        Args:
            enable_fft: Whether to use FFT acceleration
            verbose: Enable verbose logging
        """
        self.enable_fft = enable_fft
        self.verbose = verbose

        # Initialize component modules
        self.detector = LaplacianDetector()
        self.spectral = SpectralOrdering(use_fft=enable_fft)
        self.fft_ops = FFTOperations()
        self.graph_utils = GraphUtilities()

        # Cache for preprocessing results
        self.preprocessing_cache = {}

    def solve(self,
              flow_matrix: np.ndarray,
              distance_matrix: np.ndarray,
              method: str = "auto",
              config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Solve QAP with automatic preconditioning.

        Args:
            flow_matrix: QAP flow matrix F
            distance_matrix: QAP distance matrix D
            method: Optimization method to use
            config: Method-specific configuration

        Returns:
            Solution dictionary with permutation and metadata
        """
        n = len(flow_matrix)
        if config is None:
            config = {}

        # Step 1: Analyze problem structure
        analysis = self._analyze_problem_structure(flow_matrix, distance_matrix)

        if self.verbose:
            logger.info(f"Problem analysis: {analysis['summary']}")

        # Step 2: Apply preconditioning if beneficial
        if analysis['use_preconditioning']:
            initial_solution = self._generate_preconditioned_start(
                flow_matrix, distance_matrix, analysis
            )
        else:
            initial_solution = np.random.permutation(n)

        # Step 3: Select optimization method
        if method == "auto":
            method = self._select_method(analysis)

        # Step 4: Prepare solver configuration
        solver_config = self._prepare_solver_config(
            flow_matrix, distance_matrix, analysis, config
        )
        solver_config['initial_solution'] = initial_solution

        # Step 5: Run optimization
        result = self._run_optimization(
            flow_matrix, distance_matrix, method, solver_config
        )

        # Step 6: Add preconditioning metadata
        result['preconditioning'] = {
            'used': analysis['use_preconditioning'],
            'techniques': analysis.get('techniques_used', []),
            'flow_structure': analysis.get('flow_type', 'unknown'),
            'distance_structure': analysis.get('distance_type', 'unknown'),
            'speedup_estimate': analysis.get('speedup_estimate', 1.0)
        }

        return result

    def _analyze_problem_structure(self,
                                   flow_matrix: np.ndarray,
                                   distance_matrix: np.ndarray) -> Dict:
        """
        Comprehensive analysis of QAP instance structure.

        Returns dictionary with structural information and recommendations.
        """
        analysis = {
            'use_preconditioning': False,
            'techniques_used': [],
            'summary': ''
        }

        # Detect Laplacian structure
        flow_type, flow_meta = self.detector.detect_structure(flow_matrix)
        dist_type, dist_meta = self.detector.detect_structure(distance_matrix)

        analysis['flow_type'] = flow_type.value
        analysis['distance_type'] = dist_type.value
        analysis['flow_metadata'] = flow_meta
        analysis['distance_metadata'] = dist_meta

        # Check for exploitable structures
        exploitable_types = [
            GraphType.GRID_2D, GraphType.GRID_3D,
            GraphType.CYCLE, GraphType.PATH,
            GraphType.TREE, GraphType.COMPLETE
        ]

        if flow_type in exploitable_types or dist_type in exploitable_types:
            analysis['use_preconditioning'] = True

            # Select techniques based on structure
            if flow_type == GraphType.GRID_2D or dist_type == GraphType.GRID_2D:
                if self.enable_fft:
                    analysis['techniques_used'].append('FFT-accelerated operations')
                    analysis['speedup_estimate'] = 10.0  # O(n²) to O(n log n)
                else:
                    analysis['techniques_used'].append('Grid-aware spectral ordering')
                    analysis['speedup_estimate'] = 2.0

            if flow_type in [GraphType.CYCLE, GraphType.PATH]:
                analysis['techniques_used'].append('1D spectral decomposition')
                analysis['speedup_estimate'] = max(analysis.get('speedup_estimate', 1.0), 3.0)

            if flow_type == GraphType.TREE or dist_type == GraphType.TREE:
                analysis['techniques_used'].append('Tree decomposition')
                analysis['speedup_estimate'] = max(analysis.get('speedup_estimate', 1.0), 5.0)

        # Graph-based analysis for non-Laplacian matrices
        if not analysis['use_preconditioning']:
            graph_recommendations = self.graph_utils.suggest_preprocessing(
                flow_matrix, distance_matrix
            )

            if graph_recommendations['use_fft'] and self.enable_fft:
                analysis['use_preconditioning'] = True
                analysis['techniques_used'].extend(
                    graph_recommendations['specific_techniques']
                )

        # Build summary
        if analysis['use_preconditioning']:
            techniques_str = ', '.join(analysis['techniques_used'])
            analysis['summary'] = f"Preconditioning enabled: {techniques_str}"
        else:
            analysis['summary'] = "No exploitable structure detected, using standard methods"

        return analysis

    def _generate_preconditioned_start(self,
                                       flow_matrix: np.ndarray,
                                       distance_matrix: np.ndarray,
                                       analysis: Dict) -> np.ndarray:
        """
        Generate high-quality initial solution using preconditioning.

        This uses spectral methods adapted to the detected structure.
        """
        n = len(flow_matrix)

        # Try spectral ordering first
        if 'FFT-accelerated operations' in analysis.get('techniques_used', []):
            # Use FFT-based spectral ordering for grid graphs
            initial = self.spectral.generate_ordering(
                flow_matrix, distance_matrix, method='fiedler'
            )
            if self.verbose:
                logger.info("Generated FFT-based spectral initialization")

        elif 'Tree decomposition' in analysis.get('techniques_used', []):
            # Use tree-based ordering
            initial = self._tree_based_ordering(flow_matrix, distance_matrix)
            if self.verbose:
                logger.info("Generated tree-based initialization")

        else:
            # General spectral ordering
            initial = self.spectral.generate_ordering(
                flow_matrix, distance_matrix, method='multi_level'
            )
            if self.verbose:
                logger.info("Generated multi-level spectral initialization")

        return initial

    def _tree_based_ordering(self,
                             flow_matrix: np.ndarray,
                             distance_matrix: np.ndarray) -> np.ndarray:
        """
        Generate ordering for tree-structured problems.

        Uses DFS or BFS traversal for natural tree ordering.
        """
        n = len(flow_matrix)

        # Build adjacency from flow or distance matrix
        if self.detector.is_laplacian(flow_matrix):
            adj_matrix = np.diag(np.diag(flow_matrix)) - flow_matrix
        elif self.detector.is_laplacian(distance_matrix):
            adj_matrix = np.diag(np.diag(distance_matrix)) - distance_matrix
        else:
            # Use flow matrix as adjacency
            adj_matrix = np.abs(flow_matrix)

        # DFS traversal for tree ordering
        visited = np.zeros(n, dtype=bool)
        ordering = []

        def dfs(v):
            visited[v] = True
            ordering.append(v)
            neighbors = np.where(adj_matrix[v] > 0)[0]
            for u in neighbors:
                if not visited[u]:
                    dfs(u)

        # Start from vertex with minimum degree (likely a leaf)
        degrees = np.sum(adj_matrix > 0, axis=1)
        start = np.argmin(degrees)
        dfs(start)

        # Add any unvisited vertices
        for v in range(n):
            if not visited[v]:
                ordering.append(v)

        return np.array(ordering)

    def _select_method(self, analysis: Dict) -> str:
        """
        Select best optimization method based on problem structure.

        Returns method name string.
        """
        flow_type = analysis.get('flow_type', 'unknown')
        dist_type = analysis.get('distance_type', 'unknown')

        # For highly structured problems, use simulated annealing
        # with good initial solution
        if flow_type in ['grid_2d', 'grid_3d', 'cycle', 'path']:
            return 'simulated_annealing'

        # For tree structures, use tabu search
        if flow_type == 'tree' or dist_type == 'tree':
            return 'tabu_search'

        # For dense/complete graphs, use genetic algorithm
        if flow_type == 'complete' or dist_type == 'complete':
            return 'genetic_algorithm'

        # Default to simulated annealing
        return 'simulated_annealing'

    def _prepare_solver_config(self,
                               flow_matrix: np.ndarray,
                               distance_matrix: np.ndarray,
                               analysis: Dict,
                               user_config: Dict) -> Dict:
        """
        Prepare configuration for the optimization solver.

        Merges user configuration with structure-aware defaults.
        """
        n = len(flow_matrix)

        # Base configuration
        config = {
            'max_iterations': min(1000 * n, 100000),
            'verbose': self.verbose
        }

        # Structure-specific adjustments
        if 'FFT-accelerated operations' in analysis.get('techniques_used', []):
            # For FFT-accelerated problems, we can afford more iterations
            config['max_iterations'] *= 2
            config['use_fast_evaluation'] = True
            config['evaluator'] = self._create_fast_evaluator(
                flow_matrix, distance_matrix, analysis
            )

        # Method-specific parameters
        flow_type = analysis.get('flow_type', 'unknown')

        if flow_type in ['grid_2d', 'cycle']:
            # Grid and cycle graphs benefit from local search
            config['neighborhood_size'] = int(np.sqrt(n))
            config['temperature_schedule'] = 'adaptive'

        elif flow_type == 'tree':
            # Trees benefit from larger neighborhoods
            config['tabu_tenure'] = int(np.log2(n)) + 5
            config['aspiration_criterion'] = True

        # Merge with user configuration (user config takes precedence)
        config.update(user_config)

        return config

    def _create_fast_evaluator(self,
                              flow_matrix: np.ndarray,
                              distance_matrix: np.ndarray,
                              analysis: Dict):
        """
        Create fast objective evaluator using FFT when applicable.

        Returns a callable that computes QAP objective.
        """
        flow_type = analysis.get('flow_type', 'unknown')
        dist_type = analysis.get('distance_type', 'unknown')

        # Check if we can use FFT acceleration
        use_fft_flow = (flow_type == 'grid_2d' and
                       'grid_dimension' in analysis.get('flow_metadata', {}))
        use_fft_dist = (dist_type == 'grid_2d' and
                       'grid_dimension' in analysis.get('distance_metadata', {}))

        if use_fft_flow or use_fft_dist:
            # Extract grid dimensions
            if use_fft_flow:
                grid_shape = analysis['flow_metadata']['grid_dimension']
            else:
                grid_shape = analysis['distance_metadata']['grid_dimension']

            def fast_evaluator(permutation):
                """FFT-accelerated QAP objective evaluation."""
                return self.fft_ops.fast_qap_objective_grid(
                    permutation, flow_matrix, distance_matrix, grid_shape
                )

            return fast_evaluator

        # Default evaluator
        def standard_evaluator(permutation):
            """Standard QAP objective evaluation."""
            n = len(permutation)
            P = np.zeros((n, n))
            P[np.arange(n), permutation] = 1
            return np.trace(flow_matrix @ P @ distance_matrix @ P.T)

        return standard_evaluator

    def _run_optimization(self,
                         flow_matrix: np.ndarray,
                         distance_matrix: np.ndarray,
                         method: str,
                         config: Dict) -> Dict:
        """
        Run the selected optimization method.

        This integrates with existing Librex methods.
        """
        from Librex.core.interfaces import StandardizedProblem

        # Extract initial solution
        initial_solution = config.pop('initial_solution', None)

        # Create standardized problem
        n = len(flow_matrix)

        def objective_function(x):
            """QAP objective function."""
            if len(x) == n:
                # x is a permutation vector
                perm = x.astype(int)
            else:
                # x is a flattened permutation matrix
                P = x.reshape((n, n))
                perm = np.argmax(P, axis=1)

            # Use fast evaluator if available
            if 'evaluator' in config:
                return config['evaluator'](perm)
            else:
                P = np.zeros((n, n))
                P[np.arange(n), perm] = 1
                return np.trace(flow_matrix @ P @ distance_matrix @ P.T)

        problem = StandardizedProblem(
            dimension=n,
            objective_matrix=None,
            objective_function=objective_function,
            constraint_matrix=None,
            problem_metadata={
                'type': 'qap',
                'flow_matrix': flow_matrix,
                'distance_matrix': distance_matrix
            }
        )

        # Import and run the selected method
        if method == 'simulated_annealing':
            from Librex.methods.baselines.simulated_annealing import (
                simulated_annealing_optimize,
            )
            # Add initial solution to config
            if initial_solution is not None:
                config['initial_solution'] = initial_solution
            result = simulated_annealing_optimize(problem, config)

        elif method == 'tabu_search':
            from Librex.methods.baselines.tabu_search import tabu_search_optimize
            if initial_solution is not None:
                config['initial_solution'] = initial_solution
            result = tabu_search_optimize(problem, config)

        elif method == 'genetic_algorithm':
            from Librex.methods.baselines.genetic_algorithm import (
                genetic_algorithm_optimize,
            )
            if initial_solution is not None:
                # Seed population with initial solution
                config['seed_solution'] = initial_solution
            result = genetic_algorithm_optimize(problem, config)

        else:
            # Fallback to random search
            from Librex.methods.baselines.random_search import random_search_optimize
            result = random_search_optimize(problem, config)

        return result

    def benchmark_speedup(self,
                         flow_matrix: np.ndarray,
                         distance_matrix: np.ndarray) -> Dict:
        """
        Benchmark the speedup achieved by preconditioning.

        Compares preconditioned solver with standard solver.
        """
        import time

        n = len(flow_matrix)
        num_trials = 5

        # Run with preconditioning
        preconditioned_times = []
        preconditioned_objectives = []

        for _ in range(num_trials):
            start_time = time.time()
            result = self.solve(flow_matrix, distance_matrix,
                              method='simulated_annealing',
                              config={'max_iterations': 1000})
            elapsed = time.time() - start_time
            preconditioned_times.append(elapsed)
            preconditioned_objectives.append(result['objective'])

        # Run without preconditioning (standard random start)
        standard_times = []
        standard_objectives = []

        # Temporarily disable preconditioning
        original_fft = self.enable_fft
        self.enable_fft = False

        for _ in range(num_trials):
            initial = np.random.permutation(n)
            start_time = time.time()

            # Standard solve
            from Librex.methods.baselines.simulated_annealing import (
                simulated_annealing_optimize,
            )
            from Librex.core.interfaces import StandardizedProblem

            def objective(x):
                if len(x) == n:
                    perm = x.astype(int)
                else:
                    P = x.reshape((n, n))
                    perm = np.argmax(P, axis=1)
                P = np.zeros((n, n))
                P[np.arange(n), perm] = 1
                return np.trace(flow_matrix @ P @ distance_matrix @ P.T)

            problem = StandardizedProblem(
                dimension=n,
                objective_matrix=None,
                objective_function=objective,
                constraint_matrix=None,
                problem_metadata={'type': 'qap'}
            )

            result = simulated_annealing_optimize(
                problem,
                {'initial_solution': initial, 'max_iterations': 1000}
            )

            elapsed = time.time() - start_time
            standard_times.append(elapsed)
            standard_objectives.append(result['objective'])

        # Restore FFT setting
        self.enable_fft = original_fft

        # Compute statistics
        avg_preconditioned_time = np.mean(preconditioned_times)
        avg_standard_time = np.mean(standard_times)
        avg_preconditioned_obj = np.mean(preconditioned_objectives)
        avg_standard_obj = np.mean(standard_objectives)

        speedup = avg_standard_time / avg_preconditioned_time if avg_preconditioned_time > 0 else 1.0
        quality_improvement = (avg_standard_obj - avg_preconditioned_obj) / avg_standard_obj if avg_standard_obj > 0 else 0

        return {
            'speedup': speedup,
            'quality_improvement': quality_improvement,
            'preconditioned_time': avg_preconditioned_time,
            'standard_time': avg_standard_time,
            'preconditioned_objective': avg_preconditioned_obj,
            'standard_objective': avg_standard_obj
        }