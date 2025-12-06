"""
ML Recommender Integration with ORCHEX
--------------------------------------
Integrates the ML-based solver recommendation engine with ORCHEX orchestration system.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .ml_recommender import SolverRecommender, FeatureExtractor
from .libria_solvers import LibriaSolverAdapter
from .blackboard import Blackboard

logger = logging.getLogger(__name__)


class ATLASMLRecommender:
    """Integration layer between ORCHEX and ML recommender."""

    def __init__(self, blackboard: Optional[Blackboard] = None):
        """
        Initialize ORCHEX ML recommender integration.

        Args:
            blackboard: ORCHEX blackboard for shared state
        """
        self.recommender = SolverRecommender(model_type='random_forest')
        self.feature_extractor = FeatureExtractor()
        self.blackboard = blackboard
        self.solver_adapter = LibriaSolverAdapter()
        self.model_path = Path("models/atlas_recommender.pkl")
        self.performance_history = []

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load pre-trained model if available."""
        if self.model_path.exists():
            try:
                self.recommender.load_model(self.model_path)
                logger.info("Loaded pre-trained ML recommender model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

    def recommend_solver_for_task(self, task: Dict[str, Any]) -> Tuple[str, float]:
        """
        Recommend the best solver for an ORCHEX task.

        Args:
            task: ORCHEX task dictionary

        Returns:
            Tuple of (solver_name, confidence)
        """
        # Convert ORCHEX task to problem format
        problem = self._convert_task_to_problem(task)

        # Get recommendation
        solver, confidence = self.recommender.predict(problem, return_confidence=True)

        # Log to blackboard if available
        if self.blackboard:
            self.blackboard.write('ml_recommendation', {
                'task_id': task.get('id'),
                'recommended_solver': solver,
                'confidence': confidence,
                'problem_features': problem
            })

        logger.info(f"Recommended {solver} for task {task.get('id')} "
                   f"with {confidence:.2%} confidence")

        return solver, confidence

    def get_solver_ranking(self, task: Dict[str, Any], k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k solver recommendations for a task.

        Args:
            task: ORCHEX task dictionary
            k: Number of recommendations

        Returns:
            List of (solver_name, confidence) tuples
        """
        problem = self._convert_task_to_problem(task)
        rankings = self.recommender.predict_top_k(problem, k=k)

        # Store in blackboard
        if self.blackboard:
            self.blackboard.write('solver_rankings', {
                'task_id': task.get('id'),
                'rankings': rankings
            })

        return rankings

    def _convert_task_to_problem(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ORCHEX task format to ML problem format.

        Args:
            task: ORCHEX task dictionary

        Returns:
            Problem dictionary for ML recommender
        """
        problem = {
            'name': task.get('name', 'unknown'),
            'type': task.get('problem_type', 'continuous'),
            'num_variables': task.get('dimensions', {}).get('variables', 100),
            'num_constraints': task.get('dimensions', {}).get('constraints', 0),
            'num_objectives': task.get('dimensions', {}).get('objectives', 1),
        }

        # Extract constraint information
        constraints = task.get('constraints', {})
        problem['constraints'] = {
            'linear': constraints.get('linear_count', 0),
            'nonlinear': constraints.get('nonlinear_count', 0),
            'equality': constraints.get('equality_count', 0),
        }

        # Extract variable information
        variables = task.get('variables', {})
        problem['variables'] = {
            'continuous': variables.get('continuous_count', problem['num_variables']),
            'discrete': variables.get('discrete_count', 0),
            'binary': variables.get('binary_count', 0),
            'bounded': variables.get('bounded_count', 0),
        }

        # Extract objective characteristics
        objective = task.get('objective_properties', {})
        problem['objective'] = {
            'linearity': objective.get('linearity', 0.5),
            'convexity': objective.get('convexity', 0.5),
            'smoothness': objective.get('smoothness', 0.5),
            'multimodality': objective.get('multimodality', 0.5),
        }

        # Graph properties for combinatorial problems
        if 'graph' in task:
            problem['graph'] = task['graph']

        # Statistical properties
        if 'statistics' in task:
            problem['statistics'] = task['statistics']

        return problem

    def record_performance(self, task: Dict[str, Any],
                          solver: str,
                          performance_metrics: Dict[str, float]):
        """
        Record solver performance for online learning.

        Args:
            task: ORCHEX task that was solved
            solver: Solver that was used
            performance_metrics: Performance metrics (time, quality, etc.)
        """
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(performance_metrics)

        # Convert task to problem format
        problem = self._convert_task_to_problem(task)

        # Update ML model
        self.recommender.update_online(problem, solver, performance_score)

        # Store in history
        self.performance_history.append({
            'task': task,
            'solver': solver,
            'metrics': performance_metrics,
            'score': performance_score
        })

        # Log to blackboard
        if self.blackboard:
            self.blackboard.write('solver_performance', {
                'task_id': task.get('id'),
                'solver': solver,
                'performance': performance_score,
                'metrics': performance_metrics
            })

        logger.info(f"Recorded performance for {solver}: {performance_score:.3f}")

    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall performance score from metrics.

        Args:
            metrics: Performance metrics dictionary

        Returns:
            Overall performance score (0-1)
        """
        # Weighted combination of metrics
        weights = {
            'solution_quality': 0.4,
            'convergence_speed': 0.3,
            'stability': 0.2,
            'efficiency': 0.1
        }

        score = 0.0
        total_weight = 0.0

        for metric, weight in weights.items():
            if metric in metrics:
                # Normalize metric to 0-1 range
                value = min(max(metrics[metric], 0.0), 1.0)
                score += value * weight
                total_weight += weight

        # Normalize by actual weights used
        if total_weight > 0:
            score = score / total_weight

        return score

    def train_on_historical_data(self, historical_tasks: List[Dict[str, Any]]):
        """
        Train the ML model on historical ORCHEX task data.

        Args:
            historical_tasks: List of historical tasks with performance data
        """
        training_data = []

        for task_record in historical_tasks:
            task = task_record['task']
            best_solver = task_record['best_solver']

            problem = self._convert_task_to_problem(task)
            training_data.append({
                'problem': problem,
                'best_solver': best_solver
            })

        if training_data:
            logger.info(f"Training on {len(training_data)} historical tasks")
            metrics = self.recommender.train(training_data, perform_hyperopt=True)
            logger.info(f"Training accuracy: {metrics['accuracy']:.3f}")

            # Save updated model
            self.save_model()

    def save_model(self):
        """Save the current ML model."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.recommender.save_model(self.model_path)
        logger.info(f"Saved ML model to {self.model_path}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary of feature importances
        """
        if hasattr(self.recommender, 'performance_metrics'):
            return self.recommender.performance_metrics.get('feature_importance', {})
        return {}

    def analyze_solver_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze effectiveness of each solver based on historical performance.

        Returns:
            Dictionary mapping solvers to their effectiveness metrics
        """
        solver_stats = {}

        for record in self.performance_history:
            solver = record['solver']
            score = record['score']

            if solver not in solver_stats:
                solver_stats[solver] = {
                    'total_uses': 0,
                    'total_score': 0.0,
                    'best_score': 0.0,
                    'worst_score': 1.0,
                }

            stats = solver_stats[solver]
            stats['total_uses'] += 1
            stats['total_score'] += score
            stats['best_score'] = max(stats['best_score'], score)
            stats['worst_score'] = min(stats['worst_score'], score)

        # Calculate averages
        for solver, stats in solver_stats.items():
            if stats['total_uses'] > 0:
                stats['average_score'] = stats['total_score'] / stats['total_uses']

        return solver_stats


class AdaptiveSolverSelector:
    """
    Adaptive solver selection strategy combining ML recommendations
    with runtime performance monitoring.
    """

    def __init__(self, ml_recommender: ATLASMLRecommender):
        """
        Initialize adaptive selector.

        Args:
            ml_recommender: ORCHEX ML recommender instance
        """
        self.ml_recommender = ml_recommender
        self.active_solvers = {}
        self.solver_timeouts = {
            'genetic_algorithm': 300,
            'simulated_annealing': 200,
            'particle_swarm': 250,
            'gradient_descent': 100,
            'newton_method': 150,
        }

    def select_solver(self, task: Dict[str, Any]) -> str:
        """
        Select the best solver adaptively based on ML and runtime info.

        Args:
            task: ORCHEX task dictionary

        Returns:
            Selected solver name
        """
        # Get ML recommendations
        top_solvers = self.ml_recommender.get_solver_ranking(task, k=3)

        # Filter based on runtime constraints
        task_constraints = task.get('constraints', {})
        time_limit = task_constraints.get('time_limit', float('inf'))

        suitable_solvers = []
        for solver, confidence in top_solvers:
            expected_time = self.solver_timeouts.get(solver, 200)
            if expected_time <= time_limit:
                suitable_solvers.append((solver, confidence))

        if suitable_solvers:
            # Select solver with highest confidence that meets constraints
            return suitable_solvers[0][0]
        else:
            # Fallback to fastest solver
            fastest_solver = min(self.solver_timeouts.items(),
                               key=lambda x: x[1])
            return fastest_solver[0]

    def adapt_selection(self, task: Dict[str, Any],
                       solver: str,
                       runtime_metrics: Dict[str, float]):
        """
        Adapt selection strategy based on runtime performance.

        Args:
            task: ORCHEX task that was solved
            solver: Solver that was used
            runtime_metrics: Runtime performance metrics
        """
        # Update ML model with performance
        self.ml_recommender.record_performance(task, solver, runtime_metrics)

        # Update timeout estimates
        if 'execution_time' in runtime_metrics:
            current_timeout = self.solver_timeouts.get(solver, 200)
            new_time = runtime_metrics['execution_time']

            # Exponential moving average
            alpha = 0.3
            updated_timeout = alpha * new_time + (1 - alpha) * current_timeout
            self.solver_timeouts[solver] = updated_timeout

            logger.info(f"Updated timeout for {solver}: {updated_timeout:.1f}s")