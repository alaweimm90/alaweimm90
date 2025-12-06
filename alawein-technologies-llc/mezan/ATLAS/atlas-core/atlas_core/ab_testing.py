"""A/B Testing Framework for MEZAN.

This module provides a comprehensive A/B testing system with traffic splitting,
metrics collection, statistical analysis, early stopping, and both frequentist
and Bayesian approaches.

Author: Meshal Alawein
Date: 2025-11-18
"""

import hashlib
import json
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import beta, norm

from .validation import StatisticalValidator, ValidationResult


class SplitMethod(Enum):
    """Traffic splitting methods for A/B tests."""
    USER_ID = "user_id"
    SESSION_ID = "session_id"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED_RANDOM = "weighted_random"


class MetricType(Enum):
    """Types of metrics to track in experiments."""
    CONVERSION = "conversion"
    CONTINUOUS = "continuous"
    COUNT = "count"
    DURATION = "duration"
    REVENUE = "revenue"


class ExperimentStatus(Enum):
    """Status of an A/B test experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"


@dataclass
class Variant:
    """Represents a variant in an A/B test."""
    name: str
    description: str
    weight: float = 0.5  # Traffic allocation weight
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition for experiments."""
    name: str
    type: MetricType
    description: str
    is_primary: bool = False
    higher_is_better: bool = True
    minimum_detectable_effect: Optional[float] = None


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment."""
    name: str
    description: str
    variants: List[Variant]
    metrics: List[Metric]
    split_method: SplitMethod = SplitMethod.USER_ID
    target_sample_size: Optional[int] = None
    max_duration_days: Optional[int] = None
    confidence_level: float = 0.95
    power: float = 0.8
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 0.001  # P-value threshold for stopping
    enable_bayesian: bool = True
    segment_filters: Optional[Dict[str, Any]] = None


@dataclass
class ObservedData:
    """Observed data point in an experiment."""
    user_id: str
    variant: str
    metric_name: str
    value: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestExperiment:
    """Individual A/B test experiment."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize an A/B test experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.experiment_id = str(uuid.uuid4())
        self.status = ExperimentStatus.DRAFT
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Variant assignment tracking
        self.assignments: Dict[str, str] = {}  # user_id -> variant
        self.variant_counts: Dict[str, int] = defaultdict(int)

        # Data collection
        self.data: List[ObservedData] = []
        self.metric_aggregates: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Statistical validator
        self.validator = StatisticalValidator(config.confidence_level)

        # Results cache
        self._results_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None

    def start(self):
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in {self.status} status")

        self.status = ExperimentStatus.RUNNING
        self.start_time = datetime.now()
        self._results_cache = None

    def pause(self):
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Cannot pause experiment in {self.status} status")

        self.status = ExperimentStatus.PAUSED

    def resume(self):
        """Resume the experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError(f"Cannot resume experiment in {self.status} status")

        self.status = ExperimentStatus.RUNNING

    def stop(self):
        """Stop the experiment."""
        if self.status not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
            raise ValueError(f"Cannot stop experiment in {self.status} status")

        self.status = ExperimentStatus.STOPPED
        self.end_time = datetime.now()

    def complete(self):
        """Mark experiment as completed."""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()

    def assign_variant(self, user_id: str, context: Optional[Dict] = None) -> str:
        """
        Assign a variant to a user.

        Args:
            user_id: User identifier
            context: Optional context for assignment

        Returns:
            Assigned variant name
        """
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Experiment is not running")

        # Check if user already assigned
        if user_id in self.assignments:
            return self.assignments[user_id]

        # Apply segment filters if specified
        if self.config.segment_filters and context:
            if not self._matches_segment(context):
                return "control"  # Default to control for non-matching segments

        # Determine variant based on split method
        variant = self._determine_variant(user_id)

        # Track assignment
        self.assignments[user_id] = variant
        self.variant_counts[variant] += 1

        return variant

    def _determine_variant(self, identifier: str) -> str:
        """Determine variant based on split method."""
        if self.config.split_method == SplitMethod.RANDOM:
            return self._random_variant()

        elif self.config.split_method == SplitMethod.WEIGHTED_RANDOM:
            return self._weighted_random_variant()

        elif self.config.split_method in [SplitMethod.USER_ID,
                                         SplitMethod.SESSION_ID,
                                         SplitMethod.HASH_BASED]:
            return self._hash_based_variant(identifier)

        else:
            raise ValueError(f"Unknown split method: {self.config.split_method}")

    def _random_variant(self) -> str:
        """Randomly select a variant with equal probability."""
        return random.choice([v.name for v in self.config.variants])

    def _weighted_random_variant(self) -> str:
        """Select variant based on weights."""
        weights = [v.weight for v in self.config.variants]
        return random.choices(
            [v.name for v in self.config.variants],
            weights=weights,
            k=1
        )[0]

    def _hash_based_variant(self, identifier: str) -> str:
        """Deterministically assign variant based on hash."""
        hash_value = int(hashlib.md5(
            f"{self.experiment_id}:{identifier}".encode()
        ).hexdigest(), 16)

        # Normalize to [0, 1]
        normalized = (hash_value % 10000) / 10000

        # Assign based on cumulative weights
        cumulative = 0
        total_weight = sum(v.weight for v in self.config.variants)

        for variant in self.config.variants:
            cumulative += variant.weight / total_weight
            if normalized < cumulative:
                return variant.name

        return self.config.variants[-1].name

    def _matches_segment(self, context: Dict) -> bool:
        """Check if context matches segment filters."""
        for key, expected in self.config.segment_filters.items():
            if key not in context:
                return False
            if isinstance(expected, list):
                if context[key] not in expected:
                    return False
            elif context[key] != expected:
                return False
        return True

    def track_metric(self,
                    user_id: str,
                    metric_name: str,
                    value: Any,
                    metadata: Optional[Dict] = None):
        """
        Track a metric observation.

        Args:
            user_id: User identifier
            metric_name: Name of the metric
            value: Observed value
            metadata: Optional metadata
        """
        if user_id not in self.assignments:
            raise ValueError(f"User {user_id} not assigned to a variant")

        variant = self.assignments[user_id]

        observation = ObservedData(
            user_id=user_id,
            variant=variant,
            metric_name=metric_name,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.data.append(observation)
        self.metric_aggregates[metric_name][variant].append(float(value))

        # Clear cache
        self._results_cache = None

    def analyze_results(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Analyze experiment results.

        Args:
            force_refresh: Force refresh of cached results

        Returns:
            Dictionary with analysis results
        """
        # Use cache if available and fresh
        if (not force_refresh and
            self._results_cache and
            self._cache_timestamp and
            (datetime.now() - self._cache_timestamp).seconds < 60):
            return self._results_cache

        results = {
            'experiment_id': self.experiment_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_hours': self._get_duration_hours(),
            'sample_sizes': self.variant_counts,
            'metrics': {}
        }

        # Analyze each metric
        for metric in self.config.metrics:
            metric_results = self._analyze_metric(metric.name)
            results['metrics'][metric.name] = metric_results

            # Check early stopping if enabled
            if (self.config.enable_early_stopping and
                self.status == ExperimentStatus.RUNNING and
                metric.is_primary):
                should_stop = self._check_early_stopping(metric_results)
                if should_stop:
                    results['early_stopping_triggered'] = True
                    results['early_stopping_metric'] = metric.name

        # Overall recommendation
        results['recommendation'] = self._generate_recommendation(results)

        # Cache results
        self._results_cache = results
        self._cache_timestamp = datetime.now()

        return results

    def _analyze_metric(self, metric_name: str) -> Dict[str, Any]:
        """Analyze a single metric."""
        if metric_name not in self.metric_aggregates:
            return {'error': 'No data for metric'}

        metric_data = self.metric_aggregates[metric_name]

        if len(metric_data) < 2:
            return {'error': 'Need at least 2 variants'}

        # Get control and treatment data
        variants = list(metric_data.keys())
        control = variants[0]  # Assume first variant is control
        treatment = variants[1] if len(variants) > 1 else None

        if not treatment:
            return {'error': 'No treatment variant'}

        control_data = np.array(metric_data[control])
        treatment_data = np.array(metric_data[treatment])

        # Frequentist analysis
        freq_results = self.validator.t_test(treatment_data, control_data)

        analysis = {
            'control': {
                'name': control,
                'mean': np.mean(control_data),
                'std': np.std(control_data, ddof=1),
                'sample_size': len(control_data)
            },
            'treatment': {
                'name': treatment,
                'mean': np.mean(treatment_data),
                'std': np.std(treatment_data, ddof=1),
                'sample_size': len(treatment_data)
            },
            'frequentist': {
                'p_value': freq_results.p_value,
                'is_significant': freq_results.is_significant,
                'effect_size': freq_results.effect_size,
                'confidence_interval': freq_results.confidence_interval,
                'lift': ((np.mean(treatment_data) - np.mean(control_data)) /
                        np.mean(control_data) * 100 if np.mean(control_data) != 0 else 0)
            }
        }

        # Bayesian analysis if enabled
        if self.config.enable_bayesian:
            analysis['bayesian'] = self._bayesian_analysis(control_data, treatment_data)

        return analysis

    def _bayesian_analysis(self,
                          control: np.ndarray,
                          treatment: np.ndarray) -> Dict[str, Any]:
        """
        Perform Bayesian analysis.

        Args:
            control: Control group data
            treatment: Treatment group data

        Returns:
            Bayesian analysis results
        """
        # For binary outcomes (conversions)
        if all(v in [0, 1] for v in control) and all(v in [0, 1] for v in treatment):
            # Beta-Binomial model
            control_success = np.sum(control)
            control_trials = len(control)
            treatment_success = np.sum(treatment)
            treatment_trials = len(treatment)

            # Posterior distributions (uniform prior: Beta(1,1))
            control_posterior = beta(control_success + 1,
                                   control_trials - control_success + 1)
            treatment_posterior = beta(treatment_success + 1,
                                     treatment_trials - treatment_success + 1)

            # Probability that treatment is better
            samples = 10000
            control_samples = control_posterior.rvs(samples)
            treatment_samples = treatment_posterior.rvs(samples)
            prob_treatment_better = np.mean(treatment_samples > control_samples)

            # Expected lift
            expected_lift = (np.mean(treatment_samples) - np.mean(control_samples)) / np.mean(control_samples) * 100

            return {
                'prob_treatment_better': prob_treatment_better,
                'expected_lift': expected_lift,
                'control_posterior_mean': control_posterior.mean(),
                'treatment_posterior_mean': treatment_posterior.mean(),
                'control_credible_interval': control_posterior.interval(0.95),
                'treatment_credible_interval': treatment_posterior.interval(0.95)
            }

        else:
            # Continuous outcomes - use normal approximation
            control_mean, control_std = np.mean(control), np.std(control, ddof=1)
            treatment_mean, treatment_std = np.mean(treatment), np.std(treatment, ddof=1)

            # Posterior distributions (using sample statistics)
            control_posterior = norm(control_mean, control_std / np.sqrt(len(control)))
            treatment_posterior = norm(treatment_mean, treatment_std / np.sqrt(len(treatment)))

            # Monte Carlo simulation for probability
            samples = 10000
            control_samples = control_posterior.rvs(samples)
            treatment_samples = treatment_posterior.rvs(samples)
            prob_treatment_better = np.mean(treatment_samples > control_samples)

            return {
                'prob_treatment_better': prob_treatment_better,
                'expected_difference': treatment_mean - control_mean,
                'control_posterior_mean': control_mean,
                'treatment_posterior_mean': treatment_mean,
                'control_credible_interval': control_posterior.interval(0.95),
                'treatment_credible_interval': treatment_posterior.interval(0.95)
            }

    def _check_early_stopping(self, metric_results: Dict) -> bool:
        """
        Check if early stopping criteria are met.

        Args:
            metric_results: Results for the primary metric

        Returns:
            True if experiment should be stopped early
        """
        if 'frequentist' not in metric_results:
            return False

        p_value = metric_results['frequentist']['p_value']

        # Sequential testing with alpha spending
        information_fraction = self._get_information_fraction()

        # O'Brien-Fleming spending function
        z_alpha = norm.ppf(1 - self.config.early_stopping_threshold/2)
        z_current = z_alpha / np.sqrt(information_fraction) if information_fraction > 0 else np.inf

        # Convert z-score to p-value
        adjusted_threshold = 2 * (1 - norm.cdf(z_current))

        return p_value < adjusted_threshold

    def _get_information_fraction(self) -> float:
        """Calculate information fraction for sequential testing."""
        if not self.config.target_sample_size:
            # Use time-based fraction
            if not self.config.max_duration_days or not self.start_time:
                return 0

            elapsed = (datetime.now() - self.start_time).days
            return min(1.0, elapsed / self.config.max_duration_days)

        # Sample size based fraction
        current_n = sum(self.variant_counts.values())
        return min(1.0, current_n / self.config.target_sample_size)

    def _get_duration_hours(self) -> Optional[float]:
        """Get experiment duration in hours."""
        if not self.start_time:
            return None

        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds() / 3600

    def _generate_recommendation(self, results: Dict) -> Dict[str, Any]:
        """Generate recommendation based on results."""
        recommendation = {
            'action': 'continue',
            'confidence': 'low',
            'reasoning': []
        }

        # Check if we have enough data
        min_sample_size = 100  # Minimum per variant
        if any(n < min_sample_size for n in self.variant_counts.values()):
            recommendation['reasoning'].append('Insufficient sample size')
            return recommendation

        # Check primary metrics
        primary_metrics = [m for m in self.config.metrics if m.is_primary]

        if not primary_metrics:
            recommendation['reasoning'].append('No primary metric defined')
            return recommendation

        significant_improvements = 0
        for metric in primary_metrics:
            if metric.name not in results['metrics']:
                continue

            metric_result = results['metrics'][metric.name]
            if 'frequentist' not in metric_result:
                continue

            if metric_result['frequentist']['is_significant']:
                effect_size = metric_result['frequentist'].get('effect_size', 0)
                lift = metric_result['frequentist'].get('lift', 0)

                if metric.higher_is_better:
                    if lift > 0:
                        significant_improvements += 1
                        recommendation['reasoning'].append(
                            f"{metric.name} shows {lift:.2f}% improvement"
                        )
                else:
                    if lift < 0:
                        significant_improvements += 1
                        recommendation['reasoning'].append(
                            f"{metric.name} shows {abs(lift):.2f}% improvement"
                        )

        # Make recommendation
        if significant_improvements == len(primary_metrics):
            recommendation['action'] = 'deploy_treatment'
            recommendation['confidence'] = 'high'
        elif significant_improvements > 0:
            recommendation['action'] = 'continue_monitoring'
            recommendation['confidence'] = 'medium'
        else:
            recommendation['action'] = 'keep_control'
            recommendation['confidence'] = 'medium'

        return recommendation


class ABTestingFramework:
    """Main A/B testing framework for MEZAN."""

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the A/B testing framework.

        Args:
            storage_path: Path for storing experiment data
        """
        self.experiments: Dict[str, ABTestExperiment] = {}
        self.storage_path = storage_path or Path("experiments")
        self.storage_path.mkdir(exist_ok=True, parents=True)

    def create_experiment(self, config: ExperimentConfig) -> ABTestExperiment:
        """
        Create a new experiment.

        Args:
            config: Experiment configuration

        Returns:
            Created experiment
        """
        experiment = ABTestExperiment(config)
        self.experiments[experiment.experiment_id] = experiment

        # Save experiment config
        self._save_experiment(experiment)

        return experiment

    def get_experiment(self, experiment_id: str) -> Optional[ABTestExperiment]:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[Dict]:
        """
        List all experiments.

        Args:
            status: Filter by status (optional)

        Returns:
            List of experiment summaries
        """
        experiments = []
        for exp_id, exp in self.experiments.items():
            if status and exp.status != status:
                continue

            experiments.append({
                'id': exp_id,
                'name': exp.config.name,
                'status': exp.status.value,
                'start_time': exp.start_time.isoformat() if exp.start_time else None,
                'variants': len(exp.config.variants),
                'total_users': len(exp.assignments)
            })

        return experiments

    def route_traffic(self,
                     experiment_id: str,
                     user_id: str,
                     context: Optional[Dict] = None) -> str:
        """
        Route traffic for an experiment.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            context: Optional context

        Returns:
            Assigned variant
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        return experiment.assign_variant(user_id, context)

    def track_conversion(self,
                        experiment_id: str,
                        user_id: str,
                        converted: bool = True,
                        value: Optional[float] = None):
        """
        Track a conversion event.

        Args:
            experiment_id: Experiment identifier
            user_id: User identifier
            converted: Whether user converted
            value: Optional conversion value
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.track_metric(
            user_id,
            'conversion',
            1 if converted else 0,
            {'value': value} if value else None
        )

    def calculate_sample_size(self,
                             baseline_rate: float,
                             minimum_detectable_effect: float,
                             power: float = 0.8,
                             confidence_level: float = 0.95) -> int:
        """
        Calculate required sample size for an experiment.

        Args:
            baseline_rate: Baseline conversion rate
            minimum_detectable_effect: Minimum effect to detect (relative)
            power: Statistical power
            confidence_level: Confidence level

        Returns:
            Required sample size per variant
        """
        # For proportions test
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)

        # Effect size (Cohen's h)
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))

        # Use power analysis
        from statsmodels.stats.power import NormalIndPower
        power_analysis = NormalIndPower()

        n = power_analysis.solve_power(
            effect_size=abs(effect_size),
            power=power,
            alpha=1 - confidence_level
        )

        return int(np.ceil(n))

    def generate_dashboard(self, experiment_id: str, output_path: Optional[Path] = None):
        """
        Generate experiment dashboard.

        Args:
            experiment_id: Experiment identifier
            output_path: Path to save dashboard
        """
        experiment = self.get_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        results = experiment.analyze_results()

        # Create dashboard
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(f"A/B Test Dashboard: {experiment.config.name}",
                    fontsize=16, fontweight='bold')

        # Variant distribution
        ax1 = fig.add_subplot(gs[0, 0])
        variants = list(experiment.variant_counts.keys())
        counts = list(experiment.variant_counts.values())
        ax1.bar(variants, counts)
        ax1.set_title("Variant Distribution")
        ax1.set_xlabel("Variant")
        ax1.set_ylabel("Users")

        # Metric results
        metric_idx = 0
        for row in range(3):
            for col in range(3):
                if row == 0 and col == 0:
                    continue

                if metric_idx >= len(experiment.config.metrics):
                    break

                ax = fig.add_subplot(gs[row, col])
                metric = experiment.config.metrics[metric_idx]

                if metric.name in results['metrics']:
                    metric_result = results['metrics'][metric.name]

                    if 'control' in metric_result and 'treatment' in metric_result:
                        # Bar plot of means
                        means = [metric_result['control']['mean'],
                               metric_result['treatment']['mean']]
                        stds = [metric_result['control']['std'],
                               metric_result['treatment']['std']]
                        x = [0, 1]
                        ax.bar(x, means, yerr=stds, capsize=5)
                        ax.set_xticks(x)
                        ax.set_xticklabels(['Control', 'Treatment'])
                        ax.set_title(f"{metric.name}")
                        ax.set_ylabel("Value")

                        # Add significance annotation
                        if 'frequentist' in metric_result:
                            p_value = metric_result['frequentist']['p_value']
                            sig_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                            ax.text(0.5, max(means) * 1.1, sig_text,
                                   ha='center', fontsize=12)

                metric_idx += 1

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def _save_experiment(self, experiment: ABTestExperiment):
        """Save experiment configuration and data."""
        exp_path = self.storage_path / f"{experiment.experiment_id}.json"

        data = {
            'id': experiment.experiment_id,
            'config': {
                'name': experiment.config.name,
                'description': experiment.config.description,
                'variants': [
                    {'name': v.name, 'description': v.description, 'weight': v.weight}
                    for v in experiment.config.variants
                ],
                'metrics': [
                    {
                        'name': m.name,
                        'type': m.type.value,
                        'description': m.description,
                        'is_primary': m.is_primary
                    }
                    for m in experiment.config.metrics
                ],
                'split_method': experiment.config.split_method.value,
                'confidence_level': experiment.config.confidence_level
            },
            'status': experiment.status.value,
            'start_time': experiment.start_time.isoformat() if experiment.start_time else None,
            'variant_counts': dict(experiment.variant_counts)
        }

        with open(exp_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_experiments(self):
        """Load experiments from storage."""
        for exp_file in self.storage_path.glob("*.json"):
            with open(exp_file) as f:
                data = json.load(f)

            # Reconstruct experiment
            # (Simplified - would need full reconstruction logic)
            pass


# Convenience functions for quick experiments
def quick_ab_test(control_data: List[float],
                  treatment_data: List[float],
                  metric_name: str = "conversion") -> Dict[str, Any]:
    """
    Quick A/B test between control and treatment.

    Args:
        control_data: Control group data
        treatment_data: Treatment group data
        metric_name: Name of the metric

    Returns:
        Test results
    """
    # Create simple experiment config
    config = ExperimentConfig(
        name="Quick Test",
        description="Quick A/B test",
        variants=[
            Variant("control", "Control group"),
            Variant("treatment", "Treatment group")
        ],
        metrics=[
            Metric(metric_name, MetricType.CONTINUOUS, "Test metric", is_primary=True)
        ]
    )

    # Create and populate experiment
    experiment = ABTestExperiment(config)
    experiment.start()

    # Add data
    for i, value in enumerate(control_data):
        user_id = f"control_{i}"
        experiment.assignments[user_id] = "control"
        experiment.track_metric(user_id, metric_name, value)

    for i, value in enumerate(treatment_data):
        user_id = f"treatment_{i}"
        experiment.assignments[user_id] = "treatment"
        experiment.track_metric(user_id, metric_name, value)

    # Analyze
    return experiment.analyze_results()