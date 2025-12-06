#!/usr/bin/env python3
"""
CLI for training the Adaptive Learning System on historical data

This script provides a command-line interface for training meta-learning
models on historical optimization data.

Usage:
    python -m Librex.adaptive.train --data benchmark_results/history.db

Author: Meshal Alawein
Date: 2025-11-18
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

from Librex.adaptive import (
    MetaLearner,
    PerformanceDatabase,
    PerformancePredictor,
)
from Librex.adaptive.performance_db import OptimizationRun
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_historical_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load historical optimization data from various sources.

    Args:
        data_path: Path to data file or directory

    Returns:
        List of optimization run dictionaries
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    runs = []

    if path.is_file():
        if path.suffix == '.json':
            # Load JSON file
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    runs = data
                else:
                    runs = [data]

        elif path.suffix == '.db':
            # Load from SQLite database
            db = PerformanceDatabase(str(path))
            history = db.get_performance_history(limit=10000)
            runs = [
                {
                    'problem_id': run.problem_id,
                    'problem_features': run.problem_features.tolist(),
                    'algorithm': run.method_used,
                    'hyperparameters': run.hyperparameters,
                    'performance': run.solution_quality,
                    'objective_value': run.objective_value,
                    'runtime': run.runtime_seconds,
                    'n_evaluations': run.n_evaluations
                }
                for run in history
            ]
            db.close()

        elif path.suffix == '.csv':
            # Load from CSV
            import pandas as pd
            df = pd.read_csv(path)

            for _, row in df.iterrows():
                # Parse features from string if needed
                if isinstance(row.get('problem_features'), str):
                    features = json.loads(row['problem_features'])
                else:
                    features = row.get('problem_features', [])

                runs.append({
                    'problem_id': row.get('problem_id', f"problem_{_}"),
                    'problem_features': features,
                    'algorithm': row.get('algorithm', row.get('method_used')),
                    'performance': row.get('performance', row.get('solution_quality', 0.5)),
                    'objective_value': row.get('objective_value', 100.0),
                    'runtime': row.get('runtime', row.get('runtime_seconds', 10.0)),
                    'n_evaluations': row.get('n_evaluations', 1000)
                })

    elif path.is_dir():
        # Load all JSON files in directory
        for file_path in path.glob('*.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    runs.extend(data)
                else:
                    runs.append(data)

    logger.info(f"Loaded {len(runs)} optimization runs from {data_path}")
    return runs


def analyze_data_quality(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the quality and statistics of the loaded data.

    Args:
        runs: List of optimization runs

    Returns:
        Dictionary of data statistics
    """
    stats = {
        'total_runs': len(runs),
        'algorithms': {},
        'problem_sizes': [],
        'performance_range': {},
        'runtime_range': {},
        'missing_data': {
            'features': 0,
            'performance': 0,
            'hyperparameters': 0
        }
    }

    for run in runs:
        # Count algorithms
        alg = run.get('algorithm', 'unknown')
        stats['algorithms'][alg] = stats['algorithms'].get(alg, 0) + 1

        # Problem sizes
        features = run.get('problem_features', [])
        if features:
            stats['problem_sizes'].append(len(features))
        else:
            stats['missing_data']['features'] += 1

        # Performance statistics
        perf = run.get('performance')
        if perf is not None:
            if alg not in stats['performance_range']:
                stats['performance_range'][alg] = []
            stats['performance_range'][alg].append(perf)
        else:
            stats['missing_data']['performance'] += 1

        # Runtime statistics
        runtime = run.get('runtime')
        if runtime is not None:
            if alg not in stats['runtime_range']:
                stats['runtime_range'][alg] = []
            stats['runtime_range'][alg].append(runtime)

        # Check hyperparameters
        if not run.get('hyperparameters'):
            stats['missing_data']['hyperparameters'] += 1

    # Calculate aggregates
    for alg in stats['performance_range']:
        perfs = stats['performance_range'][alg]
        stats['performance_range'][alg] = {
            'mean': np.mean(perfs),
            'std': np.std(perfs),
            'min': np.min(perfs),
            'max': np.max(perfs)
        }

    for alg in stats['runtime_range']:
        runtimes = stats['runtime_range'][alg]
        stats['runtime_range'][alg] = {
            'mean': np.mean(runtimes),
            'std': np.std(runtimes),
            'min': np.min(runtimes),
            'max': np.max(runtimes)
        }

    if stats['problem_sizes']:
        stats['problem_dimensions'] = {
            'mean': np.mean(stats['problem_sizes']),
            'min': np.min(stats['problem_sizes']),
            'max': np.max(stats['problem_sizes'])
        }

    return stats


def train_models(
    runs: List[Dict[str, Any]],
    model_type: str = 'gradient_boosting',
    output_dir: str = 'models'
) -> Dict[str, Any]:
    """
    Train meta-learning models on historical data.

    Args:
        runs: List of optimization runs
        model_type: Type of prediction model to use
        output_dir: Directory to save trained models

    Returns:
        Training results dictionary
    """
    # Initialize meta-learner
    meta_learner = MetaLearner()

    # Train on historical data
    logger.info("Training meta-learner...")
    meta_learner.learn_from_history(runs)

    # Initialize performance predictor
    predictor = PerformancePredictor(model_type=model_type)

    # Prepare training data
    training_data = []
    algorithms = set()

    for run in runs:
        features = run.get('problem_features')
        algorithm = run.get('algorithm')
        performance = run.get('performance', 0.5)

        if features and algorithm:
            training_data.append((
                np.array(features),
                algorithm,
                performance
            ))
            algorithms.add(algorithm)

    if not training_data:
        raise ValueError("No valid training data found")

    # Train predictor
    logger.info(f"Training performance predictor with {len(training_data)} samples...")
    predictor.train(training_data, list(algorithms))

    # Save models
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    predictor.save_models(str(output_path))

    # Calculate training metrics
    results = {
        'n_samples': len(training_data),
        'n_algorithms': len(algorithms),
        'algorithms': list(algorithms),
        'model_type': model_type,
        'output_dir': str(output_path)
    }

    # Feature importance
    if predictor.feature_importance:
        results['feature_importance'] = {}
        for alg, importance in predictor.feature_importance.items():
            results['feature_importance'][alg] = importance.tolist()

    return results


def validate_models(
    predictor: PerformancePredictor,
    test_data: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Validate trained models on test data.

    Args:
        predictor: Trained performance predictor
        test_data: Test dataset

    Returns:
        Validation metrics
    """
    correct_predictions = 0
    total_predictions = 0
    errors = []

    for run in test_data:
        features = run.get('problem_features')
        true_algorithm = run.get('algorithm')
        true_performance = run.get('performance')

        if features and true_algorithm and true_performance:
            # Predict best algorithm
            features_array = np.array(features)
            best_alg, confidence = predictor.recommend_algorithm(features_array)

            if best_alg == true_algorithm:
                correct_predictions += 1

            # Calculate prediction error
            predictions = predictor.predict(features_array)
            if true_algorithm in predictions:
                pred_perf = predictions[true_algorithm][0]
                error = abs(pred_perf - true_performance)
                errors.append(error)

            total_predictions += 1

    metrics = {
        'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
        'mean_absolute_error': np.mean(errors) if errors else float('inf'),
        'n_test_samples': total_predictions
    }

    return metrics


def create_summary_report(
    data_stats: Dict[str, Any],
    training_results: Dict[str, Any],
    validation_metrics: Dict[str, float]
) -> str:
    """
    Create a summary report of the training process.

    Args:
        data_stats: Data quality statistics
        training_results: Training results
        validation_metrics: Validation metrics

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 70)
    report.append("ADAPTIVE LEARNING SYSTEM TRAINING REPORT")
    report.append("=" * 70)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Data statistics
    report.append("DATA STATISTICS")
    report.append("-" * 40)
    report.append(f"Total runs: {data_stats['total_runs']}")
    report.append(f"Algorithms: {len(data_stats['algorithms'])}")
    for alg, count in sorted(data_stats['algorithms'].items(), key=lambda x: x[1], reverse=True):
        report.append(f"  - {alg}: {count} runs")

    if 'problem_dimensions' in data_stats:
        dims = data_stats['problem_dimensions']
        report.append(f"Problem dimensions: {dims['mean']:.1f} (range: {dims['min']}-{dims['max']})")

    report.append(f"Missing data:")
    for field, count in data_stats['missing_data'].items():
        if count > 0:
            report.append(f"  - {field}: {count} ({count/data_stats['total_runs']*100:.1f}%)")
    report.append("")

    # Training results
    report.append("TRAINING RESULTS")
    report.append("-" * 40)
    report.append(f"Model type: {training_results['model_type']}")
    report.append(f"Training samples: {training_results['n_samples']}")
    report.append(f"Algorithms trained: {training_results['n_algorithms']}")
    report.append(f"Models saved to: {training_results['output_dir']}")

    if 'feature_importance' in training_results:
        report.append("\nFeature Importance (top features):")
        for alg, importance in training_results['feature_importance'].items():
            top_features = np.argsort(importance)[-3:][::-1]
            report.append(f"  {alg}:")
            for idx in top_features:
                if idx < len(importance):
                    report.append(f"    - Feature {idx}: {importance[idx]:.3f}")
    report.append("")

    # Validation metrics
    if validation_metrics:
        report.append("VALIDATION METRICS")
        report.append("-" * 40)
        report.append(f"Test samples: {validation_metrics['n_test_samples']}")
        report.append(f"Algorithm selection accuracy: {validation_metrics['accuracy']:.2%}")
        report.append(f"Mean absolute error: {validation_metrics['mean_absolute_error']:.4f}")
        report.append("")

    # Performance summary
    report.append("PERFORMANCE SUMMARY")
    report.append("-" * 40)
    for alg in data_stats.get('performance_range', {}).keys():
        perf = data_stats['performance_range'][alg]
        runtime = data_stats['runtime_range'].get(alg, {})
        report.append(f"{alg}:")
        report.append(f"  Performance: {perf['mean']:.3f} ± {perf['std']:.3f}")
        if runtime:
            report.append(f"  Runtime: {runtime['mean']:.1f}s ± {runtime['std']:.1f}s")

    report.append("")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Train Librex Adaptive Learning System on historical data'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to historical data (JSON, CSV, or SQLite database)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['gradient_boosting', 'random_forest', 'ridge'],
        default='gradient_boosting',
        help='Type of prediction model to use'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--validation-split',
        type=float,
        default=0.2,
        help='Fraction of data to use for validation'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save training report'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load historical data
        logger.info(f"Loading historical data from {args.data}...")
        runs = load_historical_data(args.data)

        if not runs:
            logger.error("No data loaded")
            sys.exit(1)

        # Analyze data quality
        logger.info("Analyzing data quality...")
        data_stats = analyze_data_quality(runs)

        # Split data for validation
        n_test = int(len(runs) * args.validation_split)
        np.random.shuffle(runs)
        test_data = runs[:n_test]
        train_data = runs[n_test:]

        logger.info(f"Training on {len(train_data)} samples, validating on {len(test_data)} samples")

        # Train models
        training_results = train_models(
            train_data,
            model_type=args.model_type,
            output_dir=args.output_dir
        )

        # Validate if test data available
        validation_metrics = {}
        if test_data:
            logger.info("Validating models...")
            # Load trained models for validation
            predictor = PerformancePredictor(model_type=args.model_type)
            predictor.load_models(args.output_dir)
            validation_metrics = validate_models(predictor, test_data)

        # Create report
        report = create_summary_report(data_stats, training_results, validation_metrics)

        # Print report
        print("\n" + report)

        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {args.report}")

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()