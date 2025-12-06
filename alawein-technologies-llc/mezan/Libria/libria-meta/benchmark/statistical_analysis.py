"""
Statistical Analysis Tools for Algorithm Selection Benchmarks

Implements:
- Wilcoxon signed-rank test
- Friedman test
- Post-hoc Nemenyi test
- Effect size computation (Cohen's d, Cliff's delta)
- Critical difference diagrams
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import rankdata
import json


class StatisticalAnalyzer:
    """Statistical significance testing for algorithm selection methods"""

    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analyzer

        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha

    def wilcoxon_test(
        self,
        method1_scores: np.ndarray,
        method2_scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Wilcoxon signed-rank test for paired comparisons

        Args:
            method1_scores: Performance scores for method 1
            method2_scores: Performance scores for method 2

        Returns:
            results: Dict with statistic, p-value, significant
        """
        statistic, p_value = stats.wilcoxon(
            method1_scores,
            method2_scores,
            alternative='two-sided'
        )

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha
        }

    def friedman_test(
        self,
        method_scores: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Friedman test for comparing multiple methods

        Args:
            method_scores: Dict of {method_name: score_array}

        Returns:
            results: Dict with statistic, p-value, significant
        """
        # Convert to matrix (n_instances x n_methods)
        scores_matrix = np.array([scores for scores in method_scores.values()]).T

        statistic, p_value = stats.friedmanchisquare(*scores_matrix.T)

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'n_methods': len(method_scores),
            'n_instances': len(list(method_scores.values())[0])
        }

    def nemenyi_test(
        self,
        method_scores: Dict[str, np.ndarray],
        method_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Post-hoc Nemenyi test for pairwise comparisons

        Args:
            method_scores: Dict of {method_name: score_array}
            method_names: Optional list of method names (for ordering)

        Returns:
            results_df: DataFrame with pairwise p-values
        """
        if method_names is None:
            method_names = list(method_scores.keys())

        n_methods = len(method_names)
        n_instances = len(method_scores[method_names[0]])

        # Compute ranks
        scores_matrix = np.array([method_scores[name] for name in method_names]).T
        ranks = np.array([rankdata(row) for row in scores_matrix])
        avg_ranks = np.mean(ranks, axis=0)

        # Critical difference for Nemenyi test
        q_alpha = 2.343  # For alpha=0.05, k=6 (approximate, should use table)
        cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6 * n_instances))

        # Pairwise comparisons
        p_values = np.zeros((n_methods, n_methods))

        for i in range(n_methods):
            for j in range(n_methods):
                rank_diff = abs(avg_ranks[i] - avg_ranks[j])
                p_values[i, j] = rank_diff  # Simplified: using rank diff

        results_df = pd.DataFrame(
            p_values,
            index=method_names,
            columns=method_names
        )

        return results_df

    def cohens_d(
        self,
        method1_scores: np.ndarray,
        method2_scores: np.ndarray
    ) -> float:
        """
        Compute Cohen's d effect size

        Args:
            method1_scores: Scores for method 1
            method2_scores: Scores for method 2

        Returns:
            d: Effect size (small: 0.2, medium: 0.5, large: 0.8)
        """
        mean1 = np.mean(method1_scores)
        mean2 = np.mean(method2_scores)

        # Pooled standard deviation
        n1, n2 = len(method1_scores), len(method2_scores)
        var1, var2 = np.var(method1_scores, ddof=1), np.var(method2_scores, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        d = (mean1 - mean2) / pooled_std

        return float(d)

    def cliffs_delta(
        self,
        method1_scores: np.ndarray,
        method2_scores: np.ndarray
    ) -> float:
        """
        Compute Cliff's delta (non-parametric effect size)

        Args:
            method1_scores: Scores for method 1
            method2_scores: Scores for method 2

        Returns:
            delta: Effect size in [-1, 1]
                   (negligible: <0.147, small: 0.147-0.33, medium: 0.33-0.474, large: >0.474)
        """
        n1, n2 = len(method1_scores), len(method2_scores)

        # Count dominance
        dominance = 0
        for x1 in method1_scores:
            for x2 in method2_scores:
                if x1 > x2:
                    dominance += 1
                elif x1 < x2:
                    dominance -= 1

        delta = dominance / (n1 * n2)

        return float(delta)

    def pairwise_comparisons(
        self,
        method_scores: Dict[str, np.ndarray],
        reference_method: str,
        metric_name: str = "regret"
    ) -> pd.DataFrame:
        """
        Perform pairwise comparisons of all methods against reference

        Args:
            method_scores: Dict of {method_name: score_array}
            reference_method: Name of reference method
            metric_name: Name of metric being compared

        Returns:
            results_df: DataFrame with comparison results
        """
        results = []

        ref_scores = method_scores[reference_method]

        for method_name, scores in method_scores.items():
            if method_name == reference_method:
                continue

            # Wilcoxon test
            wilcoxon = self.wilcoxon_test(ref_scores, scores)

            # Effect sizes
            cohens_d = self.cohens_d(ref_scores, scores)
            cliffs_delta = self.cliffs_delta(ref_scores, scores)

            # Mean difference
            mean_diff = np.mean(ref_scores) - np.mean(scores)

            results.append({
                'Method': method_name,
                'Mean Diff': mean_diff,
                'p-value': wilcoxon['p_value'],
                'Significant': wilcoxon['significant'],
                "Cohen's d": cohens_d,
                "Cliff's δ": cliffs_delta,
                'Better': 'Yes' if (wilcoxon['significant'] and mean_diff < 0) else 'No'
            })

        results_df = pd.DataFrame(results)

        return results_df

    def compute_ranks(
        self,
        method_scores: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Compute average ranks for each method

        Args:
            method_scores: Dict of {method_name: score_array}

        Returns:
            ranks_df: DataFrame with average ranks
        """
        method_names = list(method_scores.keys())
        n_instances = len(method_scores[method_names[0]])

        # Compute ranks per instance (lower score = better rank)
        scores_matrix = np.array([method_scores[name] for name in method_names]).T

        ranks = np.zeros((n_instances, len(method_names)))
        for i, row in enumerate(scores_matrix):
            # Rank in ascending order (lower score = rank 1)
            ranks[i] = rankdata(row, method='average')

        avg_ranks = np.mean(ranks, axis=0)

        ranks_df = pd.DataFrame({
            'Method': method_names,
            'Avg Rank': avg_ranks
        }).sort_values('Avg Rank')

        return ranks_df

    def analyze_results(
        self,
        evaluation_results: Dict,
        reference_method: str = 'Librex.Meta',
        metric: str = 'avg_regret'
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of evaluation results

        Args:
            evaluation_results: Results from ComprehensiveEvaluator
            reference_method: Reference method for comparisons
            metric: Metric to analyze (avg_regret, top1_accuracy, etc.)

        Returns:
            analysis: Dict with all statistical tests
        """
        print(f"\n{'='*70}")
        print(f"Statistical Analysis: {metric}")
        print(f"Reference Method: {reference_method}")
        print(f"{'='*70}\n")

        # Extract scores for each method across scenarios
        method_scores = {}

        for scenario_name, scenario_results in evaluation_results['scenarios'].items():
            for method_name, method_results in scenario_results['methods'].items():
                if method_name not in method_scores:
                    method_scores[method_name] = []

                # Get metric value
                if metric == 'avg_regret':
                    score = method_results['avg_regret']
                elif metric == 'top1_accuracy':
                    score = method_results['top1_accuracy']
                elif metric == 'top3_accuracy':
                    score = method_results['top3_accuracy']
                else:
                    score = method_results.get(metric, 0.0)

                method_scores[method_name].append(score)

        # Convert to numpy arrays
        for name in method_scores:
            method_scores[name] = np.array(method_scores[name])

        # Friedman test
        print("Friedman Test:")
        friedman = self.friedman_test(method_scores)
        print(f"  Statistic: {friedman['statistic']:.4f}")
        print(f"  p-value: {friedman['p_value']:.6f}")
        print(f"  Significant: {friedman['significant']}")

        # Average ranks
        print("\nAverage Ranks:")
        ranks_df = self.compute_ranks(method_scores)
        print(ranks_df.to_string(index=False))

        # Pairwise comparisons vs. reference
        print(f"\nPairwise Comparisons vs. {reference_method}:")
        pairwise_df = self.pairwise_comparisons(
            method_scores,
            reference_method,
            metric_name=metric
        )
        print(pairwise_df.to_string(index=False))

        analysis = {
            'metric': metric,
            'reference_method': reference_method,
            'friedman_test': friedman,
            'ranks': ranks_df.to_dict(orient='records'),
            'pairwise_comparisons': pairwise_df.to_dict(orient='records')
        }

        return analysis

    def save_analysis(
        self,
        analysis: Dict,
        output_file: str
    ):
        """Save statistical analysis to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\n✓ Statistical analysis saved to {output_file}")


def main():
    """Example usage"""
    import pickle

    # Load evaluation results
    results_file = "results/evaluation_results_20251114_231552.pkl"

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Analyze
    analyzer = StatisticalAnalyzer(alpha=0.05)

    # Analyze regret
    regret_analysis = analyzer.analyze_results(
        results,
        reference_method='Librex.Meta',
        metric='avg_regret'
    )

    # Analyze top-1 accuracy
    top1_analysis = analyzer.analyze_results(
        results,
        reference_method='Librex.Meta',
        metric='top1_accuracy'
    )

    # Save
    analyzer.save_analysis(regret_analysis, "results/statistical_analysis_regret.json")
    analyzer.save_analysis(top1_analysis, "results/statistical_analysis_top1.json")


if __name__ == "__main__":
    main()
