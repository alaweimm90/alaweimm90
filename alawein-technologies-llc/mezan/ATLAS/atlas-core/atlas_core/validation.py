"""Statistical Validation Framework for MEZAN.

This module provides comprehensive statistical analysis tools for validating
MEZAN's performance, including significance testing, confidence intervals,
effect size measurement, and power analysis.

Author: Meshal Alawein
Date: 2025-11-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, chi2_contingency, mannwhitneyu,
    wilcoxon, kruskal, friedmanchisquare, f_oneway
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower, ChiSquarePower
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
import warnings
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class TestType(Enum):
    """Types of statistical tests available."""
    TTEST_IND = "independent_t_test"
    TTEST_PAIRED = "paired_t_test"
    CHI_SQUARE = "chi_square"
    MANN_WHITNEY = "mann_whitney_u"
    WILCOXON = "wilcoxon_signed_rank"
    ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    FRIEDMAN = "friedman"
    MCNEMAR = "mcnemar"


class MultipleComparisonMethod(Enum):
    """Multiple comparison correction methods."""
    BONFERRONI = "bonferroni"
    SIDAK = "sidak"
    HOLM = "holm"
    HOLM_SIDAK = "holm-sidak"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    FDR_TSBH = "fdr_tsbh"  # Two-stage Benjamini-Hochberg
    FDR_TSBKY = "fdr_tsbky"  # Two-stage Benjamini-Krieger-Yekutieli


@dataclass
class ValidationResult:
    """Results from a statistical validation test."""
    test_type: TestType
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    degrees_of_freedom: Optional[float] = None
    additional_metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_type': self.test_type.value,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'confidence_level': self.confidence_level,
            'effect_size': self.effect_size,
            'confidence_interval': self.confidence_interval,
            'power': self.power,
            'sample_size': self.sample_size,
            'degrees_of_freedom': self.degrees_of_freedom,
            'additional_metrics': self.additional_metrics
        }


class StatisticalValidator:
    """Main statistical validation engine for MEZAN."""

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the validator.

        Args:
            confidence_level: Confidence level for tests (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.results_history: List[ValidationResult] = []

    def validate_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        Test for normality using Shapiro-Wilk test.

        Args:
            data: Data to test

        Returns:
            Tuple of (is_normal, p_value)
        """
        if len(data) < 3:
            warnings.warn("Sample size too small for normality test")
            return False, 0.0

        statistic, p_value = stats.shapiro(data)
        return p_value > self.alpha, p_value

    def t_test(self,
               group1: np.ndarray,
               group2: np.ndarray,
               paired: bool = False,
               equal_var: bool = True) -> ValidationResult:
        """
        Perform t-test between two groups.

        Args:
            group1: First group data
            group2: Second group data
            paired: Whether to use paired t-test
            equal_var: Whether to assume equal variance

        Returns:
            ValidationResult with test statistics
        """
        # Clean data
        group1 = np.asarray(group1)[~np.isnan(group1)]
        group2 = np.asarray(group2)[~np.isnan(group2)]

        if paired:
            if len(group1) != len(group2):
                raise ValueError("Paired t-test requires equal sample sizes")
            statistic, p_value = ttest_rel(group1, group2)
            test_type = TestType.TTEST_PAIRED
            df = len(group1) - 1
        else:
            statistic, p_value = ttest_ind(group1, group2, equal_var=equal_var)
            test_type = TestType.TTEST_IND
            if equal_var:
                df = len(group1) + len(group2) - 2
            else:
                # Welch's t-test degrees of freedom
                s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                df = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group1)-1)*np.var(group1, ddof=1) +
                              (len(group2)-1)*np.var(group2, ddof=1)) /
                             (len(group1) + len(group2) - 2))
        effect_size = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        # Calculate confidence interval for mean difference
        mean_diff = np.mean(group1) - np.mean(group2)
        se = pooled_std * np.sqrt(1/len(group1) + 1/len(group2))
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
        ci = (mean_diff - t_critical * se, mean_diff + t_critical * se)

        # Calculate power
        power_analysis = TTestPower()
        power = power_analysis.solve_power(
            effect_size=abs(effect_size),
            nobs1=len(group1),
            ratio=len(group2)/len(group1) if not paired else 1,
            alpha=self.alpha
        )

        result = ValidationResult(
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            confidence_interval=ci,
            power=power,
            sample_size=len(group1) + len(group2),
            degrees_of_freedom=df,
            additional_metrics={
                'mean_group1': np.mean(group1),
                'mean_group2': np.mean(group2),
                'std_group1': np.std(group1, ddof=1),
                'std_group2': np.std(group2, ddof=1),
                'mean_difference': mean_diff
            }
        )

        self.results_history.append(result)
        return result

    def chi_square_test(self, contingency_table: np.ndarray) -> ValidationResult:
        """
        Perform chi-square test of independence.

        Args:
            contingency_table: 2D contingency table

        Returns:
            ValidationResult with test statistics
        """
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)

        # Calculate Cramér's V as effect size
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        # Calculate power for chi-square test
        power_analysis = ChiSquarePower()
        power = power_analysis.solve_power(
            effect_size=cramers_v,
            nobs=n,
            n_bins=(contingency_table.shape[0]) * (contingency_table.shape[1]),
            alpha=self.alpha
        )

        result = ValidationResult(
            test_type=TestType.CHI_SQUARE,
            statistic=chi2,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=cramers_v,
            power=power,
            sample_size=n,
            degrees_of_freedom=dof,
            additional_metrics={
                'expected_frequencies': expected.tolist(),
                'observed_frequencies': contingency_table.tolist(),
                'cramers_v': cramers_v
            }
        )

        self.results_history.append(result)
        return result

    def mann_whitney_u_test(self, group1: np.ndarray, group2: np.ndarray) -> ValidationResult:
        """
        Perform Mann-Whitney U test (non-parametric).

        Args:
            group1: First group data
            group2: Second group data

        Returns:
            ValidationResult with test statistics
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(group1), len(group2)
        rank_biserial = 1 - (2*statistic) / (n1*n2)

        result = ValidationResult(
            test_type=TestType.MANN_WHITNEY,
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=rank_biserial,
            sample_size=n1 + n2,
            additional_metrics={
                'median_group1': np.median(group1),
                'median_group2': np.median(group2),
                'rank_biserial': rank_biserial
            }
        )

        self.results_history.append(result)
        return result

    def anova_test(self, *groups) -> ValidationResult:
        """
        Perform one-way ANOVA test.

        Args:
            *groups: Variable number of group arrays

        Returns:
            ValidationResult with test statistics
        """
        # Clean data
        clean_groups = [np.asarray(g)[~np.isnan(g)] for g in groups]

        f_stat, p_value = f_oneway(*clean_groups)

        # Calculate effect size (eta-squared)
        all_data = np.concatenate(clean_groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in clean_groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        # Degrees of freedom
        k = len(clean_groups)  # number of groups
        n = sum(len(g) for g in clean_groups)  # total sample size
        df_between = k - 1
        df_within = n - k

        result = ValidationResult(
            test_type=TestType.ANOVA,
            statistic=f_stat,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=eta_squared,
            sample_size=n,
            degrees_of_freedom=df_between,
            additional_metrics={
                'df_between': df_between,
                'df_within': df_within,
                'eta_squared': eta_squared,
                'group_means': [np.mean(g) for g in clean_groups],
                'group_sizes': [len(g) for g in clean_groups]
            }
        )

        self.results_history.append(result)
        return result

    def multiple_comparison_correction(self,
                                      p_values: List[float],
                                      method: MultipleComparisonMethod = MultipleComparisonMethod.FDR_BH
                                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple comparison correction.

        Args:
            p_values: List of p-values to correct
            method: Correction method to use

        Returns:
            Tuple of (rejected, adjusted_p_values)
        """
        rejected, p_adjusted, _, _ = multipletests(
            p_values,
            alpha=self.alpha,
            method=method.value
        )
        return rejected, p_adjusted

    def calculate_confidence_interval(self,
                                     data: np.ndarray,
                                     confidence: Optional[float] = None) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean.

        Args:
            data: Data array
            confidence: Confidence level (uses default if None)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        confidence = confidence or self.confidence_level
        mean = np.mean(data)
        se = stats.sem(data)
        interval = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return mean - interval, mean + interval

    def calculate_sample_size(self,
                            effect_size: float,
                            power: float = 0.8,
                            test_type: str = 't-test') -> int:
        """
        Calculate required sample size for desired power.

        Args:
            effect_size: Expected effect size
            power: Desired statistical power
            test_type: Type of test ('t-test' or 'chi-square')

        Returns:
            Required sample size per group
        """
        if test_type == 't-test':
            analysis = TTestPower()
            n = analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=self.alpha
            )
        elif test_type == 'chi-square':
            analysis = ChiSquarePower()
            n = analysis.solve_power(
                effect_size=effect_size,
                power=power,
                alpha=self.alpha,
                n_bins=2  # Assuming 2x2 table
            )
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        return int(np.ceil(n))

    def proportion_test(self,
                       successes1: int,
                       trials1: int,
                       successes2: int,
                       trials2: int) -> ValidationResult:
        """
        Test difference between two proportions.

        Args:
            successes1: Number of successes in group 1
            trials1: Number of trials in group 1
            successes2: Number of successes in group 2
            trials2: Number of trials in group 2

        Returns:
            ValidationResult with test statistics
        """
        # Proportions
        p1 = successes1 / trials1
        p2 = successes2 / trials2

        # Pooled proportion for test statistic
        p_pooled = (successes1 + successes2) / (trials1 + trials2)

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/trials1 + 1/trials2))

        # Z-statistic
        z_stat = (p1 - p2) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Confidence intervals for each proportion
        ci1 = proportion_confint(successes1, trials1, alpha=self.alpha, method='wilson')
        ci2 = proportion_confint(successes2, trials2, alpha=self.alpha, method='wilson')

        # Effect size (Cohen's h)
        cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

        result = ValidationResult(
            test_type=TestType.CHI_SQUARE,  # Proportions test is related to chi-square
            statistic=z_stat,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            confidence_level=self.confidence_level,
            effect_size=cohens_h,
            sample_size=trials1 + trials2,
            additional_metrics={
                'proportion1': p1,
                'proportion2': p2,
                'ci_proportion1': ci1,
                'ci_proportion2': ci2,
                'difference': p1 - p2,
                'cohens_h': cohens_h
            }
        )

        self.results_history.append(result)
        return result

    def generate_validation_report(self,
                                  output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.

        Args:
            output_path: Path to save report (optional)

        Returns:
            Dictionary containing validation report
        """
        report = {
            'summary': {
                'total_tests': len(self.results_history),
                'significant_results': sum(1 for r in self.results_history if r.is_significant),
                'confidence_level': self.confidence_level,
                'alpha': self.alpha
            },
            'tests': [r.to_dict() for r in self.results_history],
            'statistics': self._calculate_summary_statistics()
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics from results history."""
        if not self.results_history:
            return {}

        p_values = [r.p_value for r in self.results_history]
        effect_sizes = [r.effect_size for r in self.results_history if r.effect_size]
        powers = [r.power for r in self.results_history if r.power]

        return {
            'mean_p_value': np.mean(p_values),
            'median_p_value': np.median(p_values),
            'mean_effect_size': np.mean(effect_sizes) if effect_sizes else None,
            'median_effect_size': np.median(effect_sizes) if effect_sizes else None,
            'mean_power': np.mean(powers) if powers else None,
            'test_type_distribution': self._get_test_type_distribution()
        }

    def _get_test_type_distribution(self) -> Dict[str, int]:
        """Get distribution of test types used."""
        distribution = {}
        for result in self.results_history:
            test_name = result.test_type.value
            distribution[test_name] = distribution.get(test_name, 0) + 1
        return distribution

    def visualize_results(self, save_path: Optional[Path] = None):
        """
        Create visualizations of validation results.

        Args:
            save_path: Path to save visualization
        """
        if not self.results_history:
            warnings.warn("No results to visualize")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # P-value distribution
        p_values = [r.p_value for r in self.results_history]
        axes[0, 0].hist(p_values, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(x=self.alpha, color='r', linestyle='--', label=f'α={self.alpha}')
        axes[0, 0].set_xlabel('P-value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('P-value Distribution')
        axes[0, 0].legend()

        # Effect size distribution
        effect_sizes = [r.effect_size for r in self.results_history if r.effect_size]
        if effect_sizes:
            axes[0, 1].hist(effect_sizes, bins=20, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(x=0.2, color='g', linestyle='--', label='Small')
            axes[0, 1].axvline(x=0.5, color='y', linestyle='--', label='Medium')
            axes[0, 1].axvline(x=0.8, color='r', linestyle='--', label='Large')
            axes[0, 1].set_xlabel('Effect Size')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Effect Size Distribution')
            axes[0, 1].legend()

        # Test type distribution
        test_dist = self._get_test_type_distribution()
        axes[1, 0].bar(test_dist.keys(), test_dist.values())
        axes[1, 0].set_xlabel('Test Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Test Type Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Statistical power distribution
        powers = [r.power for r in self.results_history if r.power]
        if powers:
            axes[1, 1].hist(powers, bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].axvline(x=0.8, color='g', linestyle='--', label='Target (0.8)')
            axes[1, 1].set_xlabel('Statistical Power')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Statistical Power Distribution')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


# Convenience functions for common use cases
def quick_t_test(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """Quick t-test between two groups."""
    validator = StatisticalValidator()
    result = validator.t_test(np.array(group1), np.array(group2))
    return result.to_dict()


def quick_proportion_test(success1: int, total1: int,
                         success2: int, total2: int) -> Dict[str, Any]:
    """Quick proportion test between two groups."""
    validator = StatisticalValidator()
    result = validator.proportion_test(success1, total1, success2, total2)
    return result.to_dict()


def validate_improvement(baseline: List[float],
                        improved: List[float],
                        min_effect_size: float = 0.2) -> bool:
    """
    Check if improved version shows statistically significant improvement.

    Args:
        baseline: Baseline performance metrics
        improved: Improved version metrics
        min_effect_size: Minimum effect size to consider meaningful

    Returns:
        True if improvement is significant and meaningful
    """
    validator = StatisticalValidator()
    result = validator.t_test(np.array(improved), np.array(baseline))

    return (result.is_significant and
            result.effect_size is not None and
            result.effect_size > min_effect_size)