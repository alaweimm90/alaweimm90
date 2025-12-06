"""
Analytics Engine for TalAI

Core analytics functionality including metrics computation, trend analysis,
and performance tracking for hypothesis validation workflows.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import scipy.stats as stats

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Comprehensive metrics for validation analysis."""
    hypothesis_id: str
    timestamp: datetime
    success_rate: float
    avg_evidence_quality: float
    validation_duration_seconds: float
    confidence_score: float
    domain: str
    validation_mode: str
    evidence_count: int
    assumption_count: int
    peer_review_score: Optional[float] = None
    human_oversight_triggered: bool = False
    error_count: int = 0
    iteration_count: int = 1

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

@dataclass
class ResearchInsight:
    """Insights about research patterns and trends."""
    insight_type: str  # 'trend', 'anomaly', 'correlation', 'prediction'
    description: str
    confidence: float
    affected_hypotheses: List[str]
    recommendations: List[str]
    metadata: Dict[str, Any]
    generated_at: datetime

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'insight_type': self.insight_type,
            'description': self.description,
            'confidence': self.confidence,
            'affected_hypotheses': self.affected_hypotheses,
            'recommendations': self.recommendations,
            'metadata': self.metadata,
            'generated_at': self.generated_at.isoformat()
        }

class AnalyticsEngine:
    """
    Main analytics engine for TalAI validation system.

    Provides comprehensive analytics including:
    - Success rate tracking and prediction
    - Quality scoring and assessment
    - Trend analysis and anomaly detection
    - User behavior analytics
    - Research domain insights
    - A/B testing framework
    """

    def __init__(self, data_path: Optional[str] = None):
        """Initialize analytics engine."""
        self.data_path = data_path or "/tmp/talai_analytics"
        self.metrics_cache: Dict[str, ValidationMetrics] = {}
        self.insights_cache: List[ResearchInsight] = []
        self.ab_tests: Dict[str, Dict] = {}
        self.user_sessions: Dict[str, List[Dict]] = defaultdict(list)
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for analytics."""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.anomaly_detector = DBSCAN(eps=0.5, min_samples=5)

    async def track_validation(self, validation_data: Dict) -> ValidationMetrics:
        """
        Track a validation event and compute metrics.

        Args:
            validation_data: Validation event data

        Returns:
            Computed metrics
        """
        try:
            # Extract key metrics
            hypothesis_id = validation_data.get('hypothesis_id',
                                              self._generate_id(str(validation_data)))

            # Calculate success rate from evidence
            evidence = validation_data.get('evidence', [])
            success_count = sum(1 for e in evidence if e.get('supports_hypothesis', False))
            success_rate = success_count / len(evidence) if evidence else 0.0

            # Calculate average evidence quality
            quality_scores = [e.get('quality_score', 0.5) for e in evidence]
            avg_quality = np.mean(quality_scores) if quality_scores else 0.5

            # Extract timing information
            start_time = datetime.fromisoformat(validation_data.get('start_time',
                                                                   datetime.now().isoformat()))
            end_time = datetime.fromisoformat(validation_data.get('end_time',
                                                                 datetime.now().isoformat()))
            duration = (end_time - start_time).total_seconds()

            # Create metrics object
            metrics = ValidationMetrics(
                hypothesis_id=hypothesis_id,
                timestamp=start_time,
                success_rate=success_rate,
                avg_evidence_quality=avg_quality,
                validation_duration_seconds=duration,
                confidence_score=validation_data.get('confidence', 0.5),
                domain=validation_data.get('domain', 'unknown'),
                validation_mode=validation_data.get('mode', 'standard'),
                evidence_count=len(evidence),
                assumption_count=len(validation_data.get('assumptions', [])),
                peer_review_score=validation_data.get('peer_review_score'),
                human_oversight_triggered=validation_data.get('human_oversight', False),
                error_count=validation_data.get('error_count', 0),
                iteration_count=validation_data.get('iteration', 1)
            )

            # Cache metrics
            self.metrics_cache[hypothesis_id] = metrics

            # Check for insights
            await self._check_for_insights(metrics)

            logger.info(f"Tracked validation for hypothesis {hypothesis_id}")
            return metrics

        except Exception as e:
            logger.error(f"Error tracking validation: {e}")
            raise

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content."""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def _check_for_insights(self, metrics: ValidationMetrics):
        """Check if new metrics reveal any insights."""
        # Check for anomalies
        if metrics.success_rate < 0.2 and metrics.confidence_score > 0.8:
            insight = ResearchInsight(
                insight_type='anomaly',
                description='High confidence but low success rate detected',
                confidence=0.9,
                affected_hypotheses=[metrics.hypothesis_id],
                recommendations=[
                    'Review evidence collection methodology',
                    'Verify hypothesis assumptions',
                    'Consider alternative validation approaches'
                ],
                metadata={'success_rate': metrics.success_rate,
                         'confidence': metrics.confidence_score},
                generated_at=datetime.now()
            )
            self.insights_cache.append(insight)

        # Check for efficiency issues
        if metrics.validation_duration_seconds > 300:  # More than 5 minutes
            insight = ResearchInsight(
                insight_type='performance',
                description='Validation taking longer than expected',
                confidence=0.85,
                affected_hypotheses=[metrics.hypothesis_id],
                recommendations=[
                    'Consider parallel evidence collection',
                    'Optimize validation queries',
                    'Use cached results where applicable'
                ],
                metadata={'duration_seconds': metrics.validation_duration_seconds},
                generated_at=datetime.now()
            )
            self.insights_cache.append(insight)

    async def compute_success_trends(self,
                                    domain: Optional[str] = None,
                                    time_window_days: int = 30) -> Dict[str, Any]:
        """
        Compute success rate trends over time.

        Args:
            domain: Optional domain filter
            time_window_days: Time window for analysis

        Returns:
            Trend analysis results
        """
        try:
            # Filter metrics
            cutoff_time = datetime.now() - timedelta(days=time_window_days)
            relevant_metrics = [
                m for m in self.metrics_cache.values()
                if m.timestamp >= cutoff_time and
                (domain is None or m.domain == domain)
            ]

            if not relevant_metrics:
                return {'error': 'No data available for analysis'}

            # Create time series
            df = pd.DataFrame([m.to_dict() for m in relevant_metrics])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()

            # Compute rolling statistics
            window = '7D'  # 7-day rolling window
            rolling_mean = df['success_rate'].rolling(window).mean()
            rolling_std = df['success_rate'].rolling(window).std()

            # Detect trend using linear regression
            x = np.arange(len(df))
            y = df['success_rate'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Determine trend direction
            if p_value < 0.05:  # Statistically significant
                trend_direction = 'increasing' if slope > 0 else 'decreasing'
                trend_strength = abs(r_value)
            else:
                trend_direction = 'stable'
                trend_strength = 0.0

            # Identify best and worst periods
            best_period = df['success_rate'].idxmax()
            worst_period = df['success_rate'].idxmin()

            # Forecast future success rate
            forecast_days = 7
            future_x = np.arange(len(df), len(df) + forecast_days)
            forecast = slope * future_x + intercept
            forecast = np.clip(forecast, 0, 1)  # Ensure valid probability

            return {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'current_success_rate': float(df['success_rate'].iloc[-1]),
                'avg_success_rate': float(df['success_rate'].mean()),
                'success_rate_std': float(df['success_rate'].std()),
                'best_period': best_period.isoformat(),
                'best_rate': float(df['success_rate'].max()),
                'worst_period': worst_period.isoformat(),
                'worst_rate': float(df['success_rate'].min()),
                'forecast_7_days': forecast.tolist(),
                'total_validations': len(df),
                'time_window_days': time_window_days,
                'domain': domain,
                'confidence': 1 - p_value if p_value < 0.05 else 0.0
            }

        except Exception as e:
            logger.error(f"Error computing success trends: {e}")
            return {'error': str(e)}

    async def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """
        Analyze user behavior patterns.

        Args:
            user_id: User identifier

        Returns:
            User behavior analysis
        """
        try:
            user_data = self.user_sessions.get(user_id, [])

            if not user_data:
                return {'error': 'No user data available'}

            # Compute user statistics
            total_sessions = len(user_data)
            total_validations = sum(s.get('validation_count', 0) for s in user_data)
            avg_session_duration = np.mean([s.get('duration_minutes', 0) for s in user_data])

            # Domain preferences
            domains = [s.get('primary_domain') for s in user_data if s.get('primary_domain')]
            domain_counts = Counter(domains)
            preferred_domain = domain_counts.most_common(1)[0] if domain_counts else (None, 0)

            # Activity patterns
            timestamps = [datetime.fromisoformat(s['timestamp']) for s in user_data
                         if 'timestamp' in s]
            if timestamps:
                hour_counts = Counter(ts.hour for ts in timestamps)
                peak_hour = hour_counts.most_common(1)[0][0]

                weekday_counts = Counter(ts.weekday() for ts in timestamps)
                peak_weekday = weekday_counts.most_common(1)[0][0]
                weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                peak_day_name = weekday_names[peak_weekday]
            else:
                peak_hour = None
                peak_day_name = None

            # Success metrics
            success_rates = [s.get('avg_success_rate', 0) for s in user_data
                           if 'avg_success_rate' in s]
            avg_success = np.mean(success_rates) if success_rates else 0.0

            # User engagement score (0-100)
            engagement_factors = [
                min(total_sessions / 10, 1.0) * 25,  # Session frequency
                min(total_validations / 50, 1.0) * 25,  # Validation volume
                min(avg_session_duration / 30, 1.0) * 25,  # Session duration
                avg_success * 25  # Success rate
            ]
            engagement_score = sum(engagement_factors)

            return {
                'user_id': user_id,
                'total_sessions': total_sessions,
                'total_validations': total_validations,
                'avg_session_duration_minutes': float(avg_session_duration),
                'preferred_domain': preferred_domain[0],
                'domain_diversity': len(domain_counts),
                'peak_activity_hour': peak_hour,
                'peak_activity_day': peak_day_name,
                'avg_success_rate': float(avg_success),
                'engagement_score': float(engagement_score),
                'user_type': self._classify_user_type(engagement_score, avg_success),
                'recommendations': self._get_user_recommendations(engagement_score, avg_success)
            }

        except Exception as e:
            logger.error(f"Error analyzing user behavior: {e}")
            return {'error': str(e)}

    def _classify_user_type(self, engagement: float, success: float) -> str:
        """Classify user based on engagement and success."""
        if engagement > 75 and success > 0.7:
            return 'power_user'
        elif engagement > 50:
            return 'regular_user'
        elif engagement > 25:
            return 'casual_user'
        else:
            return 'new_user'

    def _get_user_recommendations(self, engagement: float, success: float) -> List[str]:
        """Get personalized recommendations for user."""
        recommendations = []

        if engagement < 50:
            recommendations.append('Explore more validation modes for better insights')
        if success < 0.5:
            recommendations.append('Review evidence quality guidelines')
            recommendations.append('Consider peer review for complex hypotheses')
        if engagement > 75 and success > 0.7:
            recommendations.append('You might benefit from advanced features')
            recommendations.append('Consider mentoring other researchers')

        return recommendations

    async def run_ab_test(self,
                         test_name: str,
                         variant_a_data: List[Dict],
                         variant_b_data: List[Dict]) -> Dict[str, Any]:
        """
        Run A/B test analysis on two variants.

        Args:
            test_name: Name of the test
            variant_a_data: Data for variant A
            variant_b_data: Data for variant B

        Returns:
            A/B test results with statistical significance
        """
        try:
            # Extract success metrics
            a_successes = [d.get('success', False) for d in variant_a_data]
            b_successes = [d.get('success', False) for d in variant_b_data]

            a_rate = np.mean(a_successes) if a_successes else 0.0
            b_rate = np.mean(b_successes) if b_successes else 0.0

            # Perform statistical test (Chi-square test)
            a_success_count = sum(a_successes)
            a_failure_count = len(a_successes) - a_success_count
            b_success_count = sum(b_successes)
            b_failure_count = len(b_successes) - b_success_count

            contingency_table = [
                [a_success_count, a_failure_count],
                [b_success_count, b_failure_count]
            ]

            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

            # Calculate effect size (Cohen's h)
            effect_size = 2 * (np.arcsin(np.sqrt(b_rate)) - np.arcsin(np.sqrt(a_rate)))

            # Determine winner
            if p_value < 0.05:
                winner = 'variant_b' if b_rate > a_rate else 'variant_a'
                confidence = (1 - p_value) * 100
            else:
                winner = 'no_significant_difference'
                confidence = 0.0

            # Calculate lift
            lift = ((b_rate - a_rate) / a_rate * 100) if a_rate > 0 else 0.0

            # Store test results
            test_results = {
                'test_name': test_name,
                'variant_a': {
                    'sample_size': len(a_successes),
                    'success_rate': float(a_rate),
                    'success_count': a_success_count
                },
                'variant_b': {
                    'sample_size': len(b_successes),
                    'success_rate': float(b_rate),
                    'success_count': b_success_count
                },
                'winner': winner,
                'p_value': float(p_value),
                'statistical_significance': p_value < 0.05,
                'confidence_percent': float(confidence),
                'effect_size': float(effect_size),
                'lift_percent': float(lift),
                'minimum_sample_size': self._calculate_sample_size(a_rate, b_rate),
                'timestamp': datetime.now().isoformat()
            }

            self.ab_tests[test_name] = test_results

            return test_results

        except Exception as e:
            logger.error(f"Error running A/B test: {e}")
            return {'error': str(e)}

    def _calculate_sample_size(self,
                              baseline_rate: float,
                              expected_rate: float,
                              power: float = 0.8,
                              alpha: float = 0.05) -> int:
        """Calculate minimum sample size for statistical significance."""
        if baseline_rate == 0 or baseline_rate == 1:
            return 1000  # Default for edge cases

        # Use power analysis formula
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)

        p1 = baseline_rate
        p2 = expected_rate
        p_avg = (p1 + p2) / 2

        n = ((z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
              z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / ((p2 - p1) ** 2)

        return max(int(np.ceil(n)), 30)  # Minimum 30 per group

    async def get_domain_insights(self, domain: str) -> Dict[str, Any]:
        """
        Get comprehensive insights for a research domain.

        Args:
            domain: Research domain

        Returns:
            Domain-specific insights
        """
        try:
            # Filter metrics by domain
            domain_metrics = [m for m in self.metrics_cache.values()
                            if m.domain == domain]

            if not domain_metrics:
                return {'error': f'No data for domain {domain}'}

            # Compute domain statistics
            success_rates = [m.success_rate for m in domain_metrics]
            quality_scores = [m.avg_evidence_quality for m in domain_metrics]
            durations = [m.validation_duration_seconds for m in domain_metrics]

            # Identify top challenges
            failed_hypotheses = [m for m in domain_metrics if m.success_rate < 0.5]
            challenge_patterns = Counter(m.validation_mode for m in failed_hypotheses)

            # Time-based patterns
            hour_distribution = Counter(m.timestamp.hour for m in domain_metrics)
            peak_hours = hour_distribution.most_common(3)

            # Correlation analysis
            if len(domain_metrics) > 10:
                correlation_matrix = np.corrcoef([
                    success_rates,
                    quality_scores,
                    [m.evidence_count for m in domain_metrics],
                    [m.confidence_score for m in domain_metrics]
                ])
                quality_success_correlation = correlation_matrix[0, 1]
            else:
                quality_success_correlation = 0.0

            return {
                'domain': domain,
                'total_validations': len(domain_metrics),
                'avg_success_rate': float(np.mean(success_rates)),
                'success_rate_std': float(np.std(success_rates)),
                'avg_quality_score': float(np.mean(quality_scores)),
                'avg_duration_seconds': float(np.mean(durations)),
                'fastest_validation': float(min(durations)),
                'slowest_validation': float(max(durations)),
                'challenge_areas': dict(challenge_patterns.most_common(5)),
                'peak_activity_hours': [{'hour': h, 'count': c} for h, c in peak_hours],
                'quality_success_correlation': float(quality_success_correlation),
                'improvement_potential': self._calculate_improvement_potential(
                    success_rates, quality_scores
                ),
                'recommendations': self._get_domain_recommendations(
                    domain, success_rates, quality_scores
                )
            }

        except Exception as e:
            logger.error(f"Error getting domain insights: {e}")
            return {'error': str(e)}

    def _calculate_improvement_potential(self,
                                       success_rates: List[float],
                                       quality_scores: List[float]) -> float:
        """Calculate potential for improvement in domain."""
        # Higher potential if low success but high quality (room to improve process)
        avg_success = np.mean(success_rates)
        avg_quality = np.mean(quality_scores)

        if avg_quality > 0.7 and avg_success < 0.5:
            return 0.8  # High potential
        elif avg_quality > 0.5 and avg_success < 0.7:
            return 0.6  # Medium potential
        else:
            return 0.3  # Low potential (either already good or quality issues)

    def _get_domain_recommendations(self,
                                   domain: str,
                                   success_rates: List[float],
                                   quality_scores: List[float]) -> List[str]:
        """Get domain-specific recommendations."""
        recommendations = []

        avg_success = np.mean(success_rates)
        avg_quality = np.mean(quality_scores)
        std_success = np.std(success_rates)

        if avg_success < 0.5:
            recommendations.append(f'Success rate below 50% in {domain} - review validation criteria')

        if std_success > 0.3:
            recommendations.append('High variance in results - standardize validation process')

        if avg_quality < 0.6:
            recommendations.append('Focus on improving evidence quality')

        if domain in ['biology', 'medicine', 'chemistry']:
            recommendations.append('Consider experimental validation for high-impact hypotheses')

        return recommendations

    async def export_analytics_report(self) -> Dict[str, Any]:
        """
        Export comprehensive analytics report.

        Returns:
            Complete analytics report
        """
        try:
            # Gather all analytics
            all_domains = list(set(m.domain for m in self.metrics_cache.values()))

            domain_reports = {}
            for domain in all_domains:
                domain_reports[domain] = await self.get_domain_insights(domain)

            # Overall statistics
            all_success_rates = [m.success_rate for m in self.metrics_cache.values()]
            all_durations = [m.validation_duration_seconds for m in self.metrics_cache.values()]

            report = {
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_validations': len(self.metrics_cache),
                    'total_insights': len(self.insights_cache),
                    'avg_success_rate': float(np.mean(all_success_rates)) if all_success_rates else 0.0,
                    'avg_duration_seconds': float(np.mean(all_durations)) if all_durations else 0.0,
                    'domains_analyzed': len(all_domains),
                    'active_ab_tests': len(self.ab_tests)
                },
                'domain_reports': domain_reports,
                'recent_insights': [i.to_dict() for i in self.insights_cache[-10:]],
                'ab_test_results': self.ab_tests,
                'performance_metrics': {
                    'cache_size': len(self.metrics_cache),
                    'memory_usage_mb': self._estimate_memory_usage() / 1024 / 1024
                }
            }

            return report

        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            return {'error': str(e)}

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of analytics engine."""
        import sys

        total_size = 0
        total_size += sys.getsizeof(self.metrics_cache)
        total_size += sys.getsizeof(self.insights_cache)
        total_size += sys.getsizeof(self.ab_tests)
        total_size += sys.getsizeof(self.user_sessions)

        return float(total_size)