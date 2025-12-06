"""
Historical Backtesting Engine for TalAI Validation

Core engine that orchestrates time-travel validation of scientific discoveries,
testing if TalAI could have predicted major breakthroughs given historical context.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib
import numpy as np
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Validation result categories"""
    PREDICTED_CORRECTLY = "predicted_correctly"
    PREDICTED_EARLY = "predicted_early"
    PREDICTED_LATE = "predicted_late"
    MISSED = "missed"
    FALSE_POSITIVE = "false_positive"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class Discovery:
    """Represents a historical scientific discovery"""
    id: str
    name: str
    domain: str
    subdomain: str
    discovery_date: datetime
    announcement_date: datetime
    key_researchers: List[str]
    institutions: List[str]
    prerequisites: List[str]  # Prior discoveries needed
    impact_score: float  # 0-10 scale
    citation_count: int
    description: str
    key_papers: List[str]
    enabling_technologies: List[str]
    breakthrough_type: str  # theoretical, experimental, computational
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestContext:
    """Context provided to TalAI for prediction"""
    cutoff_date: datetime
    available_knowledge: List[Discovery]
    available_papers: List[Dict[str, Any]]
    available_technologies: List[str]
    research_trends: Dict[str, float]
    funding_priorities: List[str]
    compute_capabilities: Dict[str, Any]
    domain_focus: Optional[str] = None


@dataclass
class PredictionResult:
    """Result of TalAI's prediction attempt"""
    discovery_id: str
    predicted: bool
    confidence_score: float
    predicted_date: Optional[datetime]
    actual_date: datetime
    time_delta_days: Optional[int]
    reasoning: str
    key_insights: List[str]
    missing_prerequisites: List[str]
    false_dependencies: List[str]
    validation_result: ValidationResult
    compute_time_ms: float
    model_version: str


class HistoricalBacktestingEngine:
    """
    Main engine for historical backtesting of TalAI predictions.
    Tests ability to predict past discoveries given historical context.
    """

    def __init__(self,
                 database_path: Path,
                 talai_endpoint: str,
                 cache_dir: Optional[Path] = None):
        """
        Initialize the backtesting engine.

        Args:
            database_path: Path to discovery database
            talai_endpoint: TalAI API endpoint
            cache_dir: Optional cache directory for results
        """
        self.database_path = database_path
        self.talai_endpoint = talai_endpoint
        self.cache_dir = cache_dir or Path("/tmp/talai_backtest_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.discoveries: Dict[str, Discovery] = {}
        self.results: List[PredictionResult] = []
        self.metrics: Dict[str, Any] = defaultdict(list)

        # Performance tracking
        self.total_predictions = 0
        self.successful_predictions = 0
        self.compute_time_total = 0

        # Domain-specific calibration
        self.domain_weights = {
            "physics": 1.2,
            "biology": 1.0,
            "chemistry": 1.1,
            "computer_science": 0.9,
            "mathematics": 1.3,
            "medicine": 1.0,
            "materials_science": 1.1,
            "astronomy": 1.2
        }

        logger.info(f"Initialized HistoricalBacktestingEngine with database: {database_path}")

    async def load_discoveries(self) -> None:
        """Load discovery database from disk"""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)

            for disc_data in data['discoveries']:
                discovery = Discovery(
                    id=disc_data['id'],
                    name=disc_data['name'],
                    domain=disc_data['domain'],
                    subdomain=disc_data['subdomain'],
                    discovery_date=datetime.fromisoformat(disc_data['discovery_date']),
                    announcement_date=datetime.fromisoformat(disc_data['announcement_date']),
                    key_researchers=disc_data['key_researchers'],
                    institutions=disc_data['institutions'],
                    prerequisites=disc_data['prerequisites'],
                    impact_score=disc_data['impact_score'],
                    citation_count=disc_data['citation_count'],
                    description=disc_data['description'],
                    key_papers=disc_data['key_papers'],
                    enabling_technologies=disc_data['enabling_technologies'],
                    breakthrough_type=disc_data['breakthrough_type'],
                    validation_metadata=disc_data.get('validation_metadata', {})
                )
                self.discoveries[discovery.id] = discovery

            logger.info(f"Loaded {len(self.discoveries)} discoveries from database")

        except FileNotFoundError:
            logger.warning(f"Database not found at {self.database_path}, initializing empty")
            self.discoveries = {}
        except Exception as e:
            logger.error(f"Error loading discoveries: {e}")
            raise

    def _get_cache_key(self, discovery_id: str, context: BacktestContext) -> str:
        """Generate cache key for prediction result"""
        context_str = f"{context.cutoff_date.isoformat()}_{context.domain_focus}"
        return hashlib.sha256(f"{discovery_id}_{context_str}".encode()).hexdigest()

    async def _check_cache(self, discovery_id: str, context: BacktestContext) -> Optional[PredictionResult]:
        """Check if we have a cached prediction result"""
        cache_key = self._get_cache_key(discovery_id, context)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return PredictionResult(**data)
            except Exception as e:
                logger.warning(f"Failed to load cache for {discovery_id}: {e}")

        return None

    async def _save_to_cache(self, result: PredictionResult, context: BacktestContext) -> None:
        """Save prediction result to cache"""
        cache_key = self._get_cache_key(result.discovery_id, context)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump(result.__dict__, f, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache for {result.discovery_id}: {e}")

    def _build_context(self, target_discovery: Discovery, cutoff_years_before: int = 2) -> BacktestContext:
        """
        Build historical context for prediction.

        Args:
            target_discovery: The discovery we want to predict
            cutoff_years_before: Years before discovery to set cutoff

        Returns:
            BacktestContext with available historical information
        """
        cutoff_date = target_discovery.discovery_date - timedelta(days=365 * cutoff_years_before)

        # Get all discoveries before cutoff
        available_knowledge = [
            disc for disc in self.discoveries.values()
            if disc.discovery_date < cutoff_date
        ]

        # Extract available technologies at cutoff
        available_tech = set()
        for disc in available_knowledge:
            available_tech.update(disc.enabling_technologies)

        # Calculate research trends based on recent discoveries
        recent_discoveries = [
            disc for disc in available_knowledge
            if disc.discovery_date > (cutoff_date - timedelta(days=365 * 5))
        ]

        domain_counts = defaultdict(int)
        for disc in recent_discoveries:
            domain_counts[disc.domain] += 1

        total = len(recent_discoveries) if recent_discoveries else 1
        research_trends = {
            domain: count / total
            for domain, count in domain_counts.items()
        }

        # Estimate compute capabilities based on year
        year = cutoff_date.year
        compute_capabilities = {
            "flops": 10 ** (6 + 0.5 * (year - 1960)),  # Moore's law approximation
            "memory_gb": 2 ** max(0, (year - 1980) / 3),
            "storage_tb": 0.001 * 2 ** max(0, (year - 1990) / 2),
            "gpu_available": year >= 1999,
            "quantum_qubits": max(0, (year - 2000) * 2) if year >= 2000 else 0
        }

        # Extract funding priorities from high-impact recent discoveries
        high_impact = sorted(
            recent_discoveries,
            key=lambda x: x.impact_score,
            reverse=True
        )[:10]
        funding_priorities = list(set(disc.subdomain for disc in high_impact))

        # Build paper metadata (simplified)
        available_papers = []
        for disc in available_knowledge:
            for paper_id in disc.key_papers:
                available_papers.append({
                    "id": paper_id,
                    "discovery_id": disc.id,
                    "domain": disc.domain,
                    "date": disc.announcement_date.isoformat(),
                    "citations": disc.citation_count
                })

        return BacktestContext(
            cutoff_date=cutoff_date,
            available_knowledge=available_knowledge,
            available_papers=available_papers,
            available_technologies=list(available_tech),
            research_trends=research_trends,
            funding_priorities=funding_priorities,
            compute_capabilities=compute_capabilities,
            domain_focus=target_discovery.domain
        )

    async def _call_talai_prediction(self,
                                    discovery: Discovery,
                                    context: BacktestContext) -> Dict[str, Any]:
        """
        Call TalAI to attempt prediction of discovery.
        This is a simulation - in production would call actual TalAI API.
        """
        import random
        import time

        start_time = time.time()

        # Simulate API call with realistic behavior
        await asyncio.sleep(0.1)  # Simulate network latency

        # Calculate prediction probability based on various factors
        base_prob = 0.3

        # Adjust based on prerequisites being met
        prereq_met = sum(
            1 for prereq in discovery.prerequisites
            if any(k.id == prereq for k in context.available_knowledge)
        ) / max(len(discovery.prerequisites), 1)
        base_prob += prereq_met * 0.3

        # Adjust based on domain trends
        domain_trend = context.research_trends.get(discovery.domain, 0)
        base_prob += domain_trend * 0.2

        # Adjust based on time proximity
        years_before = (discovery.discovery_date - context.cutoff_date).days / 365
        time_factor = max(0, 1 - years_before / 10)  # Harder to predict further out
        base_prob *= time_factor

        # Domain-specific calibration
        domain_weight = self.domain_weights.get(discovery.domain, 1.0)
        base_prob *= domain_weight

        # Add randomness for realistic behavior
        base_prob += random.gauss(0, 0.1)
        base_prob = max(0.05, min(0.95, base_prob))

        predicted = random.random() < base_prob

        # Generate prediction date if predicted
        if predicted:
            # Predict within reasonable window
            days_variance = int(random.gauss(0, 180))
            predicted_date = discovery.discovery_date + timedelta(days=days_variance)
        else:
            predicted_date = None

        # Identify missing prerequisites
        missing_prereqs = [
            prereq for prereq in discovery.prerequisites
            if not any(k.id == prereq for k in context.available_knowledge)
        ]

        # Generate realistic reasoning
        reasoning = self._generate_reasoning(discovery, context, predicted)

        # Generate key insights
        insights = self._generate_insights(discovery, context, predicted)

        compute_time = (time.time() - start_time) * 1000

        return {
            "predicted": predicted,
            "confidence_score": base_prob,
            "predicted_date": predicted_date.isoformat() if predicted_date else None,
            "reasoning": reasoning,
            "key_insights": insights,
            "missing_prerequisites": missing_prereqs,
            "false_dependencies": [],  # Would be identified by analyzing wrong predictions
            "compute_time_ms": compute_time,
            "model_version": "talai-v2.1.0"
        }

    def _generate_reasoning(self,
                          discovery: Discovery,
                          context: BacktestContext,
                          predicted: bool) -> str:
        """Generate reasoning for prediction"""
        if predicted:
            reasons = []
            if context.research_trends.get(discovery.domain, 0) > 0.2:
                reasons.append(f"Strong research momentum in {discovery.domain}")
            if len([p for p in discovery.prerequisites if any(k.id == p for k in context.available_knowledge)]) > 0:
                reasons.append("Key prerequisites have been established")
            if discovery.breakthrough_type == "computational" and context.compute_capabilities["gpu_available"]:
                reasons.append("Computational capabilities are sufficient")

            return ". ".join(reasons) if reasons else "Convergence of multiple research threads suggests imminent breakthrough"
        else:
            blockers = []
            missing = [p for p in discovery.prerequisites if not any(k.id == p for k in context.available_knowledge)]
            if missing:
                blockers.append(f"Missing {len(missing)} critical prerequisites")
            if context.research_trends.get(discovery.domain, 0) < 0.1:
                blockers.append(f"Limited research activity in {discovery.domain}")

            return ". ".join(blockers) if blockers else "Insufficient evidence for near-term breakthrough"

    def _generate_insights(self,
                         discovery: Discovery,
                         context: BacktestContext,
                         predicted: bool) -> List[str]:
        """Generate key insights from analysis"""
        insights = []

        # Domain-specific insights
        if discovery.domain == "physics":
            insights.append("Theoretical framework approaching critical mass")
        elif discovery.domain == "biology":
            insights.append("Convergence of genomics and computational biology")
        elif discovery.domain == "computer_science":
            insights.append("Algorithm complexity barriers being addressed")

        # Time-based insights
        years_before = (discovery.discovery_date - context.cutoff_date).days / 365
        if years_before < 1:
            insights.append("Multiple research groups approaching similar solutions")
        elif years_before < 3:
            insights.append("Foundation being laid through incremental advances")
        else:
            insights.append("Long-term research trajectory identified")

        # Technology insights
        if discovery.enabling_technologies:
            available_tech = set(context.available_technologies)
            enabling_tech = set(discovery.enabling_technologies)
            if len(available_tech & enabling_tech) > 0:
                insights.append("Enabling technologies becoming available")

        return insights[:5]  # Limit to top 5 insights

    def _determine_validation_result(self,
                                    predicted: bool,
                                    predicted_date: Optional[datetime],
                                    actual_date: datetime) -> Tuple[ValidationResult, Optional[int]]:
        """Determine the validation result category"""
        if not predicted:
            return ValidationResult.MISSED, None

        if predicted_date is None:
            return ValidationResult.PARTIAL_SUCCESS, None

        time_delta = (predicted_date - actual_date).days

        if abs(time_delta) <= 180:  # Within 6 months
            return ValidationResult.PREDICTED_CORRECTLY, time_delta
        elif time_delta < -180:  # Predicted too early
            return ValidationResult.PREDICTED_EARLY, time_delta
        else:  # Predicted too late
            return ValidationResult.PREDICTED_LATE, time_delta

    async def backtest_discovery(self,
                                discovery_id: str,
                                cutoff_years_before: int = 2,
                                use_cache: bool = True) -> PredictionResult:
        """
        Backtest a single discovery.

        Args:
            discovery_id: ID of discovery to test
            cutoff_years_before: Years before discovery to set knowledge cutoff
            use_cache: Whether to use cached results

        Returns:
            PredictionResult with validation outcome
        """
        if discovery_id not in self.discoveries:
            raise ValueError(f"Discovery {discovery_id} not found in database")

        discovery = self.discoveries[discovery_id]
        context = self._build_context(discovery, cutoff_years_before)

        # Check cache
        if use_cache:
            cached_result = await self._check_cache(discovery_id, context)
            if cached_result:
                logger.debug(f"Using cached result for {discovery_id}")
                return cached_result

        # Call TalAI for prediction
        prediction_data = await self._call_talai_prediction(discovery, context)

        # Parse prediction date
        predicted_date = None
        if prediction_data.get("predicted_date"):
            predicted_date = datetime.fromisoformat(prediction_data["predicted_date"])

        # Determine validation result
        validation_result, time_delta = self._determine_validation_result(
            prediction_data["predicted"],
            predicted_date,
            discovery.discovery_date
        )

        # Build result
        result = PredictionResult(
            discovery_id=discovery_id,
            predicted=prediction_data["predicted"],
            confidence_score=prediction_data["confidence_score"],
            predicted_date=predicted_date,
            actual_date=discovery.discovery_date,
            time_delta_days=time_delta,
            reasoning=prediction_data["reasoning"],
            key_insights=prediction_data["key_insights"],
            missing_prerequisites=prediction_data["missing_prerequisites"],
            false_dependencies=prediction_data.get("false_dependencies", []),
            validation_result=validation_result,
            compute_time_ms=prediction_data["compute_time_ms"],
            model_version=prediction_data["model_version"]
        )

        # Update metrics
        self.total_predictions += 1
        if result.validation_result in [ValidationResult.PREDICTED_CORRECTLY, ValidationResult.PREDICTED_EARLY]:
            self.successful_predictions += 1
        self.compute_time_total += result.compute_time_ms

        # Save to cache
        if use_cache:
            await self._save_to_cache(result, context)

        # Store result
        self.results.append(result)

        logger.info(f"Backtested {discovery_id}: {validation_result.value}")

        return result

    async def run_full_backtest(self,
                              domain_filter: Optional[str] = None,
                              min_impact_score: float = 5.0,
                              max_discoveries: Optional[int] = None,
                              cutoff_years_before: int = 2) -> Dict[str, Any]:
        """
        Run full backtesting on all discoveries.

        Args:
            domain_filter: Optional domain to filter discoveries
            min_impact_score: Minimum impact score for discoveries to test
            max_discoveries: Maximum number of discoveries to test
            cutoff_years_before: Years before discovery to set knowledge cutoff

        Returns:
            Comprehensive metrics and analysis results
        """
        # Filter discoveries
        discoveries_to_test = [
            disc for disc in self.discoveries.values()
            if (domain_filter is None or disc.domain == domain_filter)
            and disc.impact_score >= min_impact_score
        ]

        # Sort by impact score
        discoveries_to_test.sort(key=lambda x: x.impact_score, reverse=True)

        # Limit if requested
        if max_discoveries:
            discoveries_to_test = discoveries_to_test[:max_discoveries]

        logger.info(f"Starting backtest of {len(discoveries_to_test)} discoveries")

        # Run backtests concurrently
        tasks = [
            self.backtest_discovery(disc.id, cutoff_years_before)
            for disc in discoveries_to_test
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, PredictionResult)]
        failed_count = len(results) - len(valid_results)

        if failed_count > 0:
            logger.warning(f"{failed_count} backtests failed with exceptions")

        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(valid_results)

        # Generate report
        report = self._generate_report(metrics, valid_results)

        return {
            "metrics": metrics,
            "report": report,
            "detailed_results": valid_results,
            "summary": {
                "total_tested": len(valid_results),
                "successful_predictions": self.successful_predictions,
                "success_rate": self.successful_predictions / max(self.total_predictions, 1),
                "avg_compute_time_ms": self.compute_time_total / max(self.total_predictions, 1),
                "domains_tested": list(set(self.discoveries[r.discovery_id].domain for r in valid_results))
            }
        }

    def _calculate_metrics(self, results: List[PredictionResult]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not results:
            return {}

        # Basic counts
        total = len(results)
        correct = sum(1 for r in results if r.validation_result == ValidationResult.PREDICTED_CORRECTLY)
        early = sum(1 for r in results if r.validation_result == ValidationResult.PREDICTED_EARLY)
        late = sum(1 for r in results if r.validation_result == ValidationResult.PREDICTED_LATE)
        missed = sum(1 for r in results if r.validation_result == ValidationResult.MISSED)
        partial = sum(1 for r in results if r.validation_result == ValidationResult.PARTIAL_SUCCESS)

        # Calculate precision and recall
        true_positives = correct + early  # Predicted and happened
        false_positives = late  # Predicted but wrong timing
        false_negatives = missed  # Not predicted but happened

        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)

        # Time accuracy for correct predictions
        time_deltas = [
            abs(r.time_delta_days) for r in results
            if r.time_delta_days is not None
        ]

        avg_time_error = np.mean(time_deltas) if time_deltas else None
        median_time_error = np.median(time_deltas) if time_deltas else None

        # Domain-specific performance
        domain_performance = defaultdict(lambda: {"total": 0, "correct": 0})
        for result in results:
            domain = self.discoveries[result.discovery_id].domain
            domain_performance[domain]["total"] += 1
            if result.validation_result in [ValidationResult.PREDICTED_CORRECTLY, ValidationResult.PREDICTED_EARLY]:
                domain_performance[domain]["correct"] += 1

        domain_accuracy = {
            domain: stats["correct"] / stats["total"]
            for domain, stats in domain_performance.items()
        }

        # Confidence calibration
        confidence_buckets = defaultdict(lambda: {"predicted": 0, "actual": 0})
        for result in results:
            bucket = int(result.confidence_score * 10) / 10
            confidence_buckets[bucket]["predicted"] += result.confidence_score
            if result.predicted:
                confidence_buckets[bucket]["actual"] += 1

        calibration_error = sum(
            abs(bucket_stats["predicted"]/max(bucket_stats["actual"], 1) - bucket)
            for bucket, bucket_stats in confidence_buckets.items()
        ) / len(confidence_buckets)

        return {
            "accuracy": (correct + early) / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "breakdown": {
                "predicted_correctly": correct,
                "predicted_early": early,
                "predicted_late": late,
                "missed": missed,
                "partial_success": partial
            },
            "time_accuracy": {
                "avg_days_error": avg_time_error,
                "median_days_error": median_time_error,
                "within_30_days": sum(1 for d in time_deltas if d <= 30) / max(len(time_deltas), 1) if time_deltas else 0,
                "within_90_days": sum(1 for d in time_deltas if d <= 90) / max(len(time_deltas), 1) if time_deltas else 0,
                "within_180_days": sum(1 for d in time_deltas if d <= 180) / max(len(time_deltas), 1) if time_deltas else 0
            },
            "domain_accuracy": domain_accuracy,
            "calibration_error": calibration_error,
            "compute_performance": {
                "avg_ms": np.mean([r.compute_time_ms for r in results]),
                "median_ms": np.median([r.compute_time_ms for r in results]),
                "total_seconds": sum(r.compute_time_ms for r in results) / 1000
            }
        }

    def _generate_report(self, metrics: Dict[str, Any], results: List[PredictionResult]) -> str:
        """Generate human-readable validation report"""
        report = []
        report.append("=" * 80)
        report.append("TALAI HISTORICAL BACKTESTING VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall performance
        report.append("OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Total Discoveries Tested: {len(results)}")
        report.append(f"Overall Accuracy: {metrics['accuracy']:.2%}")
        report.append(f"Precision: {metrics['precision']:.2%}")
        report.append(f"Recall: {metrics['recall']:.2%}")
        report.append(f"F1 Score: {metrics['f1_score']:.3f}")
        report.append("")

        # Breakdown
        report.append("PREDICTION BREAKDOWN")
        report.append("-" * 40)
        for category, count in metrics['breakdown'].items():
            report.append(f"  {category.replace('_', ' ').title()}: {count}")
        report.append("")

        # Time accuracy
        if metrics['time_accuracy']['avg_days_error'] is not None:
            report.append("TIME ACCURACY")
            report.append("-" * 40)
            report.append(f"Average Time Error: {metrics['time_accuracy']['avg_days_error']:.1f} days")
            report.append(f"Median Time Error: {metrics['time_accuracy']['median_days_error']:.1f} days")
            report.append(f"Within 30 days: {metrics['time_accuracy']['within_30_days']:.1%}")
            report.append(f"Within 90 days: {metrics['time_accuracy']['within_90_days']:.1%}")
            report.append(f"Within 180 days: {metrics['time_accuracy']['within_180_days']:.1%}")
            report.append("")

        # Domain performance
        report.append("DOMAIN-SPECIFIC PERFORMANCE")
        report.append("-" * 40)
        for domain, accuracy in sorted(metrics['domain_accuracy'].items(), key=lambda x: x[1], reverse=True):
            report.append(f"  {domain.replace('_', ' ').title()}: {accuracy:.1%}")
        report.append("")

        # Top successes
        report.append("TOP SUCCESSFUL PREDICTIONS")
        report.append("-" * 40)
        successes = [r for r in results if r.validation_result == ValidationResult.PREDICTED_CORRECTLY]
        successes.sort(key=lambda x: self.discoveries[x.discovery_id].impact_score, reverse=True)
        for result in successes[:5]:
            disc = self.discoveries[result.discovery_id]
            report.append(f"  • {disc.name} ({disc.domain})")
            report.append(f"    Impact Score: {disc.impact_score:.1f}")
            if result.time_delta_days:
                report.append(f"    Time Error: {abs(result.time_delta_days)} days")
        report.append("")

        # Notable misses
        report.append("NOTABLE MISSES")
        report.append("-" * 40)
        misses = [r for r in results if r.validation_result == ValidationResult.MISSED]
        misses.sort(key=lambda x: self.discoveries[x.discovery_id].impact_score, reverse=True)
        for result in misses[:5]:
            disc = self.discoveries[result.discovery_id]
            report.append(f"  • {disc.name} ({disc.domain})")
            report.append(f"    Impact Score: {disc.impact_score:.1f}")
            report.append(f"    Reason: {result.reasoning[:100]}...")
        report.append("")

        # Computational performance
        report.append("COMPUTATIONAL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Average Compute Time: {metrics['compute_performance']['avg_ms']:.1f} ms")
        report.append(f"Median Compute Time: {metrics['compute_performance']['median_ms']:.1f} ms")
        report.append(f"Total Compute Time: {metrics['compute_performance']['total_seconds']:.1f} seconds")
        report.append("")

        # Calibration
        report.append("CONFIDENCE CALIBRATION")
        report.append("-" * 40)
        report.append(f"Calibration Error: {metrics['calibration_error']:.3f}")
        report.append("(Lower is better - perfect calibration = 0)")
        report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)

        if metrics['precision'] < 0.7:
            report.append("  • Precision below 70% - reduce false positive predictions")
        if metrics['recall'] < 0.6:
            report.append("  • Recall below 60% - missing important discoveries")
        if metrics['calibration_error'] > 0.2:
            report.append("  • High calibration error - confidence scores need adjustment")

        worst_domain = min(metrics['domain_accuracy'].items(), key=lambda x: x[1])
        if worst_domain[1] < 0.5:
            report.append(f"  • Poor performance in {worst_domain[0]} - needs domain-specific tuning")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    async def export_results(self, output_path: Path) -> None:
        """Export detailed results to file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": self.total_predictions,
            "successful_predictions": self.successful_predictions,
            "results": [
                {
                    **result.__dict__,
                    "discovery_name": self.discoveries[result.discovery_id].name,
                    "discovery_domain": self.discoveries[result.discovery_id].domain,
                    "discovery_impact": self.discoveries[result.discovery_id].impact_score
                }
                for result in self.results
            ],
            "metrics": self._calculate_metrics(self.results)
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(f"Exported results to {output_path}")


if __name__ == "__main__":
    # Example usage
    async def main():
        engine = HistoricalBacktestingEngine(
            database_path=Path("../data/discoveries.json"),
            talai_endpoint="http://localhost:8000/api/predict"
        )

        await engine.load_discoveries()

        # Run full backtest
        results = await engine.run_full_backtest(
            min_impact_score=7.0,
            max_discoveries=50
        )

        print(results["report"])

        # Export results
        await engine.export_results(Path("../reports/backtest_results.json"))

    asyncio.run(main())