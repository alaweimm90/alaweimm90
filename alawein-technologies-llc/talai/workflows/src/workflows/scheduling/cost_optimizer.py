"""
Cost Optimizer - Cost optimization for API calls and resources

Minimizes costs while maintaining performance and throughput.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .job_queue import Job, JobPriority


logger = logging.getLogger(__name__)


class CostModel(Enum):
    """Cost calculation models"""
    FLAT_RATE = "flat_rate"  # Fixed cost per job
    TIME_BASED = "time_based"  # Cost per execution time
    RESOURCE_BASED = "resource_based"  # Cost per resource usage
    API_BASED = "api_based"  # Cost per API call
    TIERED = "tiered"  # Tiered pricing model
    CUSTOM = "custom"  # Custom cost function


@dataclass
class PricingTier:
    """Pricing tier for tiered models"""
    name: str
    min_usage: float
    max_usage: float
    rate: float
    unit: str = ""


@dataclass
class CostConfig:
    """Configuration for cost calculation"""
    model: CostModel = CostModel.RESOURCE_BASED
    base_rate: float = 0.01  # Base cost per unit
    rates: Dict[str, float] = field(default_factory=dict)
    tiers: List[PricingTier] = field(default_factory=list)
    budget_limit: Optional[float] = None
    budget_period: timedelta = timedelta(hours=1)
    api_costs: Dict[str, float] = field(default_factory=dict)


class CostOptimizer:
    """Optimizes job execution costs"""

    def __init__(self, config: Optional[CostConfig] = None):
        self.config = config or CostConfig()
        self.cost_history: List[Dict[str, Any]] = []
        self.current_budget_usage = 0.0
        self.budget_reset_time = datetime.now() + self.config.budget_period
        self._init_default_rates()

    def _init_default_rates(self):
        """Initialize default cost rates"""
        if not self.config.rates:
            self.config.rates = {
                "cpu": 0.01,  # per core-second
                "memory": 0.001,  # per GB-second
                "gpu": 0.1,  # per GPU-second
                "api_calls": 0.002,  # per API call
                "disk": 0.0001,  # per GB
                "network": 0.00001  # per MB
            }

        if not self.config.api_costs:
            self.config.api_costs = {
                "openai": 0.002,  # per 1K tokens
                "anthropic": 0.003,  # per 1K tokens
                "google": 0.001,  # per request
                "aws": 0.0005  # per request
            }

    def calculate_job_cost(self, job: Job) -> float:
        """Calculate cost for a job"""
        cost = 0.0

        if self.config.model == CostModel.FLAT_RATE:
            cost = self.config.base_rate

        elif self.config.model == CostModel.TIME_BASED:
            execution_time = job.metrics.execution_time or 1.0
            cost = execution_time * self.config.base_rate

        elif self.config.model == CostModel.RESOURCE_BASED:
            cost = self._calculate_resource_cost(job)

        elif self.config.model == CostModel.API_BASED:
            cost = self._calculate_api_cost(job)

        elif self.config.model == CostModel.TIERED:
            cost = self._calculate_tiered_cost(job)

        elif self.config.model == CostModel.CUSTOM:
            custom_func = job.metadata.get("cost_function")
            if custom_func and callable(custom_func):
                cost = custom_func(job)
            else:
                cost = self.config.base_rate

        # Apply priority multiplier
        cost *= self._get_priority_multiplier(job.priority)

        # Record cost
        self._record_cost(job.id, cost)

        return cost

    def _calculate_resource_cost(self, job: Job) -> float:
        """Calculate cost based on resource usage"""
        cost = 0.0
        execution_time = job.metrics.execution_time or 1.0

        for resource_name, amount in job.resource_requirements.items():
            rate = self.config.rates.get(resource_name, self.config.base_rate)
            cost += amount * execution_time * rate

        # Add resource usage from metrics
        for resource_name, usage in job.metrics.resource_usage.items():
            rate = self.config.rates.get(resource_name, self.config.base_rate)
            cost += usage * rate

        return cost

    def _calculate_api_cost(self, job: Job) -> float:
        """Calculate cost based on API usage"""
        cost = 0.0

        # Check for API usage in metadata
        api_usage = job.metadata.get("api_usage", {})
        for api_name, usage_count in api_usage.items():
            rate = self.config.api_costs.get(api_name, self.config.base_rate)
            cost += usage_count * rate

        # Check for token usage (LLM APIs)
        token_usage = job.metadata.get("token_usage", 0)
        if token_usage > 0:
            api_type = job.metadata.get("api_type", "openai")
            rate = self.config.api_costs.get(api_type, 0.002)
            cost += (token_usage / 1000) * rate

        return cost

    def _calculate_tiered_cost(self, job: Job) -> float:
        """Calculate cost using tiered pricing"""
        usage = job.metrics.resource_usage.get("total", 1.0)

        for tier in self.config.tiers:
            if tier.min_usage <= usage <= tier.max_usage:
                return usage * tier.rate

        # Default to base rate if no tier matches
        return usage * self.config.base_rate

    def _get_priority_multiplier(self, priority: JobPriority) -> float:
        """Get cost multiplier based on priority"""
        multipliers = {
            JobPriority.CRITICAL: 2.0,
            JobPriority.HIGH: 1.5,
            JobPriority.NORMAL: 1.0,
            JobPriority.LOW: 0.8,
            JobPriority.BACKGROUND: 0.5
        }
        return multipliers.get(priority, 1.0)

    def can_afford_job(self, job: Job, budget: Optional[float] = None) -> bool:
        """Check if job fits within budget"""
        budget = budget or self.config.budget_limit
        if not budget:
            return True

        # Check if budget period has reset
        if datetime.now() > self.budget_reset_time:
            self.current_budget_usage = 0.0
            self.budget_reset_time = datetime.now() + self.config.budget_period

        # Estimate job cost
        estimated_cost = self.estimate_job_cost(job)

        return (self.current_budget_usage + estimated_cost) <= budget

    def estimate_job_cost(self, job: Job) -> float:
        """Estimate cost before job execution"""
        # Use historical data if available
        similar_jobs = self._find_similar_jobs(job)
        if similar_jobs:
            avg_cost = sum(j["cost"] for j in similar_jobs) / len(similar_jobs)
            return avg_cost

        # Otherwise estimate based on requirements
        estimated_time = job.metadata.get("estimated_time", 1.0)
        estimated_cost = 0.0

        for resource_name, amount in job.resource_requirements.items():
            rate = self.config.rates.get(resource_name, self.config.base_rate)
            estimated_cost += amount * estimated_time * rate

        return estimated_cost * self._get_priority_multiplier(job.priority)

    def _find_similar_jobs(self, job: Job, limit: int = 10) -> List[Dict[str, Any]]:
        """Find similar jobs from history"""
        similar = []

        for hist in reversed(self.cost_history[-100:]):  # Check last 100 jobs
            if hist.get("handler_type") == job.metadata.get("handler_type"):
                similar.append(hist)
                if len(similar) >= limit:
                    break

        return similar

    def optimize_job_scheduling(self, jobs: List[Job],
                               budget: float) -> List[Job]:
        """Optimize job scheduling to maximize value within budget"""
        # Calculate value/cost ratio for each job
        job_scores = []
        for job in jobs:
            cost = self.estimate_job_cost(job)
            value = self._calculate_job_value(job)
            if cost > 0:
                score = value / cost
            else:
                score = float('inf')

            job_scores.append((job, score, cost))

        # Sort by score (value/cost ratio)
        job_scores.sort(key=lambda x: x[1], reverse=True)

        # Select jobs within budget
        selected = []
        total_cost = 0.0

        for job, score, cost in job_scores:
            if total_cost + cost <= budget:
                selected.append(job)
                total_cost += cost

        logger.info(f"Optimized {len(jobs)} jobs, selected {len(selected)} within budget {budget}")
        return selected

    def _calculate_job_value(self, job: Job) -> float:
        """Calculate value/importance of a job"""
        # Base value from priority
        priority_values = {
            JobPriority.CRITICAL: 100,
            JobPriority.HIGH: 50,
            JobPriority.NORMAL: 20,
            JobPriority.LOW: 10,
            JobPriority.BACKGROUND: 5
        }
        value = priority_values.get(job.priority, 10)

        # Adjust for custom value
        if "value" in job.metadata:
            value = job.metadata["value"]

        # Penalty for retries
        value *= (0.9 ** job.metrics.retry_count)

        return value

    def _record_cost(self, job_id: str, cost: float):
        """Record cost in history"""
        self.current_budget_usage += cost

        self.cost_history.append({
            "job_id": job_id,
            "cost": cost,
            "timestamp": datetime.now(),
            "budget_usage": self.current_budget_usage
        })

        # Trim history if too large
        if len(self.cost_history) > 10000:
            self.cost_history = self.cost_history[-5000:]

    def get_cost_report(self, period: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get cost report for period"""
        period = period or timedelta(hours=24)
        cutoff = datetime.now() - period

        period_costs = [
            h for h in self.cost_history
            if h["timestamp"] > cutoff
        ]

        if not period_costs:
            return {
                "total_cost": 0,
                "job_count": 0,
                "avg_cost": 0,
                "budget_usage": self.current_budget_usage
            }

        total_cost = sum(h["cost"] for h in period_costs)
        avg_cost = total_cost / len(period_costs)

        return {
            "total_cost": total_cost,
            "job_count": len(period_costs),
            "avg_cost": avg_cost,
            "min_cost": min(h["cost"] for h in period_costs),
            "max_cost": max(h["cost"] for h in period_costs),
            "budget_usage": self.current_budget_usage,
            "budget_limit": self.config.budget_limit,
            "budget_remaining": (
                self.config.budget_limit - self.current_budget_usage
                if self.config.budget_limit else None
            )
        }

    def reset_budget(self):
        """Reset budget usage"""
        self.current_budget_usage = 0.0
        self.budget_reset_time = datetime.now() + self.config.budget_period
        logger.info("Budget reset")

    def update_rates(self, rates: Dict[str, float]):
        """Update cost rates"""
        self.config.rates.update(rates)
        logger.info(f"Updated cost rates: {rates}")