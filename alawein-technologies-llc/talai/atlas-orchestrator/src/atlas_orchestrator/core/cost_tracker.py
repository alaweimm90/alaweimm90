"""
Cost tracking and budget management for ORCHEX Orchestrator
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from pydantic import BaseModel, Field


class CostReport(BaseModel):
    """Cost report for a time period"""
    total_cost: float = Field(0.0, description="Total cost in USD")
    total_requests: int = Field(0, description="Total number of requests")
    cost_by_model: Dict[str, float] = Field(default_factory=dict, description="Cost breakdown by model")
    cost_by_task_type: Dict[str, float] = Field(default_factory=dict, description="Cost by task type")
    avg_cost_per_request: float = Field(0.0, description="Average cost per request")
    period_start: datetime = Field(default_factory=datetime.now, description="Report period start")
    period_end: datetime = Field(default_factory=datetime.now, description="Report period end")


@dataclass
class CostEntry:
    """Single cost entry"""
    timestamp: datetime
    model: str
    task_type: str
    cost: float
    input_tokens: int
    output_tokens: int
    task_id: str


class CostTracker:
    """
    Track costs and enforce budgets

    Features:
    - Real-time cost tracking
    - Budget enforcement
    - Cost analytics
    - Alert thresholds
    """

    def __init__(
        self,
        max_daily_cost: float = 50.0,
        max_per_request: float = 1.0,
        alert_threshold: float = 40.0
    ):
        self.max_daily_cost = max_daily_cost
        self.max_per_request = max_per_request
        self.alert_threshold = alert_threshold

        # Storage
        self._entries: List[CostEntry] = []
        self._daily_totals: Dict[str, float] = defaultdict(float)  # date -> cost
        self._model_totals: Dict[str, float] = defaultdict(float)  # model -> cost
        self._task_type_totals: Dict[str, float] = defaultdict(float)  # task_type -> cost

        # Alerts
        self._alert_triggered = False

    def record_cost(
        self,
        model: str,
        task_type: str,
        cost: float,
        input_tokens: int,
        output_tokens: int,
        task_id: str
    ) -> None:
        """Record a cost entry"""
        now = datetime.now()
        entry = CostEntry(
            timestamp=now,
            model=model,
            task_type=task_type,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            task_id=task_id
        )

        # Store entry
        self._entries.append(entry)

        # Update totals
        today = now.date().isoformat()
        self._daily_totals[today] += cost
        self._model_totals[model] += cost
        self._task_type_totals[task_type] += cost

        # Check alert threshold
        if not self._alert_triggered and self.get_today_cost() >= self.alert_threshold:
            self._alert_triggered = True
            self._trigger_alert()

    def check_budget(self, estimated_cost: float) -> bool:
        """
        Check if request would exceed budget

        Args:
            estimated_cost: Estimated cost of the request

        Returns:
            True if within budget, False otherwise

        Raises:
            BudgetExceededError: If budget would be exceeded
        """
        # Check per-request limit
        if estimated_cost > self.max_per_request:
            raise BudgetExceededError(
                f"Request cost ${estimated_cost:.4f} exceeds per-request limit ${self.max_per_request:.2f}"
            )

        # Check daily limit
        today_cost = self.get_today_cost()
        if today_cost + estimated_cost > self.max_daily_cost:
            raise BudgetExceededError(
                f"Request would exceed daily budget. Current: ${today_cost:.2f}, "
                f"Limit: ${self.max_daily_cost:.2f}, Request: ${estimated_cost:.4f}"
            )

        return True

    def get_today_cost(self) -> float:
        """Get today's total cost"""
        today = datetime.now().date().isoformat()
        return self._daily_totals.get(today, 0.0)

    def get_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> CostReport:
        """
        Generate cost report for a period

        Args:
            start_date: Start of period (default: 7 days ago)
            end_date: End of period (default: now)

        Returns:
            CostReport object
        """
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        # Filter entries
        entries = [
            e for e in self._entries
            if start_date <= e.timestamp <= end_date
        ]

        if not entries:
            return CostReport(
                period_start=start_date,
                period_end=end_date
            )

        # Calculate metrics
        total_cost = sum(e.cost for e in entries)
        cost_by_model: Dict[str, float] = defaultdict(float)
        cost_by_task_type: Dict[str, float] = defaultdict(float)

        for entry in entries:
            cost_by_model[entry.model] += entry.cost
            cost_by_task_type[entry.task_type] += entry.cost

        return CostReport(
            total_cost=total_cost,
            total_requests=len(entries),
            cost_by_model=dict(cost_by_model),
            cost_by_task_type=dict(cost_by_task_type),
            avg_cost_per_request=total_cost / len(entries) if entries else 0.0,
            period_start=start_date,
            period_end=end_date
        )

    def get_model_costs(self) -> Dict[str, float]:
        """Get cost breakdown by model"""
        return dict(self._model_totals)

    def get_task_type_costs(self) -> Dict[str, float]:
        """Get cost breakdown by task type"""
        return dict(self._task_type_totals)

    def reset_daily_tracker(self) -> None:
        """Reset daily tracking (call at midnight)"""
        today = datetime.now().date().isoformat()
        self._daily_totals[today] = 0.0
        self._alert_triggered = False

    def _trigger_alert(self) -> None:
        """Trigger cost alert (can be extended to send notifications)"""
        print(f"⚠️  COST ALERT: Daily spending has reached ${self.get_today_cost():.2f} "
              f"(threshold: ${self.alert_threshold:.2f})")

    def get_savings_report(self) -> Dict[str, any]:
        """
        Calculate savings from using cheaper models

        Returns:
            Dict with savings metrics
        """
        entries = self._entries
        if not entries:
            return {"savings": 0.0, "savings_percent": 0.0}

        actual_cost = sum(e.cost for e in entries)

        # Calculate what it would cost if all requests used GPT-4
        gpt4_cost_per_1k_in = 0.03
        gpt4_cost_per_1k_out = 0.06

        hypothetical_cost = sum(
            (e.input_tokens / 1000 * gpt4_cost_per_1k_in +
             e.output_tokens / 1000 * gpt4_cost_per_1k_out)
            for e in entries
        )

        savings = hypothetical_cost - actual_cost
        savings_percent = (savings / hypothetical_cost * 100) if hypothetical_cost > 0 else 0.0

        return {
            "actual_cost": actual_cost,
            "hypothetical_gpt4_cost": hypothetical_cost,
            "savings": savings,
            "savings_percent": savings_percent,
            "total_requests": len(entries)
        }


class BudgetExceededError(Exception):
    """Raised when budget limit is exceeded"""
    pass
