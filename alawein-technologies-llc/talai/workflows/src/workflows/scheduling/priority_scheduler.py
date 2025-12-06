"""
Priority Scheduler - Advanced priority-based scheduling

Implements sophisticated priority scheduling with starvation prevention
and dynamic priority adjustment.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import heapq

from .job_queue import Job, JobPriority, JobStatus


logger = logging.getLogger(__name__)


@dataclass
class PriorityRule:
    """Rule for priority adjustment"""
    name: str
    condition: str
    adjustment: int  # Priority adjustment value
    max_boost: int = 2  # Maximum priority boost


class PriorityScheduler:
    """Advanced priority-based scheduler"""

    def __init__(self, starvation_threshold: int = 60,
                 aging_factor: float = 0.1):
        self.starvation_threshold = starvation_threshold  # seconds
        self.aging_factor = aging_factor
        self.priority_rules: List[PriorityRule] = []
        self.job_priorities: Dict[str, float] = {}  # Dynamic priorities
        self.original_priorities: Dict[str, JobPriority] = {}
        self._init_default_rules()

    def _init_default_rules(self):
        """Initialize default priority rules"""
        # Starvation prevention
        self.add_rule(PriorityRule(
            name="starvation_prevention",
            condition="queue_time > 60",
            adjustment=-1  # Increase priority
        ))

        # Deadline approaching
        self.add_rule(PriorityRule(
            name="deadline_urgency",
            condition="deadline_remaining < 300",
            adjustment=-2
        ))

        # Failed retry boost
        self.add_rule(PriorityRule(
            name="retry_boost",
            condition="retry_count > 0",
            adjustment=-1
        ))

    def add_rule(self, rule: PriorityRule):
        """Add priority adjustment rule"""
        self.priority_rules.append(rule)
        logger.info(f"Added priority rule: {rule.name}")

    def calculate_dynamic_priority(self, job: Job) -> float:
        """Calculate dynamic priority for job"""
        # Start with base priority
        base_priority = job.priority.value

        # Store original priority
        if job.id not in self.original_priorities:
            self.original_priorities[job.id] = job.priority

        # Apply aging (priority improves over time)
        queue_time = (datetime.now() - job.created_at).total_seconds()
        age_boost = -1 * (queue_time / self.starvation_threshold) * self.aging_factor

        # Apply rules
        rule_adjustment = 0
        for rule in self.priority_rules:
            if self._evaluate_rule(rule, job):
                rule_adjustment += rule.adjustment

        # Calculate final priority
        dynamic_priority = base_priority + age_boost + rule_adjustment

        # Apply limits
        dynamic_priority = max(0, min(dynamic_priority, 10))

        # Store dynamic priority
        self.job_priorities[job.id] = dynamic_priority

        return dynamic_priority

    def _evaluate_rule(self, rule: PriorityRule, job: Job) -> bool:
        """Evaluate if rule applies to job"""
        try:
            # Build evaluation context
            context = {
                "queue_time": (datetime.now() - job.created_at).total_seconds(),
                "retry_count": job.metrics.retry_count,
                "deadline_remaining": self._get_deadline_remaining(job),
                "resource_usage": sum(job.resource_requirements.values()),
                "dependencies": len(job.dependencies)
            }

            # Evaluate condition
            return eval(rule.condition, {"__builtins__": {}}, context)
        except:
            return False

    def _get_deadline_remaining(self, job: Job) -> float:
        """Get remaining time until deadline"""
        deadline = job.metadata.get("deadline")
        if deadline:
            if isinstance(deadline, str):
                deadline = datetime.fromisoformat(deadline)
            return (deadline - datetime.now()).total_seconds()
        return float('inf')

    def get_next_job(self, jobs: List[Job]) -> Optional[Job]:
        """Get next job based on dynamic priority"""
        if not jobs:
            return None

        # Calculate dynamic priorities
        job_heap = []
        for job in jobs:
            if job.status == JobStatus.QUEUED:
                priority = self.calculate_dynamic_priority(job)
                # Use negative priority for min heap (lower value = higher priority)
                heapq.heappush(job_heap, (priority, job.created_at.timestamp(), job))

        if job_heap:
            _, _, selected_job = heapq.heappop(job_heap)
            return selected_job

        return None

    def adjust_priority(self, job_id: str, adjustment: int):
        """Manually adjust job priority"""
        if job_id in self.job_priorities:
            self.job_priorities[job_id] += adjustment
            logger.info(f"Adjusted priority for job {job_id} by {adjustment}")

    def reset_priority(self, job_id: str):
        """Reset job to original priority"""
        if job_id in self.original_priorities:
            original = self.original_priorities[job_id]
            self.job_priorities[job_id] = original.value
            logger.info(f"Reset priority for job {job_id}")

    def prevent_starvation(self, jobs: List[Job]) -> List[Job]:
        """Prevent job starvation by boosting old jobs"""
        boosted = []

        for job in jobs:
            if job.status != JobStatus.QUEUED:
                continue

            queue_time = (datetime.now() - job.created_at).total_seconds()

            if queue_time > self.starvation_threshold:
                # Boost priority
                if job.priority != JobPriority.CRITICAL:
                    old_priority = job.priority
                    job.priority = JobPriority(max(0, job.priority.value - 1))
                    logger.info(f"Boosted priority for starving job {job.id}: "
                              f"{old_priority.name} -> {job.priority.name}")
                    boosted.append(job)

        return boosted

    def balance_priorities(self, jobs: List[Job]) -> Dict[JobPriority, int]:
        """Balance job distribution across priority levels"""
        distribution = {priority: 0 for priority in JobPriority}

        for job in jobs:
            if job.status == JobStatus.QUEUED:
                distribution[job.priority] += 1

        # Check for imbalance
        total = sum(distribution.values())
        if total > 0:
            for priority, count in distribution.items():
                ratio = count / total
                if ratio > 0.5 and priority != JobPriority.CRITICAL:
                    logger.warning(f"Priority imbalance detected: "
                                 f"{priority.name} has {ratio:.1%} of jobs")

        return distribution

    def get_priority_stats(self) -> Dict[str, Any]:
        """Get priority scheduling statistics"""
        stats = {
            "total_jobs": len(self.job_priorities),
            "rules_count": len(self.priority_rules),
            "boosted_jobs": 0,
            "avg_dynamic_priority": 0
        }

        if self.job_priorities:
            priorities = list(self.job_priorities.values())
            stats["avg_dynamic_priority"] = sum(priorities) / len(priorities)

            # Count boosted jobs
            for job_id, dyn_priority in self.job_priorities.items():
                if job_id in self.original_priorities:
                    orig = self.original_priorities[job_id].value
                    if dyn_priority < orig:
                        stats["boosted_jobs"] += 1

        return stats