"""
Batch Processor - Batch processing for multiple hypotheses

Optimizes batch execution for efficiency and throughput.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

from .job_queue import Job, JobPriority


logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies"""
    SIZE_BASED = "size_based"  # Batch by fixed size
    TIME_BASED = "time_based"  # Batch by time window
    SIMILARITY = "similarity"  # Batch similar jobs
    RESOURCE = "resource"  # Batch by resource requirements
    PRIORITY = "priority"  # Batch by priority level
    HYBRID = "hybrid"  # Combine multiple strategies


@dataclass
class Batch:
    """Batch of jobs for processing"""
    id: str
    jobs: List[Job]
    strategy: BatchStrategy
    created_at: datetime
    metadata: Dict[str, Any]

    def add_job(self, job: Job) -> bool:
        """Add job to batch if compatible"""
        if self._is_compatible(job):
            self.jobs.append(job)
            return True
        return False

    def _is_compatible(self, job: Job) -> bool:
        """Check if job is compatible with batch"""
        if not self.jobs:
            return True

        if self.strategy == BatchStrategy.PRIORITY:
            # Same priority level
            return job.priority == self.jobs[0].priority

        elif self.strategy == BatchStrategy.SIMILARITY:
            # Similar handler type
            return job.metadata.get("handler_type") == self.jobs[0].metadata.get("handler_type")

        elif self.strategy == BatchStrategy.RESOURCE:
            # Similar resource requirements
            batch_resources = sum(
                sum(j.resource_requirements.values())
                for j in self.jobs
            )
            job_resources = sum(job.resource_requirements.values())
            # Check if adding won't exceed threshold
            return (batch_resources + job_resources) <= self.metadata.get("resource_limit", 100)

        return True

    def is_ready(self) -> bool:
        """Check if batch is ready for execution"""
        if self.strategy == BatchStrategy.SIZE_BASED:
            return len(self.jobs) >= self.metadata.get("batch_size", 5)

        elif self.strategy == BatchStrategy.TIME_BASED:
            elapsed = (datetime.now() - self.created_at).total_seconds()
            return elapsed >= self.metadata.get("time_window", 10)

        return len(self.jobs) > 0


class BatchProcessor:
    """Processes jobs in batches for efficiency"""

    def __init__(self, default_strategy: BatchStrategy = BatchStrategy.SIZE_BASED,
                 default_batch_size: int = 5):
        self.default_strategy = default_strategy
        self.default_batch_size = default_batch_size
        self.active_batches: Dict[str, Batch] = {}
        self.batch_handlers: Dict[str, Callable] = {}
        self._batch_counter = 0

    def create_batch(self, strategy: Optional[BatchStrategy] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Batch:
        """Create new batch"""
        strategy = strategy or self.default_strategy
        metadata = metadata or {}

        # Set defaults
        if "batch_size" not in metadata:
            metadata["batch_size"] = self.default_batch_size

        self._batch_counter += 1
        batch = Batch(
            id=f"batch_{self._batch_counter}",
            jobs=[],
            strategy=strategy,
            created_at=datetime.now(),
            metadata=metadata
        )

        self.active_batches[batch.id] = batch
        logger.info(f"Created batch {batch.id} with strategy {strategy.value}")
        return batch

    async def add_to_batch(self, job: Job,
                          batch_id: Optional[str] = None) -> str:
        """Add job to batch"""
        # Find or create batch
        if batch_id and batch_id in self.active_batches:
            batch = self.active_batches[batch_id]
        else:
            # Find compatible batch or create new one
            batch = self._find_compatible_batch(job)
            if not batch:
                batch = self.create_batch(
                    strategy=self._determine_strategy(job)
                )

        # Add job to batch
        if batch.add_job(job):
            logger.info(f"Added job {job.id} to batch {batch.id}")
            return batch.id
        else:
            # Create new batch for incompatible job
            new_batch = self.create_batch(
                strategy=self._determine_strategy(job)
            )
            new_batch.add_job(job)
            logger.info(f"Created new batch {new_batch.id} for job {job.id}")
            return new_batch.id

    def _find_compatible_batch(self, job: Job) -> Optional[Batch]:
        """Find compatible batch for job"""
        for batch in self.active_batches.values():
            if not batch.is_ready() and batch.add_job(job):
                batch.jobs.remove(job)  # Remove test add
                return batch
        return None

    def _determine_strategy(self, job: Job) -> BatchStrategy:
        """Determine best batching strategy for job"""
        # Check job metadata for hints
        if job.metadata.get("batch_strategy"):
            return BatchStrategy(job.metadata["batch_strategy"])

        # Use priority-based for high priority jobs
        if job.priority in [JobPriority.CRITICAL, JobPriority.HIGH]:
            return BatchStrategy.PRIORITY

        # Use similarity for jobs with handlers
        if job.metadata.get("handler_type"):
            return BatchStrategy.SIMILARITY

        return self.default_strategy

    async def process_batch(self, batch_id: str) -> List[Any]:
        """Process a batch of jobs"""
        if batch_id not in self.active_batches:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self.active_batches[batch_id]
        logger.info(f"Processing batch {batch_id} with {len(batch.jobs)} jobs")

        # Get appropriate handler
        handler = self._get_batch_handler(batch)

        # Process batch
        results = await handler(batch.jobs)

        # Clean up
        del self.active_batches[batch_id]

        logger.info(f"Batch {batch_id} processed successfully")
        return results

    def _get_batch_handler(self, batch: Batch) -> Callable:
        """Get handler for batch processing"""
        # Check for registered handler
        handler_type = batch.metadata.get("handler_type")
        if handler_type and handler_type in self.batch_handlers:
            return self.batch_handlers[handler_type]

        # Default batch processing
        return self._default_batch_handler

    async def _default_batch_handler(self, jobs: List[Job]) -> List[Any]:
        """Default batch processing handler"""
        results = []

        # Process jobs concurrently
        tasks = []
        for job in jobs:
            if job.handler:
                if asyncio.iscoroutinefunction(job.handler):
                    task = job.handler(job.inputs)
                else:
                    # Wrap sync handler
                    async def wrapper(h=job.handler, i=job.inputs):
                        return h(i)
                    task = wrapper()
            else:
                # Dummy processing
                async def dummy():
                    return {"job_id": job.id, "status": "processed"}
                task = dummy()

            tasks.append(task)

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def register_batch_handler(self, handler_type: str, handler: Callable):
        """Register custom batch handler"""
        self.batch_handlers[handler_type] = handler
        logger.info(f"Registered batch handler: {handler_type}")

    async def optimize_batch(self, jobs: List[Job]) -> List[Job]:
        """Optimize job ordering within batch"""
        # Sort by various criteria
        optimized = sorted(jobs, key=lambda j: (
            j.priority.value,  # Priority first
            -len(j.dependencies),  # Jobs with more dependencies
            j.created_at  # FIFO for same priority
        ))

        return optimized

    async def auto_batch(self, jobs: List[Job],
                        max_batch_size: int = 10) -> List[List[Job]]:
        """Automatically batch jobs for optimal execution"""
        batches = []
        current_batch = []

        # Sort jobs by compatibility
        sorted_jobs = sorted(jobs, key=lambda j: (
            j.priority.value,
            j.metadata.get("handler_type", ""),
            sum(j.resource_requirements.values())
        ))

        for job in sorted_jobs:
            if not current_batch:
                current_batch.append(job)
            elif len(current_batch) < max_batch_size and self._are_compatible(current_batch[0], job):
                current_batch.append(job)
            else:
                batches.append(current_batch)
                current_batch = [job]

        if current_batch:
            batches.append(current_batch)

        logger.info(f"Auto-batched {len(jobs)} jobs into {len(batches)} batches")
        return batches

    def _are_compatible(self, job1: Job, job2: Job) -> bool:
        """Check if two jobs are compatible for batching"""
        # Same priority
        if job1.priority != job2.priority:
            return False

        # Same handler type
        if job1.metadata.get("handler_type") != job2.metadata.get("handler_type"):
            return False

        # Similar resource requirements
        resources1 = sum(job1.resource_requirements.values())
        resources2 = sum(job2.resource_requirements.values())
        if abs(resources1 - resources2) > 10:  # Threshold
            return False

        return True

    def get_batch_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        stats = {
            "active_batches": len(self.active_batches),
            "total_jobs": sum(len(b.jobs) for b in self.active_batches.values()),
            "batches_by_strategy": {}
        }

        for strategy in BatchStrategy:
            count = sum(
                1 for b in self.active_batches.values()
                if b.strategy == strategy
            )
            stats["batches_by_strategy"][strategy.value] = count

        return stats