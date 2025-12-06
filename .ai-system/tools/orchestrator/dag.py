#!/usr/bin/env python3
"""DAG (Directed Acyclic Graph) engine for workflow orchestration."""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Step:
    id: str
    command: str
    depends_on: List[str]
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict] = None

class DAG:
    """Directed Acyclic Graph for workflow dependencies."""
    
    def __init__(self):
        self.steps: Dict[str, Step] = {}
        self.graph: Dict[str, Set[str]] = {}
    
    def add_step(self, step: Step):
        """Add step to DAG."""
        self.steps[step.id] = step
        self.graph[step.id] = set(step.depends_on)
    
    def get_ready_steps(self) -> List[Step]:
        """Get steps ready to execute (dependencies met)."""
        ready = []
        for step_id, step in self.steps.items():
            if step.status != StepStatus.PENDING:
                continue
            
            # Check if all dependencies are successful
            deps_met = all(
                self.steps[dep].status == StepStatus.SUCCESS
                for dep in step.depends_on
            )
            
            if deps_met:
                ready.append(step)
        
        return ready
    
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return all(
            step.status in [StepStatus.SUCCESS, StepStatus.FAILED, StepStatus.SKIPPED]
            for step in self.steps.values()
        )
    
    def has_failures(self) -> bool:
        """Check if any steps failed."""
        return any(step.status == StepStatus.FAILED for step in self.steps.values())
    
    def validate(self) -> bool:
        """Validate DAG has no cycles."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for dep in self.graph.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self.graph:
            if node not in visited:
                if has_cycle(node):
                    return False
        
        return True
