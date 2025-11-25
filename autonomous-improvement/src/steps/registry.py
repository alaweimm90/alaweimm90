"""
Step Registry for REPZ Workflow System

Manages registration and execution of all 500 workflow steps.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class WorkflowPhase(Enum):
    """Workflow phases"""
    ASSESSMENT = "assessment"
    IDEATION = "ideation"
    TESTING = "testing"
    FEEDBACK = "feedback"
    SCALABILITY = "scalability"
    OUTPUT = "output"


@dataclass
class WorkflowStep:
    """Represents a single workflow step"""
    step_id: int
    phase: WorkflowPhase
    function: Callable
    description: str
    dependencies: List[int] = None
    timeout: int = 3600  # 1 hour default
    retry_count: int = 3
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


class StepRegistry:
    """Registry for managing workflow steps"""

    def __init__(self):
        self._steps: Dict[int, WorkflowStep] = {}

    def register_step(
        self,
        step_id: int,
        phase: WorkflowPhase,
        function: Callable,
        description: str,
        dependencies: List[int] = None,
        timeout: int = 3600,
        retry_count: int = 3,
        metadata: Dict[str, Any] = None
    ):
        """Register a workflow step"""
        if step_id in self._steps:
            raise ValueError(f"Step {step_id} already registered")

        if not 1 <= step_id <= 500:
            raise ValueError(f"Step ID {step_id} out of valid range (1-500)")

        step = WorkflowStep(
            step_id=step_id,
            phase=phase,
            function=function,
            description=description,
            dependencies=dependencies or [],
            timeout=timeout,
            retry_count=retry_count,
            metadata=metadata or {}
        )

        self._steps[step_id] = step

    def get_step(self, step_id: int) -> Optional[WorkflowStep]:
        """Get a step by ID"""
        return self._steps.get(step_id)

    def get_steps_by_phase(self, phase: WorkflowPhase) -> List[WorkflowStep]:
        """Get all steps for a specific phase"""
        return [step for step in self._steps.values() if step.phase == phase]

    def get_all_steps(self) -> List[WorkflowStep]:
        """Get all registered steps"""
        return list(self._steps.values())

    def get_step_count(self) -> int:
        """Get total number of registered steps"""
        return len(self._steps)

    def validate_dependencies(self) -> List[str]:
        """Validate step dependencies"""
        errors = []

        for step_id, step in self._steps.items():
            for dep_id in step.dependencies:
                if dep_id not in self._steps:
                    errors.append(f"Step {step_id} depends on non-existent step {dep_id}")
                elif dep_id >= step_id:
                    errors.append(f"Step {step_id} cannot depend on future step {dep_id}")

        return errors

    def get_execution_order(self) -> List[int]:
        """Get steps in execution order considering dependencies"""
        # Simple topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []

        def visit(step_id: int):
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving step {step_id}")
            if step_id in visited:
                return

            temp_visited.add(step_id)

            step = self._steps.get(step_id)
            if step:
                for dep in step.dependencies:
                    visit(dep)

            temp_visited.remove(step_id)
            visited.add(step_id)
            order.append(step_id)

        # Visit all steps
        for step_id in sorted(self._steps.keys()):
            if step_id not in visited:
                visit(step_id)

        return order

    def get_phase_progress(self, phase: WorkflowPhase, completed_steps: List[int]) -> Dict[str, Any]:
        """Get progress information for a phase"""
        phase_steps = self.get_steps_by_phase(phase)
        phase_step_ids = {step.step_id for step in phase_steps}
        completed_in_phase = phase_step_ids.intersection(set(completed_steps))

        return {
            "phase": phase.value,
            "total_steps": len(phase_steps),
            "completed_steps": len(completed_in_phase),
            "progress_percentage": (len(completed_in_phase) / len(phase_steps) * 100) if phase_steps else 0,
            "remaining_steps": len(phase_steps) - len(completed_in_phase)
        }

    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all phases"""
        summary = {}
        for phase in WorkflowPhase:
            phase_steps = self.get_steps_by_phase(phase)
            summary[phase.value] = {
                "step_range": self._get_phase_range(phase),
                "total_steps": len(phase_steps),
                "description": self._get_phase_description(phase)
            }

        return summary

    def _get_phase_range(self, phase: WorkflowPhase) -> str:
        """Get the step range for a phase"""
        ranges = {
            WorkflowPhase.ASSESSMENT: "1-100",
            WorkflowPhase.IDEATION: "101-200",
            WorkflowPhase.TESTING: "201-300",
            WorkflowPhase.FEEDBACK: "301-400",
            WorkflowPhase.SCALABILITY: "401-450",
            WorkflowPhase.OUTPUT: "451-500"
        }
        return ranges.get(phase, "unknown")

    def _get_phase_description(self, phase: WorkflowPhase) -> str:
        """Get description for a phase"""
        descriptions = {
            WorkflowPhase.ASSESSMENT: "Initial assessment and benchmarking of current REPZ performance",
            WorkflowPhase.IDEATION: "Idea generation and prototyping for enhancements",
            WorkflowPhase.TESTING: "Automated testing and deployment with rollback mechanisms",
            WorkflowPhase.FEEDBACK: "Continuous integration of user feedback and monitoring",
            WorkflowPhase.SCALABILITY: "Scalability optimizations for high-traffic scenarios",
            WorkflowPhase.OUTPUT: "Final output generation (PDFs, notebooks, spreadsheets, forms)"
        }
        return descriptions.get(phase, "Unknown phase")