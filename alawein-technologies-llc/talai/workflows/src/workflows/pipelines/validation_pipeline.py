"""
Validation Pipeline - Base framework for validation pipelines

Provides core validation pipeline functionality for domain-specific implementations.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import logging


logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of validation"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationStep:
    """Single validation step"""
    name: str
    validator: Callable
    required: bool = True
    timeout: Optional[float] = None
    retry_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation"""
    step_name: str
    status: ValidationStatus
    score: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class ValidationPipeline:
    """Base validation pipeline"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: List[ValidationStep] = []
        self.results: List[ValidationResult] = []
        self.metadata: Dict[str, Any] = {}
        self.pre_processors: List[Callable] = []
        self.post_processors: List[Callable] = []

    def add_step(self, step: ValidationStep):
        """Add validation step to pipeline"""
        self.steps.append(step)
        logger.info(f"Added step {step.name} to pipeline {self.name}")

    def add_preprocessor(self, processor: Callable):
        """Add preprocessor to run before validation"""
        self.pre_processors.append(processor)

    def add_postprocessor(self, processor: Callable):
        """Add postprocessor to run after validation"""
        self.post_processors.append(processor)

    async def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation pipeline"""
        logger.info(f"Starting validation pipeline: {self.name}")
        self.results = []

        # Run preprocessors
        for processor in self.pre_processors:
            if asyncio.iscoroutinefunction(processor):
                data = await processor(data)
            else:
                data = processor(data)

        # Run validation steps
        overall_status = ValidationStatus.PASSED
        scores = []

        for step in self.steps:
            result = await self._run_step(step, data)
            self.results.append(result)

            # Update overall status
            if result.status == ValidationStatus.FAILED:
                if step.required:
                    overall_status = ValidationStatus.FAILED
                    logger.warning(f"Required step {step.name} failed")
                else:
                    if overall_status != ValidationStatus.FAILED:
                        overall_status = ValidationStatus.WARNING

            if result.score is not None:
                scores.append(result.score)

        # Calculate overall score
        overall_score = sum(scores) / len(scores) if scores else None

        # Run postprocessors
        validation_output = {
            "pipeline": self.name,
            "status": overall_status.value,
            "score": overall_score,
            "results": [self._result_to_dict(r) for r in self.results],
            "timestamp": datetime.now().isoformat()
        }

        for processor in self.post_processors:
            if asyncio.iscoroutinefunction(processor):
                validation_output = await processor(validation_output)
            else:
                validation_output = processor(validation_output)

        logger.info(f"Validation pipeline {self.name} completed: {overall_status.value}")
        return validation_output

    async def _run_step(self, step: ValidationStep,
                       data: Dict[str, Any]) -> ValidationResult:
        """Run single validation step"""
        logger.info(f"Running validation step: {step.name}")

        for attempt in range(step.retry_count):
            try:
                # Run validator with timeout
                if asyncio.iscoroutinefunction(step.validator):
                    if step.timeout:
                        result = await asyncio.wait_for(
                            step.validator(data),
                            timeout=step.timeout
                        )
                    else:
                        result = await step.validator(data)
                else:
                    # Run sync validator
                    result = step.validator(data)

                # Process result
                if isinstance(result, ValidationResult):
                    return result
                elif isinstance(result, bool):
                    return ValidationResult(
                        step_name=step.name,
                        status=ValidationStatus.PASSED if result else ValidationStatus.FAILED
                    )
                elif isinstance(result, dict):
                    return ValidationResult(
                        step_name=step.name,
                        status=ValidationStatus(result.get("status", "passed")),
                        score=result.get("score"),
                        message=result.get("message"),
                        details=result.get("details", {})
                    )
                else:
                    return ValidationResult(
                        step_name=step.name,
                        status=ValidationStatus.PASSED,
                        details={"result": result}
                    )

            except asyncio.TimeoutError:
                logger.error(f"Step {step.name} timed out (attempt {attempt + 1})")
                if attempt == step.retry_count - 1:
                    return ValidationResult(
                        step_name=step.name,
                        status=ValidationStatus.FAILED,
                        message="Validation timed out"
                    )

            except Exception as e:
                logger.error(f"Step {step.name} failed: {e} (attempt {attempt + 1})")
                if attempt == step.retry_count - 1:
                    return ValidationResult(
                        step_name=step.name,
                        status=ValidationStatus.FAILED,
                        message=str(e)
                    )

            # Wait before retry
            if attempt < step.retry_count - 1:
                await asyncio.sleep(1)

        # Should not reach here
        return ValidationResult(
            step_name=step.name,
            status=ValidationStatus.FAILED,
            message="Unknown error"
        )

    def _result_to_dict(self, result: ValidationResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "step": result.step_name,
            "status": result.status.value,
            "score": result.score,
            "message": result.message,
            "details": result.details,
            "timestamp": result.timestamp.isoformat()
        }

    def compose(self, other_pipeline: 'ValidationPipeline') -> 'ValidationPipeline':
        """Compose two pipelines into one"""
        composed = ValidationPipeline(
            name=f"{self.name}_{other_pipeline.name}",
            description=f"Composed: {self.description} + {other_pipeline.description}"
        )

        # Add all steps from both pipelines
        composed.steps.extend(self.steps)
        composed.steps.extend(other_pipeline.steps)

        # Add all processors
        composed.pre_processors.extend(self.pre_processors)
        composed.pre_processors.extend(other_pipeline.pre_processors)
        composed.post_processors.extend(self.post_processors)
        composed.post_processors.extend(other_pipeline.post_processors)

        return composed

    def to_dict(self) -> Dict[str, Any]:
        """Export pipeline configuration"""
        return {
            "name": self.name,
            "description": self.description,
            "steps": [
                {
                    "name": step.name,
                    "required": step.required,
                    "timeout": step.timeout,
                    "retry_count": step.retry_count,
                    "metadata": step.metadata
                }
                for step in self.steps
            ],
            "metadata": self.metadata
        }