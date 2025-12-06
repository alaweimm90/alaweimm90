"""
ORCHEX Stage 3: Experimentation

Complete experimentation pipeline for validated hypotheses.
"""

__version__ = "0.2.0"

from ORCHEX.experimentation.experiment_designer import (
    ExperimentDesigner,
    ExperimentDesign,
    ExperimentParameter
)

from ORCHEX.experimentation.code_generator import (
    CodeGenerator,
    GeneratedCode
)

from ORCHEX.experimentation.sandbox_executor import (
    SandboxExecutor,
    ExecutionResult,
    BatchExecutor
)

__all__ = [
    "ExperimentDesigner",
    "ExperimentDesign",
    "ExperimentParameter",
    "CodeGenerator",
    "GeneratedCode",
    "SandboxExecutor",
    "ExecutionResult",
    "BatchExecutor",
]
