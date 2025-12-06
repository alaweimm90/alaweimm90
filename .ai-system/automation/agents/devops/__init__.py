"""
DevOps Agents Module
Implementation of the 20 essential DevOps agents for CI/CD, deployment, and operations.
"""

from .types import (
    DevOpsAgentId,
    DevOpsCategory,
    DevOpsAgent,
    AgentExecutionContext,
    AgentExecutionResult,
    # Input/Output types
    PipelineOrchestratorInput,
    PipelineOrchestratorOutput,
    BuildInput,
    BuildOutput,
    TestRunnerInput,
    TestRunnerOutput,
    ContainerBuildInput,
    ContainerBuildOutput,
    ImageScanInput,
    ImageScanOutput,
    K8sDeployInput,
    K8sDeployOutput,
    ComplianceAuditInput,
    ComplianceAuditOutput,
)
from .registry import DevOpsAgentRegistry, load_devops_agents
from .executor import execute_devops_agent

__all__ = [
    # Types
    "DevOpsAgentId",
    "DevOpsCategory",
    "DevOpsAgent",
    "AgentExecutionContext",
    "AgentExecutionResult",
    # Input/Output types
    "PipelineOrchestratorInput",
    "PipelineOrchestratorOutput",
    "BuildInput",
    "BuildOutput",
    "TestRunnerInput",
    "TestRunnerOutput",
    "ContainerBuildInput",
    "ContainerBuildOutput",
    "ImageScanInput",
    "ImageScanOutput",
    "K8sDeployInput",
    "K8sDeployOutput",
    "ComplianceAuditInput",
    "ComplianceAuditOutput",
    # Registry
    "DevOpsAgentRegistry",
    "load_devops_agents",
    # Executor
    "execute_devops_agent",
]
