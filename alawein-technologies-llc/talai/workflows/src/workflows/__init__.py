"""
TalAI Intelligent Workflow Automation System

A sophisticated workflow orchestration platform with DAG-based execution,
smart scheduling, integration hub, and domain-specific validation pipelines.

Components:
- WorkflowEngine: DAG-based workflow execution with conditional branching
- SmartScheduler: Priority-based job scheduling with resource optimization
- IntegrationHub: REST clients and webhooks for external services
- ValidationPipelines: Pre-built pipelines for various domains

Author: TalAI Automation Team
Version: 1.0.0
"""

__version__ = "1.0.0"

from workflows.engine import (
    WorkflowEngine,
    WorkflowDAG,
    WorkflowNode,
    WorkflowExecutor,
    WorkflowTemplate,
    WorkflowVersion,
    ExecutionContext,
)

from workflows.scheduling import (
    SmartScheduler,
    JobQueue,
    PriorityScheduler,
    ResourceAllocator,
    BatchProcessor,
    CostOptimizer,
)

from workflows.integrations import (
    IntegrationHub,
    GitHubIntegration,
    SlackIntegration,
    EmailIntegration,
    WebhookManager,
    CustomIntegration,
)

from workflows.pipelines import (
    ValidationPipeline,
    DrugDiscoveryPipeline,
    FinancialValidationPipeline,
    ClimateSciencePipeline,
    AISafetyPipeline,
    PipelineMarketplace,
)

__all__ = [
    # Engine
    "WorkflowEngine",
    "WorkflowDAG",
    "WorkflowNode",
    "WorkflowExecutor",
    "WorkflowTemplate",
    "WorkflowVersion",
    "ExecutionContext",

    # Scheduling
    "SmartScheduler",
    "JobQueue",
    "PriorityScheduler",
    "ResourceAllocator",
    "BatchProcessor",
    "CostOptimizer",

    # Integrations
    "IntegrationHub",
    "GitHubIntegration",
    "SlackIntegration",
    "EmailIntegration",
    "WebhookManager",
    "CustomIntegration",

    # Pipelines
    "ValidationPipeline",
    "DrugDiscoveryPipeline",
    "FinancialValidationPipeline",
    "ClimateSciencePipeline",
    "AISafetyPipeline",
    "PipelineMarketplace",
]