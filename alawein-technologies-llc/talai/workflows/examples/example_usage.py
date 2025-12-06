"""
Example usage of TalAI Intelligent Workflow Automation System

Demonstrates all major components working together.
"""

import asyncio
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.workflows import (
    # Engine
    WorkflowEngine,
    WorkflowDAG,
    WorkflowNode,
    WorkflowTemplate,
    ExecutionContext,

    # Scheduling
    SmartScheduler,
    Job,
    JobPriority,
    ResourceAllocator,
    BatchProcessor,

    # Integrations
    IntegrationHub,
    GitHubIntegration,
    SlackIntegration,
    EmailIntegration,
    WebhookManager,
    Webhook,

    # Pipelines
    DrugDiscoveryPipeline,
    FinancialValidationPipeline,
    ClimateSciencePipeline,
    AISafetyPipeline,
    PipelineMarketplace
)

from src.workflows.engine.workflow_dag import NodeType, NodeCondition
from src.workflows.scheduling import SchedulingPolicy, SchedulerConfig
from src.workflows.integrations import IntegrationConfig, IntegrationType


async def example_workflow_engine():
    """Example: Create and execute a workflow with DAG"""
    print("\n=== WORKFLOW ENGINE EXAMPLE ===")

    # Create workflow engine
    engine = WorkflowEngine()

    # Create DAG
    dag = WorkflowDAG(name="research_workflow")

    # Add nodes
    data_collection = WorkflowNode(
        id="collect_data",
        name="Collect Research Data",
        type=NodeType.TASK,
        inputs={"sources": ["pubmed", "arxiv", "github"]}
    )
    dag.add_node(data_collection)

    analysis = WorkflowNode(
        id="analyze",
        name="Analyze Data",
        type=NodeType.TASK,
        inputs={"method": "statistical"}
    )
    dag.add_node(analysis)

    validation = WorkflowNode(
        id="validate",
        name="Validate Results",
        type=NodeType.TASK,
        inputs={"threshold": 0.95}
    )
    dag.add_node(validation)

    # Add edges
    dag.add_edge("start", "collect_data")
    dag.add_edge("collect_data", "analyze")
    dag.add_edge("analyze", "validate")
    dag.add_edge("validate", "end")

    # Register workflow
    from src.workflows.engine.workflow_engine import WorkflowDefinition
    workflow = WorkflowDefinition(
        id="research_001",
        name="Research Pipeline",
        description="Automated research workflow",
        dag=dag,
        version="1.0.0",
        author="TalAI"
    )

    workflow_id = engine.register_workflow(workflow)
    print(f"Registered workflow: {workflow_id}")

    # Execute workflow
    result = await engine.execute_workflow(
        workflow_id,
        inputs={"query": "AI safety research"}
    )

    print(f"Workflow completed: {result.status}")
    print(f"Duration: {result.duration_seconds:.2f} seconds")


async def example_smart_scheduling():
    """Example: Smart job scheduling with resource optimization"""
    print("\n=== SMART SCHEDULING EXAMPLE ===")

    # Configure scheduler
    config = SchedulerConfig(
        policy=SchedulingPolicy.ADAPTIVE,
        max_concurrent_jobs=5,
        enable_batching=True,
        enable_cost_optimization=True,
        cost_budget=100.0
    )

    scheduler = SmartScheduler(config)
    await scheduler.start(num_workers=3)

    # Create jobs
    jobs = []
    for i in range(10):
        priority = JobPriority.HIGH if i < 3 else JobPriority.NORMAL
        job = Job(
            name=f"validation_job_{i}",
            priority=priority,
            inputs={"hypothesis_id": f"hyp_{i}"},
            resource_requirements={"cpu": 1, "memory": 2}
        )
        jobs.append(job)

    # Schedule jobs
    for job in jobs:
        job_id = await scheduler.schedule_job(job)
        print(f"Scheduled job: {job_id} (Priority: {job.priority.name})")

    # Wait for completion
    await asyncio.sleep(5)

    # Get metrics
    metrics = scheduler.get_metrics()
    print(f"Jobs completed: {metrics['jobs_completed']}")
    print(f"Average queue time: {metrics['avg_queue_time']:.2f}s")

    await scheduler.stop()


async def example_integration_hub():
    """Example: Integration with external services"""
    print("\n=== INTEGRATION HUB EXAMPLE ===")

    # Create integration hub
    hub = IntegrationHub()
    await hub.initialize()

    # Configure GitHub integration
    from src.workflows.integrations.github_integration import GitHubConfig
    github_config = GitHubConfig(
        token="github_token",
        owner="AlaweinOS",
        repo="TalAI"
    )
    github = GitHubIntegration(github_config)

    hub.register_integration(
        "github",
        github,
        IntegrationConfig(
            name="github",
            type=IntegrationType.GITHUB
        )
    )

    # Configure Slack integration
    from src.workflows.integrations.slack_integration import SlackConfig
    slack_config = SlackConfig(
        token="slack_token",
        channel="#workflow-notifications"
    )
    slack = SlackIntegration(slack_config)

    hub.register_integration(
        "slack",
        slack,
        IntegrationConfig(
            name="slack",
            type=IntegrationType.SLACK
        )
    )

    # Send event to all integrations
    await hub.send_event(
        "workflow_completed",
        {
            "workflow_id": "research_001",
            "status": "success",
            "duration": 120
        }
    )

    print(f"Active integrations: {hub.list_integrations()}")

    # Setup webhook
    webhook_manager = WebhookManager()
    await webhook_manager.initialize()

    webhook = Webhook(
        id="wh_001",
        url="https://example.com/webhook",
        events=["workflow_completed", "job_failed"]
    )
    webhook_manager.register_webhook(webhook)

    # Trigger webhook
    await webhook_manager.trigger_event(
        "workflow_completed",
        {"workflow_id": "research_001"}
    )

    await hub.cleanup()
    await webhook_manager.cleanup()


async def example_validation_pipelines():
    """Example: Domain-specific validation pipelines"""
    print("\n=== VALIDATION PIPELINES EXAMPLE ===")

    # Drug Discovery Pipeline
    print("\n--- Drug Discovery Validation ---")
    drug_pipeline = DrugDiscoveryPipeline()

    drug_data = {
        "smiles": "CC(C)Cc1ccc(cc1)C(C)C(O)=O",
        "molecular_weight": 206.28,
        "target_id": "COX-2",
        "binding_energy": -8.5,
        "admet_properties": {
            "absorption": 0.85,
            "bioavailability": 0.75,
            "half_life_hours": 6
        },
        "toxicity": {
            "hepatotoxicity": 0.1,
            "cardiotoxicity": 0.05,
            "mutagenicity": 0.02
        }
    }

    drug_result = await drug_pipeline.validate(drug_data)
    print(f"Drug validation status: {drug_result['status']}")
    print(f"Overall score: {drug_result['score']:.2f}")

    # Financial Validation Pipeline
    print("\n--- Financial Model Validation ---")
    financial_pipeline = FinancialValidationPipeline()

    financial_data = {
        "time_series": [0.02, 0.01, -0.03, 0.04, 0.02, -0.01, 0.03],
        "returns": [0.02, 0.01, -0.03, 0.04, 0.02, -0.01, 0.03],
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.15,
        "backtest_returns": [0.01] * 50,
        "win_rate": 0.62,
        "profit_factor": 1.8,
        "num_trades": 50,
        "leverage": 1.5
    }

    financial_result = await financial_pipeline.validate(financial_data)
    print(f"Financial validation status: {financial_result['status']}")
    print(f"Overall score: {financial_result['score']:.2f}")

    # AI Safety Pipeline
    print("\n--- AI Safety Assessment ---")
    safety_pipeline = AISafetyPipeline()

    safety_data = {
        "noise_tolerance": 0.85,
        "distribution_shift_score": 0.80,
        "edge_case_performance": 0.75,
        "demographic_parity": 0.90,
        "equal_opportunity": 0.88,
        "disparate_impact": 0.95,
        "protected_groups": ["age", "gender", "race"],
        "objective_alignment": 0.92,
        "value_alignment": 0.89,
        "harmful_output_rate": 0.001
    }

    safety_result = await safety_pipeline.validate(safety_data)
    print(f"AI Safety status: {safety_result['status']}")
    print(f"Overall score: {safety_result['score']:.2f}")

    # Pipeline Marketplace
    print("\n--- Pipeline Marketplace ---")
    marketplace = PipelineMarketplace()

    from src.workflows.pipelines.pipeline_marketplace import PipelineMetadata

    # Publish pipeline to marketplace
    metadata = PipelineMetadata(
        id="drug_disco_v2",
        name="Enhanced Drug Discovery",
        description="Advanced drug discovery validation with ML",
        author="TalAI Team",
        version="2.0.0",
        category="healthcare",
        tags=["drug-discovery", "ml", "validation"]
    )

    pipeline_id = marketplace.publish_pipeline(drug_pipeline, metadata)
    print(f"Published pipeline: {pipeline_id}")

    # Search marketplace
    results = marketplace.search_pipelines(category="healthcare")
    print(f"Found {len(results)} healthcare pipelines")


async def example_complete_workflow():
    """Example: Complete workflow with all components"""
    print("\n=== COMPLETE WORKFLOW EXAMPLE ===")

    # 1. Setup components
    engine = WorkflowEngine()
    scheduler = SmartScheduler()
    hub = IntegrationHub()
    marketplace = PipelineMarketplace()

    await scheduler.start(num_workers=2)
    await hub.initialize()

    # 2. Create research workflow
    dag = WorkflowDAG(name="ai_research_validation")

    # Add validation node
    validation_node = WorkflowNode(
        id="validate_hypothesis",
        name="Validate AI Hypothesis",
        type=NodeType.TASK
    )
    dag.add_node(validation_node)
    dag.add_edge("start", "validate_hypothesis")
    dag.add_edge("validate_hypothesis", "end")

    # 3. Schedule validation job
    job = Job(
        name="ai_hypothesis_validation",
        priority=JobPriority.HIGH,
        inputs={
            "hypothesis": "LLMs can achieve human-level reasoning",
            "method": "empirical_testing"
        }
    )

    job_id = await scheduler.schedule_job(job)
    print(f"Scheduled validation job: {job_id}")

    # 4. Run AI safety validation
    safety_pipeline = AISafetyPipeline()
    safety_result = await safety_pipeline.validate({
        "noise_tolerance": 0.9,
        "distribution_shift_score": 0.85,
        "edge_case_performance": 0.8,
        "demographic_parity": 0.95,
        "equal_opportunity": 0.93,
        "disparate_impact": 1.0
    })

    print(f"Safety validation: {safety_result['status']}")

    # 5. Send notifications
    await hub.send_event(
        "validation_complete",
        {
            "job_id": job_id,
            "safety_status": safety_result['status'],
            "timestamp": datetime.now().isoformat()
        }
    )

    # Cleanup
    await scheduler.stop()
    await hub.cleanup()

    print("\nWorkflow completed successfully!")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("TalAI INTELLIGENT WORKFLOW AUTOMATION EXAMPLES")
    print("=" * 60)

    # Run examples
    await example_workflow_engine()
    await example_smart_scheduling()
    await example_integration_hub()
    await example_validation_pipelines()
    await example_complete_workflow()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())