"""
End-to-End Workflow Integration Tests for ORCHEX

Tests complete research workflows from input to publication-ready output.
"""

import pytest
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import redis
from datetime import datetime, timedelta

from atlas_core.engine import ATLASEngine
from atlas_core.agents import (
    LiteratureReviewAgent,
    HypothesisGeneratorAgent,
    ExperimentDesignerAgent,
    DataAnalysisAgent,
    SynthesisAgent,
    QualityCheckAgent,
    PublicationFormatterAgent,
    ResearchValidatorAgent,
    CitationManagerAgent,
)
from atlas_core.blackboard import ATLASBlackboard


class TestEndToEndWorkflow:
    """Test complete research workflows from start to finish."""

    @pytest.fixture
    def engine(self):
        """Create a test engine instance."""
        engine = ATLASEngine(redis_url="redis://localhost:6379/1", libria_enabled=True)
        return engine

    @pytest.fixture
    def test_workflow(self):
        """Create a test research workflow."""
        return {
            "id": str(uuid.uuid4()),
            "type": "research_publication",
            "topic": "AI Safety in Large Language Models",
            "steps": [
                {"agent": "literature_review", "task": "Review existing research"},
                {"agent": "hypothesis_generator", "task": "Generate hypotheses"},
                {"agent": "experiment_designer", "task": "Design experiments"},
                {"agent": "data_analysis", "task": "Analyze results"},
                {"agent": "synthesis", "task": "Synthesize findings"},
                {"agent": "quality_check", "task": "Quality assurance"},
                {"agent": "publication_formatter", "task": "Format for publication"},
            ],
            "constraints": {
                "max_duration": 3600,  # 1 hour
                "quality_threshold": 0.85,
                "citation_requirement": 50,
            },
        }

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_research_workflow(self, engine, test_workflow):
        """Test a complete research workflow from literature review to publication."""
        # Initialize workflow
        workflow_id = engine.create_workflow(test_workflow)
        assert workflow_id is not None

        # Start workflow execution
        result = await engine.execute_workflow(workflow_id)

        # Verify all steps completed
        assert result["status"] == "completed"
        assert len(result["completed_steps"]) == len(test_workflow["steps"])

        # Verify output quality
        quality_score = result["quality_metrics"]["overall_score"]
        assert quality_score >= test_workflow["constraints"]["quality_threshold"]

        # Verify citations
        citation_count = result["citations"]["count"]
        assert citation_count >= test_workflow["constraints"]["citation_requirement"]

        # Verify timing constraint
        duration = result["metrics"]["total_duration"]
        assert duration <= test_workflow["constraints"]["max_duration"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_dialectical_workflow(self, engine):
        """Test thesis-antithesis-synthesis dialectical workflow."""
        dialectical_config = {
            "topic": "Impact of AI on Employment",
            "type": "dialectical",
            "rounds": 3,
            "agents": {
                "thesis": "OptimisticAnalyst",
                "antithesis": "PessimisticCritic",
                "synthesis": "BalancedSynthesizer",
            },
        }

        # Execute dialectical analysis
        result = await engine.execute_dialectical_workflow(dialectical_config)

        # Verify dialectical rounds
        assert len(result["rounds"]) == 3
        for round_result in result["rounds"]:
            assert "thesis" in round_result
            assert "antithesis" in round_result
            assert "synthesis" in round_result
            assert round_result["synthesis"]["quality_score"] > 0.7

        # Verify final synthesis
        assert result["final_synthesis"]["balanced_score"] > 0.8
        assert len(result["final_synthesis"]["key_points"]) >= 5

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_agent_collaboration(self, engine):
        """Test collaboration between multiple agents."""
        collaboration_config = {
            "task": "Develop COVID-19 treatment protocol",
            "agents": [
                "medical_researcher",
                "epidemiologist",
                "clinical_trials_designer",
                "statistical_analyst",
                "regulatory_expert",
            ],
            "collaboration_mode": "parallel_with_sync",
            "sync_points": 3,
        }

        # Execute collaborative task
        result = await engine.execute_collaborative_task(collaboration_config)

        # Verify all agents participated
        assert len(result["agent_contributions"]) == len(collaboration_config["agents"])

        # Verify sync points were respected
        assert len(result["sync_history"]) == collaboration_config["sync_points"]

        # Verify consensus was reached
        assert result["consensus"]["achieved"] is True
        assert result["consensus"]["confidence"] > 0.75

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_adaptive_workflow_optimization(self, engine):
        """Test workflow optimization using Libria integration."""
        initial_workflow = {
            "task": "Climate change impact analysis",
            "agents": ["data_collector", "analyst", "modeler", "visualizer"],
            "optimization_enabled": True,
            "target_metrics": {
                "speed": 0.8,
                "quality": 0.9,
                "resource_efficiency": 0.7,
            },
        }

        # Execute with optimization
        result = await engine.execute_optimized_workflow(initial_workflow)

        # Verify optimization occurred
        assert result["optimization"]["applied"] is True
        assert result["optimization"]["improvements"]["speed"] > 0.1
        assert result["optimization"]["improvements"]["efficiency"] > 0.15

        # Verify adapted workflow is better
        assert result["final_metrics"]["speed"] >= initial_workflow["target_metrics"]["speed"]
        assert (
            result["final_metrics"]["quality"] >= initial_workflow["target_metrics"]["quality"]
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_failure_recovery(self, engine):
        """Test workflow recovery from agent failures."""
        workflow_with_failure = {
            "task": "Complex data analysis",
            "agents": ["data_loader", "faulty_processor", "analyzer", "reporter"],
            "failure_injection": {
                "agent": "faulty_processor",
                "failure_type": "timeout",
                "at_step": 2,
            },
            "recovery_strategy": "retry_with_alternative",
        }

        # Execute with expected failure
        result = await engine.execute_workflow_with_recovery(workflow_with_failure)

        # Verify recovery was triggered
        assert result["recovery"]["triggered"] is True
        assert result["recovery"]["strategy"] == "retry_with_alternative"
        assert result["recovery"]["success"] is True

        # Verify workflow completed despite failure
        assert result["status"] == "completed_with_recovery"
        assert len(result["completed_steps"]) == len(workflow_with_failure["agents"])

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_quality_gate_enforcement(self, engine):
        """Test quality gate enforcement throughout workflow."""
        quality_workflow = {
            "task": "Medical diagnosis protocol development",
            "quality_gates": [
                {"after_agent": "data_collector", "min_quality": 0.9, "metric": "completeness"},
                {"after_agent": "analyzer", "min_quality": 0.85, "metric": "accuracy"},
                {"after_agent": "validator", "min_quality": 0.95, "metric": "safety"},
            ],
            "abort_on_quality_failure": True,
        }

        # Test with passing quality gates
        result = await engine.execute_with_quality_gates(quality_workflow)
        assert result["all_gates_passed"] is True
        assert result["status"] == "completed"

        # Test with failing quality gate
        quality_workflow["quality_gates"][0]["min_quality"] = 0.99  # Unrealistic threshold
        with pytest.raises(QualityGateException):
            await engine.execute_with_quality_gates(quality_workflow)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_distributed_workflow_execution(self, engine):
        """Test distributed execution across multiple nodes."""
        distributed_config = {
            "task": "Large-scale genomic analysis",
            "distribution": {
                "strategy": "data_parallel",
                "nodes": 4,
                "partitioning": "automatic",
            },
            "agents_per_node": 3,
            "coordination": "central",
        }

        # Execute distributed workflow
        result = await engine.execute_distributed_workflow(distributed_config)

        # Verify distribution
        assert len(result["node_assignments"]) == distributed_config["distribution"]["nodes"]
        assert result["distribution"]["efficiency"] > 0.7

        # Verify coordination
        assert result["coordination"]["conflicts_resolved"] >= 0
        assert result["coordination"]["sync_successful"] is True

        # Verify results aggregation
        assert result["aggregation"]["complete"] is True
        assert result["final_result"]["partitions_merged"] == 4

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_time_monitoring_integration(self, engine):
        """Test real-time monitoring during workflow execution."""
        monitored_workflow = {
            "task": "Financial risk assessment",
            "monitoring": {
                "enabled": True,
                "metrics": ["latency", "throughput", "error_rate", "quality"],
                "interval": 1,  # seconds
                "alerts": [
                    {"metric": "latency", "threshold": 5.0, "action": "warn"},
                    {"metric": "error_rate", "threshold": 0.1, "action": "abort"},
                ],
            },
        }

        # Start monitored execution
        monitor_data = []

        async def monitor_callback(metrics):
            monitor_data.append(metrics)

        result = await engine.execute_with_monitoring(
            monitored_workflow, monitor_callback=monitor_callback
        )

        # Verify monitoring data collected
        assert len(monitor_data) > 0
        assert all("latency" in m for m in monitor_data)
        assert all("throughput" in m for m in monitor_data)

        # Verify alerts were processed
        if result.get("alerts_triggered"):
            assert result["alerts_triggered"][0]["action_taken"] in ["warned", "aborted"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_pipeline_integration(self, engine):
        """Test integration with data processing pipelines."""
        pipeline_workflow = {
            "type": "data_pipeline",
            "stages": [
                {"name": "ingestion", "source": "s3://bucket/data", "format": "parquet"},
                {"name": "validation", "rules": ["schema", "quality", "completeness"]},
                {"name": "transformation", "operations": ["normalize", "aggregate", "enrich"]},
                {"name": "analysis", "algorithms": ["statistical", "ml_inference"]},
                {"name": "output", "destination": "postgres://db/results", "format": "json"},
            ],
            "streaming": False,
            "checkpoint_enabled": True,
        }

        # Execute pipeline
        result = await engine.execute_data_pipeline(pipeline_workflow)

        # Verify all stages completed
        assert len(result["completed_stages"]) == len(pipeline_workflow["stages"])

        # Verify checkpointing worked
        assert result["checkpoints"]["created"] > 0
        assert result["checkpoints"]["recovery_possible"] is True

        # Verify data quality
        assert result["data_quality"]["score"] > 0.85
        assert result["data_quality"]["records_processed"] > 0
        assert result["data_quality"]["errors"] < result["data_quality"]["records_processed"] * 0.01

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_api_gateway_integration(self, engine):
        """Test integration with API gateway for external requests."""
        api_workflow = {
            "exposed_via_api": True,
            "endpoints": [
                {"path": "/research", "method": "POST", "workflow": "research_pipeline"},
                {"path": "/analyze", "method": "POST", "workflow": "analysis_pipeline"},
                {"path": "/status/{id}", "method": "GET", "workflow": "status_check"},
            ],
            "authentication": "jwt",
            "rate_limiting": {"requests_per_minute": 100, "burst": 20},
        }

        # Simulate API requests
        api_client = engine.get_api_client()

        # Test research endpoint
        research_response = await api_client.post(
            "/research", json={"topic": "Quantum computing applications"}
        )
        assert research_response.status_code == 202  # Accepted
        assert "workflow_id" in research_response.json()

        # Test status endpoint
        workflow_id = research_response.json()["workflow_id"]
        status_response = await api_client.get(f"/status/{workflow_id}")
        assert status_response.status_code == 200
        assert status_response.json()["status"] in ["pending", "running", "completed"]

        # Test rate limiting
        responses = []
        for _ in range(150):
            responses.append(await api_client.post("/analyze", json={"data": "test"}))

        rate_limited = sum(1 for r in responses if r.status_code == 429)
        assert rate_limited > 0  # Some requests should be rate limited