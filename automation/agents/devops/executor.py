"""
DevOps Agent Executor
Execute DevOps agents with retry logic and error handling.
"""

import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from .registry import get_registry
from .types import (
    AgentExecutionContext,
    AgentExecutionResult,
    BuildInput,
    BuildOutput,
    ComplianceAuditInput,
    ComplianceAuditOutput,
    ContainerBuildInput,
    ContainerBuildOutput,
    DevOpsAgentId,
    ImageScanInput,
    ImageScanOutput,
    K8sDeployInput,
    K8sDeployOutput,
    PipelineOrchestratorInput,
    PipelineOrchestratorOutput,
    TestRunnerInput,
    TestRunnerOutput,
)

T = TypeVar("T")


async def execute_devops_agent(ctx: AgentExecutionContext) -> AgentExecutionResult:
    """
    Execute a DevOps agent with the given context.

    Args:
        ctx: Execution context containing agent ID, input, and options.

    Returns:
        AgentExecutionResult with success status, output, and metrics.
    """
    start_time = time.time()
    retries = 0
    max_retries = ctx.retries

    registry = get_registry()
    agent = registry.get(ctx.agent_id)

    if agent is None:
        return AgentExecutionResult(
            success=False,
            error=f"Agent not found: {ctx.agent_id}",
            duration_ms=int((time.time() - start_time) * 1000),
            retries=0,
        )

    while retries < max_retries:
        try:
            result = await _run_agent(ctx.agent_id, ctx.input, ctx.env)
            return AgentExecutionResult(
                success=True,
                output=result,
                duration_ms=int((time.time() - start_time) * 1000),
                retries=retries,
            )
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                return AgentExecutionResult(
                    success=False,
                    error=str(e),
                    duration_ms=int((time.time() - start_time) * 1000),
                    retries=retries,
                )

    return AgentExecutionResult(
        success=False,
        error="Max retries exceeded",
        duration_ms=int((time.time() - start_time) * 1000),
        retries=retries,
    )


async def _run_agent(
    agent_id: DevOpsAgentId,
    input_data: Any,
    env: Dict[str, str],
) -> Any:
    """Route to specific agent executor."""
    executors: Dict[DevOpsAgentId, Callable] = {
        DevOpsAgentId.PIPELINE_ORCHESTRATOR: _execute_pipeline_orchestrator,
        DevOpsAgentId.BUILD: _execute_build,
        DevOpsAgentId.TEST_RUNNER: _execute_test_runner,
        DevOpsAgentId.CONTAINER_BUILD: _execute_container_build,
        DevOpsAgentId.IMAGE_SCAN: _execute_image_scan,
        DevOpsAgentId.K8S_DEPLOY: _execute_k8s_deploy,
        DevOpsAgentId.COMPLIANCE_AUDIT: _execute_compliance_audit,
    }

    executor = executors.get(agent_id, _execute_generic)
    return await executor(input_data, env)


# ============================================================================
# Specific Agent Implementations
# ============================================================================


async def _execute_pipeline_orchestrator(
    input_data: PipelineOrchestratorInput,
    env: Dict[str, str],
) -> PipelineOrchestratorOutput:
    """Execute Pipeline Orchestrator agent."""
    pipeline_file = input_data.get("pipelineFile", "")

    if not os.path.exists(pipeline_file):
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_file}")

    # In a real implementation, this would:
    # 1. Parse the pipeline YAML
    # 2. Resolve stage dependencies (DAG)
    # 3. Execute stages in order with parallelization
    # 4. Handle retries and approvals

    return {
        "status": "success",
        "stageLogs": {
            "checkout": "Checked out source code",
            "build": "Build completed successfully",
            "test": "All tests passed",
        },
        "artifactsIndex": [],
    }


async def _execute_build(
    input_data: BuildInput,
    env: Dict[str, str],
) -> BuildOutput:
    """Execute Build agent."""
    start_time = time.time()
    source_path = input_data.get("sourcePath", ".")

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source path not found: {source_path}")

    # In a real implementation, this would:
    # 1. Detect build system (npm, maven, gradle, etc.)
    # 2. Restore cache if cacheKey provided
    # 3. Execute build command
    # 4. Publish artifacts

    return {
        "success": True,
        "buildLogs": "Build completed successfully",
        "artifactIds": ["artifact-001"],
        "duration_ms": int((time.time() - start_time) * 1000),
    }


async def _execute_test_runner(
    input_data: TestRunnerInput,
    env: Dict[str, str],
) -> TestRunnerOutput:
    """Execute Test Runner agent."""
    test_spec = input_data.get("testSpec", "")

    if not os.path.exists(test_spec):
        raise FileNotFoundError(f"Test spec not found: {test_spec}")

    # In a real implementation, this would:
    # 1. Detect test framework
    # 2. Configure sharding if specified
    # 3. Execute tests
    # 4. Collect coverage if enabled

    coverage_report = None
    if input_data.get("coverage"):
        coverage_report = {
            "lines": 85.5,
            "branches": 78.2,
            "functions": 90.1,
            "statements": 84.3,
        }

    return {
        "passed": 42,
        "failed": 0,
        "skipped": 2,
        "junitXml": "<testsuites></testsuites>",
        "coverageReport": coverage_report,
        "flakyTests": [],
    }


async def _execute_container_build(
    input_data: ContainerBuildInput,
    env: Dict[str, str],
) -> ContainerBuildOutput:
    """Execute Container Build agent."""
    dockerfile = input_data.get("dockerfile", "Dockerfile")

    if not os.path.exists(dockerfile):
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile}")

    # In a real implementation, this would:
    # 1. Run docker build or buildx
    # 2. Generate SBOM
    # 3. Push to registry if tags include registry

    return {
        "imageDigest": "sha256:abc123...",
        "sbom": {
            "format": "cyclonedx",
            "packages": [
                {"name": "base-image", "version": "alpine:3.19"},
                {"name": "nodejs", "version": "20.10.0"},
            ],
        },
        "layers": [
            {"digest": "sha256:layer1", "size": 5000000, "command": "FROM alpine:3.19"},
            {"digest": "sha256:layer2", "size": 50000000, "command": "RUN npm install"},
        ],
        "size": 55000000,
    }


async def _execute_image_scan(
    input_data: ImageScanInput,
    env: Dict[str, str],
) -> ImageScanOutput:
    """Execute Image Scan agent."""
    # In a real implementation, this would:
    # 1. Pull image if not local
    # 2. Run Trivy or Grype
    # 3. Apply policy rules
    # 4. Generate remediation hints

    policy = input_data.get("policy", {})
    max_severity = policy.get("maxSeverity", "high")

    mock_vulns = [
        {
            "id": "CVE-2024-1234",
            "severity": "medium",
            "package": "openssl",
            "version": "1.1.1",
            "fixedIn": "1.1.2",
            "description": "Buffer overflow vulnerability",
        }
    ]

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    has_blocking = any(
        severity_order.get(v["severity"], 3) <= severity_order.get(max_severity, 1)
        for v in mock_vulns
    )

    return {
        "passed": not has_blocking,
        "vulnerabilities": mock_vulns,
        "remediationHints": ["Upgrade openssl to 1.1.2 or later"],
        "sbomLinked": True,
    }


async def _execute_k8s_deploy(
    input_data: K8sDeployInput,
    env: Dict[str, str],
) -> K8sDeployOutput:
    """Execute Kubernetes Deploy agent."""
    # In a real implementation, this would:
    # 1. Connect to cluster using kubeconfig
    # 2. Apply manifests or install Helm chart
    # 3. Wait for rollout
    # 4. Validate health probes

    namespace = input_data.get("namespace", "default")

    return {
        "rolloutStatus": "complete",
        "deployedResources": [
            {
                "kind": "Deployment",
                "name": "app",
                "namespace": namespace,
                "status": "Running",
            },
            {
                "kind": "Service",
                "name": "app-svc",
                "namespace": namespace,
                "status": "Active",
            },
        ],
        "kubectlLogs": "deployment.apps/app successfully rolled out",
    }


async def _execute_compliance_audit(
    input_data: ComplianceAuditInput,
    env: Dict[str, str],
) -> ComplianceAuditOutput:
    """Execute Compliance Audit agent."""
    policies_dir = input_data.get("policiesDir", "")

    if not os.path.exists(policies_dir):
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    # In a real implementation, this would:
    # 1. Load OPA/Conftest policies
    # 2. Scan target scope
    # 3. Generate evidence artifacts

    return {
        "passed": True,
        "violations": [],
        "evidenceArtifacts": [".atlas/evidence/compliance-report.json"],
    }


async def _execute_generic(
    input_data: Any,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Generic executor for agents without specific implementation."""
    return {
        "status": "executed",
        "message": "Generic execution completed",
        "input": input_data,
    }
