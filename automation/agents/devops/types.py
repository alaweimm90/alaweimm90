"""
DevOps Agent Type Definitions
Type definitions for the 20 essential DevOps agents.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


# ============================================================================
# Base Types
# ============================================================================


class DevOpsAgentId(str, Enum):
    """DevOps agent identifiers."""

    PIPELINE_ORCHESTRATOR = "pipeline-orchestrator"
    BUILD = "build"
    TEST_RUNNER = "test-runner"
    ARTIFACT_REPO = "artifact-repo"
    CONTAINER_BUILD = "container-build"
    IMAGE_SCAN = "image-scan"
    SECRETS = "secrets"
    INFRA_PROVISIONER = "infra-provisioner"
    CONFIG_MANAGER = "config-manager"
    K8S_DEPLOY = "k8s-deploy"
    PROGRESSIVE_DELIVERY = "progressive-delivery"
    ROLLBACK = "rollback"
    METRICS = "metrics"
    LOG_SHIPPER = "log-shipper"
    ALERT_ROUTER = "alert-router"
    TRIAGE = "triage"
    RELEASE_MANAGER = "release-manager"
    FEATURE_FLAGS = "feature-flags"
    COST_MONITOR = "cost-monitor"
    COMPLIANCE_AUDIT = "compliance-audit"


class DevOpsCategory(str, Enum):
    """DevOps agent categories."""

    PIPELINE = "pipeline"
    CONTAINER = "container"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    DEPLOYMENT = "deployment"
    OBSERVABILITY = "observability"
    RELEASE = "release"
    FINOPS = "finops"


@dataclass
class LLMConfig:
    """LLM configuration for an agent."""

    model: str = "claude-3-sonnet"
    temperature: float = 0.2
    max_tokens: Optional[int] = None


@dataclass
class DevOpsAgent:
    """DevOps agent definition."""

    id: DevOpsAgentId
    role: str
    goal: str
    backstory: str
    aliases: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    category: DevOpsCategory = DevOpsCategory.PIPELINE
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    version_compat: Dict[str, str] = field(default_factory=dict)
    security: List[str] = field(default_factory=list)


@dataclass
class AgentExecutionContext:
    """Context for agent execution."""

    agent_id: DevOpsAgentId
    input: Any
    env: Dict[str, str] = field(default_factory=dict)
    timeout: Optional[int] = None
    retries: int = 1


@dataclass
class AgentExecutionResult:
    """Result of agent execution."""

    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0
    retries: int = 0


# ============================================================================
# Agent Input/Output Types
# ============================================================================


# 1) Pipeline Orchestrator
class PipelineOrchestratorInput(TypedDict, total=False):
    """Input for Pipeline Orchestrator agent."""

    pipelineFile: str
    env: str
    secretsRef: str
    parameters: Dict[str, Any]


class ArtifactRef(TypedDict):
    """Artifact reference."""

    id: str
    url: str
    checksum: str


class PipelineOrchestratorOutput(TypedDict):
    """Output from Pipeline Orchestrator agent."""

    status: str  # 'success' | 'failure' | 'cancelled'
    stageLogs: Dict[str, str]
    artifactsIndex: List[ArtifactRef]


# 2) Build Agent
class BuildInput(TypedDict, total=False):
    """Input for Build agent."""

    sourcePath: str
    buildSpec: str
    cacheKey: str
    outputDir: str


class BuildOutput(TypedDict):
    """Output from Build agent."""

    success: bool
    buildLogs: str
    artifactIds: List[str]
    duration_ms: int


# 3) Test Runner
class CoverageReport(TypedDict):
    """Code coverage report."""

    lines: float
    branches: float
    functions: float
    statements: float


class TestRunnerInput(TypedDict, total=False):
    """Input for Test Runner agent."""

    testSpec: str
    shard: str  # 'auto' or number
    env: str
    coverage: bool
    reportDir: str


class TestRunnerOutput(TypedDict):
    """Output from Test Runner agent."""

    passed: int
    failed: int
    skipped: int
    junitXml: str
    coverageReport: Optional[CoverageReport]
    flakyTests: List[str]


# 4) Artifact Repository
class ArtifactRepoInput(TypedDict, total=False):
    """Input for Artifact Repository agent."""

    action: str  # 'push' | 'pull'
    artifactPath: str
    repoRef: str
    metadata: Dict[str, str]


class ArtifactRepoOutput(TypedDict):
    """Output from Artifact Repository agent."""

    artifactUrl: str
    checksum: str
    size: int


# 5) Container Builder
class ContainerBuildInput(TypedDict, total=False):
    """Input for Container Builder agent."""

    dockerfile: str
    context: str
    tags: List[str]
    buildArgs: Dict[str, str]
    platform: str


class Package(TypedDict, total=False):
    """SBOM package entry."""

    name: str
    version: str
    license: str


class SBOM(TypedDict):
    """Software Bill of Materials."""

    format: str  # 'spdx' | 'cyclonedx'
    packages: List[Package]


class LayerInfo(TypedDict):
    """Container image layer info."""

    digest: str
    size: int
    command: str


class ContainerBuildOutput(TypedDict):
    """Output from Container Builder agent."""

    imageDigest: str
    sbom: SBOM
    layers: List[LayerInfo]
    size: int


# 6) Image Scanner
class ScanPolicy(TypedDict, total=False):
    """Vulnerability scan policy."""

    maxSeverity: str  # 'critical' | 'high' | 'medium' | 'low'
    allowlist: List[str]
    failOnFix: bool


class ImageScanInput(TypedDict, total=False):
    """Input for Image Scanner agent."""

    imageRef: str
    policy: ScanPolicy


class Vulnerability(TypedDict, total=False):
    """Vulnerability finding."""

    id: str
    severity: str  # 'critical' | 'high' | 'medium' | 'low'
    package: str
    version: str
    fixedIn: str
    description: str


class ImageScanOutput(TypedDict):
    """Output from Image Scanner agent."""

    passed: bool
    vulnerabilities: List[Vulnerability]
    remediationHints: List[str]
    sbomLinked: bool


# 7) Secrets Management
class SecretsInput(TypedDict, total=False):
    """Input for Secrets agent."""

    secretsRef: str
    bindings: Dict[str, str]
    ttl: int


class SecretsOutput(TypedDict, total=False):
    """Output from Secrets agent."""

    envFile: str
    rotationStatus: str  # 'current' | 'rotated' | 'pending'
    expiresAt: str


# 8) Infrastructure Provisioner
class InfraProvisionerInput(TypedDict, total=False):
    """Input for Infrastructure Provisioner agent."""

    iacPath: str
    workspace: str
    action: str  # 'plan' | 'apply' | 'destroy'
    varFile: str
    vars: Dict[str, Any]


class InfraProvisionerOutput(TypedDict, total=False):
    """Output from Infrastructure Provisioner agent."""

    planDiff: str
    applyStatus: str  # 'success' | 'failure' | 'no_changes'
    resourcesCreated: int
    resourcesUpdated: int
    resourcesDestroyed: int


# 9) Configuration Manager
class ConfigManagerInput(TypedDict, total=False):
    """Input for Configuration Manager agent."""

    playbook: str
    inventory: str
    vars: Dict[str, Any]
    tags: List[str]
    limit: str


class ChangeItem(TypedDict):
    """Configuration change item."""

    host: str
    task: str
    status: str  # 'changed' | 'ok' | 'failed' | 'skipped'


class ComplianceItem(TypedDict, total=False):
    """Compliance check item."""

    rule: str
    status: str  # 'pass' | 'fail'
    details: str


class ConfigManagerOutput(TypedDict):
    """Output from Configuration Manager agent."""

    changeSet: List[ChangeItem]
    complianceReport: List[ComplianceItem]
    hostsSucceeded: int
    hostsFailed: int


# 10) Kubernetes Deploy
class K8sDeployInput(TypedDict, total=False):
    """Input for Kubernetes Deploy agent."""

    clusterRef: str
    manifests: str
    chartPath: str
    valuesFile: str
    namespace: str
    dryRun: bool


class K8sResource(TypedDict):
    """Kubernetes resource."""

    kind: str
    name: str
    namespace: str
    status: str


class K8sDeployOutput(TypedDict):
    """Output from Kubernetes Deploy agent."""

    rolloutStatus: str  # 'complete' | 'in_progress' | 'failed'
    deployedResources: List[K8sResource]
    kubectlLogs: str


# 11) Progressive Delivery
class HealthChecks(TypedDict, total=False):
    """Health check configuration."""

    maxErrorRate: float
    minSuccessRate: float
    latencyP99: int


class ProgressiveDeliveryInput(TypedDict, total=False):
    """Input for Progressive Delivery agent."""

    serviceRef: str
    strategy: str  # 'canary' | 'blue-green' | 'rolling'
    weights: List[int]
    checks: HealthChecks


class DeliveryMetrics(TypedDict):
    """Delivery metrics."""

    errorRate: float
    successRate: float
    latencyP99: int


class ProgressiveDeliveryOutput(TypedDict):
    """Output from Progressive Delivery agent."""

    status: str  # 'promoted' | 'rolled_back' | 'in_progress'
    currentWeight: int
    metrics: DeliveryMetrics


# 12) Rollback
class RollbackInput(TypedDict, total=False):
    """Input for Rollback agent."""

    deploymentId: str
    rollbackPlan: str
    confirm: bool
    monitors: List[str]


class RollbackOutput(TypedDict, total=False):
    """Output from Rollback agent."""

    status: str  # 'success' | 'failure' | 'skipped'
    rollbackedTo: str
    incidentLink: str


# 20) Compliance Audit
class ComplianceAuditInput(TypedDict, total=False):
    """Input for Compliance Audit agent."""

    policiesDir: str
    scope: str
    exceptionsFile: str


class PolicyViolation(TypedDict):
    """Policy violation."""

    policy: str
    resource: str
    severity: str
    message: str


class ComplianceAuditOutput(TypedDict):
    """Output from Compliance Audit agent."""

    passed: bool
    violations: List[PolicyViolation]
    evidenceArtifacts: List[str]
