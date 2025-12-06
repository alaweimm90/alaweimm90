"""
ORCHEX Hypothesis Evaluation Platform

A comprehensive AI-powered platform for evaluating research hypotheses
through multiple evaluation modes including adversarial testing, cross-domain
collision, evolutionary simulation, multiverse analysis, and prediction markets.
"""

__version__ = "0.1.0"
__author__ = "ORCHEX Team"
__email__ = "meshal@berkeley.edu"

# Core components
# Advanced systems
from ORCHEX.advanced_systems import (
    AdaptiveScheduler,
    BootstrapCI,
    ComparativeAnalyzer,
    MetaEvaluator,
    MLTaskRouter,
    TenantQuotaManager,
)
from ORCHEX.pii_redactor import PIIPipeline
from ORCHEX.quality_gates import GateResult, GateStatus, QualityGates
from ORCHEX.results_store import (
    DistributedCache,
    DynamicConfig,
    PreSalesArtifacts,
    ResultsAPI,
    ResultsStore,
)
from ORCHEX.shadow_eval import BudgetMonitor, ShadowEvaluator
from ORCHEX.tracing_logger import ATLASTracer, StructuredLogger

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Quality gates
    "QualityGates",
    "GateStatus",
    "GateResult",
    # PII handling
    "PIIPipeline",
    # Tracing
    "ATLASTracer",
    "StructuredLogger",
    # Shadow evaluation
    "ShadowEvaluator",
    "BudgetMonitor",
    # Results storage
    "ResultsStore",
    "ResultsAPI",
    "DistributedCache",
    "DynamicConfig",
    "PreSalesArtifacts",
    # Advanced systems
    "MetaEvaluator",
    "BootstrapCI",
    "ComparativeAnalyzer",
    "AdaptiveScheduler",
    "MLTaskRouter",
    "TenantQuotaManager",
]
