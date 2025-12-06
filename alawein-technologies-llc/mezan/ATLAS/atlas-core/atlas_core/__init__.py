"""
MEZAN / ORCHEX - Multi-Agent AI Research & Optimization Orchestration System

MEZAN = Meta-Equilibrium Zero-regret Assignment Network

Provides:
- MEZAN dual-solver balancing engine
- DeepThink 3+1 agent system (optimized: 3 parallel + 1 sequential deep)
- Intelligent solver selection with deep reasoning
- 40+ ORCHEX research agents
- Dialectical workflows
- Integration with Libria optimization solvers
- Production REST API server (FastAPI)
- Parallel execution system (multiprocessing)
- Performance monitoring & telemetry (Prometheus-compatible)
- ML-based solver recommendation
- Real-time dashboard

Version 3.5.0: Production-Grade Enterprise Features
- REST API for remote optimization
- True parallel execution with process pools
- Comprehensive telemetry and monitoring
- ML-based intelligent solver selection
- Real-time metrics dashboard
- Enterprise observability
"""

__version__ = "3.5.0"  # Updated for Production Features
__author__ = "MEZAN Research Team"

# ORCHEX core components
from atlas_core.agent import ResearchAgent, AgentConfig
from atlas_core.engine import ATLASEngine
from atlas_core.blackboard import ATLASBlackboard

# MEZAN engine components
from atlas_core.mezan_engine import (
    MezanEngine,
    BaseSolver,
    SolverConfig,
    SolverResult,
    SolverType,
    MezanState,
)

# Ultrathink parallel agents
from atlas_core.ultrathink_agents import (
    UltrathinkOrchestrator,
    AgentTask,
    AgentResult,
    TeamRole,
    TaskPriority,
    TaskStatus,
)

# MEZAN orchestrator
from atlas_core.mezan_orchestrator import (
    MezanOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    OrchestrationResult,
    create_default_orchestrator,
)

# DeepThink agents (optimized: 3 parallel + 1 sequential)
from atlas_core.deepthink_agents import (
    DeepThinkOrchestrator,
    DeepTask,
    DeepResult,
    AnalysisDepth,
    AgentRole,
)

# Intelligent MEZAN (with deep reasoning)
from atlas_core.intelligent_mezan import (
    IntelligentMezanEngine,
    IntelligentSolverPair,
    create_intelligent_mezan,
)

# Causal Reasoning Engine (opus-level intelligence)
from atlas_core.causal_engine import (
    CausalReasoningEngine,
    CausalNode,
    CausalEdge,
    create_causal_engine,
)

# Real Libria Solver Integrations
from atlas_core.libria_solvers import (
    LibriaSolverResult,
    QAPFlowSolver,
    AllocFlowSolver,
    WorkFlowSolver,
    EvoFlowSolver,
    GraphFlowSolver,
    DualFlowSolver,
    MetaFlowSolver,
    create_libria_solver,
    get_available_solvers,
    get_solver_info,
)

# Production Features (V3.5+)
try:
    from atlas_core.parallel_executor import ParallelExecutor, TaskPriority
    from atlas_core.telemetry import (
        MetricsCollector,
        Counter,
        Gauge,
        Histogram,
        ResourceMonitor,
        AlertManager,
        AlertRule,
    )
    from atlas_core.ml_recommender import (
        SolverRecommender,
        FeatureExtractor,
        ProblemFeatures,
    )
    from atlas_core.ml_integration import (
        ATLASMLRecommender,
        AdaptiveSolverSelector,
    )
    PRODUCTION_FEATURES_AVAILABLE = True
except ImportError as e:
    PRODUCTION_FEATURES_AVAILABLE = False
    import warnings
    warnings.warn(f"Production features not available: {e}")

__all__ = [
    # Version info
    "__version__",
    # ORCHEX core
    "ResearchAgent",
    "AgentConfig",
    "ATLASEngine",
    "ATLASBlackboard",
    # MEZAN engine
    "MezanEngine",
    "BaseSolver",
    "SolverConfig",
    "SolverResult",
    "SolverType",
    "MezanState",
    # Ultrathink agents (legacy - 5 teams)
    "UltrathinkOrchestrator",
    "AgentTask",
    "AgentResult",
    "TeamRole",
    "TaskPriority",
    "TaskStatus",
    # DeepThink agents (optimized - 3 parallel + 1 sequential)
    "DeepThinkOrchestrator",
    "DeepTask",
    "DeepResult",
    "AnalysisDepth",
    "AgentRole",
    # Intelligent MEZAN (with deep reasoning)
    "IntelligentMezanEngine",
    "IntelligentSolverPair",
    "create_intelligent_mezan",
    # Causal Reasoning Engine (opus-level)
    "CausalReasoningEngine",
    "CausalNode",
    "CausalEdge",
    "create_causal_engine",
    # Real Libria Solvers
    "LibriaSolverResult",
    "QAPFlowSolver",
    "AllocFlowSolver",
    "WorkFlowSolver",
    "EvoFlowSolver",
    "GraphFlowSolver",
    "DualFlowSolver",
    "MetaFlowSolver",
    "create_libria_solver",
    "get_available_solvers",
    "get_solver_info",
    # MEZAN orchestrator
    "MezanOrchestrator",
    "OrchestratorConfig",
    "OrchestratorMode",
    "OrchestrationResult",
    "create_default_orchestrator",
]

# Conditionally add production features to exports
if PRODUCTION_FEATURES_AVAILABLE:
    __all__.extend([
        # Parallel execution
        "ParallelExecutor",
        "TaskPriority",
        # Telemetry
        "MetricsCollector",
        "Counter",
        "Gauge",
        "Histogram",
        "ResourceMonitor",
        "AlertManager",
        "AlertRule",
        # ML Recommendation
        "SolverRecommender",
        "FeatureExtractor",
        "ProblemFeatures",
        "ATLASMLRecommender",
        "AdaptiveSolverSelector",
        # Feature flag
        "PRODUCTION_FEATURES_AVAILABLE",
    ])
