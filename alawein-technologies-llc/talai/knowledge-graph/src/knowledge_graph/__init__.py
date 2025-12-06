"""
Collaborative Knowledge Graph for Scientific Research

A distributed knowledge graph system for extracting, storing, and reasoning over
scientific concepts with multi-institution collaboration support.
"""

from .extraction import (
    EntityExtractor,
    RelationExtractor,
    TripleExtractor,
    PaperProcessor,
)
from .embeddings import (
    TransEEmbedding,
    RotatEEmbedding,
    ComplExEmbedding,
    EmbeddingTrainer,
)
from .reasoning import (
    KnowledgeReasoner,
    PathFinder,
    LinkPredictor,
    ConflictResolver,
)
from .query import (
    SPARQLEngine,
    CypherEngine,
    QueryOptimizer,
    FederatedQuerier,
)
from .versioning import (
    KnowledgeVersionControl,
    ConflictMerger,
    ProvenanceTracker,
    EvolutionAnalyzer,
)
from .visualization import (
    GraphVisualizer,
    InteractiveExplorer,
    CollaborativeEditor,
    DashboardGenerator,
)

__version__ = "1.0.0"
__all__ = [
    "EntityExtractor",
    "RelationExtractor",
    "TripleExtractor",
    "PaperProcessor",
    "TransEEmbedding",
    "RotatEEmbedding",
    "ComplExEmbedding",
    "EmbeddingTrainer",
    "KnowledgeReasoner",
    "PathFinder",
    "LinkPredictor",
    "ConflictResolver",
    "SPARQLEngine",
    "CypherEngine",
    "QueryOptimizer",
    "FederatedQuerier",
    "KnowledgeVersionControl",
    "ConflictMerger",
    "ProvenanceTracker",
    "EvolutionAnalyzer",
    "GraphVisualizer",
    "InteractiveExplorer",
    "CollaborativeEditor",
    "DashboardGenerator",
]