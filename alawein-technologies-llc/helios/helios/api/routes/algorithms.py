"""
Algorithm API Routes - Algorithm browsing and filtering

Provides endpoints for:
- Listing all algorithms
- Filtering by domain, type, performance
- Getting algorithm details and metrics
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

router = APIRouter()

# ============================================================================
# DATA MODELS
# ============================================================================

class AlgorithmType(str, Enum):
    """Algorithm classification type."""
    BASELINE = "baseline"
    NOVEL = "novel"
    ENSEMBLE = "ensemble"


class AlgorithmInfo(BaseModel):
    """Algorithm information model."""
    id: str
    name: str
    domain: str
    type: AlgorithmType
    quality: float
    speedup: float
    tests: int


class AlgorithmDetail(BaseModel):
    """Complete algorithm information."""
    id: str
    name: str
    domain: str
    type: AlgorithmType
    description: str
    authors: List[str]
    quality: float
    speedup: float
    tests: int
    papers: List[str]
    complexity: str
    implementation: str


# ============================================================================
# ALGORITHM DATA
# ============================================================================

ALGORITHMS_DATA = [
    # Quantum algorithms
    {"id": "q1", "name": "Grover's Algorithm", "domain": "quantum", "type": "baseline", "quality": 85, "speedup": 2.0, "tests": 12},
    {"id": "q2", "name": "VQE Hybrid", "domain": "quantum", "type": "novel", "quality": 88, "speedup": 2.3, "tests": 15},
    {"id": "q3", "name": "Quantum Annealing", "domain": "quantum", "type": "baseline", "quality": 82, "speedup": 1.8, "tests": 10},

    # Materials algorithms
    {"id": "m1", "name": "Crystal Prediction", "domain": "materials", "type": "baseline", "quality": 81, "speedup": 1.9, "tests": 14},
    {"id": "m2", "name": "Structure GNN", "domain": "materials", "type": "novel", "quality": 86, "speedup": 2.2, "tests": 16},
    {"id": "m3", "name": "Stability Estimator", "domain": "materials", "type": "novel", "quality": 84, "speedup": 2.0, "tests": 13},

    # Optimization algorithms
    {"id": "o1", "name": "Simulated Annealing", "domain": "optimization", "type": "baseline", "quality": 78, "speedup": 1.5, "tests": 20},
    {"id": "o2", "name": "Genetic Algorithm", "domain": "optimization", "type": "baseline", "quality": 79, "speedup": 1.6, "tests": 18},
    {"id": "o3", "name": "Hybrid Metaheuristic", "domain": "optimization", "type": "novel", "quality": 88, "speedup": 2.4, "tests": 25},
    {"id": "o4", "name": "Ant Colony Opt", "domain": "optimization", "type": "baseline", "quality": 80, "speedup": 1.7, "tests": 17},

    # ML algorithms
    {"id": "ml1", "name": "ResNet Variant", "domain": "ml", "type": "baseline", "quality": 89, "speedup": 2.1, "tests": 22},
    {"id": "ml2", "name": "Transformer Arch", "domain": "ml", "type": "novel", "quality": 91, "speedup": 2.5, "tests": 28},
    {"id": "ml3", "name": "Attention Mechanism", "domain": "ml", "type": "novel", "quality": 90, "speedup": 2.3, "tests": 26},

    # NAS algorithms
    {"id": "nas1", "name": "DARTS", "domain": "nas", "type": "baseline", "quality": 85, "speedup": 1.9, "tests": 12},
    {"id": "nas2", "name": "Differentiable NAS", "domain": "nas", "type": "novel", "quality": 87, "speedup": 2.2, "tests": 14},

    # Synthesis algorithms
    {"id": "s1", "name": "Retrosynthesis", "domain": "synthesis", "type": "baseline", "quality": 82, "speedup": 1.8, "tests": 10},
    {"id": "s2", "name": "Graph Generation", "domain": "synthesis", "type": "novel", "quality": 86, "speedup": 2.1, "tests": 13},

    # Graph algorithms
    {"id": "g1", "name": "Graph Neural Network", "domain": "graph", "type": "novel", "quality": 87, "speedup": 2.0, "tests": 15},
    {"id": "g2", "name": "PageRank Variant", "domain": "graph", "type": "baseline", "quality": 81, "speedup": 1.7, "tests": 11},
]


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/algorithms", response_model=List[AlgorithmInfo])
async def list_algorithms(
    domain: Optional[str] = Query(None, description="Filter by domain"),
    type: Optional[str] = Query(None, description="Filter by type (baseline, novel, ensemble)"),
    min_quality: Optional[float] = Query(None, description="Minimum quality score"),
    sort_by: str = Query("quality", description="Sort by field"),
    limit: int = Query(100, ge=1, le=1000),
) -> List[AlgorithmInfo]:
    """
    List all algorithms with optional filtering and sorting.

    Args:
        domain: Filter by domain (quantum, materials, optimization, ml, nas, synthesis, graph)
        type: Filter by algorithm type (baseline, novel, ensemble)
        min_quality: Minimum quality score
        sort_by: Field to sort by (quality, speedup, tests)
        limit: Maximum number of results

    Returns:
        List of algorithms matching filters
    """
    results = ALGORITHMS_DATA.copy()

    # Apply filters
    if domain:
        results = [a for a in results if a["domain"].lower() == domain.lower()]

    if type:
        results = [a for a in results if a["type"] == type.lower()]

    if min_quality is not None:
        results = [a for a in results if a["quality"] >= min_quality]

    # Sort
    sort_fields = {"quality": -1, "speedup": -1, "tests": -1, "name": 1}
    if sort_by in sort_fields:
        reverse = sort_fields[sort_by] == -1
        results = sorted(results, key=lambda x: x[sort_by], reverse=reverse)

    # Limit
    results = results[:limit]

    return [
        AlgorithmInfo(
            id=a["id"],
            name=a["name"],
            domain=a["domain"],
            type=a["type"],
            quality=a["quality"],
            speedup=a["speedup"],
            tests=a["tests"],
        )
        for a in results
    ]


@router.get("/algorithms/{algorithm_id}", response_model=AlgorithmDetail)
async def get_algorithm_details(algorithm_id: str) -> AlgorithmDetail:
    """
    Get detailed information about a specific algorithm.

    Args:
        algorithm_id: Algorithm identifier

    Returns:
        Complete algorithm information

    Raises:
        HTTPException: If algorithm not found
    """
    algorithm = next((a for a in ALGORITHMS_DATA if a["id"] == algorithm_id), None)

    if not algorithm:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm_id}' not found")

    return AlgorithmDetail(
        id=algorithm["id"],
        name=algorithm["name"],
        domain=algorithm["domain"],
        type=algorithm["type"],
        description=f"{algorithm['name']} - Advanced algorithm for {algorithm['domain']} domain",
        authors=["HELIOS Team"],
        quality=algorithm["quality"],
        speedup=algorithm["speedup"],
        tests=algorithm["tests"],
        papers=[f"paper_{algorithm['id']}_1.pdf", f"paper_{algorithm['id']}_2.pdf"],
        complexity=f"O(n^{2 + algorithm['quality']/50}) - O(n^3)",
        implementation="Python + Cython optimizations",
    )


@router.get("/algorithms/search/{query}")
async def search_algorithms(query: str, limit: int = 10) -> List[AlgorithmInfo]:
    """
    Search algorithms by name or description.

    Args:
        query: Search query string
        limit: Maximum results

    Returns:
        Matching algorithms
    """
    query = query.lower()
    results = [
        a for a in ALGORITHMS_DATA
        if query in a["name"].lower() or query in a["domain"].lower()
    ]

    return [
        AlgorithmInfo(
            id=a["id"],
            name=a["name"],
            domain=a["domain"],
            type=a["type"],
            quality=a["quality"],
            speedup=a["speedup"],
            tests=a["tests"],
        )
        for a in results[:limit]
    ]
