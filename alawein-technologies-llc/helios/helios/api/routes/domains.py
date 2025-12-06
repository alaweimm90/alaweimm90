"""
Domain API Routes - Research domain endpoints

Provides endpoints for:
- Listing all research domains
- Getting domain-specific details
- Domain statistics and benchmarks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

router = APIRouter()

# ============================================================================
# DATA MODELS
# ============================================================================

class DomainStats(BaseModel):
    """Domain statistics model."""
    name: str
    algorithms: int
    benchmarks: int
    recent_hypotheses: int


class DomainDetail(BaseModel):
    """Complete domain information."""
    id: str
    name: str
    description: str
    keywords: List[str]
    algorithms: int
    benchmarks: int
    recent_hypotheses: int
    learning_curve: List[float]
    state_of_art: float
    established_at: datetime


# ============================================================================
# DOMAIN DATA
# ============================================================================

DOMAINS_DATA = {
    "quantum": {
        "id": "quantum",
        "name": "Quantum Computing",
        "description": "Quantum algorithms and quantum error correction",
        "keywords": ["quantum", "gates", "circuits", "qubits", "entanglement"],
        "algorithms": 12,
        "benchmarks": 8,
        "recent_hypotheses": 45,
        "learning_curve": [0.45, 0.52, 0.61, 0.68, 0.73, 0.77, 0.80],
        "state_of_art": 0.85,
    },
    "materials": {
        "id": "materials",
        "name": "Materials Science",
        "description": "Crystal structure discovery and materials properties",
        "keywords": ["materials", "crystals", "structures", "properties", "synthesis"],
        "algorithms": 11,
        "benchmarks": 7,
        "recent_hypotheses": 38,
        "learning_curve": [0.52, 0.58, 0.65, 0.71, 0.75, 0.78, 0.81],
        "state_of_art": 0.82,
    },
    "optimization": {
        "id": "optimization",
        "name": "Combinatorial Optimization",
        "description": "QAP, TSP, and other combinatorial problems (includes Librex.QAP)",
        "keywords": ["optimization", "qap", "tsp", "combinatorial", "heuristics"],
        "algorithms": 18,
        "benchmarks": 12,
        "recent_hypotheses": 62,
        "learning_curve": [0.48, 0.55, 0.63, 0.70, 0.76, 0.81, 0.84],
        "state_of_art": 0.88,
    },
    "ml": {
        "id": "ml",
        "name": "Machine Learning",
        "description": "Neural network architectures and training algorithms",
        "keywords": ["ml", "neural", "networks", "training", "architectures"],
        "algorithms": 14,
        "benchmarks": 10,
        "recent_hypotheses": 78,
        "learning_curve": [0.55, 0.62, 0.69, 0.75, 0.80, 0.83, 0.85],
        "state_of_art": 0.89,
    },
    "nas": {
        "id": "nas",
        "name": "Neural Architecture Search",
        "description": "Automated architecture design and optimization",
        "keywords": ["nas", "architecture", "search", "automated", "design"],
        "algorithms": 9,
        "benchmarks": 6,
        "recent_hypotheses": 35,
        "learning_curve": [0.50, 0.57, 0.65, 0.72, 0.77, 0.81, 0.83],
        "state_of_art": 0.84,
    },
    "synthesis": {
        "id": "synthesis",
        "name": "Molecular Synthesis",
        "description": "Drug discovery and molecular synthesis pathways",
        "keywords": ["synthesis", "drugs", "molecules", "pathways", "discovery"],
        "algorithms": 8,
        "benchmarks": 5,
        "recent_hypotheses": 28,
        "learning_curve": [0.42, 0.49, 0.57, 0.65, 0.71, 0.76, 0.79],
        "state_of_art": 0.80,
    },
    "graph": {
        "id": "graph",
        "name": "Graph Optimization",
        "description": "Network analysis and graph algorithm optimization",
        "keywords": ["graph", "networks", "algorithms", "optimization", "analysis"],
        "algorithms": 10,
        "benchmarks": 7,
        "recent_hypotheses": 41,
        "learning_curve": [0.46, 0.53, 0.61, 0.68, 0.74, 0.78, 0.81],
        "state_of_art": 0.83,
    },
}


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/domains", response_model=List[DomainStats])
async def list_domains() -> List[DomainStats]:
    """
    List all available research domains.

    Returns:
        List of domain statistics
    """
    return [
        DomainStats(
            name=d["name"],
            algorithms=d["algorithms"],
            benchmarks=d["benchmarks"],
            recent_hypotheses=d["recent_hypotheses"],
        )
        for d in DOMAINS_DATA.values()
    ]


@router.get("/domains/{domain_id}", response_model=DomainDetail)
async def get_domain_details(domain_id: str) -> DomainDetail:
    """
    Get detailed information about a specific domain.

    Args:
        domain_id: Domain identifier (quantum, materials, optimization, ml, nas, synthesis, graph)

    Returns:
        Complete domain information

    Raises:
        HTTPException: If domain not found
    """
    domain_id = domain_id.lower()

    if domain_id not in DOMAINS_DATA:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

    data = DOMAINS_DATA[domain_id]
    return DomainDetail(
        id=data["id"],
        name=data["name"],
        description=data["description"],
        keywords=data["keywords"],
        algorithms=data["algorithms"],
        benchmarks=data["benchmarks"],
        recent_hypotheses=data["recent_hypotheses"],
        learning_curve=data["learning_curve"],
        state_of_art=data["state_of_art"],
        established_at=datetime(2025, 1, 1),
    )


@router.get("/domains/{domain_id}/stats", response_model=Dict[str, Any])
async def get_domain_stats(domain_id: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific domain.

    Args:
        domain_id: Domain identifier

    Returns:
        Domain statistics including learning curves and performance metrics
    """
    domain_id = domain_id.lower()

    if domain_id not in DOMAINS_DATA:
        raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

    data = DOMAINS_DATA[domain_id]
    return {
        "domain": data["name"],
        "algorithms_count": data["algorithms"],
        "benchmarks_count": data["benchmarks"],
        "recent_hypotheses_count": data["recent_hypotheses"],
        "learning_progress": {
            "weeks": list(range(1, 8)),
            "improvements": data["learning_curve"],
            "current": data["learning_curve"][-1],
            "target": 0.95,
        },
        "state_of_art": data["state_of_art"],
        "coverage": {
            "algorithms": f"{data['algorithms']}/20",
            "benchmarks": f"{data['benchmarks']}/15",
            "hypotheses": data["recent_hypotheses"],
        }
    }
