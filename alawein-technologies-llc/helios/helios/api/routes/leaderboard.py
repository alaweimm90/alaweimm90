"""
Leaderboard API Routes - Performance rankings and benchmarks

Provides endpoints for:
- Global algorithm rankings
- Domain-specific leaderboards
- Real-time performance metrics
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

# ============================================================================
# DATA MODELS
# ============================================================================

class LeaderboardEntry(BaseModel):
    """Leaderboard entry for an algorithm."""
    rank: int
    name: str
    domain: str
    quality: float
    speedup: float
    tests: int
    latest_score: float
    trend: str  # "up", "down", "stable"


class LeaderboardResponse(BaseModel):
    """Complete leaderboard response."""
    domain: Optional[str]
    total_entries: int
    entries: List[LeaderboardEntry]
    updated_at: datetime


# ============================================================================
# LEADERBOARD DATA
# ============================================================================

LEADERBOARD_DATA = [
    # Global top performers
    {"rank": 1, "name": "Transformer Arch", "domain": "ml", "quality": 91, "speedup": 2.5, "tests": 28, "latest": 0.91, "trend": "up"},
    {"rank": 2, "name": "Hybrid Metaheuristic", "domain": "optimization", "quality": 88, "speedup": 2.4, "tests": 25, "latest": 0.88, "trend": "stable"},
    {"rank": 3, "name": "VQE Hybrid", "domain": "quantum", "quality": 88, "speedup": 2.3, "tests": 15, "latest": 0.87, "trend": "up"},
    {"rank": 4, "name": "Attention Mechanism", "domain": "ml", "quality": 90, "speedup": 2.3, "tests": 26, "latest": 0.89, "trend": "stable"},
    {"rank": 5, "name": "Graph Generation", "domain": "synthesis", "quality": 86, "speedup": 2.1, "tests": 13, "latest": 0.85, "trend": "up"},
    {"rank": 6, "name": "ResNet Variant", "domain": "ml", "quality": 89, "speedup": 2.1, "tests": 22, "latest": 0.88, "trend": "down"},
    {"rank": 7, "name": "Structure GNN", "domain": "materials", "quality": 86, "speedup": 2.2, "tests": 16, "latest": 0.85, "trend": "stable"},
    {"rank": 8, "name": "Differentiable NAS", "domain": "nas", "quality": 87, "speedup": 2.2, "tests": 14, "latest": 0.86, "trend": "up"},
    {"rank": 9, "name": "Graph Neural Network", "domain": "graph", "quality": 87, "speedup": 2.0, "tests": 15, "latest": 0.86, "trend": "stable"},
    {"rank": 10, "name": "Stability Estimator", "domain": "materials", "quality": 84, "speedup": 2.0, "tests": 13, "latest": 0.83, "trend": "down"},
]

# Domain-specific leaderboards
DOMAIN_LEADERBOARDS = {
    "quantum": [
        {"rank": 1, "name": "VQE Hybrid", "domain": "quantum", "quality": 88, "speedup": 2.3, "tests": 15, "latest": 0.87, "trend": "up"},
        {"rank": 2, "name": "Grover's Algorithm", "domain": "quantum", "quality": 85, "speedup": 2.0, "tests": 12, "latest": 0.84, "trend": "stable"},
        {"rank": 3, "name": "Quantum Annealing", "domain": "quantum", "quality": 82, "speedup": 1.8, "tests": 10, "latest": 0.81, "trend": "down"},
    ],
    "materials": [
        {"rank": 1, "name": "Structure GNN", "domain": "materials", "quality": 86, "speedup": 2.2, "tests": 16, "latest": 0.85, "trend": "stable"},
        {"rank": 2, "name": "Stability Estimator", "domain": "materials", "quality": 84, "speedup": 2.0, "tests": 13, "latest": 0.83, "trend": "down"},
        {"rank": 3, "name": "Crystal Prediction", "domain": "materials", "quality": 81, "speedup": 1.9, "tests": 14, "latest": 0.80, "trend": "up"},
    ],
    "optimization": [
        {"rank": 1, "name": "Hybrid Metaheuristic", "domain": "optimization", "quality": 88, "speedup": 2.4, "tests": 25, "latest": 0.88, "trend": "stable"},
        {"rank": 2, "name": "Ant Colony Opt", "domain": "optimization", "quality": 80, "speedup": 1.7, "tests": 17, "latest": 0.79, "trend": "up"},
        {"rank": 3, "name": "Genetic Algorithm", "domain": "optimization", "quality": 79, "speedup": 1.6, "tests": 18, "latest": 0.78, "trend": "stable"},
    ],
    "ml": [
        {"rank": 1, "name": "Transformer Arch", "domain": "ml", "quality": 91, "speedup": 2.5, "tests": 28, "latest": 0.91, "trend": "up"},
        {"rank": 2, "name": "Attention Mechanism", "domain": "ml", "quality": 90, "speedup": 2.3, "tests": 26, "latest": 0.89, "trend": "stable"},
        {"rank": 3, "name": "ResNet Variant", "domain": "ml", "quality": 89, "speedup": 2.1, "tests": 22, "latest": 0.88, "trend": "down"},
    ],
    "nas": [
        {"rank": 1, "name": "Differentiable NAS", "domain": "nas", "quality": 87, "speedup": 2.2, "tests": 14, "latest": 0.86, "trend": "up"},
        {"rank": 2, "name": "DARTS", "domain": "nas", "quality": 85, "speedup": 1.9, "tests": 12, "latest": 0.84, "trend": "stable"},
    ],
    "synthesis": [
        {"rank": 1, "name": "Graph Generation", "domain": "synthesis", "quality": 86, "speedup": 2.1, "tests": 13, "latest": 0.85, "trend": "up"},
        {"rank": 2, "name": "Retrosynthesis", "domain": "synthesis", "quality": 82, "speedup": 1.8, "tests": 10, "latest": 0.81, "trend": "stable"},
    ],
    "graph": [
        {"rank": 1, "name": "Graph Neural Network", "domain": "graph", "quality": 87, "speedup": 2.0, "tests": 15, "latest": 0.86, "trend": "stable"},
        {"rank": 2, "name": "PageRank Variant", "domain": "graph", "quality": 81, "speedup": 1.7, "tests": 11, "latest": 0.80, "trend": "up"},
    ],
}


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.get("/leaderboard", response_model=LeaderboardResponse)
async def get_global_leaderboard(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("rank", description="Sort by: rank, quality, speedup")
) -> LeaderboardResponse:
    """
    Get global algorithm leaderboard.

    Args:
        limit: Number of results
        offset: Number of results to skip
        sort_by: Sort field (rank, quality, speedup)

    Returns:
        Global leaderboard entries
    """
    results = LEADERBOARD_DATA.copy()

    # Sort
    if sort_by == "quality":
        results = sorted(results, key=lambda x: x["quality"], reverse=True)
    elif sort_by == "speedup":
        results = sorted(results, key=lambda x: x["speedup"], reverse=True)
    # default is by rank

    # Paginate
    results = results[offset:offset + limit]

    entries = [
        LeaderboardEntry(
            rank=r["rank"],
            name=r["name"],
            domain=r["domain"],
            quality=r["quality"],
            speedup=r["speedup"],
            tests=r["tests"],
            latest_score=r["latest"],
            trend=r["trend"],
        )
        for r in results
    ]

    return LeaderboardResponse(
        domain=None,
        total_entries=len(LEADERBOARD_DATA),
        entries=entries,
        updated_at=datetime.utcnow(),
    )


@router.get("/leaderboard/{domain}", response_model=LeaderboardResponse)
async def get_domain_leaderboard(
    domain: str,
    limit: int = Query(20, ge=1, le=100),
) -> LeaderboardResponse:
    """
    Get domain-specific leaderboard.

    Args:
        domain: Domain identifier (quantum, materials, optimization, ml, nas, synthesis, graph)
        limit: Number of results

    Returns:
        Domain leaderboard entries
    """
    domain = domain.lower()

    if domain not in DOMAIN_LEADERBOARDS:
        # Return empty list for unknown domains
        return LeaderboardResponse(
            domain=domain,
            total_entries=0,
            entries=[],
            updated_at=datetime.utcnow(),
        )

    results = DOMAIN_LEADERBOARDS[domain][:limit]

    entries = [
        LeaderboardEntry(
            rank=r["rank"],
            name=r["name"],
            domain=r["domain"],
            quality=r["quality"],
            speedup=r["speedup"],
            tests=r["tests"],
            latest_score=r["latest"],
            trend=r["trend"],
        )
        for r in results
    ]

    return LeaderboardResponse(
        domain=domain,
        total_entries=len(DOMAIN_LEADERBOARDS.get(domain, [])),
        entries=entries,
        updated_at=datetime.utcnow(),
    )


@router.get("/leaderboard/stats/summary")
async def leaderboard_summary():
    """
    Get leaderboard summary statistics.

    Returns:
        Overall statistics across all domains
    """
    all_entries = LEADERBOARD_DATA

    return {
        "total_algorithms": len(all_entries),
        "average_quality": sum(e["quality"] for e in all_entries) / len(all_entries),
        "max_quality": max(e["quality"] for e in all_entries),
        "min_quality": min(e["quality"] for e in all_entries),
        "average_speedup": sum(e["speedup"] for e in all_entries) / len(all_entries),
        "total_tests": sum(e["tests"] for e in all_entries),
        "domain_count": len(DOMAIN_LEADERBOARDS),
        "trending_up": sum(1 for e in all_entries if e["trend"] == "up"),
        "trending_down": sum(1 for e in all_entries if e["trend"] == "down"),
        "stable": sum(1 for e in all_entries if e["trend"] == "stable"),
    }
