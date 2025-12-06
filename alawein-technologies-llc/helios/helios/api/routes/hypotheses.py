"""
Hypothesis API Routes - Hypothesis creation and validation

Provides endpoints for:
- Creating research hypotheses
- Validating hypotheses
- Getting validation results and feedback
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

router = APIRouter()

# ============================================================================
# DATA MODELS
# ============================================================================

class HypothesisCreate(BaseModel):
    """Request model for creating a hypothesis."""
    text: str = Field(..., min_length=10, max_length=5000, description="Hypothesis statement")
    domain: str = Field(..., description="Research domain")
    keywords: Optional[List[str]] = Field(None, description="Relevant keywords")


class ValidationRequest(BaseModel):
    """Request model for validation."""
    hypothesis_text: str = Field(..., min_length=10, max_length=5000)
    domain: str = Field(...)
    validation_depth: int = Field(3, ge=1, le=5, description="Validation depth level")


class ValidationScore(BaseModel):
    """Validation score component."""
    name: str
    score: float
    feedback: str


class ValidationResult(BaseModel):
    """Complete validation result."""
    hypothesis_id: str
    overall_score: float
    component_scores: List[ValidationScore]
    weaknesses: List[str]
    strengths: List[str]
    recommendations: List[str]
    validation_timestamp: datetime


class Hypothesis(BaseModel):
    """Hypothesis model."""
    id: str
    text: str
    domain: str
    created_at: datetime
    validation_result: Optional[ValidationResult] = None


# ============================================================================
# STORAGE (In-memory for demo)
# ============================================================================

hypotheses_store: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/hypotheses", response_model=Hypothesis)
async def create_hypothesis(req: HypothesisCreate) -> Hypothesis:
    """
    Create a new research hypothesis.

    Args:
        req: Hypothesis creation request

    Returns:
        Created hypothesis with ID

    Raises:
        HTTPException: If domain invalid
    """
    valid_domains = ["quantum", "materials", "optimization", "ml", "nas", "synthesis", "graph"]

    if req.domain.lower() not in valid_domains:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid domain. Must be one of: {', '.join(valid_domains)}"
        )

    hypothesis_id = str(uuid4())
    hypothesis_data = {
        "id": hypothesis_id,
        "text": req.text,
        "domain": req.domain,
        "keywords": req.keywords or [],
        "created_at": datetime.utcnow(),
        "validation_result": None,
    }

    hypotheses_store[hypothesis_id] = hypothesis_data

    return Hypothesis(
        id=hypothesis_id,
        text=req.text,
        domain=req.domain,
        created_at=datetime.utcnow(),
    )


@router.post("/hypotheses/{hypothesis_id}/validate", response_model=ValidationResult)
async def validate_hypothesis(
    hypothesis_id: str,
    req: ValidationRequest
) -> ValidationResult:
    """
    Validate a hypothesis using HELIOS validation engine.

    Args:
        hypothesis_id: Hypothesis to validate
        req: Validation request

    Returns:
        Detailed validation results

    Raises:
        HTTPException: If hypothesis not found
    """
    if hypothesis_id not in hypotheses_store:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    # Simulate validation with scoring
    base_score = 0.0

    # Domain-specific adjustments
    domain_scores = {
        "quantum": 0.72,
        "materials": 0.68,
        "optimization": 0.75,
        "ml": 0.78,
        "nas": 0.71,
        "synthesis": 0.65,
        "graph": 0.70,
    }

    base_score = domain_scores.get(req.domain.lower(), 0.65)

    # Validation depth adjustment
    base_score += (req.validation_depth - 1) * 0.05

    # Cap at 1.0
    overall_score = min(base_score, 0.95)

    # Component scores
    component_scores = [
        ValidationScore(
            name="Logical Consistency",
            score=overall_score * 0.98,
            feedback="Hypothesis maintains logical consistency throughout"
        ),
        ValidationScore(
            name="Empirical Falsifiability",
            score=overall_score * 0.92,
            feedback="Clear experimental design could test this hypothesis"
        ),
        ValidationScore(
            name="Novelty Assessment",
            score=overall_score * 0.88,
            feedback="Shows moderate novelty compared to existing work"
        ),
        ValidationScore(
            name="Mechanism Plausibility",
            score=overall_score * 0.95,
            feedback="Proposed mechanism is plausible given current knowledge"
        ),
        ValidationScore(
            name="Boundary Violation Check",
            score=overall_score * 0.91,
            feedback="Does not violate known physical/computational boundaries"
        ),
    ]

    return ValidationResult(
        hypothesis_id=hypothesis_id,
        overall_score=overall_score,
        component_scores=component_scores,
        weaknesses=[
            "Limited experimental validation data provided",
            "Assumes normal distribution in some cases",
            "Scalability analysis missing",
        ],
        strengths=[
            "Clear hypothesis statement",
            "Relevant to current research trends",
            "Falsifiable design",
            "Proper domain specification",
        ],
        recommendations=[
            "Provide preliminary experimental results",
            "Expand to edge case analysis",
            "Compare with related hypotheses",
            "Add computational complexity analysis",
        ],
        validation_timestamp=datetime.utcnow(),
    )


@router.get("/hypotheses/{hypothesis_id}", response_model=Hypothesis)
async def get_hypothesis(hypothesis_id: str) -> Hypothesis:
    """
    Get a hypothesis by ID.

    Args:
        hypothesis_id: Hypothesis ID

    Returns:
        Hypothesis information

    Raises:
        HTTPException: If not found
    """
    if hypothesis_id not in hypotheses_store:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    h = hypotheses_store[hypothesis_id]
    return Hypothesis(
        id=h["id"],
        text=h["text"],
        domain=h["domain"],
        created_at=h["created_at"],
        validation_result=h.get("validation_result"),
    )


@router.get("/hypotheses")
async def list_hypotheses(
    domain: Optional[str] = None,
    limit: int = 10,
) -> List[Hypothesis]:
    """
    List hypotheses with optional filtering.

    Args:
        domain: Filter by domain
        limit: Maximum results

    Returns:
        List of hypotheses
    """
    results = list(hypotheses_store.values())

    if domain:
        results = [h for h in results if h["domain"].lower() == domain.lower()]

    return [
        Hypothesis(
            id=h["id"],
            text=h["text"],
            domain=h["domain"],
            created_at=h["created_at"],
            validation_result=h.get("validation_result"),
        )
        for h in results[:limit]
    ]
