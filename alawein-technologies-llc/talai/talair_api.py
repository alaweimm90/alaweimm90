"""
TalAI Turing Challenge System - REST API Server

🎉 SURPRISE FEATURE: AI Research Orchestrator UI Backend

Provides a REST API for the web-based Research Orchestrator interface.
Users can submit hypotheses, run validations, and view results through
a beautiful web UI.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import uuid

from turing_challenge_system import TuringChallengeSystem, ValidationMode
from self_refutation import Hypothesis, HypothesisDomain

# Initialize FastAPI app
app = FastAPI(
    title="TalAI Research Orchestrator API",
    description="AI-powered research validation system with Nobel Prize-level rigor",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all origins for demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for validation jobs (use Redis in production)
validation_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize Turing Challenge System
turing_system = TuringChallengeSystem()


# =============================================================================
# Request/Response Models
# =============================================================================


class HypothesisRequest(BaseModel):
    """Request model for hypothesis submission."""

    claim: str = Field(..., description="The hypothesis claim to validate")
    domain: str = Field(..., description="Research domain (e.g., biology, physics)")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made")
    validation_mode: str = Field(default="standard", description="Validation mode")


class ValidationStatus(BaseModel):
    """Status of a validation job."""

    job_id: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    current_step: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response model for validation results."""

    job_id: str
    status: str
    hypothesis: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    overall_score: Optional[float] = None
    recommendation: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TalAI Research Orchestrator</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 {
                color: #667eea;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-align: center;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 40px;
                font-size: 1.1em;
            }
            .form-group {
                margin-bottom: 25px;
            }
            label {
                display: block;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
            }
            input, textarea, select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                min-height: 120px;
                resize: vertical;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 1.1em;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
            }
            .btn:active {
                transform: translateY(0);
            }
            .results {
                margin-top: 40px;
                padding: 30px;
                background: #f8f9fa;
                border-radius: 12px;
                display: none;
            }
            .results.show {
                display: block;
            }
            .score {
                font-size: 3em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
            }
            .score.high { color: #28a745; }
            .score.medium { color: #ffc107; }
            .score.low { color: #dc3545; }
            .recommendation {
                text-align: center;
                font-size: 1.5em;
                font-weight: 600;
                margin: 20px 0;
                padding: 15px;
                border-radius: 8px;
            }
            .recommendation.proceed { background: #d4edda; color: #155724; }
            .recommendation.revise { background: #fff3cd; color: #856404; }
            .recommendation.reject { background: #f8d7da; color: #721c24; }
            .loading {
                text-align: center;
                padding: 40px;
                display: none;
            }
            .loading.show {
                display: block;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 20px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-card {
                padding: 20px;
                background: white;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                text-align: center;
            }
            .feature-icon {
                font-size: 2em;
                margin-bottom: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 TalAI Research Orchestrator</h1>
            <p class="subtitle">Nobel Prize-level validation for your research hypotheses</p>

            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🏆</div>
                    <h3>Agent Tournaments</h3>
                    <p>Competitive selection</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">👿</div>
                    <h3>Devil's Advocate</h3>
                    <p>Adversarial testing</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🐝</div>
                    <h3>Swarm Voting</h3>
                    <p>Democratic consensus</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">✨</div>
                    <h3>Emergent Behavior</h3>
                    <p>Pattern detection</p>
                </div>
            </div>

            <form id="hypothesisForm">
                <div class="form-group">
                    <label>Hypothesis Claim *</label>
                    <textarea id="claim" required placeholder="Enter your hypothesis claim..." rows="4"></textarea>
                </div>

                <div class="form-group">
                    <label>Research Domain *</label>
                    <select id="domain" required>
                        <option value="">Select domain...</option>
                        <option value="biology">Biology</option>
                        <option value="physics">Physics</option>
                        <option value="chemistry">Chemistry</option>
                        <option value="computer_science">Computer Science</option>
                        <option value="mathematics">Mathematics</option>
                        <option value="optimization">Optimization</option>
                        <option value="machine_learning">Machine Learning</option>
                        <option value="quantum">Quantum Computing</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Evidence (one per line)</label>
                    <textarea id="evidence" placeholder="Enter supporting evidence..." rows="3"></textarea>
                </div>

                <div class="form-group">
                    <label>Assumptions (one per line)</label>
                    <textarea id="assumptions" placeholder="Enter assumptions..." rows="3"></textarea>
                </div>

                <div class="form-group">
                    <label>Validation Mode</label>
                    <select id="mode">
                        <option value="quick">Quick (1 min)</option>
                        <option value="standard" selected>Standard (5 min)</option>
                        <option value="comprehensive">Comprehensive (15 min)</option>
                        <option value="rigorous">Rigorous (30 min)</option>
                    </select>
                </div>

                <button type="submit" class="btn">🚀 Validate Hypothesis</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Running Turing Challenge validation...</p>
                <p id="currentStep">Initializing...</p>
            </div>

            <div class="results" id="results">
                <h2>Validation Results</h2>
                <div class="score" id="score">--</div>
                <div class="recommendation" id="recommendation">--</div>
                <div id="details"></div>
            </div>
        </div>

        <script>
            document.getElementById('hypothesisForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                // Get form data
                const claim = document.getElementById('claim').value;
                const domain = document.getElementById('domain').value;
                const evidence = document.getElementById('evidence').value.split('\\n').filter(e => e.trim());
                const assumptions = document.getElementById('assumptions').value.split('\\n').filter(a => a.trim());
                const mode = document.getElementById('mode').value;

                // Show loading
                document.getElementById('loading').classList.add('show');
                document.getElementById('results').classList.remove('show');

                // Submit hypothesis
                const response = await fetch('/api/validate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ claim, domain, evidence, assumptions, validation_mode: mode })
                });

                const data = await response.json();
                const jobId = data.job_id;

                // Poll for results
                const pollInterval = setInterval(async () => {
                    const statusResponse = await fetch(`/api/status/${jobId}`);
                    const status = await statusResponse.json();

                    document.getElementById('currentStep').textContent = status.current_step || 'Processing...';

                    if (status.status === 'completed') {
                        clearInterval(pollInterval);

                        const resultsResponse = await fetch(`/api/results/${jobId}`);
                        const results = await resultsResponse.json();

                        // Hide loading
                        document.getElementById('loading').classList.remove('show');

                        // Show results
                        const scoreDiv = document.getElementById('score');
                        const recDiv = document.getElementById('recommendation');

                        scoreDiv.textContent = results.overall_score.toFixed(1) + '/100';
                        scoreDiv.className = 'score';
                        if (results.overall_score >= 70) scoreDiv.classList.add('high');
                        else if (results.overall_score >= 50) scoreDiv.classList.add('medium');
                        else scoreDiv.classList.add('low');

                        recDiv.textContent = results.recommendation;
                        recDiv.className = 'recommendation';
                        if (results.recommendation === 'PROCEED') recDiv.classList.add('proceed');
                        else if (results.recommendation === 'REVISE') recDiv.classList.add('revise');
                        else recDiv.classList.add('reject');

                        document.getElementById('results').classList.add('show');
                    } else if (status.status === 'failed') {
                        clearInterval(pollInterval);
                        alert('Validation failed. Please try again.');
                        document.getElementById('loading').classList.remove('show');
                    }
                }, 2000);
            });
        </script>
    </body>
    </html>
    """


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "talair-api", "version": "1.0.0"}


@app.post("/api/validate")
async def validate_hypothesis(request: HypothesisRequest, background_tasks: BackgroundTasks):
    """Submit a hypothesis for validation."""
    job_id = str(uuid.uuid4())

    # Create hypothesis object
    hypothesis = Hypothesis(
        claim=request.claim,
        domain=HypothesisDomain(request.domain.upper()),
        evidence=request.evidence,
        assumptions=request.assumptions,
    )

    # Store job
    validation_jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "hypothesis": hypothesis,
        "mode": request.validation_mode,
        "created_at": datetime.utcnow(),
        "progress": 0.0,
        "current_step": None,
    }

    # Start validation in background
    background_tasks.add_task(run_validation, job_id)

    return {"job_id": job_id, "status": "pending"}


async def run_validation(job_id: str):
    """Run validation in background."""
    job = validation_jobs[job_id]

    try:
        job["status"] = "running"
        job["current_step"] = "Running Turing Challenge System"

        # Run validation
        result = await turing_system.validate_hypothesis_complete(
            hypothesis=job["hypothesis"],
            mode=ValidationMode(job["mode"].upper()),
        )

        # Store results
        job["status"] = "completed"
        job["results"] = result
        job["overall_score"] = result.overall_score
        job["recommendation"] = result.recommendation
        job["completed_at"] = datetime.utcnow()
        job["progress"] = 1.0

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get validation job status."""
    if job_id not in validation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = validation_jobs[job_id]

    return ValidationStatus(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        progress=job.get("progress", 0.0),
        current_step=job.get("current_step"),
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get validation results."""
    if job_id not in validation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = validation_jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Validation not yet completed")

    return ValidationResponse(
        job_id=job_id,
        status=job["status"],
        hypothesis=job["hypothesis"].dict(),
        results=job["results"].dict() if job.get("results") else None,
        overall_score=job.get("overall_score"),
        recommendation=job.get("recommendation"),
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
    )


@app.get("/api/jobs")
async def list_jobs():
    """List all validation jobs."""
    jobs = []
    for job_id, job in validation_jobs.items():
        jobs.append(
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job["created_at"],
                "hypothesis_claim": job["hypothesis"].claim[:100] + "...",
            }
        )
    return {"jobs": jobs}


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🎉 TalAI Research Orchestrator API Server")
    print("=" * 80)
    print()
    print("Starting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
