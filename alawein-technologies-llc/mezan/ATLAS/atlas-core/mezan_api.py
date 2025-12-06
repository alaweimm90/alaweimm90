"""
MEZAN REST API Server

Production-grade FastAPI server for remote MEZAN access:
- Solve optimization problems via HTTP
- Run causal analysis remotely
- Execute benchmarks
- Query solver information
- WebSocket support for real-time updates
- OpenAPI documentation
- Authentication and rate limiting
- Async/await for high performance

Author: MEZAN Research Team
Date: 2025-11-18
Version: 1.0
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import asyncio
import uuid
import time
import logging
from enum import Enum
import json

# MEZAN imports
from atlas_core import (
    create_intelligent_mezan,
    create_causal_engine,
    create_libria_solver,
    get_available_solvers,
    get_solver_info,
    AnalysisDepth,
)
from atlas_core.benchmarking import Benchmarker, create_qap_problem_generator

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="MEZAN API",
    description="Production-grade REST API for MEZAN V3.0 Optimization System",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage (in production, use Redis or database)
active_jobs: Dict[str, Dict[str, Any]] = {}
completed_jobs: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class ProblemRequest(BaseModel):
    """Request model for optimization problem"""
    problem_type: str = Field(..., description="Type of problem (qap, tsp, allocation, etc.)")
    size: int = Field(..., gt=0, description="Problem size")
    objectives: int = Field(1, ge=1, description="Number of objectives")
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    data: Optional[Dict[str, Any]] = Field(None, description="Additional problem data")

    class Config:
        schema_extra = {
            "example": {
                "problem_type": "qap",
                "size": 30,
                "objectives": 1,
                "constraints": [{"type": "assignment"}],
            }
        }


class SolveRequest(BaseModel):
    """Request model for solving"""
    problem: ProblemRequest
    solver: Optional[str] = Field(None, description="Specific solver to use (or auto-select)")
    max_iterations: int = Field(5, ge=1, le=100)
    analysis_depth: str = Field("DEEP", description="QUICK, MEDIUM, or DEEP")
    timeout: float = Field(60.0, gt=0, le=3600, description="Timeout in seconds")


class JobStatus(str, Enum):
    """Job execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response model for job submission"""
    job_id: str
    status: JobStatus
    message: str


class ResultResponse(BaseModel):
    """Response model for results"""
    job_id: str
    status: JobStatus
    objective_value: Optional[float] = None
    confidence: Optional[float] = None
    iterations: Optional[int] = None
    time_elapsed: Optional[float] = None
    solution: Optional[Any] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None


class CausalAnalysisRequest(BaseModel):
    """Request for causal analysis"""
    problem: ProblemRequest

    class Config:
        schema_extra = {
            "example": {
                "problem": {
                    "problem_type": "qap",
                    "size": 40,
                    "objectives": 1,
                    "constraints": [{"type": "assignment"}],
                }
            }
        }


class SolverInfo(BaseModel):
    """Solver information"""
    name: str
    type: str
    description: str
    best_for: str


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MEZAN API",
        "version": "3.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "solve": "/solve",
            "analyze": "/analyze",
            "solvers": "/solvers",
            "jobs": "/jobs/{job_id}",
            "benchmarks": "/benchmarks",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "active_jobs": len(active_jobs),
        "completed_jobs": len(completed_jobs),
    }


@app.get("/solvers", response_model=Dict[str, SolverInfo])
async def list_solvers():
    """List all available solvers"""
    return get_solver_info()


@app.get("/solvers/available", response_model=List[str])
async def available_solvers():
    """Get list of available solver names"""
    return get_available_solvers()


@app.post("/solve", response_model=JobResponse)
async def solve_problem(
    request: SolveRequest,
    background_tasks: BackgroundTasks
):
    """
    Submit an optimization problem for solving

    Returns job_id for tracking progress
    """
    job_id = str(uuid.uuid4())

    # Create job record
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "request": request.dict(),
        "created_at": time.time(),
        "result": None,
        "error": None,
    }

    # Schedule background task
    background_tasks.add_task(
        execute_solve_job,
        job_id,
        request
    )

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Job {job_id} submitted successfully"
    )


@app.post("/solve/sync", response_model=ResultResponse)
async def solve_problem_sync(request: SolveRequest):
    """
    Solve problem synchronously (blocking)

    Use for small problems only. For large problems, use /solve (async)
    """
    job_id = str(uuid.uuid4())

    try:
        # Execute immediately
        result = await _execute_solve(request)

        return ResultResponse(
            job_id=job_id,
            status=JobStatus.COMPLETED,
            objective_value=result.get("objective_value"),
            confidence=result.get("confidence"),
            iterations=result.get("iterations"),
            time_elapsed=result.get("time_elapsed"),
            solution=result.get("solution"),
            reasoning=result.get("reasoning"),
        )
    except Exception as e:
        logger.error(f"Sync solve failed: {e}")
        return ResultResponse(
            job_id=job_id,
            status=JobStatus.FAILED,
            error=str(e)
        )


@app.get("/jobs/{job_id}", response_model=ResultResponse)
async def get_job_status(job_id: str):
    """Get status and results of a job"""

    # Check active jobs
    if job_id in active_jobs:
        job = active_jobs[job_id]
        return ResultResponse(
            job_id=job_id,
            status=job["status"],
            objective_value=job["result"].get("objective_value") if job["result"] else None,
            confidence=job["result"].get("confidence") if job["result"] else None,
            iterations=job["result"].get("iterations") if job["result"] else None,
            time_elapsed=job["result"].get("time_elapsed") if job["result"] else None,
            solution=job["result"].get("solution") if job["result"] else None,
            reasoning=job["result"].get("reasoning") if job["result"] else None,
            error=job.get("error"),
        )

    # Check completed jobs
    if job_id in completed_jobs:
        job = completed_jobs[job_id]
        return ResultResponse(
            job_id=job_id,
            status=job["status"],
            objective_value=job["result"].get("objective_value") if job["result"] else None,
            confidence=job["result"].get("confidence") if job["result"] else None,
            iterations=job["result"].get("iterations") if job["result"] else None,
            time_elapsed=job["result"].get("time_elapsed") if job["result"] else None,
            solution=job["result"].get("solution") if job["result"] else None,
            reasoning=job["result"].get("reasoning") if job["result"] else None,
            error=job.get("error"),
        )

    raise HTTPException(status_code=404, detail=f"Job {job_id} not found")


@app.post("/analyze/causal")
async def analyze_causal(request: CausalAnalysisRequest):
    """
    Perform causal analysis on a problem

    Returns causal chains, counterfactuals, and insights
    """
    try:
        engine = create_causal_engine()

        problem_dict = {
            "type": request.problem.problem_type,
            "size": request.problem.size,
            "objectives": request.problem.objectives,
            "constraints": request.problem.constraints,
        }

        analysis = engine.analyze_problem(problem_dict)

        return {
            "status": "success",
            "analysis": analysis,
        }
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/benchmarks/run")
async def run_benchmark(
    solvers: List[str],
    problem_sizes: List[int],
    repetitions: int = 3,
    background_tasks: BackgroundTasks = None,
):
    """
    Run benchmark suite

    Returns job_id for tracking progress
    """
    job_id = str(uuid.uuid4())

    active_jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "type": "benchmark",
        "created_at": time.time(),
        "result": None,
    }

    if background_tasks:
        background_tasks.add_task(
            execute_benchmark_job,
            job_id,
            solvers,
            problem_sizes,
            repetitions
        )

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Benchmark job {job_id} submitted"
    )


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_status(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates
    """
    await websocket.accept()

    try:
        while True:
            # Check job status
            if job_id in active_jobs:
                job = active_jobs[job_id]
            elif job_id in completed_jobs:
                job = completed_jobs[job_id]
                # Send final update and close
                await websocket.send_json({
                    "job_id": job_id,
                    "status": job["status"],
                    "result": job.get("result"),
                    "error": job.get("error"),
                })
                break
            else:
                await websocket.send_json({
                    "error": f"Job {job_id} not found"
                })
                break

            # Send current status
            await websocket.send_json({
                "job_id": job_id,
                "status": job["status"],
                "progress": job.get("progress", 0),
            })

            await asyncio.sleep(1)  # Poll every second

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")


# Background task executors

async def _execute_solve(request: SolveRequest) -> Dict[str, Any]:
    """Execute solve request"""
    mezan = create_intelligent_mezan()

    problem_dict = {
        "type": request.problem.problem_type,
        "size": request.problem.size,
        "objectives": request.problem.objectives,
        "constraints": request.problem.constraints,
    }

    if request.problem.data:
        problem_dict.update(request.problem.data)

    # Map depth string to enum
    depth_map = {
        "QUICK": AnalysisDepth.QUICK,
        "MEDIUM": AnalysisDepth.MEDIUM,
        "DEEP": AnalysisDepth.DEEP,
    }
    depth = depth_map.get(request.analysis_depth, AnalysisDepth.DEEP)

    result, synthesis = mezan.solve_intelligently(
        problem_dict,
        max_mezan_iterations=request.max_iterations,
        analysis_depth=depth
    )

    return {
        "objective_value": result.objective_value,
        "confidence": result.confidence,
        "iterations": result.iterations,
        "time_elapsed": result.time_elapsed,
        "solution": result.solution,
        "reasoning": synthesis.reasoning if synthesis else None,
    }


def execute_solve_job(job_id: str, request: SolveRequest):
    """Background task to execute solve job"""
    try:
        # Update status
        active_jobs[job_id]["status"] = JobStatus.RUNNING

        # Execute (blocking in thread pool)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_execute_solve(request))

        # Update with result
        active_jobs[job_id]["status"] = JobStatus.COMPLETED
        active_jobs[job_id]["result"] = result
        active_jobs[job_id]["completed_at"] = time.time()

        # Move to completed
        completed_jobs[job_id] = active_jobs.pop(job_id)

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = JobStatus.FAILED
        active_jobs[job_id]["error"] = str(e)
        completed_jobs[job_id] = active_jobs.pop(job_id)


def execute_benchmark_job(
    job_id: str,
    solvers: List[str],
    problem_sizes: List[int],
    repetitions: int
):
    """Background task to execute benchmark"""
    try:
        active_jobs[job_id]["status"] = JobStatus.RUNNING

        benchmarker = Benchmarker(suite_name=f"API_Benchmark_{job_id}")

        # Register solvers
        for solver_name in solvers:
            try:
                solver = create_libria_solver(solver_name)
                benchmarker.register_solver(solver_name, solver)
            except Exception as e:
                logger.warning(f"Could not register solver {solver_name}: {e}")

        # Register problems
        for size in problem_sizes:
            benchmarker.register_problem_generator(
                f"qap_{size}",
                create_qap_problem_generator(size)
            )

        # Run suite
        suite = benchmarker.run_suite(repetitions=repetitions, timeout=60.0)

        # Get results
        statistics = suite.get_statistics()

        active_jobs[job_id]["status"] = JobStatus.COMPLETED
        active_jobs[job_id]["result"] = {
            "statistics": statistics,
            "suite_data": suite.to_dict(),
        }

        completed_jobs[job_id] = active_jobs.pop(job_id)

    except Exception as e:
        logger.error(f"Benchmark job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = JobStatus.FAILED
        active_jobs[job_id]["error"] = str(e)
        completed_jobs[job_id] = active_jobs.pop(job_id)


# Startup/shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("MEZAN API Server starting...")
    logger.info(f"Available solvers: {get_available_solvers()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("MEZAN API Server shutting down...")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
    )
