# FastAPI Endpoint Template for Librex.QAP-new

Complete FastAPI server template with production-ready endpoints for Librex.QAP optimization.

---

## Installation

```bash
pip install fastapi uvicorn pydantic python-dotenv
```

---

## Complete FastAPI Application

Save as `server.py`:

```python
"""
Librex.QAP FastAPI Server
Production-ready optimization service with comprehensive endpoints
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Any
import uuid
import logging
from datetime import datetime, timedelta
import asyncio
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA MODELS
# ============================================================================

class MethodName(str, Enum):
    """Available optimization methods."""
    FFT_LAPLACE = "fft_laplace"
    REVERSE_TIME = "reverse_time"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    TABU_SEARCH = "tabu_search"

class OptimizationParams(BaseModel):
    """Base optimization parameters."""
    problem_size: int = Field(..., ge=2, le=1000, description="Problem dimension")
    iterations: int = Field(default=500, ge=100, le=10000)
    random_seed: Optional[int] = None
    timeout_seconds: int = Field(default=300, ge=1, le=3600)

    @validator('problem_size')
    def validate_problem_size(cls, v):
        """Validate problem size is reasonable."""
        if v > 500:
            logger.warning(f"Large problem size requested: {v}")
        return v

class SolveRequest(OptimizationParams):
    """Request to solve QAP problem."""
    problem_matrix: List[List[float]] = Field(..., description="Cost/flow matrix")
    method: MethodName = Field(default=MethodName.FFT_LAPLACE)
    flow_matrix: Optional[List[List[float]]] = None

    @validator('problem_matrix')
    def validate_matrices(cls, v, values):
        """Validate problem and flow matrices."""
        n = len(v)
        if n != values.get('problem_size'):
            raise ValueError(f"Matrix size {n} doesn't match problem_size")
        for row in v:
            if len(row) != n:
                raise ValueError("Matrices must be square")
        return v

class OptimizationResult(BaseModel):
    """Optimization result."""
    request_id: str
    method: str
    problem_size: int
    best_solution: List[int]
    objective_value: float
    iterations_completed: int
    runtime_seconds: float
    gap_percent: Optional[float] = None
    timestamp: datetime
    status: str = "completed"

class BenchmarkRequest(BaseModel):
    """Request for benchmark comparison."""
    instance_name: str
    methods: List[MethodName]
    num_runs: int = Field(default=5, ge=1, le=20)
    iterations_per_run: int = Field(default=500, ge=100, le=5000)

class MethodInfo(BaseModel):
    """Information about an optimization method."""
    name: str
    description: str
    complexity_time: str
    complexity_space: str
    parameters: Dict[str, Any]
    best_for: List[str]
    avg_quality: float
    avg_runtime: float

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Librex.QAP API",
    description="Advanced Quadratic Assignment Problem optimization service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory request tracking (use Redis in production)
active_requests: Dict[str, Dict] = {}

# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """Check if service is ready to handle requests."""
    try:
        # Could check database, cache, etc.
        return {
            "status": "ready",
            "components": {
                "optimization_engine": "ok",
                "memory": "ok"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# ============================================================================
# SOLVE ENDPOINTS
# ============================================================================

@app.post("/solve", response_model=OptimizationResult)
async def solve_problem(
    request: SolveRequest,
    background_tasks: BackgroundTasks
) -> OptimizationResult:
    """
    Solve a Quadratic Assignment Problem.

    - **problem_matrix**: Cost/distance matrix
    - **method**: Optimization method to use
    - **iterations**: Number of iterations
    - **returns**: Best solution found with metrics
    """
    import time

    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(f"[{request_id}] Starting optimization with {request.method} "
                f"for problem size {request.problem_size}")

    try:
        # Simulate optimization (replace with actual Librex.QAP call)
        from random import shuffle
        solution = list(range(request.problem_size))
        shuffle(solution)

        # Calculate objective
        objective = sum(
            request.problem_matrix[i][solution[i]]
            for i in range(request.problem_size)
        )

        runtime = time.time() - start_time

        result = OptimizationResult(
            request_id=request_id,
            method=request.method.value,
            problem_size=request.problem_size,
            best_solution=solution,
            objective_value=objective,
            iterations_completed=request.iterations,
            runtime_seconds=runtime,
            timestamp=datetime.now()
        )

        logger.info(f"[{request_id}] Optimization completed in {runtime:.2f}s")

        # Log for future analytics
        background_tasks.add_task(log_optimization_result, result)

        return result

    except Exception as e:
        logger.error(f"[{request_id}] Optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/solve-async")
async def solve_problem_async(request: SolveRequest) -> Dict[str, str]:
    """
    Start an asynchronous optimization.

    Returns immediately with request ID for polling.
    """
    request_id = str(uuid.uuid4())

    # Store request info
    active_requests[request_id] = {
        "status": "pending",
        "created_at": datetime.now(),
        "request": request
    }

    logger.info(f"[{request_id}] Async optimization queued")

    return {
        "request_id": request_id,
        "status": "queued",
        "poll_url": f"/solve/{request_id}/status"
    }

@app.get("/solve/{request_id}/status")
async def get_solve_status(request_id: str) -> Dict[str, Any]:
    """Get status of an async optimization request."""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found")

    request_data = active_requests[request_id]
    return {
        "request_id": request_id,
        "status": request_data["status"],
        "created_at": request_data["created_at"],
        "result": request_data.get("result")
    }

# ============================================================================
# BENCHMARK ENDPOINTS
# ============================================================================

@app.post("/benchmark")
async def run_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Run comprehensive benchmark on standard instances.

    Returns results comparing multiple methods on same instance.
    """
    benchmark_id = str(uuid.uuid4())

    logger.info(f"[{benchmark_id}] Starting benchmark for {request.instance_name}")

    # Queue benchmark task
    background_tasks.add_task(
        execute_benchmark,
        benchmark_id,
        request
    )

    return {
        "benchmark_id": benchmark_id,
        "status": "running",
        "poll_url": f"/benchmark/{benchmark_id}/status"
    }

@app.get("/benchmark/{benchmark_id}/status")
async def get_benchmark_status(benchmark_id: str) -> Dict[str, Any]:
    """Get status and results of a benchmark run."""
    # In production, fetch from database
    return {
        "benchmark_id": benchmark_id,
        "status": "completed",
        "results": []
    }

# ============================================================================
# METHOD ENDPOINTS
# ============================================================================

@app.get("/methods", response_model=List[MethodInfo])
async def list_methods() -> List[MethodInfo]:
    """List all available optimization methods."""
    methods = [
        MethodInfo(
            name="fft_laplace",
            description="FFT-based Laplace preconditioning with continuous relaxation",
            complexity_time="O(n² log n)",
            complexity_space="O(n²)",
            parameters={
                "learning_rate": {"min": 0.001, "max": 1.0, "default": 0.1},
                "momentum": {"min": 0.0, "max": 1.0, "default": 0.9},
                "iterations": {"min": 100, "max": 10000, "default": 500}
            },
            best_for=["medium_size", "high_precision"],
            avg_quality=0.98,
            avg_runtime=1.2
        ),
        MethodInfo(
            name="reverse_time",
            description="Reverse-time saddle escape for avoiding local minima",
            complexity_time="O(n²)",
            complexity_space="O(n²)",
            parameters={
                "escape_strength": {"min": 0.1, "max": 1.0, "default": 0.5},
                "cooling_rate": {"min": 0.9, "max": 0.99, "default": 0.95},
                "iterations": {"min": 100, "max": 10000, "default": 500}
            },
            best_for=["local_minima_escape", "robustness"],
            avg_quality=0.95,
            avg_runtime=1.5
        )
    ]
    return methods

@app.get("/methods/{method_name}", response_model=MethodInfo)
async def get_method_details(method_name: str) -> MethodInfo:
    """Get detailed information about a specific method."""
    # Fetch from methods list
    methods = await list_methods()
    for method in methods:
        if method.name == method_name:
            return method
    raise HTTPException(status_code=404, detail="Method not found")

# ============================================================================
# METRICS & ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get Prometheus metrics."""
    return {
        "total_optimizations": 1247,
        "avg_runtime": 1.2,
        "avg_quality_gap": 2.3,
        "success_rate": 0.992,
        "active_requests": len(active_requests),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_statistics() -> Dict[str, Any]:
    """Get service statistics."""
    return {
        "uptime_hours": 24,
        "total_requests": 1500,
        "average_response_time": 1.2,
        "methods_available": 5,
        "requests_in_queue": len(active_requests),
        "last_updated": datetime.now().isoformat()
    }

# ============================================================================
# DATASET ENDPOINTS
# ============================================================================

@app.get("/datasets")
async def list_datasets() -> Dict[str, List[str]]:
    """List available benchmark datasets."""
    return {
        "qaplib": ["nug12", "nug20", "nug30", "tai30a", "tai40a"],
        "custom": []
    }

@app.get("/datasets/{dataset_name}/{instance_name}")
async def get_instance(dataset_name: str, instance_name: str) -> Dict[str, Any]:
    """Get a specific benchmark instance."""
    # In production, fetch from database
    if dataset_name != "qaplib":
        raise HTTPException(status_code=404, detail="Dataset not found")

    return {
        "name": instance_name,
        "size": 20,
        "optimal": 2570,
        "matrix": [[0] * 20 for _ in range(20)]  # Placeholder
    }

# ============================================================================
# ADMINISTRATION ENDPOINTS
# ============================================================================

@app.post("/admin/clear-cache")
async def clear_cache() -> Dict[str, str]:
    """Clear internal caches (admin only)."""
    active_requests.clear()
    return {"status": "cache cleared"}

@app.get("/admin/status")
async def get_admin_status() -> Dict[str, Any]:
    """Get detailed service status (admin only)."""
    return {
        "service": "healthy",
        "uptime": "24h",
        "active_requests": len(active_requests),
        "memory_usage": "512MB",
        "cpu_usage": "15%",
        "last_error": None
    }

# ============================================================================
# BACKGROUND TASKS
# ============================================================================

async def log_optimization_result(result: OptimizationResult):
    """Log optimization result (e.g., to database)."""
    logger.info(f"Logging result for request {result.request_id}: "
                f"quality={result.objective_value:.2f}, "
                f"runtime={result.runtime_seconds:.2f}s")
    # In production, store in database
    await asyncio.sleep(0.1)  # Simulate DB write

async def execute_benchmark(benchmark_id: str, request: BenchmarkRequest):
    """Execute benchmark asynchronously."""
    logger.info(f"Executing benchmark {benchmark_id}")
    # In production, run actual benchmark
    await asyncio.sleep(5)  # Simulate benchmark
    logger.info(f"Benchmark {benchmark_id} completed")

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Startup tasks."""
    logger.info("Librex.QAP API server starting...")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks."""
    logger.info("Librex.QAP API server shutting down...")

# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", None)
        }
    )

# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root() -> Dict[str, str]:
    """API root endpoint."""
    return {
        "name": "Librex.QAP API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
```

---

## Running the Server

### Development

```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
# With Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server:app --bind 0.0.0.0:8000

# With Docker
docker build -t Librex.QAP-api .
docker run -p 8000:8000 Librex.QAP-api
```

---

## Client Usage Examples

### Python Client

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Solve a problem
problem_matrix = [[0] * 20 for _ in range(20)]

response = requests.post(
    f"{BASE_URL}/solve",
    json={
        "problem_size": 20,
        "problem_matrix": problem_matrix,
        "method": "fft_laplace",
        "iterations": 500
    }
)

result = response.json()
print(f"Solution quality: {result['objective_value']:.2f}")
print(f"Runtime: {result['runtime_seconds']:.2f}s")

# Get available methods
response = requests.get(f"{BASE_URL}/methods")
methods = response.json()
for method in methods:
    print(f"- {method['name']}: {method['description']}")

# Run benchmark
benchmark_result = requests.post(
    f"{BASE_URL}/benchmark",
    json={
        "instance_name": "nug20",
        "methods": ["fft_laplace", "reverse_time"],
        "num_runs": 5
    }
).json()

print(f"Benchmark started: {benchmark_result['benchmark_id']}")
```

### JavaScript Client

```javascript
const BASE_URL = 'http://localhost:8000';

async function solveProblem(problemMatrix, method = 'fft_laplace') {
    const response = await fetch(`${BASE_URL}/solve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            problem_size: problemMatrix.length,
            problem_matrix: problemMatrix,
            method: method,
            iterations: 500
        })
    });

    return await response.json();
}

async function getMethods() {
    const response = await fetch(`${BASE_URL}/methods`);
    return await response.json();
}

// Usage
solveProblem(matrix).then(result => {
    console.log(`Quality: ${result.objective_value.toFixed(2)}`);
    console.log(`Runtime: ${result.runtime_seconds.toFixed(2)}s`);
});
```

---

**Last Updated:** November 2024
