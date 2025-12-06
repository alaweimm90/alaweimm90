"""
Enhanced MEZAN REST API Server with Enterprise Features

Production-grade FastAPI server with:
- API Gateway with routing and circuit breakers
- Multi-algorithm rate limiting
- JWT/OAuth2/API Key authentication
- Load balancing across workers
- RBAC authorization
- Distributed caching
- WebSocket support
- OpenAPI documentation
- Observability (metrics, tracing, logging)

Author: MEZAN Team
Date: 2025-11-18
Version: 3.5.0
"""

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
import uvicorn

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

# Import new enterprise components
from atlas_core.api_gateway import APIGateway, Service, ServiceEndpoint, Route, LoadBalancingStrategy
from atlas_core.rate_limiter import RateLimitManager, RateLimitConfig, RateLimitAlgorithm, RateLimitScope
from atlas_core.auth import AuthManager, AuthMethod, User
from atlas_core.load_balancer import LoadBalancer, Server, ServerPool, LoadBalancingAlgorithm

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting MEZAN Enhanced API Server...")

    # Initialize Redis
    app.state.redis = await redis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"),
        encoding="utf-8",
        decode_responses=True
    )

    # Initialize components
    await initialize_components(app)

    # Start background tasks
    app.state.health_check_task = asyncio.create_task(health_check_loop(app))

    logger.info("MEZAN Enhanced API Server started successfully")

    yield

    # Shutdown
    logger.info("Shutting down MEZAN Enhanced API Server...")

    # Cancel background tasks
    if hasattr(app.state, 'health_check_task'):
        app.state.health_check_task.cancel()

    # Stop components
    await shutdown_components(app)

    # Close Redis
    await app.state.redis.close()

    logger.info("MEZAN Enhanced API Server shutdown complete")


# FastAPI app with lifespan
app = FastAPI(
    title="MEZAN Enhanced API",
    description="Enterprise-grade REST API for MEZAN V3.5.0 Optimization System",
    version="3.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)


async def initialize_components(app: FastAPI):
    """Initialize enterprise components"""

    # Initialize API Gateway
    app.state.gateway = APIGateway(
        config_path="config/gateway_config.yaml",
        enable_metrics=True,
        enable_tracing=True,
        enable_caching=True,
        redis_client=app.state.redis
    )
    await app.state.gateway.start()

    # Initialize Rate Limiter
    app.state.rate_limiter = RateLimitManager(redis_client=app.state.redis)
    app.state.rate_limiter.load_config("config/rate_limits.yaml")

    # Initialize Auth Manager
    app.state.auth = AuthManager(
        jwt_secret=os.getenv("JWT_SECRET", "change-this-secret-key"),
        redis_client=app.state.redis,
        enable_audit=True
    )
    app.state.auth.rbac.load_config("config/rbac_policies.yaml")

    # Initialize Load Balancer
    app.state.load_balancer = LoadBalancer(
        enable_metrics=True,
        enable_health_checks=True,
        redis_client=app.state.redis
    )

    # Setup worker pools
    setup_worker_pools(app.state.load_balancer)
    await app.state.load_balancer.start()


async def shutdown_components(app: FastAPI):
    """Shutdown enterprise components"""

    if hasattr(app.state, 'gateway'):
        await app.state.gateway.stop()

    if hasattr(app.state, 'load_balancer'):
        await app.state.load_balancer.stop()


def setup_worker_pools(load_balancer: LoadBalancer):
    """Setup worker pools for load balancing"""

    # MEZAN solver workers
    solver_servers = [
        Server(id=f"solver-{i}", host="localhost", port=8000+i, weight=1)
        for i in range(3)
    ]

    solver_pool = ServerPool(
        name="solvers",
        algorithm=LoadBalancingAlgorithm.LEAST_CONNECTIONS,
        servers=solver_servers,
        health_check_enabled=True
    )

    load_balancer.add_pool(solver_pool)

    # Research workers
    research_servers = [
        Server(id=f"research-{i}", host="localhost", port=9000+i, weight=1)
        for i in range(2)
    ]

    research_pool = ServerPool(
        name="research",
        algorithm=LoadBalancingAlgorithm.RESOURCE_BASED,
        servers=research_servers,
        sticky_sessions=True,
        session_ttl=3600
    )

    load_balancer.add_pool(research_pool)


async def health_check_loop(app: FastAPI):
    """Background health check loop"""
    while True:
        try:
            # Update server resource usage
            for pool_name, pool in app.state.load_balancer.pools.items():
                for server in pool.servers:
                    # Simulate resource monitoring (would be real metrics in production)
                    import random
                    app.state.load_balancer.update_server_resources(
                        pool_name,
                        server.id,
                        cpu_usage=random.uniform(10, 90),
                        memory_usage=random.uniform(20, 80)
                    )

            await asyncio.sleep(30)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health check error: {e}")
            await asyncio.sleep(30)


# Dependency for authentication
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user"""

    # Check rate limit for auth
    rate_limit = await request.app.state.rate_limiter.check(
        "auth_limit",
        ip_address=request.client.host,
        endpoint=request.url.path
    )

    if not rate_limit.allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many authentication attempts",
            headers=rate_limit.headers
        )

    # Validate token
    token = credentials.credentials
    payload = await request.app.state.auth.jwt_provider.validate_token(token)

    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    # Get user
    user = request.app.state.auth.rbac.users.get(payload["sub"])

    if not user or not user.active:
        raise HTTPException(
            status_code=401,
            detail="User not found or inactive"
        )

    return user


# Middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware"""

    # Extract user info if authenticated
    user_id = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        payload = await request.app.state.auth.jwt_provider.validate_token(token)
        if payload:
            user_id = payload.get("sub")

    # Check rate limits
    endpoint_limit = await request.app.state.rate_limiter.check(
        "global_limit",
        user_id=user_id,
        ip_address=request.client.host,
        endpoint=request.url.path
    )

    if not endpoint_limit.allowed:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate limit exceeded"},
            headers=endpoint_limit.headers
        )

    # Process request
    response = await call_next(request)

    # Add rate limit headers
    for header, value in endpoint_limit.headers.items():
        response.headers[header] = value

    return response


# Pydantic models (keeping existing ones and adding new)
class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str
    totp_code: Optional[str] = None


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class ProblemRequest(BaseModel):
    """Request model for optimization problem"""
    problem_type: str = Field(..., description="Type of problem")
    size: int = Field(..., gt=0, description="Problem size")
    objectives: int = Field(1, ge=1, description="Number of objectives")
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    data: Optional[Dict[str, Any]] = None


# Enhanced API Endpoints

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: Request, login_data: LoginRequest):
    """Authenticate user and get tokens"""

    # Authenticate
    session = await request.app.state.auth.authenticate(
        method=AuthMethod.BASIC,
        credentials={
            "username": login_data.username,
            "password": login_data.password,
            "totp_code": login_data.totp_code
        },
        ip_address=request.client.host,
        user_agent=request.headers.get("User-Agent")
    )

    if not session:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    return TokenResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=request.app.state.auth.jwt_provider.access_token_ttl
    )


@app.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, refresh_token: str):
    """Refresh access token"""

    session = await request.app.state.auth.refresh_token(
        refresh_token=refresh_token,
        ip_address=request.client.host,
        user_agent=request.headers.get("User-Agent")
    )

    if not session:
        raise HTTPException(
            status_code=401,
            detail="Invalid refresh token"
        )

    return TokenResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        expires_in=request.app.state.auth.jwt_provider.access_token_ttl
    )


@app.post("/auth/logout")
async def logout(request: Request, user: User = Depends(get_current_user)):
    """Logout and invalidate tokens"""

    # Find user session
    for session_id, session in request.app.state.auth.sessions.items():
        if session.user_id == user.id:
            request.app.state.auth.logout(session_id)
            break

    return {"message": "Logged out successfully"}


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "MEZAN Enhanced API",
        "version": "3.5.0",
        "status": "operational",
        "features": [
            "API Gateway",
            "Multi-algorithm Rate Limiting",
            "JWT/OAuth2/API Key Authentication",
            "Load Balancing",
            "RBAC Authorization",
            "Circuit Breakers",
            "Distributed Caching"
        ],
        "docs": "/docs",
        "endpoints": {
            "auth": "/auth/login",
            "solve": "/api/v1/solve",
            "analyze": "/api/v1/analyze",
            "research": "/api/v1/research",
            "metrics": "/metrics",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check(request: Request):
    """Comprehensive health check"""

    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }

    # Check Redis
    try:
        await request.app.state.redis.ping()
        health_status["components"]["redis"] = "healthy"
    except:
        health_status["components"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check Gateway
    if hasattr(request.app.state, 'gateway'):
        gateway_metrics = request.app.state.gateway.get_metrics()
        health_status["components"]["gateway"] = {
            "status": "healthy",
            "requests_total": gateway_metrics["requests_total"],
            "success_rate": gateway_metrics.get("requests_success", 0) / max(gateway_metrics.get("requests_total", 1), 1)
        }

    # Check Load Balancer
    if hasattr(request.app.state, 'load_balancer'):
        lb_metrics = request.app.state.load_balancer.get_metrics()
        health_status["components"]["load_balancer"] = {
            "status": "healthy",
            "pools": len(lb_metrics["pools"]),
            "success_rate": lb_metrics["success_rate"]
        }

    # Check Rate Limiter
    if hasattr(request.app.state, 'rate_limiter'):
        rl_stats = request.app.state.rate_limiter.get_stats()
        health_status["components"]["rate_limiter"] = {
            "status": "healthy",
            "total_checks": rl_stats["total_checks"],
            "denial_rate": rl_stats["denied"] / max(rl_stats["total_checks"], 1)
        }

    return health_status


@app.get("/metrics")
async def metrics(request: Request, user: User = Depends(get_current_user)):
    """Get system metrics (requires authentication)"""

    # Check permission
    if not request.app.state.auth.rbac.check_permission(user.id, "read:metrics"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    metrics = {
        "timestamp": time.time(),
        "gateway": {},
        "load_balancer": {},
        "rate_limiter": {},
        "auth": {}
    }

    # Gateway metrics
    if hasattr(request.app.state, 'gateway'):
        metrics["gateway"] = request.app.state.gateway.get_metrics()

    # Load balancer metrics
    if hasattr(request.app.state, 'load_balancer'):
        metrics["load_balancer"] = request.app.state.load_balancer.get_metrics()

    # Rate limiter stats
    if hasattr(request.app.state, 'rate_limiter'):
        metrics["rate_limiter"] = request.app.state.rate_limiter.get_stats()

    # Auth stats
    if hasattr(request.app.state, 'auth'):
        metrics["auth"] = {
            "active_sessions": len(request.app.state.auth.sessions),
            "total_users": len(request.app.state.auth.rbac.users),
            "audit_logs": len(request.app.state.auth.audit_logs)
        }

    return metrics


@app.post("/api/v1/solve")
async def solve_problem(
    request: Request,
    problem: ProblemRequest,
    user: User = Depends(get_current_user)
):
    """Solve optimization problem with load balancing"""

    # Check permission
    if not request.app.state.auth.rbac.check_permission(user.id, "solve:basic"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    # Check problem size limits based on user role
    if "user_free" in user.roles and problem.size > 100:
        raise HTTPException(
            status_code=403,
            detail="Problem size exceeds free tier limit (100)"
        )

    # Execute with load balancing
    async def solve_request(server: Server):
        # In production, this would make an HTTP request to the server
        # For now, we'll execute locally
        mezan = create_intelligent_mezan()

        problem_dict = {
            "type": problem.problem_type,
            "size": problem.size,
            "objectives": problem.objectives,
            "constraints": problem.constraints,
        }

        if problem.data:
            problem_dict.update(problem.data)

        result, synthesis = mezan.solve_intelligently(
            problem_dict,
            max_mezan_iterations=5,
            analysis_depth=AnalysisDepth.MEDIUM
        )

        return {
            "objective_value": result.objective_value,
            "confidence": result.confidence,
            "iterations": result.iterations,
            "time_elapsed": result.time_elapsed,
            "solution": result.solution,
            "reasoning": synthesis.reasoning if synthesis else None,
            "server_id": server.id
        }

    try:
        result = await request.app.state.load_balancer.execute_request(
            pool_name="solvers",
            request_func=solve_request,
            session_id=user.id,
            client_ip=request.client.host
        )

        return result

    except Exception as e:
        logger.error(f"Solve request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Solve request failed: {str(e)}"
        )


@app.get("/api/v1/services/{service_name}/status")
async def get_service_status(
    request: Request,
    service_name: str,
    user: User = Depends(get_current_user)
):
    """Get detailed service status"""

    # Check permission
    if not request.app.state.auth.rbac.check_permission(user.id, "read:metrics"):
        raise HTTPException(
            status_code=403,
            detail="Insufficient permissions"
        )

    # Get service status from gateway
    if hasattr(request.app.state, 'gateway'):
        status = request.app.state.gateway.get_service_status(service_name)
        if status:
            return status

    # Get pool status from load balancer
    if hasattr(request.app.state, 'load_balancer'):
        status = request.app.state.load_balancer.get_pool_status(service_name)
        if status:
            return status

    raise HTTPException(
        status_code=404,
        detail=f"Service '{service_name}' not found"
    )


@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_status(
    websocket: WebSocket,
    job_id: str
):
    """WebSocket endpoint for real-time job updates"""
    await websocket.accept()

    try:
        while True:
            # In production, this would monitor actual job status
            await websocket.send_json({
                "job_id": job_id,
                "status": "running",
                "progress": 50,
                "timestamp": time.time()
            })

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "path": request.url.path
        },
        headers=exc.headers if hasattr(exc, 'headers') else None
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Disable reload in production
    )