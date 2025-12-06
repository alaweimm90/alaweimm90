"""
TalAI Enterprise API
Production-ready enterprise application with authentication, billing, multi-tenancy, and admin features
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Import routers
from auth_routes import auth_router
from billing_routes import billing_router
from admin_routes import admin_router
from tenant_routes import tenant_router
from api_routes import api_router

# Import services and configurations
from database import init_database, engine
from auth_service import init_redis
from admin_service import admin_service
from tenancy_service import tenant_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application configuration
class AppConfig:
    TITLE = "TalAI Enterprise API"
    VERSION = "1.0.0"
    DESCRIPTION = """
    Enterprise-grade API for TalAI platform with:
    - JWT-based authentication with OAuth2 support
    - Role-based access control (RBAC)
    - Multi-tenancy with isolation strategies
    - Stripe billing integration
    - Comprehensive admin dashboard
    - Usage tracking and analytics
    """

    # CORS settings
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    # Security
    TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "localhost,*.talai.com").split(",")

    # API settings
    API_PREFIX = "/api/v1"
    DOCS_URL = "/docs"
    REDOC_URL = "/redoc"
    OPENAPI_URL = "/openapi.json"

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""

    # Startup
    logger.info("Starting TalAI Enterprise API...")

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")

        # Initialize Redis
        await init_redis()
        logger.info("Redis initialized")

        # Initialize services
        await admin_service.initialize()
        await tenant_manager.initialize()
        logger.info("Services initialized")

        # Warm up caches
        logger.info("Application startup complete")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down TalAI Enterprise API...")

    # Close database connections
    await engine.dispose()

    logger.info("Application shutdown complete")

# Create FastAPI application
app = FastAPI(
    title=AppConfig.TITLE,
    version=AppConfig.VERSION,
    description=AppConfig.DESCRIPTION,
    docs_url=AppConfig.DOCS_URL,
    redoc_url=AppConfig.REDOC_URL,
    openapi_url=AppConfig.OPENAPI_URL,
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=AppConfig.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Total-Count", "X-Request-ID"]
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=AppConfig.TRUSTED_HOSTS
)

# Add rate limiting
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    # Don't expose internal errors in production
    if os.getenv("ENVIRONMENT") == "production":
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error occurred"}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc)}
        )

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "service": "talai-enterprise"}

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check"""
    from database import check_database_health

    db_health = await check_database_health()

    return {
        "status": "healthy",
        "service": "talai-enterprise",
        "version": AppConfig.VERSION,
        "database": db_health
    }

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Include routers
app.include_router(
    auth_router,
    prefix=f"{AppConfig.API_PREFIX}/auth",
    tags=["Authentication"]
)

app.include_router(
    billing_router,
    prefix=f"{AppConfig.API_PREFIX}/billing",
    tags=["Billing"]
)

app.include_router(
    admin_router,
    prefix=f"{AppConfig.API_PREFIX}/admin",
    tags=["Admin"]
)

app.include_router(
    tenant_router,
    prefix=f"{AppConfig.API_PREFIX}/tenants",
    tags=["Tenants"]
)

app.include_router(
    api_router,
    prefix=f"{AppConfig.API_PREFIX}",
    tags=["API"]
)

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "TalAI Enterprise API",
        "version": AppConfig.VERSION,
        "documentation": AppConfig.DOCS_URL,
        "health": "/health",
        "metrics": "/metrics"
    }

# API information endpoint
@app.get(f"{AppConfig.API_PREFIX}")
async def api_info():
    """API information"""
    return {
        "version": AppConfig.VERSION,
        "endpoints": {
            "auth": f"{AppConfig.API_PREFIX}/auth",
            "billing": f"{AppConfig.API_PREFIX}/billing",
            "admin": f"{AppConfig.API_PREFIX}/admin",
            "tenants": f"{AppConfig.API_PREFIX}/tenants"
        },
        "documentation": {
            "swagger": AppConfig.DOCS_URL,
            "redoc": AppConfig.REDOC_URL,
            "openapi": AppConfig.OPENAPI_URL
        }
    }

# Custom 404 handler
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "The requested resource was not found",
            "path": request.url.path
        }
    )

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests"""
    import uuid

    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response

# Log requests middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    import time

    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.3f}s"
    )

    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)

    return response

if __name__ == "__main__":
    import uvicorn

    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )