"""
REPZ Autonomous Improvement Workflow System

A fully autonomous, self-executing 500-step workflow for iteratively improving
REPZ—an integrated AI-powered athletic performance platform.

This system operates in YOLO mode with zero user interventions, incorporating
AI-driven decision-making, ethical data handling, and compliance with regulations.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from .config.settings import get_settings
from .workflow.engine import WorkflowEngine
from .monitoring.metrics import MetricsCollector
from .database.connection import init_database
from .ai.decision_engine import AIDecisionEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
workflow_engine: Optional[WorkflowEngine] = None
metrics_collector: Optional[MetricsCollector] = None
ai_engine: Optional[AIDecisionEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global workflow_engine, metrics_collector, ai_engine

    # Startup
    logger.info("Starting REPZ Improvement Workflow System...")

    settings = get_settings()

    # Initialize database
    await init_database(settings.database_url)

    # Initialize components
    metrics_collector = MetricsCollector()
    ai_engine = AIDecisionEngine()
    workflow_engine = WorkflowEngine(
        ai_engine=ai_engine,
        metrics_collector=metrics_collector
    )

    # Start background monitoring
    asyncio.create_task(metrics_collector.start_collection())

    logger.info("REPZ Improvement Workflow System started successfully")

    yield

    # Shutdown
    logger.info("Shutting down REPZ Improvement Workflow System...")
    if workflow_engine:
        await workflow_engine.shutdown()
    if metrics_collector:
        await metrics_collector.stop_collection()
    logger.info("REPZ Improvement Workflow System shut down")

# Create FastAPI app
app = FastAPI(
    title="REPZ Autonomous Improvement Workflow",
    description="Fully autonomous 500-step workflow for REPZ platform improvement",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class WorkflowStatus(BaseModel):
    current_step: int
    total_steps: int = 500
    phase: str
    status: str
    progress_percentage: float
    current_task: str
    estimated_completion: Optional[str] = None

class WorkflowCommand(BaseModel):
    action: str  # start, pause, resume, stop, reset
    parameters: Optional[Dict[str, Any]] = None

class SystemHealth(BaseModel):
    status: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "REPZ Autonomous Improvement Workflow System",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check() -> SystemHealth:
    """System health check"""
    if not all([workflow_engine, metrics_collector, ai_engine]):
        raise HTTPException(status_code=503, detail="System not fully initialized")

    return SystemHealth(
        status="healthy",
        components={
            "workflow_engine": "operational",
            "metrics_collector": "operational",
            "ai_engine": "operational",
            "database": "connected"
        },
        metrics=await metrics_collector.get_system_metrics()
    )

@app.get("/workflow/status")
async def get_workflow_status() -> WorkflowStatus:
    """Get current workflow status"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    status = await workflow_engine.get_status()
    return WorkflowStatus(**status)

@app.post("/workflow/control")
async def control_workflow(command: WorkflowCommand, background_tasks: BackgroundTasks):
    """Control workflow execution"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    try:
        if command.action == "start":
            background_tasks.add_task(workflow_engine.start_workflow)
            return {"message": "Workflow starting...", "action": "start"}
        elif command.action == "pause":
            await workflow_engine.pause_workflow()
            return {"message": "Workflow paused", "action": "pause"}
        elif command.action == "resume":
            await workflow_engine.resume_workflow()
            return {"message": "Workflow resumed", "action": "resume"}
        elif command.action == "stop":
            await workflow_engine.stop_workflow()
            return {"message": "Workflow stopped", "action": "stop"}
        elif command.action == "reset":
            await workflow_engine.reset_workflow()
            return {"message": "Workflow reset", "action": "reset"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {command.action}")
    except Exception as e:
        logger.error(f"Workflow control error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/steps/{step_id}")
async def get_step_details(step_id: int):
    """Get details for a specific step"""
    if not workflow_engine:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    if not 1 <= step_id <= 500:
        raise HTTPException(status_code=400, detail="Step ID must be between 1 and 500")

    step_info = await workflow_engine.get_step_info(step_id)
    if not step_info:
        raise HTTPException(status_code=404, detail=f"Step {step_id} not found")

    return step_info

@app.get("/metrics")
async def get_metrics():
    """Get system and workflow metrics"""
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics collector not available")

    return await metrics_collector.get_all_metrics()

@app.get("/ai/insights")
async def get_ai_insights():
    """Get AI-driven insights and recommendations"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not available")

    return await ai_engine.get_current_insights()

@app.post("/ai/decision")
async def request_ai_decision(context: Dict[str, Any]):
    """Request AI decision for a specific context"""
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI engine not available")

    try:
        decision = await ai_engine.make_decision(context)
        return {"decision": decision}
    except Exception as e:
        logger.error(f"AI decision error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )