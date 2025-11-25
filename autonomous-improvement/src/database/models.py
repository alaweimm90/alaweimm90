"""
Database models for REPZ Workflow System
"""

from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey
from sqlalchemy.orm import relationship

from .connection import Base


class WorkflowState(Base):
    """Model for storing workflow execution state"""
    __tablename__ = "workflow_states"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String(50), unique=True, index=True, default="repz_main")
    current_step = Column(Integer, default=0)
    status = Column(String(20), default="idle")  # idle, running, paused, stopped, completed, error
    phase = Column(String(20), default="assessment")
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Progress tracking
    completed_steps = Column(JSON, default=list)  # List of completed step IDs
    failed_steps = Column(JSON, default=list)     # List of failed step IDs
    step_results = Column(JSON, default=dict)     # Dict of step_id -> result

    # Metadata
    total_steps = Column(Integer, default=500)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class StepExecution(Base):
    """Model for tracking individual step executions"""
    __tablename__ = "step_executions"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String(50), index=True, default="repz_main")
    step_id = Column(Integer, index=True)
    phase = Column(String(20))
    description = Column(String(255))

    # Execution details
    start_time = Column(DateTime)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    success = Column(Boolean, default=False)
    error_message = Column(Text, nullable=True)

    # Results and metadata
    result_data = Column(JSON, nullable=True)
    metadata = Column(JSON, default=dict)

    # Retry information
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)

    created_at = Column(DateTime, default=datetime.utcnow)


class WorkflowMetrics(Base):
    """Model for storing workflow performance metrics"""
    __tablename__ = "workflow_metrics"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String(50), index=True, default="repz_main")
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # System metrics
    cpu_percent = Column(Float, nullable=True)
    memory_percent = Column(Float, nullable=True)
    disk_percent = Column(Float, nullable=True)
    network_bytes_sent = Column(Integer, nullable=True)
    network_bytes_recv = Column(Integer, nullable=True)

    # Workflow metrics
    active_steps = Column(Integer, default=0)
    completed_steps_count = Column(Integer, default=0)
    failed_steps_count = Column(Integer, default=0)
    average_step_duration = Column(Float, nullable=True)

    # Performance indicators
    success_rate = Column(Float, nullable=True)
    throughput_steps_per_hour = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class AIDecision(Base):
    """Model for storing AI decision history"""
    __tablename__ = "ai_decisions"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(String(50), index=True, default="repz_main")
    step_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Decision details
    decision_type = Column(String(50))  # analysis, optimization, risk_assessment
    context = Column(JSON)  # Input context for decision
    decision = Column(JSON)  # AI decision output

    # Performance metrics
    confidence_score = Column(Float, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    # Outcome tracking
    implemented = Column(Boolean, default=False)
    outcome_success = Column(Boolean, nullable=True)
    outcome_notes = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class SystemHealth(Base):
    """Model for system health monitoring"""
    __tablename__ = "system_health"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Component status
    database_status = Column(String(20), default="unknown")
    api_status = Column(String(20), default="unknown")
    worker_status = Column(String(20), default="unknown")
    monitoring_status = Column(String(20), default="unknown")

    # Performance metrics
    response_time_ms = Column(Float, nullable=True)
    error_rate_percent = Column(Float, nullable=True)
    uptime_percent = Column(Float, nullable=True)

    # Resource usage
    cpu_usage_percent = Column(Float, nullable=True)
    memory_usage_percent = Column(Float, nullable=True)
    disk_usage_percent = Column(Float, nullable=True)

    # Health score (0-100)
    overall_health_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    """Model for audit logging"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Event details
    event_type = Column(String(50), index=True)
    event_category = Column(String(50), index=True)
    user_id = Column(String(50), nullable=True, index=True)
    session_id = Column(String(50), nullable=True)

    # Event data
    event_data = Column(JSON)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)

    # Severity and impact
    severity = Column(String(20), default="info")  # debug, info, warning, error, critical
    impact_level = Column(String(20), default="low")  # low, medium, high, critical

    # Additional context
    workflow_id = Column(String(50), nullable=True)
    step_id = Column(Integer, nullable=True)
    component = Column(String(50), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)