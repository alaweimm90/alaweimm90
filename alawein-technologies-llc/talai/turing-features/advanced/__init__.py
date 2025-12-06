#!/usr/bin/env python3
"""
Advanced AI Capabilities for TalAI Turing Challenge

This module integrates sophisticated AI orchestration, adaptive learning,
automated reporting, and real-time collaboration features.
"""

from .multi_agent_coordination import (
    MultiAgentCoordinationSystem,
    BaseAgent,
    ResearchAgent,
    CoordinatorAgent,
    AgentState,
    ConsensusAlgorithm,
    AgentMessage,
    MessageType,
    AgentPerformanceMetrics
)

from .adaptive_learning import (
    AdaptiveLearner,
    ValidationRecord,
    LearningStrategy,
    FeedbackType,
    LearningPattern,
    ModelPerformance,
    MetaLearningInsight
)

from .report_generation import (
    ReportGenerator,
    ReportData,
    ReportFormat,
    ReportSection,
    VisualizationConfig
)

from .realtime_collaboration import (
    CollaborationHub,
    TeamWorkspaceManager,
    WebSocketConnection,
    CollaborationSession,
    User,
    UserRole,
    Comment,
    Annotation,
    HypothesisVersion,
    EventType
)

__all__ = [
    # Multi-Agent Coordination
    'MultiAgentCoordinationSystem',
    'BaseAgent',
    'ResearchAgent',
    'CoordinatorAgent',
    'AgentState',
    'ConsensusAlgorithm',
    'AgentMessage',
    'MessageType',
    'AgentPerformanceMetrics',

    # Adaptive Learning
    'AdaptiveLearner',
    'ValidationRecord',
    'LearningStrategy',
    'FeedbackType',
    'LearningPattern',
    'ModelPerformance',
    'MetaLearningInsight',

    # Report Generation
    'ReportGenerator',
    'ReportData',
    'ReportFormat',
    'ReportSection',
    'VisualizationConfig',

    # Real-time Collaboration
    'CollaborationHub',
    'TeamWorkspaceManager',
    'WebSocketConnection',
    'CollaborationSession',
    'User',
    'UserRole',
    'Comment',
    'Annotation',
    'HypothesisVersion',
    'EventType'
]

__version__ = '1.0.0'