"""
Services package for parallel workflow system.

This package provides specialized services for:
- Claude reasoning and code analysis
- Parallel compilation and testing coordination
- Async deployment with Docker integration
- Integration coordination and service orchestration
"""

from .claude_reasoning import handle_claude_analysis_task
from .build_test_coordinator import handle_compilation_task, handle_testing_task
from .deployment_service import handle_deployment_task
from .integration_coordinator import IntegrationCoordinator, get_integration_coordinator, register_handlers_with_executor

__all__ = [
    'handle_claude_analysis_task',
    'handle_compilation_task',
    'handle_testing_task',
    'handle_deployment_task',
    'IntegrationCoordinator',
    'get_integration_coordinator',
    'register_handlers_with_executor'
]
