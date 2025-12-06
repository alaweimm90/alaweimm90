"""High-level LLM client wrapper for ORCHEX Orchestrator.

This module provides a small, reusable interface for other tal-ai systems
(ORCHEX Autonomous Research, Turingo, etc.) to call the orchestrator
without needing to construct Task objects or manage a local Orchestrator
instance.

Usage (inside async code):

    from atlas_orchestrator.llm_client import llm_call
    from atlas_orchestrator.core.task import TaskType

    result = await llm_call(
        prompt="Summarize the latest work on QAP solvers",
        task_type=TaskType.RESEARCH,
    )
    print(result.content)

The orchestrator is created lazily on first use, and configuration (API
keys, routing strategy, budgets) is resolved via OrchestratorConfig,
which already knows how to read environment variables.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from atlas_orchestrator.core.orchestrator import Orchestrator
from atlas_orchestrator.core.task import Task, TaskResult, TaskType


_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Return a process-wide Orchestrator instance.

    This uses default OrchestratorConfig behavior (including env vars)
    and creates the orchestrator lazily on first use.
    """

    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


async def llm_call(
    prompt: str,
    task_type: TaskType = TaskType.SIMPLE,
    *,
    context: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None,
) -> TaskResult:
    """Execute a single LLM task via ORCHEX Orchestrator.

    Parameters
    ----------
    prompt:
        Main prompt to send to the model.
    task_type:
        TaskType describing the intent (affects routing).
    context:
        Optional additional context.
    max_tokens:
        Maximum tokens for the response.
    temperature:
        Sampling temperature.
    metadata:
        Optional dictionary of metadata to attach to the task.
    """

    orchestrator = get_orchestrator()
    task = Task(
        prompt=prompt,
        task_type=task_type,
        context=context,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata=metadata or {},
    )
    return await orchestrator.execute(task)
