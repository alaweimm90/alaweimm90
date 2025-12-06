"""High-level client wrapper for TuringoRodeo.

This module provides a small, reusable interface for other tal-ai systems
(e.g., ORCHEX Autonomous Research) to interact with Turingo without
needing to manage CLI args or instantiate TuringoRodeo directly.

Usage (async):

    from turingo_client import async_solve

    result = await async_solve(
        problem_type="qap",
        instance="tai30a",
    )

Usage (sync / script):

    from turingo_client import solve

    result = solve(problem_type="qap", instance="tai30a")
    print(result.to_dict())
"""

from __future__ import annotations

import asyncio
from typing import List, Optional

from turingo import TuringoRodeo, TuringoResult


_turingo: Optional[TuringoRodeo] = None


def get_turingo(config_path: Optional[str] = None) -> TuringoRodeo:
    """Return a process-wide TuringoRodeo instance.

    Parameters
    ----------
    config_path:
        Optional path to a JSON config file. If provided and a Turingo
        instance already exists, the existing instance is returned; the
        config_path is only used on first construction.
    """

    global _turingo
    if _turingo is None:
        _turingo = TuringoRodeo(config_path=config_path)
    return _turingo


async def async_solve(
    problem_type: str,
    instance: str,
    *,
    paradigms: Optional[List[str]] = None,
    time_limit_hours: float = 2.0,
    validation_level: str = "rigorous",
    config_path: Optional[str] = None,
) -> TuringoResult:
    """Solve a single optimization problem instance using Turingo.

    This is the recommended entry point from async code.
    """

    turingo = get_turingo(config_path=config_path)
    return await turingo.solve(
        problem_type=problem_type,
        instance=instance,
        paradigms=paradigms,
        time_limit_hours=time_limit_hours,
        validation_level=validation_level,
    )


def solve(
    problem_type: str,
    instance: str,
    *,
    paradigms: Optional[List[str]] = None,
    time_limit_hours: float = 2.0,
    validation_level: str = "rigorous",
    config_path: Optional[str] = None,
) -> TuringoResult:
    """Synchronous wrapper around :func:`async_solve`.

    Useful for scripts or environments where you don't already have an
    asyncio event loop.
    """

    return asyncio.run(
        async_solve(
            problem_type=problem_type,
            instance=instance,
            paradigms=paradigms,
            time_limit_hours=time_limit_hours,
            validation_level=validation_level,
            config_path=config_path,
        )
    )
