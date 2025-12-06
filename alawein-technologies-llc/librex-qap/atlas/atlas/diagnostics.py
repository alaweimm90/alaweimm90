"""
Diagnostics helpers for the ORCHEX research stack.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Dict, List


CRITICAL_MODULES = {
    "Self-Refutation Protocol": "self_refutation",
    "200-Question Interrogation": "interrogation",
    "Hall of Failures": "hall_of_failures",
    "ORCHEX Orchestrator": "atlas_orchestrator",
}

OPTIONAL_MODULES = {
    "Meta-Learning Agents": "meta_learning",
}

API_KEYS = {
    "ANTHROPIC_API_KEY": "Anthropic (Claude)",
    "OPENAI_API_KEY": "OpenAI (GPT-4)",
    "GOOGLE_API_KEY": "Google (Gemini)",
}


def _module_available(module_name: str) -> bool:
    """Return True if module import target exists."""
    return importlib.util.find_spec(module_name) is not None


def _env_present(var: str) -> bool:
    """Return True if env var looks populated (non-empty)."""
    value = os.getenv(var)
    return bool(value and value.strip())


def run_diagnostics() -> Dict[str, object]:
    """
    Run dependency + configuration diagnostics.

    Returns:
        Dict with check results, suitable for CLI or API consumption.
    """
    checks: List[Dict[str, object]] = []

    for label, module_name in CRITICAL_MODULES.items():
        available = _module_available(module_name)
        checks.append(
            {
                "name": label,
                "module": module_name,
                "critical": True,
                "ok": available,
                "hint": "" if available else f"Install or add `{module_name}` to PYTHONPATH.",
            }
        )

    for label, module_name in OPTIONAL_MODULES.items():
        available = _module_available(module_name)
        checks.append(
            {
                "name": label,
                "module": module_name,
                "critical": False,
                "ok": available,
                "hint": "Optional feature disabled" if not available else "",
            }
        )

    api_keys = []
    for var, provider in API_KEYS.items():
        present = _env_present(var)
        api_keys.append(
            {
                "variable": var,
                "provider": provider,
                "ok": present,
                "hint": "" if present else f"Set {var} to call {provider} via the orchestrator.",
            }
        )

    healthy = all(check["ok"] for check in checks if check["critical"])

    return {
        "healthy": healthy,
        "checks": checks,
        "api_keys": api_keys,
    }


def summarize_diagnostics(result: Dict[str, object]) -> str:
    """Return a plain-text summary for CLI output."""
    lines = []
    lines.append("ORCHEX Diagnostics")
    lines.append("=================")

    status = "PASS" if result.get("healthy") else "FAIL"
    lines.append(f"Status: {status}")
    lines.append("")
    lines.append("Modules:")
    for check in result.get("checks", []):
        mark = "✅" if check["ok"] else ("❌" if check["critical"] else "⚠️")
        lines.append(f"  {mark} {check['name']} ({check['module']})")
        if not check["ok"] and check.get("hint"):
            lines.append(f"     → {check['hint']}")

    lines.append("")
    lines.append("API Keys:")
    for api in result.get("api_keys", []):
        mark = "✅" if api["ok"] else "⚠️"
        lines.append(f"  {mark} {api['provider']} ({api['variable']})")
        if not api["ok"] and api.get("hint"):
            lines.append(f"     → {api['hint']}")

    return "\n".join(lines)
