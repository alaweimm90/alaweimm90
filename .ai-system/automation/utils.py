"""
Shared utilities for the automation package.

This module provides common functionality used across the automation system,
including file operations, configuration management, and routing logic.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def get_automation_path() -> Path:
    """
    Get the automation assets path.

    Priority:
    1. AUTOMATION_PATH environment variable
    2. Sibling automation directory (for automation-ts)
    3. Current directory

    Returns:
        Path to automation assets directory
    """
    if env_path := os.getenv("AUTOMATION_PATH"):
        return Path(env_path)

    # Check for sibling automation directory (for automation-ts usage)
    current = Path(__file__).parent
    sibling = current.parent / "automation"
    if sibling.exists() and sibling.is_dir():
        return sibling

    # Fallback to current directory
    return current


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """
    Load a YAML file safely.

    Args:
        file_path: Path to the YAML file

    Returns:
        Parsed YAML content as dict, or empty dict if file not found/error
    """
    try:
        if not file_path.exists():
            return {}
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_markdown_file(file_path: Path) -> Optional[str]:
    """
    Load a markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        File content as string, or None if not found/error
    """
    try:
        if not file_path.exists():
            return None
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return None


def list_files_recursive(directory: Path, extension: Optional[str] = None) -> List[Path]:
    """
    Recursively list all files in a directory.

    Args:
        directory: Directory to search
        extension: Optional file extension filter (e.g., '.md', '.yaml')

    Returns:
        List of file paths
    """
    if not directory.exists():
        return []

    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if extension and not filename.endswith(extension):
                continue
            files.append(Path(root) / filename)
    return files


class TaskRouter:
    """
    Centralized task routing logic.

    This consolidates the routing implementation used in multiple places.
    """

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load orchestration configuration."""
        config_file = get_automation_path() / "orchestration" / "config" / "orchestration.yaml"
        return load_yaml_file(config_file)

    def route_task(self, task_description: str) -> Dict[str, Any]:
        """
        Route a task based on description.

        Args:
            task_description: Natural language task description

        Returns:
            Routing result with category, confidence, tools, etc.
        """
        task_lower = task_description.lower()

        keywords = self.config.get("tool_routing", {}).get("intent_extraction", {}).get("keywords", {})
        rules = self.config.get("tool_routing", {}).get("rules", {})

        # Score each category
        scores = {}
        for category, kws in keywords.items():
            score = sum(1 for kw in kws if kw in task_lower)
            if score > 0:
                scores[category] = score

        if not scores:
            return {
                "success": False,
                "message": "Could not classify task",
                "task": task_description
            }

        # Best match
        best_category = max(scores, key=scores.get)
        word_count = max(len(task_lower.split()), 1)
        confidence = min(scores[best_category] / word_count, 1.0)

        tools = rules.get(best_category, {}).get("tools", [])
        threshold = rules.get(best_category, {}).get("confidence_threshold", 0.6)

        return {
            "success": True,
            "task": task_description,
            "detected_type": best_category,
            "confidence": confidence,
            "meets_threshold": confidence >= threshold,
            "recommended_tools": tools,
            "primary_tool": tools[0] if tools else None,
            "scores": scores
        }


# Global router instance
_router = None

def get_task_router() -> TaskRouter:
    """Get the global task router instance."""
    global _router
    if _router is None:
        _router = TaskRouter()
    return _router


def route_task(task_description: str) -> Dict[str, Any]:
    """
    Convenience function to route a task.

    Args:
        task_description: Task description

    Returns:
        Routing result
    """
    return get_task_router().route_task(task_description)
