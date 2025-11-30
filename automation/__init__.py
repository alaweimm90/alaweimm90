"""
Automation Package - Centralized AI Assets Management

This package provides:
- Prompts: System, project, and task prompts
- Agents: Agent definitions and templates
- Workflows: Workflow definitions and templates
- Orchestration: Routing and patterns
- Tools: Tool registry

Usage:
    from automation import load_agent, load_workflow, load_prompt

    agent = load_agent("scientist_agent")
    workflow = load_workflow("code_review")
    prompt = load_prompt("system", "orchestrator")
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__version__ = "1.0.0"
__all__ = [
    "load_agent",
    "load_workflow",
    "load_prompt",
    "load_orchestration_config",
    "load_tools_config",
    "get_agents",
    "get_workflows",
    "route_task",
]

AUTOMATION_PATH = Path(__file__).parent


def _load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _load_markdown(file_path: Path) -> str:
    """Load a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============== AGENTS ==============

def get_agents() -> Dict[str, Any]:
    """Get all agent definitions."""
    agents_file = AUTOMATION_PATH / "agents" / "config" / "agents.yaml"
    config = _load_yaml(agents_file)
    return config.get("agents", {})


def load_agent(name: str) -> Optional[Dict[str, Any]]:
    """Load a specific agent by name."""
    agents = get_agents()
    return agents.get(name)


# ============== WORKFLOWS ==============

def get_workflows() -> Dict[str, Any]:
    """Get all workflow definitions."""
    workflows_file = AUTOMATION_PATH / "workflows" / "config" / "workflows.yaml"
    config = _load_yaml(workflows_file)
    return config.get("workflows", {})


def load_workflow(name: str) -> Optional[Dict[str, Any]]:
    """Load a specific workflow by name."""
    workflows = get_workflows()
    return workflows.get(name)


# ============== PROMPTS ==============

def load_prompt(category: str, name: str) -> Optional[str]:
    """
    Load a prompt by category and name.

    Args:
        category: One of 'system', 'project', 'tasks'
        name: Prompt name (without .md extension)

    Returns:
        Prompt content as string, or None if not found
    """
    prompt_path = AUTOMATION_PATH / "prompts" / category / f"{name}.md"
    if prompt_path.exists():
        return _load_markdown(prompt_path)
    return None


def get_prompts(category: str = None) -> Dict[str, str]:
    """
    Get all prompts, optionally filtered by category.

    Args:
        category: Optional category filter ('system', 'project', 'tasks')

    Returns:
        Dict mapping prompt names to their content
    """
    prompts = {}
    categories = [category] if category else ["system", "project", "tasks"]

    for cat in categories:
        cat_path = AUTOMATION_PATH / "prompts" / cat
        if cat_path.exists():
            for f in cat_path.glob("*.md"):
                prompts[f.stem] = _load_markdown(f)

    return prompts


# ============== ORCHESTRATION ==============

def load_orchestration_config() -> Dict[str, Any]:
    """Load the orchestration configuration."""
    config_file = AUTOMATION_PATH / "orchestration" / "config" / "orchestration.yaml"
    return _load_yaml(config_file)


def load_pattern(name: str) -> Optional[Dict[str, Any]]:
    """Load an orchestration pattern by name."""
    pattern_file = AUTOMATION_PATH / "orchestration" / "patterns" / f"{name}.yaml"
    if pattern_file.exists():
        return _load_yaml(pattern_file)
    return None


def route_task(task_description: str) -> Dict[str, Any]:
    """
    Route a task to the appropriate handler.

    Args:
        task_description: Natural language task description

    Returns:
        Dict with routing information including:
        - detected_type: The classified task type
        - confidence: Confidence score
        - recommended_tools: List of recommended tools
        - scores: All category scores
    """
    config = load_orchestration_config()
    task = task_description.lower()

    keywords = config.get("tool_routing", {}).get("intent_extraction", {}).get("keywords", {})
    rules = config.get("tool_routing", {}).get("rules", {})

    # Score each category
    scores = {}
    for category, kws in keywords.items():
        score = sum(1 for kw in kws if kw in task)
        if score > 0:
            scores[category] = score

    if not scores:
        return {
            "success": False,
            "message": "Could not determine task type",
            "task": task_description
        }

    # Get best match
    best_category = max(scores, key=scores.get)
    word_count = max(len(task.split()), 1)
    confidence = min(scores[best_category] / word_count, 1.0)

    tools = rules.get(best_category, {}).get("tools", [])

    return {
        "success": True,
        "task": task_description,
        "detected_type": best_category,
        "confidence": confidence,
        "recommended_tools": tools,
        "primary_tool": tools[0] if tools else None,
        "scores": scores
    }


# ============== TOOLS ==============

def load_tools_config() -> Dict[str, Any]:
    """Load the tools configuration."""
    tools_file = AUTOMATION_PATH / "tools" / "config" / "tools.yaml"
    return _load_yaml(tools_file)


def get_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a specific tool definition."""
    config = load_tools_config()
    return config.get("tools", {}).get(name)
