#!/usr/bin/env python3
"""
Automation CLI - Manage prompts, agents, workflows, and orchestration.

Usage:
    python -m automation.cli prompts list
    python -m automation.cli agents list
    python -m automation.cli workflows run code_review --input file.py
    python -m automation.cli route "fix the authentication bug"
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Base path for automation assets
AUTOMATION_PATH = Path(__file__).parent


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_markdown(file_path: Path) -> str:
    """Load a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# ============== PROMPTS ==============

def cmd_prompts_list(args):
    """List all available prompts."""
    prompts_path = AUTOMATION_PATH / "prompts"

    categories = {
        "system": prompts_path / "system",
        "project": prompts_path / "project",
        "tasks": prompts_path / "tasks"
    }

    print("=" * 60)
    print("AVAILABLE PROMPTS")
    print("=" * 60)

    total = 0
    for category, path in categories.items():
        if not path.exists():
            continue

        files = list(path.glob("*.md"))
        if not files:
            continue

        print(f"\n{category.upper()} ({len(files)} prompts)")
        print("-" * 40)

        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.stem:<40} ({size_kb:.1f} KB)")
            total += 1

    print(f"\n{'=' * 60}")
    print(f"Total: {total} prompts")
    return 0


def cmd_prompts_show(args):
    """Show a specific prompt."""
    prompts_path = AUTOMATION_PATH / "prompts"

    # Search in all categories
    for category in ["system", "project", "tasks"]:
        path = prompts_path / category / f"{args.name}.md"
        if path.exists():
            content = load_markdown(path)
            print(content)
            return 0

    print(f"Error: Prompt '{args.name}' not found", file=sys.stderr)
    return 1


def cmd_prompts_search(args):
    """Search prompts by keyword."""
    prompts_path = AUTOMATION_PATH / "prompts"
    query = args.query.lower()

    results = []
    for category in ["system", "project", "tasks"]:
        category_path = prompts_path / category
        if not category_path.exists():
            continue

        for f in category_path.glob("*.md"):
            content = load_markdown(f).lower()
            if query in content or query in f.stem.lower():
                results.append((category, f.stem, f))

    if not results:
        print(f"No prompts found matching '{args.query}'")
        return 1

    print(f"Found {len(results)} prompts matching '{args.query}':")
    print("-" * 40)
    for category, name, path in results:
        print(f"  [{category}] {name}")

    return 0


# ============== AGENTS ==============

def cmd_agents_list(args):
    """List all available agents."""
    agents_file = AUTOMATION_PATH / "agents" / "config" / "agents.yaml"

    if not agents_file.exists():
        print("Error: agents.yaml not found", file=sys.stderr)
        return 1

    config = load_yaml(agents_file)
    agents = config.get("agents", {})
    categories = config.get("categories", {})

    print("=" * 60)
    print("AVAILABLE AGENTS")
    print("=" * 60)

    for cat_name, cat_info in categories.items():
        print(f"\n{cat_name.upper()}: {cat_info.get('description', '')}")
        print("-" * 40)

        for agent_name in cat_info.get("agents", []):
            agent = agents.get(agent_name, {})
            role = agent.get("role", "Unknown")
            print(f"  {agent_name:<25} - {role}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(agents)} agents")
    return 0


def cmd_agents_show(args):
    """Show details of a specific agent."""
    agents_file = AUTOMATION_PATH / "agents" / "config" / "agents.yaml"
    config = load_yaml(agents_file)
    agents = config.get("agents", {})

    if args.name not in agents:
        print(f"Error: Agent '{args.name}' not found", file=sys.stderr)
        return 1

    agent = agents[args.name]

    print(f"Agent: {args.name}")
    print("=" * 40)
    print(f"Role: {agent.get('role', 'N/A')}")
    print(f"Goal: {agent.get('goal', 'N/A')}")
    print(f"\nBackstory:\n{agent.get('backstory', 'N/A')}")
    print(f"\nTools: {', '.join(agent.get('tools', []))}")
    print(f"\nLLM Config:")
    for k, v in agent.get('llm_config', {}).items():
        print(f"  {k}: {v}")

    return 0


# ============== WORKFLOWS ==============

def cmd_workflows_list(args):
    """List all available workflows."""
    workflows_file = AUTOMATION_PATH / "workflows" / "config" / "workflows.yaml"

    if not workflows_file.exists():
        print("Error: workflows.yaml not found", file=sys.stderr)
        return 1

    config = load_yaml(workflows_file)
    workflows = config.get("workflows", {})
    categories = config.get("categories", {})

    print("=" * 60)
    print("AVAILABLE WORKFLOWS")
    print("=" * 60)

    for cat_name, cat_info in categories.items():
        print(f"\n{cat_name.upper()}: {cat_info.get('description', '')}")
        print("-" * 40)

        for wf_name in cat_info.get("workflows", []):
            wf = workflows.get(wf_name, {})
            desc = wf.get("description", wf.get("name", "Unknown"))[:50]
            pattern = wf.get("pattern", "unknown")
            print(f"  {wf_name:<25} [{pattern}]")
            print(f"    {desc}")

    print(f"\n{'=' * 60}")
    print(f"Total: {len(workflows)} workflows")
    return 0


def cmd_workflows_show(args):
    """Show details of a specific workflow."""
    workflows_file = AUTOMATION_PATH / "workflows" / "config" / "workflows.yaml"
    config = load_yaml(workflows_file)
    workflows = config.get("workflows", {})

    if args.name not in workflows:
        print(f"Error: Workflow '{args.name}' not found", file=sys.stderr)
        return 1

    wf = workflows[args.name]

    print(f"Workflow: {args.name}")
    print("=" * 40)
    print(f"Pattern: {wf.get('pattern', 'N/A')}")
    print(f"Description: {wf.get('description', 'N/A')}")

    print(f"\nStages:")
    for i, stage in enumerate(wf.get("stages", []), 1):
        print(f"  {i}. {stage.get('name', 'unnamed')}")
        print(f"     Agent: {stage.get('agent', 'N/A')}")
        print(f"     Action: {stage.get('action', 'N/A')[:60]}...")

    print(f"\nSuccess Criteria:")
    for criterion in wf.get("success_criteria", []):
        print(f"  - {criterion}")

    return 0


# ============== ORCHESTRATION ==============

def cmd_route(args):
    """Route a task to the appropriate handler."""
    orch_file = AUTOMATION_PATH / "orchestration" / "config" / "orchestration.yaml"
    config = load_yaml(orch_file)

    task = args.task.lower()
    keywords = config.get("tool_routing", {}).get("intent_extraction", {}).get("keywords", {})
    rules = config.get("tool_routing", {}).get("rules", {})

    # Score each category
    scores = {}
    for category, kws in keywords.items():
        score = sum(1 for kw in kws if kw in task)
        if score > 0:
            scores[category] = score

    if not scores:
        print("Could not determine task type. Please provide more context.")
        return 1

    # Get best match
    best_category = max(scores, key=scores.get)
    confidence = scores[best_category] / max(len(task.split()), 1)

    tools = rules.get(best_category, {}).get("tools", [])

    print("=" * 60)
    print("TASK ROUTING RESULT")
    print("=" * 60)
    print(f"\nTask: {args.task}")
    print(f"\nDetected Type: {best_category}")
    print(f"Confidence: {min(confidence, 1.0):.0%}")
    print(f"\nRecommended Tools:")
    for i, tool in enumerate(tools, 1):
        marker = "→" if i == 1 else " "
        print(f"  {marker} {tool}")

    print(f"\nAll Scores:")
    for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {score}")

    return 0


def cmd_patterns_list(args):
    """List available orchestration patterns."""
    patterns_path = AUTOMATION_PATH / "orchestration" / "patterns"

    if not patterns_path.exists():
        print("Error: patterns directory not found", file=sys.stderr)
        return 1

    print("=" * 60)
    print("ORCHESTRATION PATTERNS (Anthropic)")
    print("=" * 60)

    for f in sorted(patterns_path.glob("*.yaml")):
        config = load_yaml(f)
        name = config.get("name", f.stem)
        desc = config.get("description", "").split("\n")[0][:60]

        print(f"\n{name}")
        print("-" * 40)
        print(f"  {desc}")

        use_when = config.get("use_when", [])[:3]
        if use_when:
            print(f"\n  Use when:")
            for item in use_when:
                print(f"    • {item}")

    return 0


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(
        description="Automation CLI - Manage AI assets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Prompts commands
    prompts_parser = subparsers.add_parser("prompts", help="Manage prompts")
    prompts_sub = prompts_parser.add_subparsers(dest="action")

    prompts_sub.add_parser("list", help="List all prompts")

    show_parser = prompts_sub.add_parser("show", help="Show a prompt")
    show_parser.add_argument("name", help="Prompt name (without .md)")

    search_parser = prompts_sub.add_parser("search", help="Search prompts")
    search_parser.add_argument("query", help="Search query")

    # Agents commands
    agents_parser = subparsers.add_parser("agents", help="Manage agents")
    agents_sub = agents_parser.add_subparsers(dest="action")

    agents_sub.add_parser("list", help="List all agents")

    agent_show = agents_sub.add_parser("show", help="Show agent details")
    agent_show.add_argument("name", help="Agent name")

    # Workflows commands
    workflows_parser = subparsers.add_parser("workflows", help="Manage workflows")
    workflows_sub = workflows_parser.add_subparsers(dest="action")

    workflows_sub.add_parser("list", help="List all workflows")

    wf_show = workflows_sub.add_parser("show", help="Show workflow details")
    wf_show.add_argument("name", help="Workflow name")

    # Route command
    route_parser = subparsers.add_parser("route", help="Route a task")
    route_parser.add_argument("task", help="Task description")

    # Patterns command
    subparsers.add_parser("patterns", help="List orchestration patterns")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to handlers
    if args.command == "prompts":
        if args.action == "list":
            return cmd_prompts_list(args)
        elif args.action == "show":
            return cmd_prompts_show(args)
        elif args.action == "search":
            return cmd_prompts_search(args)
        else:
            prompts_parser.print_help()
            return 1

    elif args.command == "agents":
        if args.action == "list":
            return cmd_agents_list(args)
        elif args.action == "show":
            return cmd_agents_show(args)
        else:
            agents_parser.print_help()
            return 1

    elif args.command == "workflows":
        if args.action == "list":
            return cmd_workflows_list(args)
        elif args.action == "show":
            return cmd_workflows_show(args)
        else:
            workflows_parser.print_help()
            return 1

    elif args.command == "route":
        return cmd_route(args)

    elif args.command == "patterns":
        return cmd_patterns_list(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
