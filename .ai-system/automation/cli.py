#!/usr/bin/env python3
"""Automation CLI - Manage prompts, agents, workflows."""

import argparse
import sys
from pathlib import Path

BASE_PATH = Path(__file__).parent


def load_yaml_file(file_path):
    try:
        import yaml
        print(f"Loading YAML file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def cmd_prompts_list(args):
    prompts_path = BASE_PATH / "prompts"
    categories = {"system": prompts_path / "system", "project": prompts_path / "project", "tasks": prompts_path / "tasks"}
    print("AVAILABLE PROMPTS")
    print("=" * 40)
    total = 0
    for category, path in categories.items():
        if not path.exists():
            continue
        files = list(path.glob("*.md"))
        if not files:
            continue
        total += len(files)
        print(f"\n{category.upper()} ({len(files)} prompts)")
    print(f"\nTotal: {total} prompts")
    return 0


def cmd_agents_list(args):
    agents_config = BASE_PATH / "agents" / "config" / "agents.yaml"
    if not agents_config.exists():
        print("Agents config not found")
        return 1
    data = load_yaml_file(agents_config)
    if not data:
        return 1
    agents = data.get("agents", [])
    print("AVAILABLE AGENTS")
    print("=" * 40)
    if isinstance(agents, dict):
        for name, info in agents.items():
            role = info.get('role', '') if isinstance(info, dict) else ''
            print(f"  {name:<30} {role}")
        print(f"\nTotal: {len(agents)} agents")
    else:
        for agent in agents:
            print(f"  {agent.get('name', 'unknown'):<30} {agent.get('role', '')}")
        print(f"\nTotal: {len(agents)} agents")
    return 0


def cmd_workflows_list(args):
    workflows_config = BASE_PATH / "workflows" / "config" / "workflows.yaml"
    if not workflows_config.exists():
        print("Workflows config not found")
        return 1
    data = load_yaml_file(workflows_config)
    if not data:
        return 1
    workflows = data.get("workflows", {})
    print("AVAILABLE WORKFLOWS")
    print("=" * 40)
    if isinstance(workflows, dict):
        for name, info in workflows.items():
            desc = info.get('description', '')[:50] if isinstance(info, dict) else str(info)[:50]
            print(f"  {name:<25} {desc}")
        print(f"\nTotal: {len(workflows)} workflows")
    else:
        for wf in workflows:
            print(f"  {wf.get('name', 'unknown'):<25} {wf.get('description', '')[:50]}")
        print(f"\nTotal: {len(workflows)} workflows")
    return 0


def cmd_route(args):
    keywords = {"prompts": ["prompt", "template"], "agents": ["agent", "persona"], "workflows": ["workflow", "pipeline"], "debugging": ["debug", "fix", "error", "bug"]}
    scores = {}
    task_desc = args.task.lower()
    for category, kws in keywords.items():
        score = sum(1 for kw in kws if kw in task_desc)
        if score > 0:
            scores[category] = score
    if not scores:
        print("Could not determine task type.")
        return 1
    best = max(scores, key=scores.get)
    print(f"Routed task \"{args.task}\" to type: {best} (confidence: {min(scores[best]/3.0, 1.0):.1f})")
    return 0


def cmd_patterns(args):
    print("ORCHESTRATION PATTERNS (Anthropic)")
    print("=" * 40)
    for name, desc in [("prompt-chaining", "Chain prompts sequentially"), ("routing", "Route to specialized handlers"), ("parallelization", "Run tasks in parallel"), ("orchestrator-workers", "Central orchestrator with workers"), ("evaluator-optimizer", "Iterative refinement loop")]:
        print(f"\n{name}\n" + "-" * 40 + f"\n  {desc}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Automation CLI")
    subparsers = parser.add_subparsers(dest="command")
    prompts_p = subparsers.add_parser("prompts")
    prompts_p.add_subparsers(dest="prompts_cmd").add_parser("list")
    agents_p = subparsers.add_parser("agents")
    agents_p.add_subparsers(dest="agents_cmd").add_parser("list")
    workflows_p = subparsers.add_parser("workflows")
    workflows_p.add_subparsers(dest="workflows_cmd").add_parser("list")
    route_p = subparsers.add_parser("route")
    route_p.add_argument("task")
    subparsers.add_parser("patterns")
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    if args.command == "prompts" and getattr(args, 'prompts_cmd', None) == "list":
        return cmd_prompts_list(args)
    if args.command == "agents" and getattr(args, 'agents_cmd', None) == "list":
        return cmd_agents_list(args)
    if args.command == "workflows" and getattr(args, 'workflows_cmd', None) == "list":
        return cmd_workflows_list(args)
    if args.command == "route":
        return cmd_route(args)
    if args.command == "patterns":
        return cmd_patterns(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
