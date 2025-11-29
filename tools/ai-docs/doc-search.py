#!/usr/bin/env python3
"""
AI Coding Tools Documentation Search Utility

Searches through downloaded AI tool documentation for keywords and patterns.
Useful for finding specific features, APIs, or implementation details.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Generator, List, Tuple

# Default documentation directory
DOCS_DIR = Path(__file__).parent.parent.parent / "docs" / "ai-coding-tools"
CATALOG_PATH = DOCS_DIR / "catalog.json"


def load_catalog() -> dict:
    """Load the documentation catalog."""
    if CATALOG_PATH.exists():
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def find_markdown_files(base_path: Path) -> Generator[Path, None, None]:
    """Find all markdown files in the documentation directory."""
    for pattern in ["**/*.md", "**/*.mdx", "**/*.rst"]:
        yield from base_path.glob(pattern)


def search_in_file(
    file_path: Path, pattern: str, context_lines: int = 2
) -> List[Tuple[int, str]]:
    """Search for pattern in a file and return matching lines with context."""
    matches = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            regex = re.compile(pattern, re.IGNORECASE)

            for i, line in enumerate(lines):
                if regex.search(line):
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    context = "".join(lines[start:end])
                    matches.append((i + 1, context))
    except Exception as e:
        pass  # Silently skip files that can't be read
    return matches


def search_docs(
    query: str,
    tool_filter: str = None,
    file_type: str = "md",
    context_lines: int = 2,
    max_results: int = 20,
) -> None:
    """Search documentation for a query."""
    results_count = 0

    print(f"\n{'='*60}")
    print(f"Searching for: {query}")
    print(f"{'='*60}\n")

    for doc_file in find_markdown_files(DOCS_DIR):
        # Apply tool filter if specified
        if tool_filter and tool_filter.lower() not in str(doc_file).lower():
            continue

        # Apply file type filter
        if file_type and not doc_file.suffix.endswith(file_type):
            continue

        matches = search_in_file(doc_file, query, context_lines)

        if matches:
            relative_path = doc_file.relative_to(DOCS_DIR)
            print(f"\n📄 {relative_path}")
            print("-" * 40)

            for line_num, context in matches:
                print(f"  Line {line_num}:")
                for ctx_line in context.strip().split("\n"):
                    print(f"    {ctx_line}")
                print()

                results_count += 1
                if results_count >= max_results:
                    print(f"\n[Showing first {max_results} results]")
                    return

    if results_count == 0:
        print("No matches found.")
    else:
        print(f"\nTotal matches: {results_count}")


def list_tools() -> None:
    """List all documented tools."""
    catalog = load_catalog()

    print("\n" + "=" * 60)
    print("AI Coding Tools Documentation Catalog")
    print("=" * 60 + "\n")

    for category_id, category in catalog.get("categories", {}).items():
        print(f"\n📁 {category.get('name', category_id)}")
        print("-" * 40)

        for tool in category.get("tools", []):
            status = "✅" if tool.get("status") == "cloned" else "🔗"
            print(f"  {status} {tool.get('name')}")
            if tool.get("local_path"):
                print(f"     └─ Local: {tool.get('local_path')}")
            else:
                print(f"     └─ URL: {tool.get('url', 'N/A')}")


def compare_features(tools: List[str]) -> None:
    """Compare features between tools."""
    catalog = load_catalog()
    all_tools = []

    for category in catalog.get("categories", {}).values():
        all_tools.extend(category.get("tools", []))

    # Find requested tools
    found_tools = []
    for tool in all_tools:
        if any(t.lower() in tool.get("id", "").lower() for t in tools):
            found_tools.append(tool)

    if not found_tools:
        print("No matching tools found.")
        return

    # Collect all features
    all_features = set()
    for tool in found_tools:
        all_features.update(tool.get("features", []))

    print("\n" + "=" * 60)
    print("Feature Comparison")
    print("=" * 60 + "\n")

    # Header
    header = "Feature".ljust(25) + " | ".join(t.get("name", "")[:12].center(12) for t in found_tools)
    print(header)
    print("-" * len(header))

    # Features
    for feature in sorted(all_features):
        row = feature.ljust(25)
        for tool in found_tools:
            has_feature = feature in tool.get("features", [])
            row += ("✅" if has_feature else "❌").center(14)
        print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Search AI coding tools documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python doc-search.py search "MCP server"
  python doc-search.py search "function calling" --tool openai
  python doc-search.py list
  python doc-search.py compare claude copilot cursor
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documentation")
    search_parser.add_argument("query", help="Search query (regex supported)")
    search_parser.add_argument("--tool", "-t", help="Filter by tool name")
    search_parser.add_argument(
        "--context", "-c", type=int, default=2, help="Context lines (default: 2)"
    )
    search_parser.add_argument(
        "--max", "-m", type=int, default=20, help="Max results (default: 20)"
    )

    # List command
    subparsers.add_parser("list", help="List all documented tools")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare tool features")
    compare_parser.add_argument("tools", nargs="+", help="Tools to compare")

    args = parser.parse_args()

    if args.command == "search":
        search_docs(
            args.query,
            tool_filter=args.tool,
            context_lines=args.context,
            max_results=args.max,
        )
    elif args.command == "list":
        list_tools()
    elif args.command == "compare":
        compare_features(args.tools)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
