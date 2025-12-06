#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


def tree(root: Path, max_depth: int = 3) -> str:
    lines = []
    prefix_stack = []

    def walk(dir_path: Path, depth: int):
        if depth > max_depth:
            return
        entries = sorted([p for p in dir_path.iterdir() if not p.name.startswith('.')], key=lambda p: (p.is_file(), p.name.lower()))
        for i, p in enumerate(entries):
            is_last = i == len(entries) - 1
            branch = '└── ' if is_last else '├── '
            prefix = ''.join(prefix_stack) + branch
            lines.append(f"{prefix}{p.name}")
            if p.is_dir():
                prefix_stack.append('    ' if is_last else '│   ')
                walk(p, depth + 1)
                prefix_stack.pop()

    lines.append(str(root.resolve()))
    walk(root, 1)
    return '\n'.join(lines)


def write_tree_md(root: Path, out_file: Path, max_depth: int = 3):
    content = [f"# Directory Tree for {root}", "", "```", tree(root, max_depth), "```", ""]
    out_file.write_text('\n'.join(content))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    targets = [
        (repo_root, repo_root / 'docs' / 'TREE.md'),
        (repo_root / 'Libria' / 'libria-meta', repo_root / 'Libria' / 'libria-meta' / 'docs' / 'REPO_TREE.md'),
    ]
    for root, outp in targets:
        outp.parent.mkdir(parents=True, exist_ok=True)
        write_tree_md(root, outp, max_depth=3)
    print("Wrote tree docs:")
    for _, outp in targets:
        print(" -", outp)


if __name__ == '__main__':
    main()

