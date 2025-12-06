#!/usr/bin/env python3
"""
Generate a GitHub Issues CSV from repository TODO backlogs.

Reads:
- Libria/libria-meta/TODO_QAPFlow.md
- BACKLOG_IMPROVEMENTS.md (if present)

Outputs:
- issues.csv with columns: Title,Body,Labels

Usage:
  python scripts/generate_github_issues.py > issues.csv
  # Then import into GitHub repository Issues (via CSV import)
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parents[1]


def parse_numbered_tasks(md_path: Path, default_labels: List[str]) -> List[Tuple[str, str, List[str]]]:
    tasks: List[Tuple[str, str, List[str]]] = []
    if not md_path.exists():
        return tasks
    text = md_path.read_text(encoding='utf-8', errors='replace')
    lines = text.splitlines()
    current_section = None
    for line in lines:
        if re.match(r"^\d+\) ", line):
            current_section = line
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", line.strip())
        if m:
            num = m.group(1)
            title = m.group(2).strip()
            labels = list(default_labels)
            # Heuristic labeling based on keywords
            lower = title.lower()
            if any(k in lower for k in ["bench", "benchmark", "qaplib"]):
                labels.append('benchmark')
            if any(k in lower for k in ["schema", "docs", "guide", "document"]):
                labels.append('docs')
            if any(k in lower for k in ["ci", "workflow", "github"]):
                labels.append('ci')
            if any(k in lower for k in ["orchestrator", "router", "http"]):
                labels.append('orchestrator')
            if any(k in lower for k in ["mode", "backend", "wire", "adapter"]):
                labels.append('backend')
            body = f"From {md_path.name} (task {num}).\n\nSection: {current_section or ''}\n\nDetails: {title}"
            tasks.append((title, body, labels))
    return tasks


def main():
    todo = ROOT / 'Libria' / 'libria-meta' / 'TODO_QAPFlow.md'
    backlog = ROOT / 'BACKLOG_IMPROVEMENTS.md'
    rows: List[Tuple[str, str, List[str]]] = []
    rows.extend(parse_numbered_tasks(todo, ['qapflow']))
    rows.extend(parse_numbered_tasks(backlog, ['mezan']))

    w = csv.writer(fp := open('issues.csv', 'w', newline='', encoding='utf-8'))
    w.writerow(['Title', 'Body', 'Labels'])
    for title, body, labels in rows:
        w.writerow([title, body, ','.join(labels)])
    fp.close()
    print(f"Wrote {len(rows)} issues to issues.csv")


if __name__ == '__main__':
    main()

