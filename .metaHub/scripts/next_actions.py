#!/usr/bin/env python3
"""Generate prioritized next actions based on current state."""
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent.parent
REPORT = ROOT / ".metaHub" / "reports" / "tool-deduplication.json"

data = json.loads(REPORT.read_text())
recs = data["recommendations"]

# Count by action type
consolidate = [r for r in recs if "CONSOLIDATE" in r["action"]]
use_global = [r for r in recs if "USE GLOBAL" in r["action"]]

# Count by category
categories = Counter()
for r in consolidate:
    cat = r["action"].split()[1].rstrip(":")
    categories[cat] += 1

print("=== PRIORITY ACTIONS ===\n")

print("1. CONSOLIDATE CI WORKFLOWS (64 projects)")
print("   Impact: Highest - affects all projects")
print("   Effort: Low - reusable workflows exist")
print("   Command: Create .github/workflows/reusable-ci.yml")
print()

print("2. CONSOLIDATE LINT CONFIGS (28 projects)")
print("   Impact: High - code quality consistency")
print("   Effort: Low - 2 configs already done")
print("   Command: python consolidate_tools.py --category=lint")
print()

print("3. CONSOLIDATE TEST CONFIGS (21 projects)")
print("   Impact: Medium - testing standardization")
print("   Effort: Low - vitest.config.ts exists")
print("   Command: python consolidate_tools.py --category=test")
print()

print("4. CONSOLIDATE DOCKER (19 projects)")
print("   Impact: Medium - deployment consistency")
print("   Effort: Medium - need base templates")
print("   Command: Create tools/docker/Dockerfile.{python,node}")
print()

print("5. MERGE SIMILAR PROJECTS")
print("   Impact: High - reduce maintenance burden")
print("   Effort: High - requires code review")
print("   Candidates:")
print("   - alawein-business: 7 TypeScript projects")
print("   - alawein-science: 5 Python projects")
print("   - AlaweinOS: 10 Python + 4 TypeScript projects")
print()

print("=== QUICK WINS (Next 30 min) ===")
print("- Add tools/config/ruff.toml (Python lint)")
print("- Add tools/config/playwright.config.ts (E2E)")
print("- Run consolidate_tools.py again")
print()

print("=== THIS WEEK ===")
print("- Create reusable CI workflows")
print("- Consolidate all lint/test/format configs")
print("- Document global tool usage in README")
print()

print("=== THIS MONTH ===")
print("- Merge duplicate projects (discuss #4)")
print("- Migrate all projects to global tools")
print("- Remove 111 redundant config files")
