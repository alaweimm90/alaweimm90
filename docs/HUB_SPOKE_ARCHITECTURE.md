# Hub-Spoke Architecture Guide

## The Professional Enterprise Pattern

This is how professional software organizations structure codebases to maximize reuse, maintainability, and user experience while preserving all development work.

## Core Principle

**Separate "What Users See" from "How It Was Built"**

```
User Experience Layer (Clean, Polished)
         ↓
    Hub Layer (Reusable Core)
         ↓
Development Archive (Full History)
```

## Architecture Overview

### The Hub (Control Center)

**Location:** `.metaHub/` or `tools/` or `shared/`

**Contains:**
- ✅ Core libraries (shared code)
- ✅ Standard CLIs (reusable tools)
- ✅ Services (APIs, orchestration)
- ✅ Utilities (helpers, common functions)
- ✅ Templates (project scaffolding)
- ✅ Schemas (validation, standards)

**Purpose:** Single source of truth for all shared functionality

### The Spokes (Projects)

**Location:** `organizations/*/PROJECT_NAME/`

**Contains:**
- ✅ Project-specific logic (unique features)
- ✅ Configuration (project settings)
- ✅ User-facing interfaces (CLI wrappers, UIs)
- ✅ Documentation (user guides)
- ✅ Tests (integration tests)

**Purpose:** Thin wrappers that compose hub functionality for specific use cases

### The Archive (Development History)

**Location:** `.archive/` or `PROJECT_NAME/.dev-archive/`

**Contains:**
- ✅ Original development code (how features were built)
- ✅ Experiments and prototypes (R&D work)
- ✅ Build artifacts (compilation history)
- ✅ Migration records (what moved where)
- ✅ Git history snapshots (full provenance)

**Purpose:** Preserve all work with full traceability

## Your GitHub Repository Structure

### Current State (Scattered)

```
organizations/
├── AlaweinOS/
│   ├── Benchmarks/
│   │   ├── src/benchmark_utils.py       # Duplicated!
│   │   ├── scripts/run_bench.py
│   │   └── helpers/perf_monitor.py
│   ├── MEZAN/
│   │   ├── utils/benchmark_utils.py     # Duplicate!
│   │   └── tools/performance.py
│   └── Optilibria/
│       └── benchmark/runner.py          # Duplicate!
```

**Problem:** Same functionality scattered across 3+ projects

### Target State (Hub-Spoke)

```
.metaHub/                              # THE HUB (Control Center)
├── libs/                              # Shared libraries
│   ├── benchmarking/
│   │   ├── __init__.py
│   │   ├── core.py                    # Consolidated benchmark logic
│   │   ├── monitors.py                # Performance monitoring
│   │   └── reporters.py               # Result reporting
│   ├── optimization/
│   ├── data_processing/
│   └── common/
├── clis/                              # Standard CLIs
│   ├── bench                          # Benchmark CLI (uses libs/benchmarking)
│   ├── optimize                       # Optimization CLI
│   └── analyze                        # Analysis CLI
├── services/                          # Shared services
│   ├── orchestrator/
│   └── monitor/
└── schemas/                           # Validation schemas

organizations/                         # THE SPOKES (User-facing)
├── AlaweinOS/
│   ├── Benchmarks/
│   │   ├── config/                    # Project-specific config
│   │   ├── scripts/
│   │   │   └── run_benchmark.sh       # Thin wrapper: calls hub CLI
│   │   └── README.md                  # User documentation
│   ├── MEZAN/
│   │   └── Uses hub libs directly
│   └── Optilibria/
│       └── Uses hub libs directly

.archive/                              # THE ARCHIVE (Development History)
├── benchmarks-development/
│   ├── MANIFEST.json                  # What, when, why
│   ├── MIGRATION.md                   # What moved where
│   ├── original-src/                  # All original code
│   └── git-history.txt                # Full commit log
```

## Implementation Pattern

### Step 1: Extract to Hub (Refactor)

**Identify shared functionality:**

```bash
# Find duplicate functions across projects
grep -r "def benchmark" organizations/*/

# Identify common patterns
grep -r "class.*Monitor" organizations/*/
```

**Extract to hub:**

```python
# Before (scattered):
# organizations/AlaweinOS/Benchmarks/utils.py
def run_benchmark(target, iterations=100):
    # benchmark logic
    pass

# organizations/AlaweinOS/MEZAN/perf.py
def run_benchmark(target, iterations=100):
    # same logic!
    pass

# After (consolidated in hub):
# .metaHub/libs/benchmarking/core.py
def run_benchmark(target, iterations=100, **kwargs):
    """
    Unified benchmark runner used across all projects.

    Originally developed in:
    - AlaweinOS/Benchmarks (initial implementation)
    - MEZAN (performance variant)
    - Optilibria (optimization variant)

    Consolidated: 2025-01-15
    Migration: See .archive/benchmarks-development/MIGRATION.md
    """
    # Single implementation
    pass
```

### Step 2: Create Hub CLIs (User Interface)

**Build standard CLI tools:**

```python
# .metaHub/clis/bench/main.py
"""
Benchmark CLI - Standard interface for all benchmarking.

Uses: .metaHub/libs/benchmarking
Replaces: Scattered benchmark scripts across projects
"""

import click
from metahub.libs.benchmarking import run_benchmark, generate_report

@click.group()
def cli():
    """Universal benchmarking CLI for all projects."""
    pass

@cli.command()
@click.argument('target')
@click.option('--iterations', default=100)
def run(target, iterations):
    """Run benchmark on target."""
    results = run_benchmark(target, iterations)
    generate_report(results)

@cli.command()
def history():
    """Show benchmark history."""
    # Load from centralized results DB
    pass

if __name__ == '__main__':
    cli()
```

**Compile to executable:**

```bash
# Build standalone CLI
pyinstaller --onefile --name bench .metaHub/clis/bench/main.py

# Result: Single executable
dist/bench

# Users run:
./bench run my-target --iterations 1000
```

### Step 3: Convert Projects to Thin Wrappers (Spokes)

**Projects become configuration + documentation:**

```bash
# organizations/AlaweinOS/Benchmarks/
# Now a thin wrapper around hub functionality

# config/benchmarks.yaml
targets:
  - name: mezan-optimizer
    type: python
    iterations: 1000
  - name: atlas-api
    type: api
    iterations: 500

# scripts/run_all_benchmarks.sh
#!/bin/bash
# Thin wrapper - just calls hub CLI
../../.metaHub/clis/bench run-config config/benchmarks.yaml

# README.md
# AlaweinOS Benchmarks

This project provides benchmark configurations for AlaweinOS projects.

## Usage
```bash
# Run all configured benchmarks
./scripts/run_all_benchmarks.sh

# Run specific benchmark
bench run mezan-optimizer --iterations 1000
```

## Architecture
- Core benchmarking logic: `.metaHub/libs/benchmarking/`
- CLI tool: `.metaHub/clis/bench`
- Development history: `.archive/benchmarks-development/`
```

### Step 4: Archive Development Artifacts (Preserve History)

**Preserve everything with full traceability:**

```bash
# .archive/benchmarks-development/MANIFEST.json
{
  "project": "AlaweinOS/Benchmarks",
  "consolidation_date": "2025-01-15",
  "hub_location": ".metaHub/libs/benchmarking/",
  "cli_location": ".metaHub/clis/bench",
  "original_files": [
    "organizations/AlaweinOS/Benchmarks/src/benchmark_utils.py",
    "organizations/AlaweinOS/Benchmarks/scripts/run_bench.py",
    "organizations/AlaweinOS/Benchmarks/helpers/perf_monitor.py"
  ],
  "git_commits": [
    "abc123 - Initial benchmark implementation",
    "def456 - Add performance monitoring",
    "ghi789 - Refactor for reusability"
  ],
  "migration_steps": "See MIGRATION.md",
  "rebuild_instructions": "See REBUILD.md"
}

# .archive/benchmarks-development/MIGRATION.md
# Benchmark Consolidation Migration

## What Happened
Original benchmark code scattered across 3 projects was consolidated into hub.

## Where Things Moved

| Original Location | New Location | Type |
|-------------------|--------------|------|
| Benchmarks/src/benchmark_utils.py | .metaHub/libs/benchmarking/core.py | Library |
| Benchmarks/scripts/run_bench.py | .metaHub/clis/bench | CLI |
| Benchmarks/helpers/perf_monitor.py | .metaHub/libs/benchmarking/monitors.py | Library |

## How to Use Now

**Old way (deprecated):**
```bash
cd organizations/AlaweinOS/Benchmarks
python scripts/run_bench.py --target foo
```

**New way:**
```bash
bench run foo
# or
cd organizations/AlaweinOS/Benchmarks
./scripts/run_all_benchmarks.sh  # Wrapper around hub CLI
```

## How to Rebuild Original

If you need the original scattered implementation:
```bash
cd .archive/benchmarks-development/original-src
# Full original code is here
```

## Git History
Full commit history preserved in git-history.txt
```

### Step 5: Distribution (User-Ready)

**What users get:**

```bash
# Clean installation
npm install -g @metahub/bench-cli
# or
pip install metahub-benchmarking

# Simple usage
bench run my-project
bench report
bench history

# No need to know about:
# - Original development structure
# - How features were built
# - Migration history
# Just works!
```

## Benefits of Hub-Spoke Pattern

### ✅ For Users
- **Simple:** Single CLI, clear documentation
- **Consistent:** Same interface everywhere
- **Fast:** Optimized, compiled tools
- **Reliable:** Well-tested core libraries

### ✅ For Developers
- **No Duplication:** Write once, use everywhere
- **Easy Maintenance:** Fix bugs in one place
- **Clear Structure:** Know where everything lives
- **Fast Development:** Compose from hub components

### ✅ For Organization
- **Preserved Work:** Nothing deleted, full history
- **Traceable:** Know where every feature came from
- **Professional:** Enterprise-grade architecture
- **Scalable:** Easy to add new projects

## Real-World Example: Your Benchmarking Use Case

### Before (Scattered)

**AlaweinOS/Benchmarks/** (19 files)
- Custom benchmark scripts
- Performance monitors
- Result reporters

**AlaweinOS/MEZAN/**
- Same benchmark logic (copy-paste)
- Slightly different interface

**AlaweinOS/Optilibria/**
- Yet another benchmark implementation

**Problem:**
- 3x duplication
- Inconsistent results
- Hard to maintain

### After (Hub-Spoke)

**Hub (`.metaHub/`):**
```
.metaHub/
├── libs/benchmarking/           # Core library (1 implementation)
│   ├── core.py                  # Unified benchmark engine
│   ├── monitors.py              # Performance monitoring
│   └── reporters.py             # Result reporting
└── clis/bench                   # Single CLI tool
```

**Spokes (Projects):**
```
organizations/AlaweinOS/Benchmarks/
├── config/benchmarks.yaml       # Just configuration
└── scripts/run.sh               # Thin wrapper

organizations/AlaweinOS/MEZAN/
└── Uses .metaHub/libs/benchmarking directly

organizations/AlaweinOS/Optilibria/
└── Uses .metaHub/libs/benchmarking directly
```

**Archive:**
```
.archive/benchmarks-development/
├── MANIFEST.json                # Full metadata
├── MIGRATION.md                 # What moved where
├── REBUILD.md                   # Rebuild instructions
└── original-src/                # All 19 original files
    └── (preserved forever)
```

**Result:**
- ✅ 19 files → 1 CLI + 3 config files
- ✅ 3 implementations → 1 implementation
- ✅ All history preserved
- ✅ User-ready distribution
- ✅ Professional architecture

## Implementation Workflow

### For Each Project (e.g., Benchmarks)

```bash
# 1. Analysis (Find what's shared)
cd organizations/AlaweinOS/Benchmarks
grep -r "def " src/ | sort > /tmp/functions.txt
# Identify reusable functions

# 2. Extract to Hub
mkdir -p .metaHub/libs/benchmarking
# Move shared code to hub
cp src/benchmark_utils.py .metaHub/libs/benchmarking/core.py
# Refactor and consolidate

# 3. Build Hub CLI
mkdir -p .metaHub/clis/bench
# Create CLI that uses hub libs

# 4. Update Project (Make it thin)
# Keep only: config, project-specific scripts, README
rm -rf src/ helpers/ utils/  # Move to archive, not delete!

# 5. Archive Original
mkdir -p .archive/benchmarks-development/original-src
cp -r <old-structure> .archive/benchmarks-development/original-src/
# Create manifest and migration docs

# 6. Test
# Verify CLI works
# Verify archive is complete
# Update documentation

# 7. Commit
git add .metaHub/ .archive/ organizations/AlaweinOS/Benchmarks/
git commit -m "refactor: consolidate Benchmarks into hub architecture

- Extract core benchmark logic to .metaHub/libs/benchmarking
- Create unified 'bench' CLI in .metaHub/clis/bench
- Convert Benchmarks project to thin config wrapper
- Archive original implementation in .archive/benchmarks-development

Nothing deleted, all history preserved.
"
```

## Hub Organization Structure

### Recommended Layout

```
.metaHub/                          # Control Center
├── libs/                          # Shared libraries
│   ├── benchmarking/
│   ├── optimization/
│   ├── data_processing/
│   ├── ml_pipelines/
│   └── common/
├── clis/                          # Standard CLIs
│   ├── bench/
│   ├── optimize/
│   ├── analyze/
│   └── orchestrate/
├── services/                      # Shared services
│   ├── api_server/
│   ├── job_queue/
│   └── monitoring/
├── schemas/                       # Validation schemas
│   ├── repo-schema.json
│   └── config-schema.json
├── templates/                     # Project templates
│   ├── python-project/
│   └── typescript-project/
├── scripts/                       # Hub automation
│   ├── enforce.py
│   ├── validate.py
│   └── build_clis.py
└── docs/                          # Hub documentation
    ├── HUB_ARCHITECTURE.md
    └── CLI_REFERENCE.md

.archive/                          # Development History
├── benchmarks-development/
├── mezan-experiments/
└── {project}-development/

organizations/                     # User-Facing Projects
├── AlaweinOS/
│   ├── Benchmarks/               # Thin wrapper
│   ├── MEZAN/                    # Consumes hub
│   └── ...
├── alaweimm90-business/
└── alaweimm90-science/
```

## Migration Checklist

For each project being rationalized:

### Phase 1: Analysis
- [ ] Identify shared functionality (what can go in hub)
- [ ] Identify unique functionality (stays in project)
- [ ] List all dependencies
- [ ] Document current usage patterns

### Phase 2: Hub Extraction
- [ ] Create hub library module
- [ ] Move shared code to hub
- [ ] Remove duplication
- [ ] Add comprehensive tests
- [ ] Document hub APIs

### Phase 3: CLI Creation
- [ ] Design CLI interface
- [ ] Implement CLI using hub libs
- [ ] Add tests for CLI
- [ ] Build executable
- [ ] Document CLI usage

### Phase 4: Project Conversion
- [ ] Remove code now in hub
- [ ] Create thin wrapper scripts
- [ ] Add project-specific config
- [ ] Update README for new structure
- [ ] Update import statements

### Phase 5: Archival
- [ ] Copy all original files to archive
- [ ] Create manifest JSON
- [ ] Write migration documentation
- [ ] Write rebuild instructions
- [ ] Preserve git history

### Phase 6: Validation
- [ ] Test hub libs work correctly
- [ ] Test CLI works correctly
- [ ] Test project still functions
- [ ] Verify archive is complete
- [ ] Update all documentation

### Phase 7: Deployment
- [ ] Commit changes
- [ ] Tag release
- [ ] Update CI/CD pipelines
- [ ] Notify team
- [ ] Monitor for issues

## Key Principles

### Never Delete, Always Archive
```
❌ git rm old_code.py
✅ mv old_code.py .archive/project-dev/original-src/
✅ git add .archive/project-dev/
```

### Preserve Traceability
Every archived file must have:
- Original location
- Git commit history
- Why it was moved
- Where it moved to
- How to rebuild it

### Make It User-Friendly
Users should never need to know about:
- Internal refactoring
- Migration history
- Development artifacts

They just use the clean CLI/API.

### Maintain Professional Standards
- Comprehensive documentation
- Full test coverage
- Semantic versioning
- CI/CD automation

---

**This is how enterprises build maintainable, scalable software.**

