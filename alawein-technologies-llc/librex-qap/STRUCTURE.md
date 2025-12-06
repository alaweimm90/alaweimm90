# Librex.QAP-new Directory Structure Guide

Complete guide to understanding and navigating the Librex.QAP-new project structure.

---

## Directory Map

```
Librex.QAP-new/
│
├── 📄 Root Documentation (START HERE)
│   ├── README.md                 ← Quick start & overview
│   ├── PROJECT.md                ← Complete project details
│   ├── STRUCTURE.md              ← This file
│   ├── DEVELOPMENT.md            ← How to develop
│   ├── CONTRIBUTING.md           ← How to contribute
│   └── CHANGELOG.md              ← Version history
│
├── 🔧 Configuration Files
│   ├── pyproject.toml            ← Python package configuration
│   ├── pytest.ini                ← Testing configuration
│   ├── Makefile                  ← Development commands
│   ├── LICENSE                   ← MIT License
│   └── .gitignore                ← Git ignore rules
│
├── 📊 Librex.QAP/                 [PRIMARY: Optimization Engine]
│   ├── README.md                 ← Librex.QAP module guide
│   ├── __init__.py               ← Package initialization
│   ├── core/                     ← CORE OPTIMIZATION
│   │   ├── __init__.py
│   │   ├── pipeline.py           ← Main optimization pipeline (CENTRAL)
│   │   └── [Tests: test_pipeline_exhaustive.py]
│   ├── methods/                  ← ALGORITHMS
│   │   ├── __init__.py
│   │   ├── novel.py              ← 7 novel methods
│   │   ├── baselines.py          ← 9 baseline algorithms
│   │   ├── metadata.py           ← Method metadata & registry
│   │   └── [Tests: test_methods.py]
│   ├── utils.py                  ← Core utilities (~1000 LOC)
│   ├── validation.py             ← Validation framework
│   ├── logging_config.py         ← Logging setup
│   ├── benchmarking_suite.py     ← Benchmarking utilities
│   ├── championship_visualizer.py ← Result visualization
│   ├── breakthrough_pursuit.py   ← Advanced optimization pursuit
│   ├── pipeline_dispatcher.py    ← Pipeline routing
│   ├── plots.py                  ← Plotting utilities
│   ├── run_championship.py       ← Championship runner
│   └── tables.py                 ← Table generation
│
├── 🤖 ORCHEX/                     [PRIMARY: Autonomous Research System]
│   ├── README.md                 ← ORCHEX module guide
│   ├── ORCHEX/                    ← Main ORCHEX module
│   │   ├── __init__.py           ← Agent registry & initialization
│   │   ├── brainstorming/        ← Hypothesis generation
│   │   │   ├── __init__.py
│   │   │   └── brainstorm_engine.py
│   │   ├── experimentation/      ← Experiment design & execution
│   │   │   ├── __init__.py
│   │   │   ├── code_generator.py
│   │   │   ├── experiment_designer.py
│   │   │   └── sandbox_executor.py
│   │   ├── learning/             ← Learning mechanisms
│   │   │   ├── __init__.py
│   │   │   └── advanced_bandits.py
│   │   ├── orchestration/        ← Workflow orchestration
│   │   │   ├── __init__.py
│   │   │   ├── workflow_orchestrator.py
│   │   │   ├── intent_classifier.py
│   │   │   └── problem_types.py
│   │   ├── publication/          ← Paper generation (v0.2+)
│   │   │   ├── __init__.py
│   │   │   └── paper_generator.py
│   │   ├── cli.py                ← Command-line interface
│   │   ├── diagnostics.py        ← Diagnostic tools
│   │   ├── hypothesis_generator.py
│   │   ├── performance_utils.py
│   │   ├── protocol.py           ← Core protocol definitions
│   │   └── [Tests: test_integration.py]
│   └── uaro/                     ← Universal solver integration
│       ├── __init__.py
│       ├── atlas_integration.py
│       ├── explainability.py
│       ├── marketplace.py
│       ├── reasoning_primitives.py
│       └── universal_solver.py
│
├── ✅ tests/                     [TESTING: All test files]
│   ├── __init__.py
│   ├── test_pipeline_exhaustive.py  ← Pipeline tests (512 lines)
│   ├── test_methods.py              ← Method validation (477 lines)
│   ├── test_integration.py          ← ORCHEX-Librex.QAP integration (206 lines)
│   ├── test_benchmarks.py           ← Benchmark tests (70 lines)
│   ├── test_utils_core.py           ← Utility tests (593 lines)
│   └── test_validation.py           ← Validation tests (281 lines)
│   [Total: 149 tests, ~2,100 LOC]
│
├── 💡 examples/                  [USAGE: Example scripts]
│   ├── README.md                 ← Examples guide
│   ├── 01_sudoku_solver.py       ← Optimization example
│   ├── 02_path_planning.py       ← Path finding example
│   ├── 03_n_queens.py            ← N-Queens problem
│   ├── 04_logic_puzzle.py        ← Logic puzzle solver
│   ├── 05_optimization.py        ← Librex.QAP optimization example
│   ├── 06_atlas_uaro_integration.py ← ORCHEX integration example
│   ├── personality_agents_demo.py   ← Personality agents demo
│   └── proofs/                   ← Proof files for examples
│       ├── logic_puzzle_proof.md
│       ├── n_queens_proof.md
│       ├── path_planning_proof.md
│       └── tsp_proof.md
│
├── 📦 data/                      [DATA: Benchmark instances]
│   └── qaplib/                   ← QAPLIB benchmark instances
│       ├── README.md             ← Data guide
│       ├── MANIFEST.md           ← Manifest of instances
│       ├── validate_format.py    ← Data validation script
│       ├── chr12c.dat, chr20a.dat, chr20b.dat     ← Small (12-20)
│       ├── had12.dat, had20.dat                   ← Small (12-20)
│       ├── nug12.dat, nug20.dat                   ← Small (12-20)
│       ├── rou20.dat, tai12a.dat, tai20a.dat      ← Small-Medium
│       ├── ste36a.dat, tai30a.dat                 ← Medium
│       ├── tai40a.dat, tai50a.dat                 ← Large (40-50)
│       ├── DOWNLOAD_REPORT.txt
│       └── LARGE_INSTANCES_DOWNLOAD_REPORT.md
│
├── 📚 docs/                      [ACTIVE DEVELOPMENT DOCS]
│   ├── development/              ← Development notes
│   │   └── [Add architecture & design notes here]
│   └── guides/                   ← Feature guides
│       └── [Add how-to guides here]
│
└── 📁 .archive/                  [HISTORICAL REFERENCE]
    ├── README.md                 ← Archive guide
    ├── docs/                     ← Archived documentation (50+ files)
    │   ├── ORCHEX/                ← ORCHEX architecture docs
    │   │   ├── ATLAS_ARCHITECTURE.md
    │   │   ├── DEVELOPER_GUIDE.md
    │   │   ├── INTEGRATION_GUIDE.md
    │   │   ├── QUICK_START_GUIDE.md
    │   │   ├── MASTER_EXECUTION_PLAN.md
    │   │   ├── COMPREHENSIVE_ACTION_PLAN.md
    │   │   ├── CYCLES_27-41_FINAL_REPORT.md
    │   │   ├── PILOT_EXECUTION_PLAN.md
    │   │   └── README.md
    │   └── Librex.QAP/             ← Librex.QAP docs
    │       ├── FORMULA_REFERENCES.md (KEY!)
    │       ├── AGENT*_*.md        ← Quality reports
    │       ├── CODE_*.md          ← Code reviews
    │       ├── PUBLICATION_*.md   ← Certification
    │       ├── TODO.md
    │       └── ... (40+ docs)
    └── results/                   ← Archived benchmark results
        ├── BENCHMARK_EXECUTIVE_SUMMARY.md
        ├── INITIAL_BENCHMARK_ANALYSIS.md
        ├── LITERATURE_COMPARISON.md
        ├── N20_BENCHMARK_ANALYSIS.md
        ├── initial_benchmark_results.csv
        ├── n20_benchmark_results.csv
        └── initial_benchmark_summary.txt
```

---

## Quick Navigation Guide

### I Want To...

**Understand the project**
→ Start with `README.md`, then `PROJECT.md`

**Set up for development**
→ `README.md` → `DEVELOPMENT.md` → `make install-dev`

**Add a new optimization method**
→ `CONTRIBUTING.md` → `docs/guides/adding-methods.md` → edit `Librex.QAP/methods/novel.py`

**Use the library**
→ `examples/05_optimization.py` → `Librex.QAP/core/pipeline.py` docstrings

**Understand validation**
→ `PROJECT.md` (ORCHEX section) → `ORCHEX/` files → `test_integration.py`

**Extend ORCHEX**
→ `ORCHEX/ORCHEX/__init__.py` → docs/guides/extending-agents.md (coming)

**View archived info**
→ `.archive/docs/` (for historical reference only)

**Find benchmark data**
→ `data/qaplib/` → 14 instances ready to use

**Run tests**
→ `make test` → `tests/` folder structure

**Check git history**
→ `CHANGELOG.md` → `git log`

---

## File Roles & Responsibilities

### Core Architecture Files

| File | Purpose | Owner | Status |
|------|---------|-------|--------|
| `Librex.QAP/core/pipeline.py` | Main optimization | Librex.QAP | CRITICAL |
| `Librex.QAP/methods/novel.py` | Novel methods | Librex.QAP | Active |
| `Librex.QAP/methods/baselines.py` | Baseline methods | Librex.QAP | Reference |
| `ORCHEX/ORCHEX/__init__.py` | Agent registry | ORCHEX | CRITICAL |
| `ORCHEX/orchestration/workflow_orchestrator.py` | Main orchestrator | ORCHEX | Critical |

### Configuration

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python package config (dependencies, metadata) |
| `pytest.ini` | Testing configuration |
| `Makefile` | Development commands |
| `.gitignore` | Git ignore rules |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Quick start (entry point) |
| `PROJECT.md` | Complete overview |
| `STRUCTURE.md` | This file (navigation) |
| `DEVELOPMENT.md` | Development workflow |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history |

### By Directory Responsibility

**Librex.QAP/** (Optimization)
- Core: Pipeline, methods, utilities
- Tests: Pipeline exhaustive, methods, utils
- Owner: Optimization researcher

**ORCHEX/** (Research Validation)
- Core: Orchestration, agents, learning
- Tests: Integration, benchmarks
- Owner: Research systems designer

**tests/** (Quality Assurance)
- All test files
- Owner: Development team

**examples/** (Usage & Learning)
- Example scripts and proofs
- Owner: Community

**data/** (Benchmarks)
- Benchmark instances
- Owner: Reference (don't modify)

**docs/** (Development Docs)
- guides/: How-to guides
- development/: Architecture notes
- Owner: Everyone (collaborative)

**.archive/** (Reference Only)
- Historical documentation
- Old results
- Owner: Reference (read-only)

---

## Data Flow & Dependencies

```
┌─────────────────────────────────────────────┐
│  User/ORCHEX                                 │
│  Provides QAP problem instance              │
└──────────────┬────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│  Librex.QAP/core/pipeline.py                 │
│  (Main optimization pipeline)               │
└──────────────┬────────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    ┌────────┐    ┌──────────┐
    │ Novel  │    │ Baseline │
    │ Methods│    │ Methods  │
    └────┬───┘    └────┬─────┘
         │             │
         └──────┬──────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Librex.QAP/utils.py      │
    │ (Utilities & helpers)   │
    └────────────┬────────────┘
                 │
                 ▼
    ┌──────────────────────────────┐
    │ Return OptimizationResult    │
    │ (solution, objective, etc.)  │
    └──────────────┬───────────────┘
                   │
    ┌──────────────┴─────────────┐
    │                            │
    ▼                            ▼
ORCHEX Validation         User/Benchmark
(test_integration.py)    (examples/)
```

---

## Module Dependencies

### Librex.QAP Dependencies
```
Librex.QAP/
├── numpy, scipy       (numerical computation)
├── pandas             (data handling)
├── matplotlib/plotly  (visualization)
└── tests → pytest
```

### ORCHEX Dependencies
```
ORCHEX/
├── numpy, scipy       (numerical computation)
├── requests           (HTTP for literature search - optional)
└── tests → pytest
```

### Development Tools
```
Development/Testing
├── pytest             (testing framework)
├── black              (code formatting)
├── ruff               (linting)
├── mypy               (type checking)
├── pytest-cov         (coverage)
└── Other: numpy, scipy, pandas
```

---

## How Files Are Organized

### By Layer

**Core Engine Layer**
- `Librex.QAP/core/pipeline.py` - Central coordination
- `ORCHEX/orchestration/workflow_orchestrator.py` - Central coordination

**Implementation Layer**
- `Librex.QAP/methods/` - Optimization algorithms
- `ORCHEX/ORCHEX/brainstorming/` - Research generation
- `ORCHEX/ORCHEX/learning/` - Agent learning

**Support Layer**
- `Librex.QAP/utils.py` - Core utilities
- `ORCHEX/performance_utils.py` - Performance tools
- `**/validation.py` - Validation tools

**Testing Layer**
- `tests/test_*.py` - Unit & integration tests

**Interface Layer**
- `examples/` - Usage examples
- `ORCHEX/cli.py` - Command-line interface

### By Purpose

**Computation**
- All files in `Librex.QAP/methods/`
- `Librex.QAP/core/pipeline.py`
- `ORCHEX/learning/`

**Orchestration**
- `Librex.QAP/core/pipeline.py`
- `ORCHEX/orchestration/`

**Analysis**
- `Librex.QAP/benchmarking_suite.py`
- `Librex.QAP/championship_visualizer.py`
- `ORCHEX/diagnostics.py`

**Validation**
- `Librex.QAP/validation.py`
- `tests/` all files

---

## Creation & Modification Guide

### When Adding New Code

**Step 1: Determine Category**
- Optimization → `Librex.QAP/`
- Research → `ORCHEX/`
- Testing → `tests/`
- Example → `examples/`

**Step 2: Find Right Location**
- Core algorithm → `methods/` or `brainstorming/`
- Utility → `utils.py` or new `module.py`
- Test → `test_*.py` with matching name

**Step 3: Follow Structure**
- Add docstrings
- Import from `__init__.py`
- Register in metadata if needed
- Write tests in `tests/`

**Step 4: Document**
- Update `CHANGELOG.md`
- Create guide in `docs/guides/` if needed
- Update README in that directory

### When Refactoring

**Step 1: Plan**
- Identify what's changing
- Plan new structure
- Update tests first (TDD)

**Step 2: Implement**
- Make changes
- Run `make check-all`
- Update documentation

**Step 3: Document**
- Update `CHANGELOG.md`
- Update affected READMEs
- Update docstrings

### When Fixing Issues

**Step 1: Reproduce**
- Create test that fails
- Fix the issue
- Test passes

**Step 2: Document**
- Update `CHANGELOG.md`
- Link to issue number
- Commit with clear message

---

## Files to NEVER Modify

```
.archive/                     (Reference only, read-only)
data/qaplib/*.dat             (Benchmark data, read-only)
.git/                         (Managed by git)
.gitignore                    (Only if updating ignore rules)
LICENSE                       (Unless changing license)
```

---

## Files That Need Regular Updates

```
CHANGELOG.md                  (After every change)
docs/guides/*.md              (When adding features)
README.md (in directories)    (When reorganizing)
examples/                     (Keep synchronized with code)
```

---

## Adding New Directories

When adding a new subdirectory:

1. Create folder: `mkdir -p new_folder`
2. Create `__init__.py` in folder
3. Create `README.md` explaining its purpose
4. Add files following conventions
5. Update `STRUCTURE.md`
6. Commit with clear message

Example:
```bash
mkdir -p Librex.QAP/new_module
touch Librex.QAP/new_module/__init__.py
# Write Librex.QAP/new_module/README.md
# Add implementation files
```

---

## Testing a Complete File Tree

```bash
# Verify structure is valid
make test                    # Run all tests
make lint                    # Check imports
make format-check            # Check formatting

# Find issues
find . -name "*.py" -exec python -m py_compile {} \;
```

---

## Quick Reference: File Sizes

| Directory | Files | Purpose |
|-----------|-------|---------|
| Librex.QAP/ | ~20 | Optimization (primary) |
| ORCHEX/ | ~25 | Research validation (primary) |
| tests/ | 6 | Quality assurance |
| examples/ | 8 | Usage examples |
| data/qaplib/ | 14 | Benchmark data |
| docs/ | ~2 | Active development docs |
| .archive/docs/ | ~50 | Historical reference |
| Root | 6 | Configuration |

---

## Navigation Tips

1. **Always start with README.md** in the directory
2. **Check __init__.py** for module exports
3. **Follow docstrings** for implementation details
4. **Look in tests/test_*.py** for usage examples
5. **Check examples/** for complete workflows
6. **Refer to .archive/docs/** for historical context

---

## Directory Health Checklist

- ✅ No orphaned files
- ✅ Consistent naming (lowercase-with-hyphens for dirs, lowercase_with_underscores for files)
- ✅ README.md in major directories
- ✅ __init__.py in all Python packages
- ✅ All imports resolvable
- ✅ No circular dependencies
- ✅ Tests for all modules
- ✅ Documentation up-to-date
- ✅ CHANGELOG.md current
- ✅ .archive/ for historical only

---

## Summary: Why This Structure Works

| Aspect | Solution |
|--------|----------|
| **Clarity** | Clear separation: Librex.QAP vs ORCHEX |
| **Scalability** | Easy to add new methods/agents |
| **Testing** | Tests parallel main structure |
| **Documentation** | READMEs at every level |
| **Navigation** | Consistent patterns throughout |
| **Maintenance** | Active vs archived separation |
| **Collaboration** | Clear ownership & boundaries |
| **Growth** | Room for expansion without refactoring |

---

## Next Steps

1. **Exploring?** Start with `README.md`, then this file
2. **Developing?** Read `DEVELOPMENT.md` and check Makefile
3. **Contributing?** Read `CONTRIBUTING.md` and pick an area
4. **Extending?** Check `examples/` and docs/guides/ for patterns

---

**Happy navigating!** This structure is designed to make finding and modifying code intuitive and professional. 🚀

---

Last Updated: November 2024
Structure Version: 1.0
Status: Production-Ready
