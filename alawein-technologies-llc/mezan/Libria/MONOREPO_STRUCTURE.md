# Itqān Libria Suite - Monorepo Structure

> **Repository Name**: `itqan-libria` or `libria-suite`
> **License**: Dual License (Apache 2.0 for core, Commercial for advanced features)
> **Language**: Python 3.10+

---

## Directory Structure

```
itqan-libria/
├── README.md                          # Main suite overview
├── LICENSE                            # Dual licensing info
├── pyproject.toml                     # Monorepo build config
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                     # CI/CD for all packages
│   │   ├── benchmarks.yml             # Automated benchmarking
│   │   └── publish.yml                # PyPI publishing
│   └── ISSUE_TEMPLATE/
│
├── docs/                              # Unified documentation
│   ├── index.md                       # Documentation homepage
│   ├── getting-started.md
│   ├── architecture.md
│   ├── solver-comparison.md
│   └── api/
│       ├── core.md
│       ├── Librex.QAP.md
│       ├── Librex.Flow.md
│       └── ...
│
├── packages/                          # All solver packages
│   ├── libria-core/                   # Shared interfaces & utilities
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/
│   │   │   └── libria_core/
│   │   │       ├── __init__.py
│   │   │       ├── base.py            # LibriaSolver base class
│   │   │       ├── types.py           # Shared types
│   │   │       ├── metrics.py         # Performance tracking
│   │   │       └── utils.py           # Common utilities
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── Librex.QAP/                     # Agent-task assignment (QAP)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md                   # Research paper draft
│   │   ├── src/
│   │   │   └── Librex.QAP/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Main QAP solver
│   │   │       ├── gpu_kernels.py     # CUDA/CuPy kernels
│   │   │       ├── synergy_model.py   # Agent synergy modeling
│   │   │       └── benchmarks.py      # QAPLIB benchmark runner
│   │   ├── tests/
│   │   │   ├── test_solver.py
│   │   │   ├── test_gpu.py
│   │   │   └── test_benchmarks.py
│   │   └── benchmarks/
│   │       ├── qaplib/                # QAPLIB instances
│   │       └── results/               # Benchmark results
│   │
│   ├── Librex.Flow/                    # Workflow routing (MAB)
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md
│   │   ├── src/
│   │   │   └── Librex.Flow/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Confidence-aware routing
│   │   │       ├── bandits.py         # Thompson Sampling variants
│   │   │       ├── quality_model.py   # Validation quality objectives
│   │   │       └── benchmarks.py
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── Librex.Alloc/                   # Resource allocation
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md
│   │   ├── src/
│   │   │   └── Librex.Alloc/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Constrained TS
│   │   │       ├── constraints.py     # Budget/capacity handling
│   │   │       └── benchmarks.py
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── Librex.Graph/                   # Network topology
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md
│   │   ├── src/
│   │   │   └── Librex.Graph/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Info-theoretic topology
│   │   │       ├── metrics.py         # Network efficiency metrics
│   │   │       └── benchmarks.py
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── Librex.Meta/                    # Meta-optimization
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md
│   │   ├── src/
│   │   │   └── Librex.Meta/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Solver selection
│   │   │       ├── tournament.py      # Agent tournaments
│   │   │       ├── meta_features.py   # Problem characterization
│   │   │       └── benchmarks.py
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   ├── Librex.Dual/                    # Adversarial robustness
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── PAPER.md
│   │   ├── src/
│   │   │   └── Librex.Dual/
│   │   │       ├── __init__.py
│   │   │       ├── solver.py          # Min-max optimization
│   │   │       ├── adversary.py       # Adversarial agent
│   │   │       └── benchmarks.py
│   │   ├── tests/
│   │   └── benchmarks/
│   │
│   └── Librex.Evo/                     # Evolutionary NAS
│       ├── pyproject.toml
│       ├── README.md
│       ├── PAPER.md
│       ├── src/
│       │   └── Librex.Evo/
│       │       ├── __init__.py
│       │       ├── solver.py          # Agent architecture search
│       │       ├── operators.py       # Genetic operators
│       │       └── benchmarks.py
│       ├── tests/
│       └── benchmarks/
│
├── examples/                          # Usage examples
│   ├── quickstart.py
│   ├── atlas_integration.py
│   ├── custom_solver.py
│   └── benchmarking.py
│
├── benchmarks/                        # Cross-solver benchmarks
│   ├── suite_comparison.py
│   ├── ablation_studies.py
│   └── results/
│
└── scripts/                           # Development scripts
    ├── setup_dev.sh
    ├── run_all_tests.sh
    ├── run_all_benchmarks.sh
    └── generate_docs.sh
```

---

## Package Dependencies

### libria-core (Foundation)
```toml
[project]
name = "libria-core"
version = "0.1.0"
dependencies = [
    "numpy>=1.24",
    "scipy>=1.10",
    "pydantic>=2.0",
]

[project.optional-dependencies]
gpu = ["cupy>=12.0"]
viz = ["matplotlib>=3.7", "seaborn>=0.12"]
```

### Individual Solvers (Example: Librex.QAP)
```toml
[project]
name = "Librex.QAP"
version = "0.1.0"
dependencies = [
    "libria-core>=0.1.0",
    "numba>=0.57",         # JIT compilation
]

[project.optional-dependencies]
gpu = ["cupy>=12.0", "cupy-cuda11x"]
benchmarks = ["pandas>=2.0", "tqdm>=4.65"]
```

---

## Installation Patterns

### For Users (PyPI)
```bash
# Install core only
pip install libria-core

# Install specific solver
pip install Librex.QAP

# Install full suite
pip install itqan-libria[all]

# Install with GPU support
pip install Librex.QAP[gpu]
```

### For Developers (Monorepo)
```bash
# Clone and setup
git clone https://github.com/yourorg/itqan-libria.git
cd itqan-libria

# Install all packages in editable mode
pip install -e packages/libria-core
pip install -e packages/Librex.QAP
pip install -e packages/Librex.Flow
# ... repeat for all solvers

# Or use script
./scripts/setup_dev.sh
```

---

## Shared Base Class (libria-core/src/libria_core/base.py)

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel

class SolverConfig(BaseModel):
    """Base configuration for all Libria solvers"""
    seed: int = 42
    max_iterations: int = 1000
    timeout_seconds: Optional[float] = None
    enable_gpu: bool = False
    log_level: str = "INFO"

class SolverResult(BaseModel):
    """Standardized result format"""
    solution: Any
    objective_value: float
    iterations: int
    time_seconds: float
    metadata: Dict[str, Any] = {}

class LibriaSolver(ABC):
    """Base class for all Itqān Libria solvers"""

    def __init__(self, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()
        self._setup_logging()

    @abstractmethod
    def solve(self, problem: Any) -> SolverResult:
        """Solve the optimization problem"""
        pass

    @abstractmethod
    def benchmark(self, instances: list) -> Dict[str, Any]:
        """Run benchmark suite"""
        pass

    def _setup_logging(self):
        """Configure logging"""
        pass
```

---

## CI/CD Pipeline (.github/workflows/ci.yml)

```yaml
name: Itqān Libria CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
        package:
          - libria-core
          - Librex.QAP
          - Librex.Flow
          - Librex.Alloc
          - Librex.Graph
          - Librex.Meta
          - Librex.Dual
          - Librex.Evo

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e packages/libria-core
          pip install -e packages/${{ matrix.package }}[dev]

      - name: Run tests
        run: |
          cd packages/${{ matrix.package }}
          pytest tests/ --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'

    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: ./scripts/run_all_benchmarks.sh

      - name: Compare with baseline
        run: python benchmarks/suite_comparison.py
```

---

## Publishing Strategy

### Open Source (Free Tier)
- **libria-core**: Apache 2.0
- **Librex.QAP**: Apache 2.0 (basic implementation)
- **Librex.Flow**: Apache 2.0 (basic implementation)

### Commercial (Premium Features)
- **Librex.QAP-pro**: GPU acceleration, synergy modeling
- **Librex.Meta**: Full solver selection (basic free)
- **Librex.Dual**: Commercial only
- **Librex.Evo**: Commercial only

### PyPI Publishing
```bash
# Build all packages
python -m build packages/libria-core
python -m build packages/Librex.QAP
# ...

# Publish to PyPI
twine upload packages/*/dist/*
```

---

## Development Workflow

### Adding a New Solver

1. **Create package directory**:
   ```bash
   mkdir -p packages/newsolver/{src/newsolver,tests,benchmarks}
   ```

2. **Copy pyproject.toml template**:
   ```bash
   cp packages/Librex.QAP/pyproject.toml packages/newsolver/
   # Edit dependencies and metadata
   ```

3. **Implement solver**:
   ```python
   # packages/newsolver/src/newsolver/solver.py
   from libria_core.base import LibriaSolver, SolverResult

   class NewSolver(LibriaSolver):
       def solve(self, problem):
           # Implementation
           pass
   ```

4. **Write tests**:
   ```python
   # packages/newsolver/tests/test_solver.py
   import pytest
   from newsolver import NewSolver

   def test_basic_solve():
       solver = NewSolver()
       result = solver.solve(problem)
       assert result.objective_value > 0
   ```

5. **Add benchmarks**:
   ```python
   # packages/newsolver/benchmarks.py
   def run_benchmarks():
       # Benchmark suite
       pass
   ```

6. **Update monorepo docs**:
   - Add to `docs/api/newsolver.md`
   - Update `docs/solver-comparison.md`
   - Update main `README.md`

---

## Repository Setup Checklist

### Initial Setup
- [ ] Create GitHub repository: `itqan-libria`
- [ ] Set up branch protection (main, develop)
- [ ] Configure GitHub Actions secrets (PyPI tokens)
- [ ] Set up Codecov integration
- [ ] Create project board for roadmap tracking

### Package Setup
- [ ] Implement `libria-core` base classes
- [ ] Set up package templates for consistency
- [ ] Configure pre-commit hooks (black, ruff, mypy)
- [ ] Create testing infrastructure (pytest, fixtures)
- [ ] Set up documentation generation (Sphinx/MkDocs)

### Infrastructure
- [ ] Set up CI/CD pipeline for all packages
- [ ] Configure automated benchmarking
- [ ] Set up dependency management (Dependabot)
- [ ] Create Docker containers for reproducibility
- [ ] Set up benchmark result tracking

---

## Monorepo Management Tools

### Recommended: `Poetry` with workspace support
```toml
# Root pyproject.toml
[tool.poetry]
name = "itqan-libria"
version = "0.1.0"

[tool.poetry.dependencies]
python = "^3.10"

[tool.poetry.group.dev.dependencies]
pytest = "^7.4"
black = "^23.0"
ruff = "^0.1"
mypy = "^1.6"

# Workspace configuration
[tool.poetry.workspace]
packages = ["packages/*"]
```

### Alternative: `pip` with editable installs
```bash
# Install all packages in development mode
for pkg in packages/*/; do
    pip install -e "$pkg"
done
```

---

## Next Steps

1. **This Week**: Set up repository structure and libria-core
2. **Week 2**: Implement Librex.QAP and integrate with monorepo
3. **Week 3**: Add Librex.Flow and establish CI/CD patterns
4. **Week 4**: Complete remaining solvers using established patterns

---

## Resources

- **Monorepo Best Practices**: https://monorepo.tools
- **Poetry Workspaces**: https://python-poetry.org/docs/managing-dependencies/
- **GitHub Actions**: https://docs.github.com/en/actions
- **PyPI Publishing**: https://packaging.python.org/tutorials/packaging-projects/

---

*Monorepo structure designed for maximum developer productivity and research reproducibility*
*Created: November 14, 2025*
