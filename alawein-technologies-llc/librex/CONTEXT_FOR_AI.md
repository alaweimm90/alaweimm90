# Librex - Context for AI Assistants

**Project**: Librex - Universal Optimization Framework  
**Version**: 1.0.0 (Beta)  
**Status**: Pre-publication, Active Development  
**Last Updated**: 2025-01-27

---

## 📋 Quick Context

Librex is a **universal optimization framework** providing a unified interface for solving optimization problems across different domains. Currently focused on combinatorial optimization (QAP, TSP) with extensible architecture for continuous, multi-objective, and constraint optimization.

**Key Characteristics**:
- 🎯 Domain-agnostic architecture via Universal Adapter Pattern
- 🔬 Research-grade with patent-pending novel methods
- 📊 95% test coverage, 138 tests passing
- 📚 Comprehensive documentation (25 theory chapters)
- 🚀 Pre-publication status (papers in preparation)

---

## 🎯 Project Overview

### Purpose
Provide a unified, extensible framework for optimization that:
1. Works across any domain (QAP, TSP, scheduling, portfolio, etc.)
2. Supports multiple optimization methods (5 implemented, 15+ planned)
3. Enables fair comparison and benchmarking
4. Facilitates research into novel optimization algorithms

### Current Capabilities
- ✅ QAP Domain Adapter (95% coverage, QAPLIB benchmarks)
- ✅ TSP Domain Adapter (98% coverage)
- ✅ 5 Optimization Methods (Random, SA, Local, GA, Tabu)
- ✅ AI Method Selector (intelligent algorithm selection)
- ✅ Visualization Tools (publication-quality plots)
- ✅ Comprehensive Test Suite (138 tests)

### In Development
- 🚧 FFT-Laplace Preconditioning (Patent Pending)
- 🚧 Reverse-Time Saddle Escape (Patent Pending)
- 🚧 Additional Domain Adapters (Portfolio, Scheduling)
- 🚧 Extended Method Library (15+ algorithms)

---

## 🏗️ Architecture Overview

### Universal Adapter Pattern

The core architectural innovation is the **Universal Adapter Pattern**, which enforces strict separation of concerns:

```
┌─────────────────┐
│ Domain Problem  │ (QAP, TSP, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Domain Adapter  │ (encode, validate, decode)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Standardized    │ (dimension, objective_fn, bounds)
│ Problem         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Method          │ (domain-agnostic optimization)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Solution        │
└─────────────────┘
```

**Key Principle**: Methods see only `objective_fn`, `dimension`, `bounds`, `is_discrete`. They have **zero knowledge** of domain semantics.

### Directory Structure

```
Librex/
├── core/
│   ├── interfaces/          # Universal optimization interfaces
│   │   ├── problem.py       # StandardizedProblem, Solution
│   │   ├── method.py        # Method base class
│   │   └── adapter.py       # DomainAdapter base class
│   └── registry/            # Domain & method registration
│       ├── domain_registry.py
│       └── method_registry.py
│
├── adapters/
│   ├── qap/                 # QAP adapter (95% coverage)
│   │   ├── adapter.py       # QAPAdapter
│   │   ├── loader.py        # QAPLIB loader
│   │   └── validator.py     # QAP validation
│   └── tsp/                 # TSP adapter (98% coverage)
│       ├── adapter.py       # TSPAdapter
│       └── validator.py     # TSP validation
│
├── methods/
│   ├── baselines/           # Classical methods (99%+ coverage)
│   │   ├── random_search.py
│   │   ├── simulated_annealing.py
│   │   ├── local_search.py
│   │   ├── genetic_algorithm.py
│   │   └── tabu_search.py
│   └── novel/               # Novel algorithms (research-grade)
│       ├── fft_laplace.py   # FFT-Laplace Preconditioning
│       └── reverse_time_saddle.py  # Reverse-Time Saddle Escape
│
├── utils/
│   └── qaplib_loader.py     # 138 benchmark instances
│
├── visualization/           # Publication-quality plotting
│   ├── convergence.py
│   ├── tours.py
│   └── comparisons.py
│
├── ai/
│   └── method_selector.py   # Intelligent method selection
│
└── Librex.QAP/              # QAP convenience layer
    └── __init__.py         # optimize_qap() facade
```

---

## 📊 Method Taxonomy

### Status Tags

All methods are classified with explicit status tags:

- **implemented**: Method has concrete implementation in codebase, fully tested
- **embedded**: Method exists as sub-component inside another method
- **spec_only**: Method is documented conceptually but not yet implemented

**CRITICAL**: Always reference methods with their status tag to avoid confusion.

### Implemented Methods (5)

| Method | Family | Status | Coverage | Code Location |
|--------|--------|--------|----------|---------------|
| Random Search | Baseline | implemented | 100% | `methods/baselines/random_search.py` |
| Simulated Annealing | Metaheuristic | implemented | 99% | `methods/baselines/simulated_annealing.py` |
| Local Search | Hill Climbing | implemented | 99% | `methods/baselines/local_search.py` |
| Genetic Algorithm | Evolutionary | implemented | 100% | `methods/baselines/genetic_algorithm.py` |
| Tabu Search | Metaheuristic | implemented | 100% | `methods/baselines/tabu_search.py` |

### Novel Methods (Research-Grade)

| Method | Paper | Status | Code Location | Tests |
|--------|-------|--------|---------------|-------|
| FFT-Laplace Preconditioning (FLPGF) | Paper 1 | implemented | `methods/novel/fft_laplace.py` | `tests/unit/test_fft_laplace.py` |
| Reverse-Time Saddle Escape | Patent 002 | implemented | `methods/novel/reverse_time_saddle.py` | `tests/unit/test_reverse_time_saddle.py` |
| IMEX on Birkhoff Polytope | Paper 1 | embedded | FFT-Laplace dynamics | Indirect coverage |
| Eigen-Mode Saddle Escaper (EMSE) | Paper 1 | embedded | `SaddlePointDetector` | `test_reverse_time_saddle.py` |
| Reverse-Time Homotopy Warm-Start (RTH) | Paper 1 | spec_only | - | - |
| Fractional-Order Flow (Frac-IMEX) | Paper 2 | spec_only | - | - |
| Entropy-Regularized Projection (ECCP) | Paper 2 | spec_only | - | - |
| GNN-Regularized Birkhoff Flow (GNN-BF) | Paper 3 | spec_only | - | - |

**See**: `docs/theory/NOVEL_METHODS.md` for complete inventory (18 methods total)

### Method Families

1. **Baseline Methods**: Random, Local Search
2. **Metaheuristics**: SA, Tabu Search
3. **Evolutionary**: Genetic Algorithm
4. **Novel Continuous**: FFT-Laplace, Reverse-Time Saddle
5. **Planned**: Ant Colony, Particle Swarm, Quantum-Inspired

---

## 🔧 Domain Adapter Architecture

### Adapter Contract

Every domain adapter must implement:

```python
class DomainAdapter:
    def encode_problem(self, domain_problem) -> StandardizedProblem:
        """Convert domain-specific problem to standardized form."""
        
    def compute_objective(self, solution: Solution) -> float:
        """Evaluate objective function for a solution."""
        
    def validate_solution(self, solution: Solution) -> ValidationResult:
        """Check if solution satisfies all constraints."""
        
    def decode_solution(self, solution: Solution) -> DomainSolution:
        """Convert standardized solution back to domain representation."""
```

### Current Adapters

**QAPAdapter** (`adapters/qap/adapter.py`):
- Encodes QAP as permutation problem
- Objective: `trace(F * P * D * P^T)` where P is permutation matrix
- Validation: Checks permutation is valid (no duplicates, all indices present)
- QAPLIB integration: 138 benchmark instances

**TSPAdapter** (`adapters/tsp/adapter.py`):
- Encodes TSP as permutation of cities
- Objective: Sum of distances along tour
- Validation: Checks Hamiltonian cycle
- Supports both coordinate and distance matrix input

### Adding New Domains

To add a new domain (e.g., scheduling, portfolio):

1. Create adapter class implementing the contract
2. Register with `DomainAdapterRegistry`
3. Add validation tests
4. Add benchmark instances
5. Document in `docs/domains/`

**All existing methods automatically work with new domains** via the Universal Adapter Pattern.

---

## 🧪 Testing Strategy

### Test Structure

```
tests/
├── unit/                    # Unit tests (51 tests, core profile)
│   ├── test_qap_adapter.py
│   ├── test_tsp_adapter.py
│   ├── test_methods.py
│   ├── test_fft_laplace.py
│   └── test_reverse_time_saddle.py
│
├── integration/             # Integration tests
│   ├── test_adapter_method_integration.py
│   └── test_novel_methods_integration.py
│
├── e2e/                     # End-to-end tests
│   └── test_end_to_end_flow.py
│
└── performance/             # Performance/regression tests
    └── test_benchmarks.py
```

### Test Profiles

**Core Profile** (fast, always run):
```bash
pytest -m "unit and not fft_laplace and not theorem"
```
- Runs in < 30 seconds
- No optional dependencies required
- Should always pass on standard dev machine

**Full Profile** (comprehensive):
```bash
pytest tests/
```
- Includes all tests
- May require optional dependencies
- Some tests may be skipped if deps unavailable

**Novel Methods Profile** (research-grade):
```bash
pytest -m fft_laplace
```
- Tests FFT-Laplace method
- May fail during active development
- Performance and convergence checks

**Theorem Prover Profile** (requires z3):
```bash
pytest -m theorem
```
- Requires `pip install .[theorem]`
- Tests formal verification components
- Skipped automatically if z3 not available

### Test Markers

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.e2e          # End-to-end test
@pytest.mark.performance  # Performance test
@pytest.mark.fft_laplace  # FFT-Laplace specific
@pytest.mark.theorem      # Requires z3-solver
@pytest.mark.benchmark    # Requires pytest-benchmark
```

### Coverage Requirements

- **Minimum**: 85% overall coverage
- **Target**: 95% coverage
- **Current**: 95% coverage (51/51 core tests passing)

---

## 🔐 Exception Handling Policy

### "Option B" Pattern

Librex follows an **explicit exception handling policy** documented in planning docs:

**Outer Orchestrators/Agents**:
- ✅ **Broad exception handlers allowed** (`except Exception`)
- Purpose: Robustness in multi-stage workflows
- Must log errors comprehensively
- Should attempt graceful degradation

**Inner Helpers/Methods**:
- ⚠️ **Narrow exception types only**
- Use specific exceptions where failure modes are known
- Example: `ValueError` for invalid input, `ConvergenceError` for optimization failure
- Document expected exceptions in docstrings

**Example**:
```python
# ✅ GOOD: Outer orchestrator
def run_optimization_pipeline(problem, methods):
    results = []
    for method in methods:
        try:
            result = optimize(problem, method)
            results.append(result)
        except Exception as e:
            logger.error(f"Method {method} failed: {e}")
            results.append(None)  # Graceful degradation
    return results

# ✅ GOOD: Inner helper
def validate_qap_matrices(flow, distance):
    if flow.shape != distance.shape:
        raise ValueError(f"Matrix shape mismatch: {flow.shape} vs {distance.shape}")
    if not np.allclose(flow, flow.T):
        raise ValueError("Flow matrix must be symmetric")
```

**CRITICAL**: Do NOT propose removing broad exception handlers in orchestrators without understanding this policy.

---

## 📦 Optional Dependencies

### Dependency Groups

Librex uses **optional dependency groups** to keep core lightweight:

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "ruff", "mypy"]
quantum = ["qiskit>=0.40.0", "pennylane>=0.30.0"]
ml = ["torch>=2.0.0", "optuna>=3.0.0", "ray[tune]>=2.0.0"]
docs = ["sphinx>=5.0.0", "sphinx-rtd-theme>=1.2.0"]
theorem = ["z3-solver>=4.12.0"]
```

### Installation

```bash
# Core only (minimal)
pip install Librex

# With development tools
pip install Librex[dev]

# With quantum support
pip install Librex[quantum]

# With ML/hyperparameter tuning
pip install Librex[ml]

# With theorem prover (formal verification)
pip install Librex[theorem]

# All optional dependencies
pip install Librex[dev,quantum,ml,docs,theorem]
```

### Graceful Degradation

**CRITICAL RULE**: All optional dependencies must be handled gracefully.

**Pattern**:
```python
# ✅ GOOD: Graceful degradation
try:
    from z3 import Solver, Int
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

def verify_with_z3(problem):
    if not Z3_AVAILABLE:
        logger.warning("z3-solver not available, skipping verification")
        return None
    # Use z3...
```

**Test Handling**:
```python
# ✅ GOOD: Skip tests when optional dep unavailable
@pytest.mark.theorem
@pytest.mark.skipif(not Z3_AVAILABLE, reason="z3-solver not installed")
def test_theorem_prover():
    # Test code...
```

### Platform-Specific Dependencies

Some dependencies are platform-specific:

- **resource** module: Unix/Linux only (not available on Windows)
- **CUDA**: Requires NVIDIA GPU and drivers
- **z3-solver**: May have platform-specific build requirements

**Always wrap platform-specific imports**:
```python
try:
    import resource
    RESOURCE_AVAILABLE = True
except ImportError:
    RESOURCE_AVAILABLE = False  # Windows
```

---

## 📚 Documentation Structure

### Theory Documentation (25 Chapters)

Located in `docs/`:

1. **Part I: Foundations** (Chapters 1-4)
   - Introduction, Problem Definitions, Theoretical Foundations, Metrics

2. **Part II: Exact Methods** (Chapters 5-8)
   - ILP, Branch & Bound, Cutting Planes, Dynamic Programming

3. **Part III: Heuristics** (Chapters 9-11)
   - Greedy, Insertion, GRASP

4. **Part IV: Metaheuristics** (Chapters 12-18)
   - Local Search, VNS, ILS, SA, Tabu, GA, ACO

5. **Part V: Advanced** (Chapters 19-22)
   - Learning-Based, Matheuristics, LNS, Parallel

6. **Part VI: Applications** (Chapters 23-25)
   - Real-World, Software Tools, Future Directions

**IMPORTANT**: Not all methods described in chapters are implemented. Check `NOVEL_METHODS.md` for implementation status.

### API Documentation

- **User Guide**: `docs/USER_GUIDE.md`
- **Domain Adapter Architecture**: `docs/theory/DOMAIN_ADAPTER_ARCHITECTURE.md`
- **Novel Methods Inventory**: `docs/theory/NOVEL_METHODS.md`
- **Patent Summaries**: `docs/theory/PATENT_FFT_LAPLACE.md`, etc.

### Code Documentation

- All public functions have Google-style docstrings
- Type hints on all functions
- Examples in docstrings where helpful

---

## 🚀 Publication Status

### Pre-Publication

Librex is in **pre-publication status** with papers in preparation:

**Paper 1**: "FFT-Laplace Preconditioned Flows on the Birkhoff Polytope"
- Target: ICML/IPCO 2025
- Status: In preparation
- Methods: FFT-Laplace, IMEX, RTH, EMSE

**Paper 2**: "Fractional-Order IMEX Dynamics with Reverse-Time Continuation"
- Target: ICLR/AISTATS 2025
- Status: In preparation
- Methods: Frac-IMEX, PP-RTC, ECCP, EGT

**Paper 3**: "Structure-Aware Priors: TDA/GNN Regularization"
- Target: IPCO/Mathematical Programming 2025
- Status: In preparation
- Methods: GNN-BF, PHDI, ITES, QSPR

### Patent Applications

**Patent 001**: FFT-Laplace Preconditioning
- Status: Patent Pending
- Claims: O(n² log n) algorithm for QAP

**Patent 002**: Reverse-Time Saddle Escape
- Status: Patent Pending
- Claims: Time-reversed gradient dynamics for saddle point escape

**Patent 003**: Universal Domain Adapter Architecture
- Status: In preparation
- Claims: Extensible framework design

### Implications for Changes

**CRITICAL CONSIDERATIONS**:

1. **No Breaking Changes to Novel Methods**
   - FFT-Laplace and Reverse-Time Saddle are patent-pending
   - Changes must not affect patent claims
   - Maintain mathematical correctness

2. **Method Naming Must Be Preserved**
   - Published method names cannot change
   - Internal refactoring OK, but public API stable

3. **Documentation Must Align with Papers**
   - Theory docs are SSOT for paper content
   - Code must match documented algorithms
   - Any drift must be documented

4. **Test Coverage Must Remain High**
   - Novel methods must maintain >90% coverage
   - Convergence tests are critical for publication

---

## 🎯 Development Guidelines

### When Working on Librex

1. **Read Context First**
   - This file (CONTEXT_FOR_AI.md)
   - NOVEL_METHODS.md for method status
   - CONTRIBUTING.md for code standards

2. **Respect Status Tags**
   - Don't implement spec_only methods without approval
   - Don't claim embedded methods are standalone
   - Check NOVEL_METHODS.md before making claims

3. **Maintain Test Coverage**
   - Add tests for new features
   - Maintain >85% coverage (target 95%)
   - Use appropriate test markers

4. **Handle Optional Dependencies**
   - Keep them optional with graceful degradation
   - Add skip markers to tests
   - Document in README

5. **Follow Exception Policy**
   - Broad handlers in orchestrators OK
   - Narrow handlers in helpers
   - Log comprehensively

6. **Preserve Public API**
   - No breaking changes without deprecation
   - Follow semantic versioning
   - Provide migration guides

### Common Tasks

**Adding a New Method**:
1. Create method file in `methods/baselines/` or `methods/novel/`
2. Implement `Method` interface
3. Register with `MethodRegistry`
4. Add unit tests (>85% coverage)
5. Add integration tests
6. Update NOVEL_METHODS.md
7. Document in `docs/methods/`

**Adding a New Domain**:
1. Create adapter in `adapters/[domain]/`
2. Implement `DomainAdapter` interface
3. Add validation logic
4. Register with `DomainAdapterRegistry`
5. Add unit tests
6. Add benchmark instances
7. Document in `docs/domains/`

**Refactoring Embedded Methods**:
1. Check NOVEL_METHODS.md for status
2. Create refactoring plan (see REFACTORING_ROADMAP.md)
3. Ensure no breaking changes to public API
4. Maintain test coverage
5. Update documentation
6. Get approval from research team

---

## 📞 Support & Resources

### Key Files

- **This File**: `CONTEXT_FOR_AI.md` - AI assistant context
- **README**: `README.md` - Project overview
- **Contributing**: `CONTRIBUTING.md` - Development guidelines
- **Novel Methods**: `docs/theory/NOVEL_METHODS.md` - Method inventory
- **Architecture**: `docs/theory/DOMAIN_ADAPTER_ARCHITECTURE.md`

### Getting Help

1. Check relevant documentation
2. Review test files for examples
3. Check NOVEL_METHODS.md for method status
4. Review CONTRIBUTING.md for standards
5. Open issue if needed

### Useful Commands

```bash
# Run core tests (fast)
pytest -m "unit and not fft_laplace and not theorem"

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=Librex --cov-report=term

# Run specific test file
pytest tests/unit/test_qap_adapter.py -v

# Format code
black Librex/ tests/

# Lint
ruff check Librex/ tests/

# Type check
mypy Librex/
```

---

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-27 | Initial CONTEXT_FOR_AI.md created |

---

**Last Updated**: 2025-01-27  
**Maintainer**: Meshal Alawein (meshal@berkeley.edu)  
**Status**: Pre-publication, Active Development

---

*This context file is the single source of truth for AI assistants working on Librex. Always consult this file before making significant changes.*
