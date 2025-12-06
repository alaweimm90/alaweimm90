# HELIOS Project Structure

**Version**: 0.1.0 MVP
**Last Updated**: 2025-11-19

## 📁 Directory Organization

HELIOS follows a clean, hierarchical structure with all code contained within the `helios/` package directory:

```
CLAUDE-CODE/
├── README.md                    # Main project documentation
├── PROJECT.md                   # Project overview & vision
├── STRUCTURE.md                 # This file - directory guide
├── CONTRIBUTING.md              # Contribution guidelines
├── pyproject.toml              # Python package configuration
├── LICENSE                      # MIT License
│
└── helios/                      # Main package directory
    ├── __init__.py             # Package entry point (unified API)
    ├── README.md               # Package-level documentation
    │
    ├── core/                   # Core HELIOS system (5 modules)
    │   ├── __init__.py         # Core module registry
    │   │
    │   ├── discovery/          # Hypothesis generation from literature
    │   │   ├── __init__.py
    │   │   ├── hypothesis_generator.py
    │   │   └── brainstorm_engine.py
    │   │
    │   ├── validation/         # Turing-inspired falsification strategies
    │   │   ├── __init__.py
    │   │   └── turing/         # Turing validation suite
    │   │       ├── __init__.py
    │   │       ├── base.py
    │   │       ├── validator.py
    │   │       ├── interrogator.py
    │   │       ├── scorer.py
    │   │       ├── models.py
    │   │       ├── protocol.py
    │   │       ├── cli.py
    │   │       ├── question_loader.py
    │   │       └── strategies/
    │   │           ├── logical_contradiction.py
    │   │           ├── empirical_counter_example.py
    │   │           ├── analogical_falsification.py
    │   │           ├── boundary_violation.py
    │   │           └── mechanism_implausibility.py
    │   │
    │   ├── learning/           # Meta-learning & Hall of Failures
    │   │   ├── __init__.py
    │   │   ├── meta_learner.py
    │   │   ├── agent_personality.py
    │   │   ├── hall_of_failures.py
    │   │   ├── bandit.py
    │   │   ├── advanced_bandits.py
    │   │   ├── database.py
    │   │   ├── models.py
    │   │   ├── protocol.py
    │   │   ├── trajectory_recorder.py
    │   │   ├── lesson_extractor.py
    │   │   ├── classifier.py
    │   │   └── similarity_matcher.py
    │   │
    │   ├── agents/             # 7 personality-based research agents
    │   │   └── __init__.py
    │   │
    │   ├── orchestration/      # ORCHEX workflow engine
    │   │   ├── __init__.py
    │   │   ├── workflow_orchestrator.py
    │   │   ├── experiment_designer.py
    │   │   ├── code_generator.py
    │   │   ├── sandbox_executor.py
    │   │   ├── paper_generator.py
    │   │   ├── intent_classifier.py
    │   │   ├── hypothesis_generator.py
    │   │   ├── cli.py
    │   │   ├── protocol.py
    │   │   ├── performance_utils.py
    │   │   └── diagnostics.py
    │   │
    │   ├── diagnostics.py      # System diagnostics utilities
    │   └── performance_utils.py # Performance monitoring
    │
    ├── domains/                # 7 Research domains
    │   ├── __init__.py         # Domain registry with DOMAINS dict
    │   │
    │   ├── quantum/            # Quantum computing research
    │   │   ├── __init__.py     # QuantumDomain class
    │   │   ├── README.md
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   ├── materials/          # Materials science discovery
    │   │   ├── __init__.py     # MaterialsDomain class
    │   │   ├── README.md
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   ├── optimization/       # Combinatorial optimization (with Librex.QAP tool)
    │   │   ├── __init__.py     # OptimizationDomain class
    │   │   ├── README.md
    │   │   ├── Librex.QAP/      # Optimization tool (7 methods + 9 baselines)
    │   │   │   ├── __init__.py
    │   │   │   ├── methods/
    │   │   │   └── baselines/
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   ├── ml/                 # Machine learning research
    │   │   ├── __init__.py     # MLDomain class
    │   │   ├── README.md
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   ├── nas/                # Neural architecture search
    │   │   ├── __init__.py     # NASDomain class
    │   │   ├── README.md
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   ├── synthesis/          # Drug discovery
    │   │   ├── __init__.py     # SynthesisDomain class
    │   │   ├── README.md
    │   │   ├── benchmarks.py
    │   │   └── examples/
    │   │
    │   └── graph/              # Graph optimization
    │       ├── __init__.py     # GraphDomain class
    │       ├── README.md
    │       ├── benchmarks.py
    │       └── examples/
    │
    ├── docs/                   # Project documentation
    │   ├── ARCHITECTURE.md      # System design & design decisions
    │   ├── GETTING_STARTED.md   # Setup & installation guide
    │   ├── API.md              # Complete API reference
    │   ├── DOMAINS.md          # Detailed domain explanations
    │   ├── CONTRIBUTING.md     # Development contribution guide
    │   └── examples/           # Documentation examples
    │
    ├── examples/               # Example code & notebooks
    │   ├── basic_usage.py
    │   ├── hypothesis_generation.py
    │   ├── hypothesis_validation.py
    │   ├── meta_learning.py
    │   ├── domain_specific/
    │   │   ├── quantum_research.py
    │   │   ├── materials_research.py
    │   │   ├── optimization_research.py
    │   │   ├── ml_research.py
    │   │   ├── nas_research.py
    │   │   ├── synthesis_research.py
    │   │   └── graph_research.py
    │   └── notebooks/
    │       └── Getting_Started.ipynb
    │
    ├── scripts/                # Utility scripts
    │   ├── setup.sh           # Development environment setup
    │   ├── run.sh             # Local development launcher
    │   ├── test.sh            # Test runner with coverage
    │   ├── format.sh          # Code formatting (black, isort)
    │   └── lint.sh            # Code linting (flake8, mypy)
    │
    ├── tests/                  # Test suite
    │   ├── conftest.py        # Pytest fixtures & configuration
    │   ├── __init__.py
    │   │
    │   ├── unit/              # Unit tests (by module)
    │   │   ├── test_discovery.py
    │   │   ├── test_validation.py
    │   │   ├── test_learning.py
    │   │   ├── test_agents.py
    │   │   ├── test_orchestration.py
    │   │   └── test_domains.py
    │   │
    │   └── integration/       # Integration tests
    │       ├── test_workflow.py
    │       ├── test_domain_integration.py
    │       └── test_end_to_end.py
    │
    └── docker/                # Container configuration
        ├── Dockerfile
        └── docker-compose.yml

```

---

## 🎯 Module Purpose Guide

### Core System (`helios/core/`)

The five core modules form the heart of HELIOS:

#### 1. **discovery/** - Hypothesis Generation
- Searches academic literature
- Generates novel hypotheses using LLM integration
- Files: `hypothesis_generator.py`, `brainstorm_engine.py`

#### 2. **validation/turing/** - Falsification Suite
- Implements 5 Turing-inspired strategies:
  - **Logical Contradiction** - Find logical inconsistencies
  - **Empirical Counter-Example** - Test against real data
  - **Analogical Falsification** - Compare to similar domains
  - **Boundary Violation** - Test edge cases
  - **Mechanism Implausibility** - Evaluate underlying mechanisms
- **200-Question Interrogation Framework** - Deep hypothesis probing
- Files: `base.py`, `validator.py`, `interrogator.py`, `scorer.py`

#### 3. **learning/** - Meta-Learning System
- Hall of Failures database (learns from rejected hypotheses)
- 7 personality-based agents with bandit optimization
- Advanced learning algorithms (UCB1, Thompson sampling)
- Files: `meta_learner.py`, `agent_personality.py`, `bandit.py`, `hall_of_failures.py`

#### 4. **agents/** - Personality-Based Research Agents
- 7 distinct research personalities:
  1. Conservative - Risk-averse, validates thoroughly
  2. Creative - Generates novel combinations
  3. Rigorous - Enforces strict methodological standards
  4. Pragmatic - Focuses on applicable results
  5. Skeptic - Challenges all assumptions
  6. Specialist - Domain-deep expert
  7. Generalist - Cross-domain connector
- Learns which agents work best for each domain
- File: `agent_personality.py`

#### 5. **orchestration/** - ORCHEX Workflow Engine
- Coordinates hypothesis generation → validation → learning
- Generates experiment code automatically
- Executes experiments in sandboxed environment
- Generates publication-ready papers
- Files: `workflow_orchestrator.py`, `experiment_designer.py`, `code_generator.py`, `sandbox_executor.py`, `paper_generator.py`

---

### Research Domains (`helios/domains/`)

Each domain extends the core HELIOS system for a specific research area:

| Domain | Purpose | Key Tools |
|--------|---------|-----------|
| **quantum** | Quantum computing research | Qiskit, Cirq |
| **materials** | Materials science discovery | PyMatGen, ASE |
| **optimization** | Combinatorial optimization | Librex.QAP (7 novel methods) |
| **ml** | Machine learning research | PyTorch, scikit-learn |
| **nas** | Neural architecture search | NAS-Bench-101 |
| **synthesis** | Drug discovery & synthesis | RDKit |
| **graph** | Graph optimization | NetworkX |

Each domain provides:
- Problem definitions and benchmarks
- Domain-specific validation rules
- Example experiments
- Baseline solutions
- Integration with core HELIOS system

---

## 🔄 How to Navigate

### Finding Core Functionality
```
Looking for hypothesis generation?     → helios/core/discovery/
Looking for validation strategies?     → helios/core/validation/turing/
Looking for meta-learning?             → helios/core/learning/
Looking for workflow orchestration?    → helios/core/orchestration/
Looking for research agents?           → helios/core/agents/
```

### Finding Domain Code
```
Looking for quantum research?          → helios/domains/quantum/
Looking for optimization tools?        → helios/domains/optimization/
Looking for drug discovery?            → helios/domains/synthesis/
Looking for ML research?               → helios/domains/ml/
```

### Finding Documentation
```
System architecture?                   → helios/docs/ARCHITECTURE.md
Getting started?                       → helios/docs/GETTING_STARTED.md
API reference?                         → helios/docs/API.md
Domain explanations?                   → helios/docs/DOMAINS.md
Contribution guidelines?               → helios/docs/CONTRIBUTING.md
```

### Finding Examples
```
Basic usage?                           → helios/examples/basic_usage.py
Quantum research example?              → helios/examples/domain_specific/quantum_research.py
Full workflow example?                 → helios/examples/notebooks/Getting_Started.ipynb
```

---

## 📝 Key Files Explained

### Root Level
- **README.md** - Main project overview (users start here)
- **PROJECT.md** - Comprehensive project vision and roadmap
- **STRUCTURE.md** - This file - directory organization
- **CONTRIBUTING.md** - Contribution guidelines
- **pyproject.toml** - Python package metadata and dependencies
- **LICENSE** - MIT License text

### Package Level (`helios/`)
- **helios/__init__.py** - Unified API exports
  - Imports all public classes/functions for easy access
  - Example: `from helios import HypothesisGenerator, TuringValidator`

- **helios/README.md** - Package documentation (technical overview)

### Core System (`helios/core/`)
- **helios/core/__init__.py** - Core module registry
- Each submodule has its own `__init__.py` exporting public API
- Clean separation of concerns across 5 modules

### Domains (`helios/domains/`)
- **helios/domains/__init__.py** - DOMAINS registry
  - Maps domain names to classes: `{'quantum': QuantumDomain, ...}`
- Each domain's `__init__.py` defines its public API
- Each domain has README.md explaining its purpose

### Documentation (`helios/docs/`)
- Comprehensive guides for users and developers
- API reference with examples
- Architecture decisions and design rationale
- Contribution guidelines

### Testing (`helios/tests/`)
- Unit tests for individual modules
- Integration tests for workflows
- Fixtures and test utilities in conftest.py
- Coverage reports generated with pytest-cov

### Scripts (`helios/scripts/`)
- Automation for setup, testing, formatting, linting
- Executable shell scripts for common operations
- Documentation for each script

---

## 🚀 Adding New Components

### Adding a Domain
1. Create `helios/domains/new_domain/`
2. Add domain class in `__init__.py`
3. Register in `helios/domains/__init__.py`
4. Add domain-specific validation rules
5. Create examples and benchmarks
6. Update documentation

### Adding Core Functionality
1. Choose appropriate core module:
   - New hypothesis generation? → `discovery/`
   - New validation strategy? → `validation/turing/`
   - New learning technique? → `learning/`
   - New agent type? → `agents/`
   - New workflow? → `orchestration/`
2. Create feature file in appropriate module
3. Export in module's `__init__.py`
4. Add tests in `tests/unit/`
5. Update `helios/docs/API.md`

### Adding Tests
1. Unit tests → `tests/unit/test_module.py`
2. Integration tests → `tests/integration/test_feature.py`
3. Fixtures → `tests/conftest.py`
4. Run: `bash helios/scripts/test.sh`

---

## 📊 Quick Statistics

| Metric | Value |
|--------|-------|
| **Total LOC** | 20,582+ |
| **Core Modules** | 5 (discovery, validation, learning, agents, orchestration) |
| **Research Domains** | 7 (quantum, materials, optimization, ML, NAS, synthesis, graph) |
| **Validation Strategies** | 5 (Turing suite) |
| **Personality Agents** | 7 |
| **Documentation Files** | 7+ |
| **Test Coverage** | >60% (target: >80%) |
| **Python Version** | 3.8+ |

---

## 🔗 Cross-References

**See also:**
- [PROJECT.md](PROJECT.md) - Project vision and roadmap
- [README.md](README.md) - Quick start guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [helios/docs/](helios/docs/) - Complete documentation
- [pyproject.toml](pyproject.toml) - Dependencies and configuration

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
