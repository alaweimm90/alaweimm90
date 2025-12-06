# HELIOS Package

**Hypothesis Exploration & Learning Intelligence Orchestration System**

**Version**: 0.1.0 MVP | **Status**: Active Development

---

## 📦 What is HELIOS?

HELIOS is an autonomous research discovery platform that:

1. **Generates** novel hypotheses from academic literature
2. **Validates** hypotheses rigorously using 5 Turing-inspired falsification strategies
3. **Learns** from failures through a meta-learning system
4. **Improves** with 7 personality-based research agents
5. **Publishes** research-ready results with proofs and benchmarks

```python
from helios import HypothesisGenerator, TuringValidator

# Generate hypotheses
generator = HypothesisGenerator()
hypotheses = generator.generate("quantum computing")

# Validate rigorously
validator = TuringValidator()
results = validator.validate(hypotheses)

# Use strongest hypotheses
strong = [h for h in results if h.score > 70]
```

---

## 🏗️ Package Structure

```
helios/
├── core/              # 5 core modules
│   ├── discovery/     # Hypothesis generation
│   ├── validation/    # Turing falsification suite
│   ├── learning/      # Meta-learning system
│   ├── agents/        # 7 personality agents
│   └── orchestration/ # Workflow engine (ORCHEX)
│
├── domains/           # 7 research domains
│   ├── quantum/       # Quantum computing
│   ├── materials/     # Materials science
│   ├── optimization/  # Combinatorial optimization
│   ├── ml/           # Machine learning
│   ├── nas/          # Neural architecture search
│   ├── synthesis/    # Drug discovery
│   └── graph/        # Graph optimization
│
├── docs/              # Complete documentation
├── examples/          # Usage examples
├── scripts/           # Utility scripts
├── tests/             # Test suite
└── docker/            # Container configuration
```

---

## 🚀 Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With all optional dependencies
pip install -e ".[all,dev]"

# Or specific domains
pip install -e ".[quantum,ml]"
```

### Basic Usage

```python
# Hypothesis generation and validation
from helios import HypothesisGenerator, TuringValidator

generator = HypothesisGenerator()
validator = TuringValidator()

# Generate hypotheses
hypotheses = generator.generate("Your research topic")

# Validate with Turing suite
results = validator.validate(hypotheses)

# Filter strong hypotheses
strong = [h for h in results if h.score > 70]
```

### Domain-Specific Research

```python
from helios.domains import DOMAINS

# Load a domain
quantum_domain = DOMAINS['quantum']()

# Generate domain-specific problems
problems = quantum_domain.generate_benchmark_problems()

# Use domain in research workflow
from helios.core.orchestration import WorkflowOrchestrator
orchestrator = WorkflowOrchestrator(domain=quantum_domain)
```

### Full Workflow Example

```python
from helios.core.orchestration import WorkflowOrchestrator
from helios.domains import DOMAINS

# Setup orchestrator with a domain
domain = DOMAINS['quantum']()
orchestrator = WorkflowOrchestrator(domain=domain)

# Run full research workflow
results = orchestrator.execute(
    research_topic="Quantum error correction",
    num_hypotheses=10,
    validation_strategy="turing",  # Use Turing validation suite
    learning_enabled=True,  # Enable meta-learning
)

# Results include:
# - Generated hypotheses
# - Validation scores
# - Learned lessons
# - Publication-ready paper
```

---

## 🎯 Core Modules

### 1. Discovery (`helios.core.discovery`)

Generates novel hypotheses from literature searches.

```python
from helios.core.discovery import HypothesisGenerator

generator = HypothesisGenerator(
    llm_provider="openai",  # or "anthropic", "google"
    num_hypotheses=5,
)

hypotheses = generator.generate(
    topic="Machine learning optimization",
    domain="ml",
)
```

**Key features:**
- Literature search integration (ArXiv, OpenAI API)
- LLM-based hypothesis generation
- Domain-aware generation
- Novelty scoring

**Files:**
- `discovery/__init__.py` - Public API
- `discovery/hypothesis_generator.py` - Main generator
- `discovery/brainstorm_engine.py` - Creative hypothesis generation

### 2. Validation (`helios.core.validation.turing`)

Validates hypotheses using 5 Turing-inspired falsification strategies.

```python
from helios.core.validation.turing import TuringValidator

validator = TuringValidator()

# Validate with all strategies
results = validator.validate(hypotheses)

# Or use specific strategy
logical_contradiction = validator.validate(
    hypotheses,
    strategy="logical_contradiction"
)
```

**5 Falsification Strategies:**

1. **Logical Contradiction** - Find logical inconsistencies
   - Checks internal consistency
   - Identifies contradictory claims
   - Score: 0-100 (higher = more consistent)

2. **Empirical Counter-Example** - Test against real data
   - Validates against benchmark datasets
   - Tests edge cases
   - Score: 0-100 (higher = more empirically sound)

3. **Analogical Falsification** - Compare to similar domains
   - Finds analogies in other fields
   - Tests transferability
   - Score: 0-100 (higher = more analogically valid)

4. **Boundary Violation** - Test edge cases
   - Tests extreme values
   - Checks domain boundaries
   - Score: 0-100 (higher = more robust)

5. **Mechanism Implausibility** - Evaluate mechanisms
   - Assesses underlying mechanisms
   - Checks biophysical plausibility
   - Score: 0-100 (higher = more plausible)

**Plus: 200-Question Interrogation Framework**
- Deep probing of hypothesis weaknesses
- Generates critical questions
- Identifies unstated assumptions

**Files:**
- `validation/turing/__init__.py` - Public API
- `validation/turing/validator.py` - Main validator
- `validation/turing/interrogator.py` - Question framework
- `validation/turing/base.py` - Strategy base class
- `validation/turing/scorer.py` - Scoring logic
- `validation/turing/logical_contradiction.py` - Strategy 1
- `validation/turing/empirical_counter_example.py` - Strategy 2
- `validation/turing/analogical_falsification.py` - Strategy 3
- `validation/turing/boundary_violation.py` - Strategy 4
- `validation/turing/mechanism_implausibility.py` - Strategy 5

### 3. Learning (`helios.core.learning`)

Meta-learning system that learns from hypothesis validation results.

```python
from helios.core.learning import MetaLearner

learner = MetaLearner()

# Learn from validation results
learner.learn_from_validation(hypotheses, validation_results)

# Get agent recommendations
recommendations = learner.recommend_agent(topic="quantum computing")

# Query Hall of Failures
lessons = learner.query_failures(
    domain="quantum",
    similarity_threshold=0.7
)
```

**Key components:**

- **Hall of Failures** - Database of failed hypotheses and lessons learned
- **Meta-Learning** - Learns which approaches work best
- **Bandit Algorithms** - UCB1, Thompson sampling for agent selection
- **7 Personality Agents** - Different research personalities:
  1. Conservative - Risk-averse validation
  2. Creative - Novel hypothesis combinations
  3. Rigorous - Strict methodological standards
  4. Pragmatic - Focus on applications
  5. Skeptic - Challenge all assumptions
  6. Specialist - Domain expertise
  7. Generalist - Cross-domain connections

**Files:**
- `learning/__init__.py` - Public API
- `learning/meta_learner.py` - Meta-learning coordinator
- `learning/hall_of_failures.py` - Failure database
- `learning/agent_personality.py` - 7 personality agents
- `learning/bandit.py` - Bandit algorithms
- `learning/advanced_bandits.py` - Advanced algorithms (UCB1, Thompson)
- `learning/database.py` - Persistent storage
- `learning/trajectory_recorder.py` - Learning history

### 4. Agents (`helios.core.agents`)

7 personality-based research agents that collaborate.

```python
from helios.core.agents import ResearchAgentOrchestrator

orchestrator = ResearchAgentOrchestrator()

# Get agent recommendations
conservative = orchestrator.agents['conservative']
results = conservative.validate(hypothesis)

# Run with all agents
multi_results = orchestrator.validate_with_all(hypothesis)

# Smart agent selection
best_agent = orchestrator.select_best_agent(
    topic="quantum computing",
    based_on="learning_history"
)
```

**Agent personalities:**
- **Conservative** - Accepts only rigorously proven hypotheses
- **Creative** - Generates novel combinations and connections
- **Rigorous** - Enforces strict methodological standards
- **Pragmatic** - Focuses on practical applications
- **Skeptic** - Questions all assumptions
- **Specialist** - Deep domain expertise
- **Generalist** - Bridges across domains

**Files:**
- `agents/__init__.py` - Public API and orchestrator
- `agents/agent_personality.py` - Agent definitions (in learning/)

### 5. Orchestration (`helios.core.orchestration`)

ORCHEX workflow engine coordinating the full research process.

```python
from helios.core.orchestration import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator()

# Full research workflow
results = orchestrator.run_research(
    topic="Quantum error correction",
    domain="quantum",
    num_hypotheses=10,
    validation_strategies=["turing"],
    enable_meta_learning=True,
    publish=True,  # Generate paper
)

# Or step-by-step
hypotheses = orchestrator.generate_hypotheses(topic)
validated = orchestrator.validate(hypotheses)
learned = orchestrator.learn_from_results(validated)
paper = orchestrator.generate_paper(
    topic=topic,
    hypotheses=hypotheses,
    results=validated,
)
```

**Key features:**
- Workflow coordination (ORCHEX engine)
- Automatic experiment design
- Code generation and execution
- Hypothesis-to-paper pipeline
- Sandbox execution for safety

**Files:**
- `orchestration/__init__.py` - Public API
- `orchestration/workflow_orchestrator.py` - Main engine
- `orchestration/experiment_designer.py` - Experiment design
- `orchestration/code_generator.py` - Code generation
- `orchestration/sandbox_executor.py` - Safe execution
- `orchestration/paper_generator.py` - Publication generation

---

## 🌐 Research Domains

Each domain extends HELIOS for specific research areas:

### Quantum (`helios.domains.quantum`)

Quantum computing and quantum algorithms research.

```python
from helios.domains.quantum import QuantumDomain

domain = QuantumDomain()
problems = domain.generate_benchmark_problems()
```

**Dependencies:** Qiskit, Cirq

### Materials (`helios.domains.materials`)

Materials science and crystal structure discovery.

**Dependencies:** PyMatGen, ASE

### Optimization (`helios.domains.optimization`)

Combinatorial optimization problems with Librex.QAP tool.

```python
from helios.domains.optimization import OptimizationDomain

domain = OptimizationDomain()
# Includes Librex.QAP with 7 novel methods + 9 baselines
```

**Dependencies:** PuLP, Librex.QAP

### ML (`helios.domains.ml`)

Machine learning and neural network research.

**Dependencies:** PyTorch, scikit-learn

### NAS (`helios.domains.nas`)

Neural architecture search and AutoML.

**Dependencies:** NAS-Bench-101, PyTorch

### Synthesis (`helios.domains.synthesis`)

Drug discovery and molecular synthesis.

**Dependencies:** RDKit

### Graph (`helios.domains.graph`)

Graph optimization and network problems.

**Dependencies:** NetworkX

---

## 📖 Documentation

**Root-level guides:**
- [PROJECT.md](../PROJECT.md) - Project vision and roadmap
- [STRUCTURE.md](../STRUCTURE.md) - Directory organization
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines

**Package-level guides:**
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - Setup guide
- [docs/API.md](docs/API.md) - Complete API reference
- [docs/DOMAINS.md](docs/DOMAINS.md) - Domain explanations

**Examples:**
- [examples/basic_usage.py](examples/basic_usage.py) - Quick start
- [examples/domain_specific/](examples/domain_specific/) - Domain examples
- [examples/notebooks/](examples/notebooks/) - Jupyter notebooks

---

## 🧪 Testing

```bash
# Run all tests
bash scripts/test.sh

# Run specific test file
pytest tests/unit/test_discovery.py

# Run with coverage
pytest --cov=helios --cov-report=html tests/

# Run only integration tests
pytest tests/integration/
```

**Coverage targets:**
- **Current**: >60%
- **Target**: >80%
- **Critical paths**: 100%

---

## 🔧 Development

```bash
# Format code
bash scripts/format.sh

# Lint code
bash scripts/lint.sh

# Run tests
bash scripts/test.sh

# Run all
bash scripts/test.sh && bash scripts/lint.sh
```

**Development dependencies:**
- pytest >= 7.0
- pytest-cov >= 3.0
- black >= 22.0
- isort >= 5.10
- flake8 >= 4.0
- mypy >= 0.950

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| **Package LOC** | 20,582+ |
| **Core modules** | 5 |
| **Domains** | 7 |
| **Validation strategies** | 5 |
| **Personality agents** | 7 |
| **Test coverage** | >60% (target: >80%) |
| **Python version** | 3.8+ |

---

## 🤝 Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Development setup
- Code standards
- Testing guidelines
- PR process
- Domain extension

---

## 📝 License

- **Free Tier**: MIT License
- **Commercial Tier**: Proprietary License

See [LICENSE](../LICENSE) for details.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AlaweinOS/CLAUDE-CODE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AlaweinOS/CLAUDE-CODE/discussions)
- **Documentation**: [helios/docs/](docs/)

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
**Status**: Active Development
