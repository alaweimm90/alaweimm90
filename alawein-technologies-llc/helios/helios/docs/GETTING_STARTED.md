# Getting Started with HELIOS

**Version**: 0.1.0 MVP
**Last Updated**: 2025-11-19

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **pip**: Latest version
- **git**: For cloning the repository
- **API Keys**: OpenAI, Anthropic, or Google for LLM features (optional for basic testing)

Check your Python version:
```bash
python --version  # Should be 3.8+
```

---

## 🚀 Installation

### Option 1: Basic Installation

For development or basic usage:

```bash
# Clone repository
git clone https://github.com/AlaweinOS/CLAUDE-CODE.git
cd CLAUDE-CODE

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate      # On macOS/Linux
# or
venv\Scripts\activate         # On Windows

# Install with basic dependencies
pip install -e .
```

### Option 2: With Specific Domains

Install with specific research domains:

```bash
# Quantum domain only
pip install -e ".[quantum]"

# ML and NAS domains
pip install -e ".[ml,nas]"

# Quantum and Materials
pip install -e ".[quantum,materials]"
```

**Available domains**: `quantum`, `materials`, `optimization`, `ml`, `nas`, `synthesis`, `graph`

### Option 3: With All Domains + Development Tools

For full development experience:

```bash
# Install everything
pip install -e ".[all,dev]"
```

This includes:
- All 7 research domains
- Development tools (pytest, black, isort, flake8, mypy)
- Testing and coverage utilities

### Option 4: Using Development Script

Automated setup:

```bash
# Make script executable
chmod +x helios/scripts/setup.sh

# Run setup
bash helios/scripts/setup.sh
```

---

## ✅ Verify Installation

### Check HELIOS Import

```bash
python -c "import helios; print('✓ HELIOS installed successfully')"
```

### Run Quick Test

```python
# test_helios.py
from helios import HypothesisGenerator

# Create generator
generator = HypothesisGenerator()
print("✓ HypothesisGenerator instantiated")

# List available domains
from helios.domains import DOMAINS
print(f"✓ Available domains: {list(DOMAINS.keys())}")
```

Run it:
```bash
python test_helios.py
```

### Run Full Test Suite

```bash
# Run all tests with coverage
bash helios/scripts/test.sh

# Or with pytest directly
pytest helios/tests/ -v
```

---

## 🔑 API Configuration

### Setting Up LLM Providers

HELIOS supports multiple LLM providers. Create `.env` file in project root:

```bash
# Copy example file
cp .env.example .env

# Edit with your API keys
nano .env  # or your preferred editor
```

### .env File Template

```ini
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4  # or gpt-3.5-turbo

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Configuration
GOOGLE_API_KEY=your_google_api_key_here

# HELIOS Configuration
HELIOS_LLM_PROVIDER=openai  # or anthropic, google
HELIOS_NUM_HYPOTHESES=5
HELIOS_VALIDATION_TIMEOUT=300
```

**Note**: API keys are optional for basic testing and examples.

---

## 📚 Basic Usage Examples

### Example 1: Generate Hypotheses

```python
from helios import HypothesisGenerator

# Create generator
generator = HypothesisGenerator()

# Generate hypotheses for a research topic
hypotheses = generator.generate(
    topic="Quantum error correction techniques",
    num_hypotheses=5
)

# Display results
for h in hypotheses:
    print(f"[{h['id']}] {h['text']}")
    print(f"  Novelty: {h.get('novelty_score', 'N/A')}")
    print()
```

### Example 2: Validate Hypotheses

```python
from helios import HypothesisGenerator, TuringValidator

# Generate hypotheses
generator = HypothesisGenerator()
hypotheses = generator.generate("Machine learning optimization")

# Validate with Turing suite
validator = TuringValidator()
results = validator.validate(hypotheses)

# Display validation results
for result in results:
    print(f"Hypothesis {result['hypothesis_id']}:")
    print(f"  Overall Score: {result['overall_score']:.1f}/100")
    print(f"  Logical: {result['logical_score']:.1f}")
    print(f"  Empirical: {result['empirical_score']:.1f}")
    print(f"  Analogical: {result['analogical_score']:.1f}")
    print(f"  Boundary: {result['boundary_score']:.1f}")
    print(f"  Mechanism: {result['mechanism_score']:.1f}")
    if result.get('weaknesses'):
        print(f"  Weaknesses:")
        for w in result['weaknesses'][:3]:
            print(f"    - {w}")
    print()
```

### Example 3: Full Workflow

```python
from helios.core.orchestration import WorkflowOrchestrator
from helios.domains import DOMAINS

# Select domain
domain = DOMAINS['quantum']()

# Create orchestrator
orchestrator = WorkflowOrchestrator(domain=domain)

# Run full research workflow
results = orchestrator.execute(
    topic="Quantum error correction",
    num_hypotheses=5,
    validation_strategy="turing",
    enable_meta_learning=True,
)

# Results contain:
print(f"Generated {len(results['hypotheses'])} hypotheses")
print(f"Validated hypotheses with scores")
print(f"Learned from {len(results['learning_results'])} results")

# Check for strong hypotheses
strong = [h for h in results['hypotheses']
          if results['validation_results'][h['id']]['overall_score'] > 70]
print(f"Strong hypotheses (score > 70): {len(strong)}")
```

### Example 4: Domain-Specific Research

```python
from helios.domains import DOMAINS

# Load quantum domain
quantum = DOMAINS['quantum']()

# Get domain info
print(f"Domain: {quantum.display_name}")
print(f"Description: {quantum.description}")

# Generate benchmark problems
problems = quantum.generate_benchmark_problems()
print(f"Benchmark problems: {len(problems)}")

# Get validation rules
rules = quantum.get_validation_rules()
print(f"Validation rules: {len(rules)}")

# Get example hypotheses
examples = quantum.get_example_hypotheses()
print(f"Example hypotheses: {len(examples)}")
```

### Example 5: Meta-Learning

```python
from helios.core.learning import MetaLearner

# Create meta-learner
learner = MetaLearner()

# After validation, learn from results
learner.learn_from_validation(hypotheses, validation_results)

# Get agent recommendations
best_agent = learner.recommend_agent(
    topic="Quantum error correction",
    domain="quantum"
)
print(f"Recommended agent: {best_agent}")

# Query Hall of Failures for similar failed hypotheses
similar_failures = learner.query_failures(
    hypothesis=current_hypothesis,
    domain="quantum",
    similarity_threshold=0.7
)

if similar_failures:
    print(f"Found {len(similar_failures)} similar failures:")
    for failure in similar_failures:
        print(f"  - {failure['reason_for_failure']}")
```

---

## 🎯 Common Tasks

### Run Tests

```bash
# Run all tests
bash helios/scripts/test.sh

# Run specific test file
pytest helios/tests/unit/test_discovery.py -v

# Run with coverage
pytest --cov=helios helios/tests/

# Generate HTML coverage report
pytest --cov=helios --cov-report=html helios/tests/
# Open htmlcov/index.html in browser
```

### Format Code

```bash
# Auto-format with black and isort
bash helios/scripts/format.sh

# Or manually
black helios/
isort helios/
```

### Lint Code

```bash
# Check for style violations
bash helios/scripts/lint.sh

# Or manually
flake8 helios/
mypy helios/
```

### Run Examples

```bash
# Navigate to examples
cd helios/examples

# Run basic usage example
python basic_usage.py

# Run hypothesis generation example
python hypothesis_generation.py

# Run domain-specific example
python domain_specific/quantum_research.py
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to helios/examples/notebooks/
# Open Getting_Started.ipynb
```

---

## 🔧 Configuration

### Project Configuration

Main configuration in `pyproject.toml`:

```toml
[project]
name = "helios"
version = "0.1.0"
description = "HELIOS: Autonomous research discovery"

[project.optional-dependencies]
quantum = ["qiskit>=0.37", "cirq>=1.0"]
ml = ["torch>=1.9", "scikit-learn>=1.0"]
# ... other domains
```

### Development Configuration

Development tools configured in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py38']

[tool.isort]
profile = "black"
line_length = 100

[tool.pytest.ini_options]
testpaths = ["helios/tests"]
addopts = "--cov=helios --cov-report=term-missing"
```

---

## 📁 Project Structure

Quick reference:

```
CLAUDE-CODE/
├── helios/                  # Main package
│   ├── core/               # 5 core modules
│   │   ├── discovery/      # Hypothesis generation
│   │   ├── validation/     # Turing validation
│   │   ├── learning/       # Meta-learning
│   │   ├── agents/         # Research agents
│   │   └── orchestration/  # ORCHEX workflow
│   │
│   ├── domains/            # 7 research domains
│   ├── docs/              # Documentation
│   ├── examples/          # Usage examples
│   ├── tests/             # Test suite
│   ├── scripts/           # Utility scripts
│   └── docker/            # Container config
│
├── PROJECT.md             # Project overview
├── STRUCTURE.md           # Directory guide
├── CONTRIBUTING.md        # Contribution guide
├── pyproject.toml         # Package config
└── LICENSE                # MIT License
```

See [STRUCTURE.md](../STRUCTURE.md) for detailed directory guide.

---

## 🚨 Troubleshooting

### Import Error: "No module named 'helios'"

```bash
# Make sure you're in project directory
cd CLAUDE-CODE

# Install in development mode
pip install -e .

# Or verify installation
pip show helios
```

### "ModuleNotFoundError: No module named 'qiskit'"

The quantum domain is optional. Install it:

```bash
pip install -e ".[quantum]"
```

### API Key Not Found

Create `.env` file in project root:

```bash
cp .env.example .env
# Edit .env with your API keys
```

HELIOS loads from environment variables automatically.

### Test Failures

```bash
# Update dependencies
pip install --upgrade pip
pip install -e ".[all,dev]"

# Run tests with verbose output
pytest helios/tests/ -vv

# Check Python version
python --version  # Should be 3.8+
```

### Performance Issues

```bash
# Reduce number of hypotheses generated
generator = HypothesisGenerator(num_hypotheses=3)

# Set validation timeout
validator = TuringValidator(timeout=60)

# Use smaller domains
domain = DOMAINS['ml']()  # Smaller than quantum
```

---

## 📖 Next Steps

1. **Run Examples**: Try examples in `helios/examples/`
2. **Explore Domains**: Learn about specific research domains in `helios/docs/DOMAINS.md`
3. **Read Architecture**: Understand system design in `helios/docs/ARCHITECTURE.md`
4. **Write Code**: Start using HELIOS in your research
5. **Contribute**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guide

---

## 🆘 Support

- **Documentation**: [helios/docs/](.)
- **Issues**: [GitHub Issues](https://github.com/AlaweinOS/CLAUDE-CODE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AlaweinOS/CLAUDE-CODE/discussions)

---

## 📝 License

HELIOS is licensed under:
- **MIT License** (free tier)
- **Proprietary License** (commercial)

See [LICENSE](../../LICENSE) for details.

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
