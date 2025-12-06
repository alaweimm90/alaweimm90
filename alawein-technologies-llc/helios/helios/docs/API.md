# HELIOS API Reference

**Version**: 0.1.0 MVP
**Last Updated**: 2025-11-19

---

## 📚 API Overview

HELIOS provides a unified API for autonomous research discovery. All public classes and functions are importable from the main `helios` package.

```python
from helios import (
    # Discovery
    HypothesisGenerator,

    # Validation
    TuringValidator,

    # Learning
    MetaLearner,

    # Orchestration
    WorkflowOrchestrator,

    # Domains
    DOMAINS,
)
```

---

## 🔍 Core Classes

### HypothesisGenerator

**Module**: `helios.core.discovery`

**Purpose**: Generate novel research hypotheses

#### Constructor
```python
HypothesisGenerator(
    llm_provider: str = "openai",      # openai, anthropic, or google
    num_hypotheses: int = 5,
    model: str = "gpt-4",              # or gpt-3.5-turbo for faster
    temperature: float = 0.7,           # 0.0-1.0, higher = more creative
    timeout: int = 60,                  # seconds
)
```

#### Methods

**generate()**
```python
def generate(
    topic: str,
    num_hypotheses: int = None,        # Override default
    domain: str = None,                # quantum, ml, materials, etc.
    style: str = "novel",              # novel or safe
    context: str = None,               # Additional context
) -> List[Dict[str, Any]]:
    """
    Generate hypotheses for research topic.

    Args:
        topic: Research topic description (required)
        num_hypotheses: Number of hypotheses to generate
        domain: Specific research domain
        style: Generation style (novel or safe)
        context: Additional context for generation

    Returns:
        List of hypothesis dictionaries:
        {
            'id': str,
            'text': str,
            'domain': str,
            'novelty_score': float,      # 0-100
            'source_papers': List[str],
            'metadata': Dict[str, Any],
        }

    Raises:
        ValueError: If topic is empty
        TimeoutError: If generation takes too long
        ApiError: If LLM API fails
    """
    pass
```

**Example**:
```python
generator = HypothesisGenerator()
hypotheses = generator.generate(
    topic="Quantum error correction",
    domain="quantum",
    style="novel"
)
print(f"Generated {len(hypotheses)} hypotheses")
```

---

### TuringValidator

**Module**: `helios.core.validation.turing`

**Purpose**: Validate hypotheses using 5 falsification strategies

#### Constructor
```python
TuringValidator(
    strategies: List[str] = None,  # Specific strategies, or all by default
    timeout: int = 120,            # Total validation timeout (seconds)
    interrogation_enabled: bool = True,  # Enable 200-question framework
)
```

#### Methods

**validate()**
```python
def validate(
    hypotheses: Union[List[Dict], List[str]],
    domain: str = None,
    include_interrogation: bool = None,  # Override constructor setting
) -> List[Dict[str, Any]]:
    """
    Validate hypotheses using Turing suite.

    Args:
        hypotheses: List of hypothesis dicts or text strings
        domain: Domain for validation context
        include_interrogation: Include 200-question framework

    Returns:
        List of validation results:
        {
            'hypothesis_id': str,
            'logical_score': float,        # 0-100
            'empirical_score': float,      # 0-100
            'analogical_score': float,     # 0-100
            'boundary_score': float,       # 0-100
            'mechanism_score': float,      # 0-100
            'overall_score': float,        # 0-100 (weighted)
            'weaknesses': List[str],
            'interrogation_results': Dict,
            'validation_timestamp': str,
        }

    Raises:
        ValueError: If hypotheses list is empty
        TimeoutError: If validation exceeds timeout
    """
    pass
```

**validate_strategy()**
```python
def validate_strategy(
    hypothesis: Union[Dict, str],
    strategy: str,  # logical, empirical, analogical, boundary, mechanism
) -> Dict[str, Any]:
    """
    Validate hypothesis with specific strategy.

    Args:
        hypothesis: Hypothesis to validate
        strategy: Strategy name

    Returns:
        {
            'strategy': str,
            'score': float,     # 0-100
            'details': Dict,
        }
    """
    pass
```

**Example**:
```python
validator = TuringValidator()
results = validator.validate(hypotheses)

for r in results:
    print(f"Score: {r['overall_score']:.1f}")
    print(f"Weaknesses: {r['weaknesses']}")
```

---

### MetaLearner

**Module**: `helios.core.learning`

**Purpose**: Meta-learning system for continuous improvement

#### Constructor
```python
MetaLearner(
    persistence: str = "memory",  # memory or database
    database_path: str = None,     # Path to database if persistence=database
)
```

#### Methods

**learn_from_validation()**
```python
def learn_from_validation(
    hypotheses: List[Dict],
    validation_results: List[Dict],
) -> Dict[str, Any]:
    """
    Learn from validation results.

    Args:
        hypotheses: Original hypotheses
        validation_results: Validation results

    Returns:
        {
            'agents_updated': int,
            'lessons_learned': int,
            'hall_of_failures_updated': bool,
        }
    """
    pass
```

**recommend_agent()**
```python
def recommend_agent(
    topic: str = None,
    domain: str = None,
) -> str:
    """
    Get agent recommendation based on learning.

    Args:
        topic: Research topic
        domain: Research domain

    Returns:
        Agent name (conservative, creative, rigorous, etc.)
    """
    pass
```

**query_failures()**
```python
def query_failures(
    hypothesis: Union[Dict, str] = None,
    domain: str = None,
    similarity_threshold: float = 0.7,
    limit: int = 10,
) -> List[Dict]:
    """
    Query Hall of Failures for similar failures.

    Args:
        hypothesis: Current hypothesis (optional)
        domain: Research domain
        similarity_threshold: Minimum similarity (0-1)
        limit: Max results

    Returns:
        List of similar failures:
        {
            'hypothesis_text': str,
            'domain': str,
            'reason_for_failure': str,
            'lessons_learned': List[str],
            'similarity': float,
        }
    """
    pass
```

**Example**:
```python
learner = MetaLearner()
learner.learn_from_validation(hypotheses, validation_results)

agent = learner.recommend_agent(domain="quantum")
print(f"Best agent: {agent}")

failures = learner.query_failures(domain="quantum")
print(f"Found {len(failures)} similar failures")
```

---

### WorkflowOrchestrator

**Module**: `helios.core.orchestration`

**Purpose**: ORCHEX workflow engine for full research pipeline

#### Constructor
```python
WorkflowOrchestrator(
    domain: BaseDomain = None,
    config: Dict = None,
)
```

#### Methods

**execute()**
```python
def execute(
    research_topic: str,
    num_hypotheses: int = 5,
    validation_strategy: str = "turing",  # turing or custom
    enable_meta_learning: bool = True,
    generate_paper: bool = False,
    domain: str = None,  # Override constructor domain
) -> Dict[str, Any]:
    """
    Execute full research workflow.

    Args:
        research_topic: Main research topic
        num_hypotheses: Number of hypotheses to generate
        validation_strategy: Validation approach
        enable_meta_learning: Enable learning
        generate_paper: Generate publication
        domain: Research domain

    Returns:
        {
            'hypotheses': List[Dict],
            'validation_results': List[Dict],
            'learning_results': Dict,
            'paper': str,  # If generate_paper=True
            'execution_time': float,  # seconds
        }

    Raises:
        ValueError: If topic is empty
        RuntimeError: If workflow fails
    """
    pass
```

**generate_hypotheses()**
```python
def generate_hypotheses(
    topic: str,
    num: int = 5,
    domain: str = None,
) -> List[Dict]:
    """Generate hypotheses only."""
    pass
```

**validate_hypotheses()**
```python
def validate_hypotheses(
    hypotheses: List[Dict],
    strategy: str = "turing",
) -> List[Dict]:
    """Validate hypotheses only."""
    pass
```

**learn_from_results()**
```python
def learn_from_results(
    hypotheses: List[Dict],
    validation_results: List[Dict],
) -> Dict:
    """Update meta-learning from results."""
    pass
```

**generate_paper()**
```python
def generate_paper(
    topic: str,
    hypotheses: List[Dict],
    validation_results: List[Dict],
) -> str:
    """Generate publication-ready paper."""
    pass
```

**Example**:
```python
orchestrator = WorkflowOrchestrator()
results = orchestrator.execute(
    research_topic="Quantum error correction",
    num_hypotheses=10,
    enable_meta_learning=True,
    generate_paper=True,
)

print(f"Generated {len(results['hypotheses'])} hypotheses")
print(f"Top score: {max(r['overall_score'] for r in results['validation_results'])}")
```

---

## 🌐 Domain API

### DOMAINS Registry

**Module**: `helios.domains`

```python
from helios.domains import DOMAINS

# List available domains
print(DOMAINS.keys())
# dict_keys(['quantum', 'materials', 'optimization', 'ml', 'nas', 'synthesis', 'graph'])

# Load a domain
quantum_domain = DOMAINS['quantum']()
ml_domain = DOMAINS['ml']()
```

### BaseDomain Interface

All domains inherit from `BaseDomain`:

```python
class BaseDomain:
    name: str                  # Domain identifier
    display_name: str          # Display name
    description: str           # Domain description

    def generate_benchmark_problems(self) -> List[Problem]:
        """Generate benchmark problems for this domain."""
        pass

    def validate_solution(self, solution: Any) -> bool:
        """Validate a proposed solution."""
        pass

    def get_validation_rules(self) -> List[ValidationRule]:
        """Get domain-specific validation rules."""
        pass

    def get_example_hypotheses(self) -> List[Hypothesis]:
        """Get example hypotheses in this domain."""
        pass

    def get_benchmarks(self) -> Dict[str, Benchmark]:
        """Get available benchmarks."""
        pass
```

### Quantum Domain

```python
from helios.domains.quantum import QuantumDomain

domain = QuantumDomain()

# Generate benchmark problems
problems = domain.generate_benchmark_problems()

# Get quantum-specific validation rules
rules = domain.get_validation_rules()

# Get example hypotheses
examples = domain.get_example_hypotheses()

# Get available benchmarks
benchmarks = domain.get_benchmarks()
```

### ML Domain

```python
from helios.domains.ml import MLDomain

domain = MLDomain()
# ... same interface as QuantumDomain
```

**All domains**: quantum, materials, optimization, ml, nas, synthesis, graph

---

## 📊 Data Models

### Hypothesis Model

```python
{
    'id': str,                  # Unique identifier
    'text': str,               # Hypothesis statement
    'domain': str,             # Research domain
    'source_papers': [str],    # Referenced papers
    'novelty_score': float,    # 0-100
    'creation_timestamp': str, # ISO 8601
    'metadata': {              # Additional data
        'category': str,
        'confidence': float,
    }
}
```

### ValidationResult Model

```python
{
    'hypothesis_id': str,
    'logical_score': float,         # 0-100
    'empirical_score': float,       # 0-100
    'analogical_score': float,      # 0-100
    'boundary_score': float,        # 0-100
    'mechanism_score': float,       # 0-100
    'overall_score': float,         # 0-100 (weighted avg)
    'weaknesses': [str],            # List of identified issues
    'interrogation_results': {      # 200-question results
        'num_questions': int,
        'num_weaknesses_found': int,
        'critical_questions': [str],
    },
    'validation_timestamp': str,    # ISO 8601
}
```

### AgentRecommendation Model

```python
{
    'agent_name': str,
    'expected_score': float,        # 0-100
    'confidence': float,            # 0-1
    'reason': str,                  # Why recommended
    'performance_history': [float], # Past scores
}
```

---

## 🔄 Common Workflows

### Workflow 1: Basic Hypothesis Generation & Validation

```python
from helios import HypothesisGenerator, TuringValidator

# Generate
generator = HypothesisGenerator()
hypotheses = generator.generate("Your research topic")

# Validate
validator = TuringValidator()
results = validator.validate(hypotheses)

# Filter strong
strong = [h for h, r in zip(hypotheses, results)
          if r['overall_score'] > 70]
```

### Workflow 2: Domain-Specific Research

```python
from helios import WorkflowOrchestrator
from helios.domains import DOMAINS

# Create orchestrator with domain
domain = DOMAINS['quantum']()
orchestrator = WorkflowOrchestrator(domain=domain)

# Run workflow
results = orchestrator.execute(
    research_topic="Quantum error correction",
    num_hypotheses=10,
)
```

### Workflow 3: Learning from Failures

```python
from helios import MetaLearner

# After validation
learner = MetaLearner()
learner.learn_from_validation(hypotheses, results)

# Get agent recommendation
best_agent = learner.recommend_agent(domain="quantum")

# Query similar failures
failures = learner.query_failures(domain="quantum")
```

### Workflow 4: Full Pipeline with Paper

```python
from helios import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator()

# Execute with paper generation
results = orchestrator.execute(
    research_topic="Topic",
    domain="quantum",
    enable_meta_learning=True,
    generate_paper=True,
)

# Results include paper
paper = results['paper']
print(paper)
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Configuration
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
HELIOS_LLM_PROVIDER=openai

# HELIOS Configuration
HELIOS_NUM_HYPOTHESES=5
HELIOS_VALIDATION_TIMEOUT=300
HELIOS_DATABASE_PATH=/path/to/db
HELIOS_DEBUG=false
```

### Programmatic Configuration

```python
from helios import HypothesisGenerator

generator = HypothesisGenerator(
    llm_provider="openai",
    num_hypotheses=5,
    model="gpt-4",
    temperature=0.7,
    timeout=60,
)
```

---

## 🔢 Constants & Enums

### Available Domains

```python
AVAILABLE_DOMAINS = [
    'quantum',       # Quantum computing
    'materials',     # Materials science
    'optimization',  # Combinatorial optimization
    'ml',           # Machine learning
    'nas',          # Neural architecture search
    'synthesis',    # Drug discovery
    'graph',        # Graph optimization
]
```

### Validation Strategies

```python
VALIDATION_STRATEGIES = [
    'logical',           # Logical contradiction
    'empirical',        # Empirical counter-example
    'analogical',       # Analogical falsification
    'boundary',         # Boundary violation
    'mechanism',        # Mechanism implausibility
]
```

### Agent Names

```python
AGENT_NAMES = [
    'conservative',
    'creative',
    'rigorous',
    'pragmatic',
    'skeptic',
    'specialist',
    'generalist',
]
```

---

## 📈 Return Types

### Success Response

All methods return structured data:

```python
# List responses
[
    {'id': '...', 'score': 0.95, ...},
    {'id': '...', 'score': 0.87, ...},
]

# Dict responses
{
    'status': 'success',
    'data': {...},
    'execution_time': 1.23,
}
```

### Error Handling

```python
try:
    hypotheses = generator.generate(topic)
except ValueError as e:
    print(f"Invalid input: {e}")
except TimeoutError as e:
    print(f"Timeout: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## 🔐 Type Hints

HELIOS uses comprehensive type hints:

```python
from typing import Dict, List, Any, Optional, Union

def generate(
    topic: str,
    num_hypotheses: int = 5,
    domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    ...
```

---

## 📖 See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup guide
- [DOMAINS.md](DOMAINS.md) - Domain details
- [../CONTRIBUTING.md](../CONTRIBUTING.md) - Contributing guide

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
