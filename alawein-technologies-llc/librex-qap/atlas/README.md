# ORCHEX: Autonomous Research Validation System

The autonomous research system - rigorous validation of optimization methods through personality-based agents.

## What Is This Directory?

ORCHEX is the **autonomous research validation engine**. It validates optimization methods rigorously using 7 personality-based agents, learns from failures, and continuously improves.

**Quick Facts:**
- ~2,000 lines of Python code
- 7 personality-based research agents
- Self-refutation framework (Popperian falsification)
- 200-question interrogation protocol
- Hall of Failures learning system
- Meta-learning for agent improvement

## Directory Contents

```
ORCHEX/
├── README.md                      ← You are here
├── ORCHEX/                         ← Main ORCHEX module
│   ├── __init__.py                ← Agent registry & initialization
│   ├── brainstorming/             ← Hypothesis generation
│   │   ├── brainstorm_engine.py   ← Generate hypotheses
│   │   └── [tests]
│   ├── experimentation/           ← Experiment design & execution
│   │   ├── code_generator.py      ← Generate experiment code
│   │   ├── experiment_designer.py ← Design experiments
│   │   ├── sandbox_executor.py    ← Safe execution
│   │   └── [tests]
│   ├── learning/                  ← Learning mechanisms
│   │   ├── advanced_bandits.py    ← UCB1 multi-armed bandit
│   │   └── [tests]
│   ├── orchestration/             ← Workflow orchestration
│   │   ├── workflow_orchestrator.py ← Main orchestrator
│   │   ├── intent_classifier.py   ← Intent classification
│   │   ├── problem_types.py       ← Problem type definitions
│   │   └── [tests]
│   ├── publication/               ← Paper generation (v0.2+)
│   │   └── paper_generator.py
│   ├── cli.py                     ← Command-line interface
│   ├── diagnostics.py             ← Diagnostic tools
│   ├── hypothesis_generator.py    ← Hypothesis generation core
│   ├── performance_utils.py       ← Performance monitoring
│   └── protocol.py                ← Core protocols
│
└── uaro/                          ← Universal solver integration
    ├── atlas_integration.py       ← Integration layer
    ├── explainability.py          ← Explanation generation
    ├── marketplace.py             ← Capability marketplace
    ├── reasoning_primitives.py    ← Reasoning tools
    └── universal_solver.py        ← Universal solver wrapper
```

## The 7 Personality Agents

ORCHEX has 7 unique agents that collaborate in research:

| Agent | Role | Strictness | Superpower |
|-------|------|-----------|-----------|
| 😠 **Grumpy Refuter** | Self-refutation | 0.9 | Finds flaws ruthlessly |
| 🤨 **Skeptical Steve** | Interrogation | 0.8 | Asks 200 tough questions |
| 🤦 **Failure Frank** | Learning | 0.7 | Remembers all past mistakes |
| 😄 **Optimistic Oliver** | Generation | 0.2 | Dreaming up new ideas |
| 😰 **Cautious Cathy** | Risk | 0.75 | Identifies all risks |
| 🤓 **Pedantic Pete** | Review | 0.85 | Rigorous peer reviewer |
| 🎉 **Enthusiastic Emma** | Design | 0.4 | Creative experiment designer |

### How They Work Together

```
Optimistic Oliver         → Generates 5-10 hypotheses
         ↓
Skeptical Steve          → Interrogates with 200 questions
         ↓
Grumpy Refuter           → Attempts self-refutation
         ↓
Enthusiastic Emma        → Designs experiments
         ↓
Pedantic Pete            → Peer review
         ↓
Cautious Cathy           → Risk assessment
         ↓
Failure Frank            → Records in Hall of Failures
         ↓
Result                   → Validated or rejected
```

## Quick Start

### Installation

```bash
# From project root
pip install -e .
pip install -e ".[dev]"
```

### Basic Usage

```python
from ORCHEX.orchestration import WorkflowOrchestrator

# Create orchestrator
orchestrator = WorkflowOrchestrator(topic="optimization")

# Generate hypotheses
hypotheses = orchestrator.generate_hypotheses(count=5)

# Validate with all agents
validation_results = orchestrator.validate_all(hypotheses)

# Learn from results
for hypothesis, result in zip(hypotheses, validation_results):
    if result.is_valid:
        print(f"✓ {hypothesis.title}")
    else:
        print(f"✗ {hypothesis.title}: {result.failure_reason}")
```

### Validating Librex.QAP Methods

```python
from ORCHEX.orchestration import WorkflowOrchestrator
from Librex.QAP.core import OptimizationPipeline

# Generate hypothesis about a new method
hypothesis = orchestrator.hypothesize_method(
    name="quantum_annealing",
    expected_speedup=2.0
)

# Validate with ORCHEX agents
result = orchestrator.validate_hypothesis(hypothesis)

# Record learning
if not result.is_valid:
    hall_of_failures.record(hypothesis, result.reason)
```

## Key Files Explained

### `ORCHEX/__init__.py` ⭐ (AGENT REGISTRY)

Central initialization file that registers all agents:

**What it does:**
1. Initializes all 7 personality agents
2. Registers capabilities
3. Sets up learning systems
4. Initializes Hall of Failures

**Key Classes:**
- `PersonalityAgent` - Base agent class
- `AgentRegistry` - Agent management
- `AgentCapabilities` - Capability definition

### `orchestration/workflow_orchestrator.py` (MAIN ORCHESTRATOR)

Central orchestration engine that coordinates all agents:

```python
orchestrator = WorkflowOrchestrator(topic="optimization")

# Generate hypotheses
hypotheses = orchestrator.generate_hypotheses()

# Validate with agents
results = orchestrator.validate_all(hypotheses)

# Learn from results
orchestrator.learn_from_validation(results)
```

### `brainstorming/brainstorm_engine.py`

Hypothesis generation system:

**Capabilities:**
- Literature search integration
- Gap identification
- Hypothesis generation (5-10 per topic)
- Novelty scoring

**Usage:**
```python
from ORCHEX.brainstorming import BrainstormEngine

engine = BrainstormEngine()
hypotheses = engine.generate(topic="QAP optimization")
```

### `learning/advanced_bandits.py`

Multi-armed bandit learning system:

**How it works:**
- Treats agents as "arms"
- Uses UCB1 algorithm
- Learns which agents are most effective
- Continuously improves agent selection

**Usage:**
```python
from ORCHEX.learning import AdvancedBandits

bandits = AdvancedBandits()
selected_agents = bandits.select_agents(num=3)  # Best 3 agents
```

### `experimentation/` (v0.2.0)

Experiment design and execution:

- **code_generator.py** - Generate experiment code
- **experiment_designer.py** - Design experiments
- **sandbox_executor.py** - Safe execution environment

(Coming in v0.2.0 release)

### `publication/` (v0.2.0)

Paper generation:

- Generate research papers from findings
- Automated citation management
- Summary generation

(Coming in v0.2.0 release)

## The Validation Process

### Self-Refutation (Popperian Falsification)

Five strategies for testing:

1. **Boundary Testing** - Push to limits
2. **Contradiction Search** - Find logical flaws
3. **Assumption Critique** - Question assumptions
4. **Counterexample Generation** - Find edge cases
5. **Comparative Analysis** - Compare with alternatives

### Interrogation Protocol

Skeptical Steve asks 200+ questions:

```
Question Categories:
├── Methodological (30%)   - Is the method sound?
├── Empirical (40%)        - Do results match?
├── Theoretical (20%)      - Does theory support?
└── Practical (10%)        - Is it useful?
```

### Integration with Librex.QAP

```python
# Validate an optimization method
from Librex.QAP.core import OptimizationPipeline
from ORCHEX.orchestration import WorkflowOrchestrator

pipeline = OptimizationPipeline(size=20)
result = pipeline.solve(problem, method="fft_laplace")

# ORCHEX validates the result
orchestrator = WorkflowOrchestrator()
validation = orchestrator.validate_optimization_result(result)

if validation.is_sound:
    print("✓ Method validated!")
else:
    print(f"Issues found: {validation.issues}")
```

## Extending ORCHEX

### Adding a New Agent

1. **Create agent class**:
   ```python
   class MyCustomAgent(PersonalityAgent):
       def __init__(self):
           super().__init__(name="Custom Agent", strictness=0.5)

       def validate(self, hypothesis):
           # Validation logic
           return ValidationResult(...)
   ```

2. **Register in `ORCHEX/__init__.py`**:
   ```python
   from ORCHEX.agents import MyCustomAgent

   agent_registry = AgentRegistry()
   agent_registry.register(MyCustomAgent())
   ```

3. **Add capabilities**:
   ```python
   agent.add_capability("hypothesis_validation")
   agent.add_capability("risk_assessment")
   ```

4. **Write tests**:
   ```python
   def test_custom_agent():
       agent = MyCustomAgent()
       result = agent.validate(hypothesis)
       assert result.is_valid or not result.is_valid  # Some result
   ```

### Improving Agent Learning

1. **Analyze Hall of Failures**:
   ```python
   from ORCHEX.learning import HallOfFailures

   failures = HallOfFailures()
   patterns = failures.analyze_patterns()
   ```

2. **Update agent behavior**:
   ```python
   agent.update_strategy(patterns)
   agent.improve_heuristics()
   ```

3. **Measure improvement**:
   ```python
   metrics = agent.get_performance_metrics()
   print(f"Accuracy: {metrics['accuracy']}")
   ```

## Running Tests

### All Tests

```bash
# From project root
make test                  # Full test suite

# Or specific to ORCHEX
pytest tests/test_integration.py -v
```

### Key Test Files

**tests/test_integration.py** (206 lines)
- ORCHEX-Librex.QAP integration
- Hypothesis validation
- Agent coordination
- Learning verification

## Real-World Usage Examples

### Scenario 1: Validating a New Optimization Method

```python
from ORCHEX.orchestration import WorkflowOrchestrator

# Create hypothesis about new method
hypothesis = {
    "title": "FFT-Laplace Preconditioning",
    "claim": "Achieves 100x speedup on medium QAP instances",
    "mechanism": "FFT acceleration of Laplacian",
}

# Let ORCHEX validate
orchestrator = WorkflowOrchestrator()
validation = orchestrator.validate(hypothesis)

# Results
print(validation.report())
# Result: ✓ VALID (with caveats)
# Issues: Not reliable on small instances
```

### Scenario 2: Learning from Past Failures

```python
from ORCHEX.learning import HallOfFailures

failures = HallOfFailures()

# Load past failed hypotheses
past_failures = failures.get_similar(current_hypothesis)

# Learn from them
lessons = failures.extract_lessons(past_failures)

# Apply learning
if "assumption" in lessons:
    print(f"Watch out: {lessons['assumption']}")
```

### Scenario 3: Multi-Agent Collaboration

```python
from ORCHEX.orchestration import WorkflowOrchestrator

orchestrator = WorkflowOrchestrator()

# Run full validation pipeline
result = orchestrator.full_validation(
    hypothesis=hypothesis,
    budget=1000,  # Max iterations
    timeout=3600  # Max seconds
)

# Get report from each agent
for agent_name, agent_result in result.agent_reports.items():
    print(f"{agent_name}: {agent_result.verdict}")
```

## Architecture Highlights

### Agent Coordination

```
WorkflowOrchestrator
├── Initialize Agents
├── For each hypothesis:
│   ├── Optimistic Oliver: Generate ideas
│   ├── Skeptical Steve: Interrogate
│   ├── Grumpy Refuter: Try to refute
│   ├── Enthusiastic Emma: Design test
│   ├── Pedantic Pete: Review
│   ├── Cautious Cathy: Assess risks
│   └── Failure Frank: Learn lessons
└── Return comprehensive validation
```

### Learning Loop

```
┌─────────────────────────┐
│  New Hypothesis         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Validate with Agents   │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Record Failure/Success │
│  (Hall of Failures)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Update Agent Learning  │
│  (Meta-learning)        │
└────────────┬────────────┘
             │
             ▼
     Next iteration improved!
```

## Troubleshooting

### Agent not responding?
```python
from ORCHEX.orchestration import WorkflowOrchestrator

orch = WorkflowOrchestrator()
status = orch.check_agent_status()
for agent, is_active in status.items():
    print(f"{agent}: {'✓' if is_active else '✗'}")
```

### Validation taking too long?
```python
# Use time limits
result = orchestrator.validate(hypothesis, timeout=300)
```

### Want to see agent reasoning?
```python
# Enable verbose logging
orchestrator.enable_verbose_logging()
result = orchestrator.validate(hypothesis)
# Now you'll see detailed agent reasoning
```

## Contributing

We welcome contributions! See `CONTRIBUTING.md` for:
- How to add new agents
- Testing requirements
- Documentation standards

## Related Documentation

- **PROJECT.md** - Complete project overview
- **STRUCTURE.md** - Directory structure guide
- **DEVELOPMENT.md** - Development workflow
- **CONTRIBUTING.md** - Contribution guidelines
- **.archive/docs/ORCHEX/** - Historical documentation

## Key Concepts

### Hypothesis
A testable claim about optimization methods

### Validation Result
The output of agent validation with verdicts and reasoning

### Hall of Failures
Database of past failures to learn from

### Agent Strictness
How harsh agents are in validation (0.0-1.0)

## Performance

| Agent | Speed | Thoroughness | Notes |
|-------|-------|--------------|-------|
| Optimistic Oliver | Fast | Low | Generates ideas |
| Skeptical Steve | Medium | High | 200 questions |
| Grumpy Refuter | Medium | Very High | Tries hard to break |
| Enthusiastic Emma | Medium | Medium | Creative testing |
| Pedantic Pete | Slow | Very High | Thorough review |
| Cautious Cathy | Fast | High | Risk-focused |
| Failure Frank | Fast | Medium | Lookup-based |

## Status & Roadmap

**Current (v0.1.0):**
- ✅ All 7 personality agents
- ✅ Hypothesis generation
- ✅ Validation framework
- ✅ Hall of Failures
- ✅ Meta-learning basics

**Next (v0.2.0):**
- [ ] Full experimentation (code gen, sandbox)
- [ ] Paper generation
- [ ] Advanced learning strategies
- [ ] API server

## Authors & Citation

**Author:** Meshal Alawein

**Citation:**
```bibtex
@software{atlas_2024,
  title = {ORCHEX: Autonomous Research Validation System},
  author = {Alawein, Meshal},
  year = {2024},
  url = {https://github.com/AlaweinOS/AlaweinOS/tree/main/Librex.QAP-new/ORCHEX},
  note = {7 personality-based research agents with self-improvement}
}
```

## License

MIT License - See `LICENSE` in project root

---

**Happy researching!** 🚀

Questions? Check `PROJECT.md` or `STRUCTURE.md` for more information.

Last Updated: November 2024
