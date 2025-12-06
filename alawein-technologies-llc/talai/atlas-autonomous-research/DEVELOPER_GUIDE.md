# 🔧 ORCHEX Developer Guide

**Guide for extending and customizing ORCHEX**

---

## 🎯 Quick Links

- **Architecture**: See [ATLAS_ARCHITECTURE.md](ATLAS_ARCHITECTURE.md)
- **Integration**: See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)
- **Examples**: See [examples/](examples/)

---

## 🏗️ Project Structure

```
ORCHEX-autonomous-research/
├── src/ORCHEX/
│   ├── protocol.py           # Main ORCHEX orchestrator
│   ├── hypothesis_generator.py  # Hypothesis generation
│   ├── cli.py                # Command-line interface
│   └── __init__.py
│
├── examples/
│   ├── basic_usage.py
│   ├── personality_agents_demo.py
│   └── custom_agent_example.py
│
├── tests/                    # Unit tests (TODO)
│
└── docs/
    ├── ATLAS_ARCHITECTURE.md
    ├── INTEGRATION_GUIDE.md
    └── DEVELOPER_GUIDE.md (this file)
```

**Dependencies** (from other projects):

```
02-PROJECTS/turing-features/
├── self-refutation/         # Self-Refutation Protocol
├── interrogation/           # 200-Question Interrogation
├── hall-of-failures/        # Hall of Failures Database
└── meta-learning/           # Meta-Learning Core
```

---

## 🚀 Getting Started

### 1. Installation (Development Mode)

```bash
# Install ORCHEX in development mode
cd 02-PROJECTS/ORCHEX-autonomous-research
pip install -e .

# Install dependencies
cd ../turing-features/self-refutation && pip install -e .
cd ../interrogation && pip install -e .
cd ../hall-of-failures && pip install -e .
cd ../meta-learning && pip install -e .
```

### 2. Run Tests

```bash
# TODO: Create test suite
pytest tests/
```

### 3. Try Examples

```bash
# Basic usage
python examples/basic_usage.py

# Personality agents demo
python examples/personality_agents_demo.py
```

---

## 🎭 Adding Custom Agents

### Step 1: Define Agent Personality

```python
from meta_learning import create_custom_agent, AgentMood

my_agent = create_custom_agent(
    name="Hyper-Critical Helen",
    role="code_review",  # Can be any role
    mood=AgentMood.PEDANTIC,
    catchphrase="I found 47 bugs in your 'Hello World'.",
    emoji="🔍",
    description="Obsessive code reviewer who finds bugs everywhere.",

    # Personality traits (0-1)
    strictness=0.99,    # Extremely strict
    creativity=0.1,     # Not creative
    optimism=0.05,      # Very pessimistic
    verbosity=0.95      # Very wordy
)
```

### Step 2: Add to Agent Roster

**Option A: Runtime Addition**

```python
from meta_learning.agent_personality import AGENT_ROSTER

# Add to roster at runtime
AGENT_ROSTER["hyper_critical_helen"] = my_agent
```

**Option B: Modify Source** (for permanent addition)

Edit `02-PROJECTS/turing-features/meta-learning/src/meta_learning/agent_personality.py`:

```python
AGENT_ROSTER = {
    # ... existing agents ...

    "hyper_critical_helen": AgentPersonality(
        name="Hyper-Critical Helen",
        role="code_review",
        mood=AgentMood.PEDANTIC,
        catchphrase="I found 47 bugs in your 'Hello World'.",
        emoji="🔍",
        description="Obsessive code reviewer.",
        strictness=0.99,
        creativity=0.1,
        optimism=0.05,
        verbosity=0.95,
    ),
}
```

### Step 3: Use in ORCHEX

```python
# Agent will automatically be available for "code_review" role
agent = meta_learning.select_agent("code_review", domain)
```

---

## 🧪 Adding New Validation Stages

### Example: Add Code Quality Check

**Step 1: Create Validator**

```python
# In src/ORCHEX/validators/code_quality.py

class CodeQualityChecker:
    """Check code quality with personality agents"""

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    async def check_code(self, code: str) -> dict:
        """
        Check code quality

        Returns:
            dict with 'score', 'issues', 'suggestions'
        """
        # Your code quality logic here
        score = self._analyze_code(code)
        issues = self._find_issues(code)
        suggestions = self._generate_suggestions(code)

        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions
        }
```

**Step 2: Integrate into ORCHEX Protocol**

```python
# In src/ORCHEX/protocol.py

from ORCHEX.validators.code_quality import CodeQualityChecker

class ATLASProtocol:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add code quality checker
        self.code_quality_checker = CodeQualityChecker(orchestrator)

    async def run_research(self, ...):
        # ... existing stages ...

        # NEW STAGE: Code Quality Check
        print(f"\n{'─'*80}")
        print("STAGE 2.5: CODE QUALITY CHECK")
        print(f"{'─'*80}\n")

        if self.enable_meta_learning:
            # Select agent for code review
            review_agent = self.meta_learning.select_agent(
                "code_review",
                domain
            )
            print(f"  {review_agent.get_greeting()}")

        # Run check
        result = await self.code_quality_checker.check_code(
            generated_code
        )

        if self.enable_meta_learning:
            # Record action
            self.meta_learning.record_agent_action(
                agent=review_agent,
                action_type="code_review",
                input_data={"code": generated_code},
                output_data=result,
                success=result['score'] >= 70.0,
                score=result['score']
            )

            # Agent reacts
            self.meta_learning.agent_react(review_agent, result['score'])
```

---

## 🔌 Adding New Experiment Types

### Current: Only computational experiments planned

**To add wet lab experiments** (future):

```python
# In src/ORCHEX/experiments/wet_lab.py

class WetLabExperiment:
    """Interface for physical experiments"""

    async def design_experiment(self, hypothesis):
        """Design physical experiment protocol"""
        pass

    async def execute(self, protocol):
        """Submit to lab automation system"""
        pass

    async def analyze_results(self, data):
        """Analyze experimental data"""
        pass
```

**Integration:**

```python
# In protocol.py

if hypothesis.requires_wet_lab:
    wet_lab = WetLabExperiment()
    protocol = await wet_lab.design_experiment(hypothesis)
    results = await wet_lab.execute(protocol)
    analysis = await wet_lab.analyze_results(results)
```

---

## 📊 Customizing Learning Algorithms

### Current: UCB1 Multi-Armed Bandit

**To swap learning algorithm:**

**Step 1: Implement Interface**

```python
# In meta_learning/bandit.py

class ThompsonSamplingBandit:
    """Alternative to UCB1 using Thompson Sampling"""

    def __init__(self, agent_ids, contexts):
        self.agent_ids = agent_ids
        self.contexts = contexts

        # Beta distribution parameters
        self.alpha = {ctx: {aid: 1 for aid in agent_ids}
                     for ctx in contexts}
        self.beta = {ctx: {aid: 1 for aid in agent_ids}
                    for ctx in contexts}

    def select_agent(self, context: str) -> str:
        """Select agent via Thompson Sampling"""
        import numpy as np

        samples = {}
        for agent_id in self.agent_ids:
            # Sample from beta distribution
            alpha = self.alpha[context][agent_id]
            beta = self.beta[context][agent_id]
            samples[agent_id] = np.random.beta(alpha, beta)

        # Select agent with highest sample
        return max(samples, key=samples.get)

    def update(self, agent_id: str, reward: float, context: str):
        """Update beta distribution"""
        if reward >= 0.5:  # Success
            self.alpha[context][agent_id] += 1
        else:  # Failure
            self.beta[context][agent_id] += 1
```

**Step 2: Use in MetaLearningProtocol**

```python
# In meta_learning/protocol.py

from meta_learning.bandit import ThompsonSamplingBandit

class MetaLearningProtocol:
    def __init__(self, bandit_type="ucb1", ...):
        # ... existing code ...

        # Select bandit algorithm
        if bandit_type == "ucb1":
            self.bandit = ContextualBandit(...)
        elif bandit_type == "thompson":
            self.bandit = ThompsonSamplingBandit(...)
        else:
            raise ValueError(f"Unknown bandit: {bandit_type}")
```

---

## 🎨 Customizing Agent Reactions

### Current: Mood-based reactions

**To add context-aware reactions:**

```python
# In meta_learning/agent_personality.py

class AgentPersonality(BaseModel):
    # ... existing fields ...

    def react_to_hypothesis(self, hypothesis: str, score: float) -> str:
        """Context-aware reaction based on hypothesis content"""

        # Detect hypothesis type
        if "neural" in hypothesis.lower():
            if self.mood == AgentMood.GRUMPY:
                return f"{self.emoji} More neural networks? *eye roll*"

        elif "quantum" in hypothesis.lower():
            if self.mood == AgentMood.SKEPTICAL:
                return f"{self.emoji} Quantum this, quantum that..."

        # Fall back to score-based reaction
        return self.react_to_score(score)
```

---

## 🔐 Adding Safety Features

### Example: Budget Constraints

```python
# In src/ORCHEX/safety/budget.py

class BudgetMonitor:
    """Monitor and limit research costs"""

    def __init__(self, max_cost_usd: float = 200.0):
        self.max_cost = max_cost_usd
        self.total_cost = 0.0

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if operation is within budget"""
        if self.total_cost + estimated_cost > self.max_cost:
            raise BudgetExceededError(
                f"Operation costs ${estimated_cost:.2f}, "
                f"would exceed budget of ${self.max_cost:.2f}"
            )
        return True

    def record_cost(self, cost: float):
        """Record actual cost"""
        self.total_cost += cost
```

**Integration:**

```python
# In protocol.py

self.budget_monitor = BudgetMonitor(max_cost_usd=200.0)

# Before expensive operation
self.budget_monitor.check_budget(estimated_llm_cost)

# After operation
self.budget_monitor.record_cost(actual_cost)
```

---

## 📝 Adding New Output Formats

### Current: Markdown reports, JSON data

**To add LaTeX output:**

```python
# In src/ORCHEX/exporters/latex.py

class LatexExporter:
    """Export research to LaTeX manuscript"""

    def export_project(self, project: ResearchProject) -> str:
        """Generate LaTeX manuscript"""

        latex = r"""
\documentclass{article}
\usepackage{amsmath}
\title{""" + project.topic + r"""}
\author{ORCHEX Autonomous Research System}
\date{\today}

\begin{document}
\maketitle

\begin{abstract}
""" + self._generate_abstract(project) + r"""
\end{abstract}

\section{Introduction}
""" + self._generate_intro(project) + r"""

\section{Methodology}
""" + self._generate_methods(project) + r"""

\section{Results}
""" + self._generate_results(project) + r"""

\section{Discussion}
""" + self._generate_discussion(project) + r"""

\end{document}
"""
        return latex
```

---

## 🧪 Testing Guidelines

### Unit Tests

```python
# In tests/test_agent_selection.py

import pytest
from meta_learning import MetaLearningProtocol

def test_agent_selection_cold_start():
    """Test agent selection with no prior data"""
    ml = MetaLearningProtocol(trajectory_path=":memory:")

    agent = ml.select_agent("self_refutation", "optimization")

    assert agent is not None
    assert agent.role == "self_refutation"

def test_agent_learning():
    """Test that agents improve over time"""
    ml = MetaLearningProtocol(trajectory_path=":memory:")

    # Simulate 20 successful projects with agent A
    for _ in range(20):
        agent = ml.select_agent("self_refutation", "optimization")
        ml.bandit.update("grumpy_refuter", reward=0.9, context="optimization")

    # Agent A should now be preferred
    selections = [
        ml.select_agent("self_refutation", "optimization").name
        for _ in range(10)
    ]

    assert selections.count("Grumpy Refuter") >= 8  # Selected 80%+ of time
```

### Integration Tests

```python
# In tests/test_integration.py

async def test_full_research_pipeline():
    """Test complete research workflow"""
    from ORCHEX import ATLASProtocol

    protocol = ATLASProtocol(
        output_base_dir="/tmp/test_atlas",
        enable_meta_learning=True
    )

    project = await protocol.run_research(
        topic="Test hypothesis generation",
        domain="computer_science",
        num_hypotheses=3
    )

    assert len(project.hypothesis_candidates) == 3
    assert project.output_dir.exists()
```

---

## 📚 Documentation Standards

### Code Comments

```python
def complex_function(param1: str, param2: int) -> dict:
    """
    Brief description of what function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        dict with keys 'result', 'metadata'

    Raises:
        ValueError: If param2 < 0

    Example:
        >>> complex_function("test", 5)
        {'result': ..., 'metadata': ...}
    """
    pass
```

### Module Docstrings

```python
"""
Module Name

Brief description of module purpose.

Key Classes:
    - ClassName1: What it does
    - ClassName2: What it does

Key Functions:
    - function1(): What it does
    - function2(): What it does

Example:
    >>> from module import ClassName1
    >>> obj = ClassName1()
    >>> obj.method()
"""
```

---

## 🐛 Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now all ORCHEX operations will log verbosely
protocol = ATLASProtocol(...)
```

### Inspect Trajectories

```python
from meta_learning import TrajectoryRecorder

recorder = TrajectoryRecorder("trajectories.jsonl")
trajectories = recorder.load_trajectories()

for traj in trajectories:
    print(f"\n{traj.topic}:")
    for action in traj.actions:
        print(f"  {action.agent_name}: {action.score:.1f}/100")
```

### Force Specific Agents

```python
# Bypass bandit, use specific agent
agent = meta_learning.select_agent(
    role="self_refutation",
    domain="optimization",
    force_agent="grumpy_refuter"  # Always use this one
)
```

### Disable Meta-Learning

```python
# Simplify debugging by removing meta-learning
protocol = ATLASProtocol(enable_meta_learning=False)
```

---

## 🚀 Performance Optimization

### Parallel Hypothesis Generation

```python
# Generate multiple hypotheses in parallel
import asyncio

tasks = [
    hypothesis_generator.generate_single_hypothesis(topic, domain)
    for _ in range(5)
]

hypotheses = await asyncio.gather(*tasks)
```

### Cache Literature Results

```python
# Cache Semantic Scholar results
from functools import lru_cache

@lru_cache(maxsize=100)
def search_papers_cached(query: str):
    return search_papers(query)
```

### Batch Agent Actions

```python
# Record multiple actions at once
actions = [...]  # List of action records
recorder.record_actions_batch(actions)
```

---

## 📦 Publishing Your Extension

### 1. Create Standalone Package

```python
# my-ORCHEX-extension/setup.py

setup(
    name="ORCHEX-code-reviewer",
    version="0.1.0",
    install_requires=[
        "ORCHEX-autonomous-research>=0.1.0",
        "turing-meta-learning>=0.1.0"
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"}
)
```

### 2. Register Agent

```python
# In your extension's __init__.py

from meta_learning import AGENT_ROSTER
from my_extension.agents import CodeReviewerAgent

# Auto-register on import
AGENT_ROSTER["code_reviewer"] = CodeReviewerAgent
```

### 3. Document

- README with usage examples
- Integration guide
- API reference

---

## 🎯 Contribution Guidelines

### Before Contributing

1. Read [ATLAS_ARCHITECTURE.md](ATLAS_ARCHITECTURE.md)
2. Check existing issues/PRs
3. Discuss major changes first

### Code Style

- Python 3.9+
- Type hints required
- Black formatter
- Ruff linter

### Pull Request Process

1. Create feature branch: `git checkout -b feature/my-feature`
2. Make changes with tests
3. Update documentation
4. Submit PR with description

---

## 📊 Metrics & Monitoring

### Track System Performance

```python
# Log all research metrics
import json

metrics = {
    "hypotheses_generated": len(project.hypothesis_candidates),
    "hypotheses_validated": len(project.validated_hypotheses),
    "failures": len(project.failures),
    "duration_seconds": (end_time - start_time).total_seconds(),
    "cost_usd": total_cost,
    "agent_performance": meta_learning.get_learning_summary()
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)
```

---

## 🔮 Future Development Areas

### High Priority
1. **Stage 3: Experimentation** - Code generation and execution
2. **Stage 4: Publication** - LaTeX paper generation
3. **Real Peer Review** - Integration with review platforms

### Medium Priority
4. **Multi-Agent Debates** - Agents collaborate on hypotheses
5. **Transfer Learning** - Cross-domain knowledge transfer
6. **Agent Evolution** - Genetic algorithm for agent optimization

### Low Priority
7. **Web Interface** - Visual dashboard for ORCHEX
8. **Cloud Integration** - AWS/GCP deployment
9. **Community Marketplace** - Share/download custom agents

---

## 📞 Getting Help

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: ORCHEX@example.com (hypothetical)
- **Docs**: This file + other guides in `docs/`

---

**Happy Building! 🚀**

*ORCHEX is designed to be extended. We can't wait to see what you create!*
