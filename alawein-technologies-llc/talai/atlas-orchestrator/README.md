# 🤖 ORCHEX Orchestrator

**Multi-model AI orchestration with intelligent routing and cost optimization**

Part of [Project ORCHEX](../../MASTER_SUPER_PLAN.md) - Autonomous Theorist & Laboratory Autonomous System

---

## 🎯 What It Does

ORCHEX Orchestrator intelligently routes AI tasks across multiple models (Claude, GPT-4, Gemini, local models) to optimize for:
- **Cost**: Use cheaper models for simple tasks
- **Quality**: Use premium models for complex reasoning
- **Speed**: Parallel execution where possible
- **Context**: Match model strengths to task types

---

## ✨ Features

- 🎯 **Intelligent Routing** - Automatic model selection based on task type
- 💰 **Cost Optimization** - Track and minimize API costs
- 🔄 **Fallback Chains** - Automatic retry with alternative models
- ⚡ **Parallel Execution** - Run independent tasks concurrently
- 📊 **Cost Dashboard** - Real-time cost tracking and analytics
- 🧠 **Context Management** - Optimal use of 200K context windows
- 🔌 **Plugin System** - Easy to add new models

---

## 🚀 Quick Start

### Installation

```bash
pip install ORCHEX-orchestrator
```

### Basic Usage

```python
from atlas_orchestrator import Orchestrator, Task

# Initialize orchestrator
orchestrator = Orchestrator(
    anthropic_api_key="your-key",
    openai_api_key="your-key",
    google_api_key="your-key"
)

# Create a task
task = Task(
    prompt="Explain quantum entanglement in simple terms",
    task_type="explanation",
    max_tokens=500
)

# Execute (automatically routes to best model)
result = orchestrator.execute(task)
print(result.content)
print(f"Cost: ${result.cost:.4f}")
print(f"Model used: {result.model}")
```

### Intelligent Routing

```python
# Code task -> Routes to Claude (best for code)
code_task = Task(
    prompt="Write a Python function to calculate fibonacci",
    task_type="code"
)

# Research task -> Routes to GPT-4 (best for research)
research_task = Task(
    prompt="Summarize recent advances in quantum computing",
    task_type="research"
)

# Creative task -> Routes to appropriate model
creative_task = Task(
    prompt="Write a short story about AI",
    task_type="creative"
)

# Execute all
results = orchestrator.execute_batch([code_task, research_task, creative_task])
```

### Cost Tracking

```python
# Get cost report
report = orchestrator.get_cost_report()
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Total requests: {report.total_requests}")
print(f"Average cost per request: ${report.avg_cost_per_request:.4f}")

# Cost breakdown by model
for model, cost in report.cost_by_model.items():
    print(f"{model}: ${cost:.2f}")
```

---

## 📋 Task Types

The orchestrator automatically selects the best model based on task type:

| Task Type | Best Model | Reason |
|-----------|------------|--------|
| `code` | Claude Sonnet | Superior code generation, 200K context |
| `research` | GPT-4 | Excellent research synthesis |
| `analysis` | Claude Sonnet | Deep analytical reasoning |
| `creative` | GPT-4 | Strong creative writing |
| `simple` | GPT-3.5 | Fast and cheap for simple tasks |
| `long_context` | Claude Opus | 200K context window |

---

## 🎨 Advanced Features

### Fallback Chains

```python
# If primary fails, try alternatives
task = Task(
    prompt="Complex reasoning task",
    task_type="analysis",
    fallback_chain=["claude-opus", "gpt-4", "claude-sonnet"]
)

result = orchestrator.execute(task)
```

### Parallel Execution

```python
# Execute multiple independent tasks in parallel
tasks = [
    Task(prompt="Task 1", task_type="code"),
    Task(prompt="Task 2", task_type="research"),
    Task(prompt="Task 3", task_type="analysis"),
]

results = orchestrator.execute_parallel(tasks)
```

### Cost Budget

```python
# Set cost budget
orchestrator.set_budget(max_daily_cost=10.0)

# Raises BudgetExceededError if limit reached
result = orchestrator.execute(expensive_task)
```

### Context Window Management

```python
# Automatically splits large contexts
long_task = Task(
    prompt="Analyze this document...",
    context="[100,000 words of text]",
    task_type="analysis"
)

# Orchestrator handles splitting and reassembly
result = orchestrator.execute(long_task)
```

---

## 🔧 Configuration

Create `atlas_config.yaml`:

```yaml
orchestrator:
  default_model: claude-sonnet-4
  enable_fallback: true
  max_retries: 3
  timeout: 60

cost_limits:
  max_daily_cost: 50.0
  max_per_request: 1.0
  alert_threshold: 40.0

routing_rules:
  code:
    primary: claude-sonnet-4
    fallback: [gpt-4, claude-opus]
  research:
    primary: gpt-4
    fallback: [claude-opus, claude-sonnet-4]
  simple:
    primary: gpt-3.5-turbo
    fallback: [claude-haiku]

models:
  claude-opus:
    input_cost_per_1k: 0.015
    output_cost_per_1k: 0.075
    max_tokens: 200000

  claude-sonnet-4:
    input_cost_per_1k: 0.003
    output_cost_per_1k: 0.015
    max_tokens: 200000

  gpt-4:
    input_cost_per_1k: 0.03
    output_cost_per_1k: 0.06
    max_tokens: 128000

  gpt-3.5-turbo:
    input_cost_per_1k: 0.0005
    output_cost_per_1k: 0.0015
    max_tokens: 16000
```

---

## 📊 Cost Analysis Example

```python
from atlas_orchestrator import Orchestrator, CostAnalyzer

orchestrator = Orchestrator()

# Run various tasks
for i in range(100):
    task = generate_random_task()
    orchestrator.execute(task)

# Analyze cost efficiency
analyzer = CostAnalyzer(orchestrator)
report = analyzer.generate_report()

print(f"""
Cost Analysis Report
====================
Total Spent: ${report.total_cost:.2f}
Potential Cost (all GPT-4): ${report.all_gpt4_cost:.2f}
Savings: ${report.savings:.2f} ({report.savings_percent:.1f}%)

Model Distribution:
- Claude: {report.claude_percent:.1f}%
- GPT-4: {report.gpt4_percent:.1f}%
- GPT-3.5: {report.gpt35_percent:.1f}%

Optimization Score: {report.optimization_score}/100
""")
```

---

## 🧪 Testing

```bash
# Run targeted unit tests with the local sources
PYTHONPATH=src pytest tests

# With coverage (writes htmlcov/)
PYTHONPATH=src pytest --cov=atlas_orchestrator --cov-report=html

# Type checking
mypy src/atlas_orchestrator
```

### Debug Flags

- `ORCHESTRATOR_DEBUG_ROUTING=1` — prints the selected model + routing strategy for every task so you can see why a model was chosen.
- `ATLAS_PREFLIGHT=1` (from the ORCHEX CLI) — ensures orchestrator imports succeed before long research runs.

---

## 🗺️ Roadmap

- [x] Multi-model orchestration
- [x] Intelligent routing
- [x] Cost tracking
- [x] Fallback chains
- [ ] Local model support (Ollama)
- [ ] Streaming responses
- [ ] Fine-tuning integration
- [ ] Advanced caching
- [ ] Cost prediction
- [ ] A/B testing framework

---

## 📚 Documentation

Full documentation: https://yourorg.github.io/ORCHEX-orchestrator

- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api.md)
- [Cost Optimization Guide](docs/cost-optimization.md)
- [Model Comparison](docs/model-comparison.md)

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🌟 Part of Project ORCHEX

This is the **AI Orchestrator** component of [Project ORCHEX](../../MASTER_SUPER_PLAN.md), a Nobel-level autonomous research platform.

**Other ORCHEX Components**:
- Self-Refutation Protocol
- 200-Question Interrogation Framework
- Hall of Failures Database
- Meta-Learning Core
- Knowledge Graph System
- Research Dashboard

---

**Built with ❤️ for the future of autonomous research**
