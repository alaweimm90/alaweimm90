# ORCHEX Engine - Quick Start Guide

Get up and running with the ORCHEX Engine in 5 minutes!

---

## 🚀 Quick Installation

### Step 1: Navigate to Directory
```bash
cd /mnt/c/Users/mesha/Downloads/Important/ORCHEX/ORCHEX-core
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Dependencies**:
- redis>=4.0.0
- numpy>=1.21.0
- pandas>=1.3.0
- pytest>=7.0.0

---

## 🎯 5-Minute Demo

### Run the Demo Script
```bash
python demo.py
```

**What it does**:
- Initializes ORCHEX Engine
- Registers 10 research agents
- Assigns a test task
- Executes a thesis-antithesis-synthesis workflow
- Shows agent statistics

**Expected Output**:
```
======================================================================
ORCHEX Engine Demo - Multi-Agent Research Orchestration
======================================================================

1. Initializing ORCHEX Engine...
2025-11-14 - atlas_core.engine - INFO - ORCHEX Engine initialized

2. Registering 10 research agents...
2025-11-14 - atlas_core.agent - INFO - Initialized synthesis agent: agent_00
...

3. ORCHEX Engine Statistics:
   Total agents: 10
   Libria enabled: False
   Agents by type:
     - synthesis: 2
     - literature_review: 1
     - hypothesis_generation: 2
     ...

5. Executing Thesis-Antithesis-Synthesis workflow...
   Workflow Results:
   - Thesis quality: 0.88
   - Antithesis quality: 0.86
   - Synthesis quality: 0.85

======================================================================
Demo completed successfully!
======================================================================
```

---

## 📝 Basic Usage

### Example 1: Register Agents and Assign Tasks

```python
from atlas_core.engine import ATLASEngine
from atlas_core.agents import create_agent

# Initialize ORCHEX Engine (without Libria for now)
ORCHEX = ATLASEngine(libria_enabled=False)

# Create and register agents
for i in range(5):
    agent = create_agent(
        agent_type="synthesis",
        agent_id=f"synth_agent_{i}",
        skill_level=0.8,
        max_tasks=5
    )
    ORCHEX.register_agent(agent)

print(f"Registered {len(ORCHEX.agents)} agents")

# Create a task
task = {
    "task_id": "task_001",
    "task_type": "synthesis",
    "complexity": 0.7,
    "priority": 0.9
}

# Assign task to best agent
agent_id = ORCHEX.assign_task(task)
print(f"Task assigned to: {agent_id}")

# Execute task
result = ORCHEX.execute_task(task, agent_id)
print(f"Task completed with quality: {result.get('quality', 0):.2f}")
```

### Example 2: Execute Dialectical Workflow

```python
from atlas_core.engine import ATLASEngine
from atlas_core.agents import create_agent

# Initialize
ORCHEX = ATLASEngine(libria_enabled=False)

# Register required agent types
agent_types = [
    "hypothesis_generation",
    "critical_analysis",
    "synthesis"
]

for agent_type in agent_types:
    agent = create_agent(
        agent_type=agent_type,
        agent_id=f"{agent_type}_agent"
    )
    ORCHEX.register_agent(agent)

# Execute workflow
result = ORCHEX.execute_workflow(
    workflow_type="thesis_antithesis_synthesis",
    inputs={"topic": "neural_architecture_search"}
)

# Access results
print(f"Thesis: {result['thesis']}")
print(f"Antithesis: {result['antithesis']}")
print(f"Synthesis: {result['synthesis']}")
```

### Example 3: Agent Statistics

```python
# Get engine statistics
stats = ORCHEX.get_stats()
print(f"Total agents: {stats['total_agents']}")
print(f"Agents by type: {stats['agents_by_type']}")

# Get individual agent statistics
for agent_id, agent in ORCHEX.agents.items():
    agent_stats = agent.get_stats()
    if agent_stats['total_tasks'] > 0:
        print(f"\n{agent_id}:")
        print(f"  Tasks completed: {agent_stats['total_tasks']}")
        print(f"  Success rate: {agent_stats['success_rate']:.1%}")
        print(f"  Average quality: {agent_stats['avg_quality']:.2f}")
```

---

## 🧪 Running Tests

### Run All Tests
```bash
cd ORCHEX-core
pytest tests/ -v
```

### Run Specific Tests
```bash
# Engine tests only
pytest tests/test_engine.py -v

# Agent tests only
pytest tests/test_agents.py -v

# With coverage
pytest tests/ --cov=atlas_core --cov-report=html
```

**Expected Output**:
```
tests/test_engine.py::test_engine_initialization PASSED
tests/test_engine.py::test_agent_registration PASSED
tests/test_engine.py::test_multiple_agents PASSED
tests/test_engine.py::test_task_assignment PASSED
tests/test_engine.py::test_dialectical_workflow PASSED
tests/test_engine.py::test_engine_stats PASSED
tests/test_agents.py::test_agent_config PASSED
tests/test_agents.py::test_synthesis_agent PASSED
...
=============== 15 passed in 0.5s ===============
```

---

## 🔧 Configuration

### With Libria Integration (Week 2+)

```python
# Enable Libria integration
ORCHEX = ATLASEngine(
    redis_url="redis://localhost:6379/0",
    libria_enabled=True
)

# ORCHEX will automatically use Libria solvers:
# - Librex.QAP for task assignment
# - Librex.Flow for workflow routing
# - Librex.Graph for topology optimization
```

### Custom Redis Configuration

```python
from atlas_core.blackboard import ATLASBlackboard

# Connect to custom Redis instance
blackboard = ATLASBlackboard("redis://remote-host:6379/1")

# Test connection
if blackboard.ping():
    print("Connected to Redis!")
```

---

## 📚 Available Agent Types

| Agent Type | Description |
|------------|-------------|
| `synthesis` | Synthesizes information from multiple sources |
| `literature_review` | Reviews and summarizes literature |
| `hypothesis_generation` | Generates testable hypotheses |
| `critical_analysis` | Provides critical analysis |
| `validation` | Validates research findings |
| `data_analysis` | Analyzes data and statistics |
| `methodology` | Designs research methodologies |
| `ethics_review` | Reviews ethical implications |

### Creating Agents

```python
# Method 1: Factory function (recommended)
agent = create_agent(
    agent_type="synthesis",
    agent_id="agent_001",
    specialization="machine_learning",
    skill_level=0.85,
    max_tasks=5,
    model="claude-3-opus"
)

# Method 2: Direct instantiation
from atlas_core.agent import AgentConfig
from atlas_core.agents import SynthesisAgent

config = AgentConfig(
    agent_id="agent_002",
    agent_type="synthesis",
    specialization="NLP",
    skill_level=0.9,
    max_tasks=3,
    model="gpt-4-turbo"
)
agent = SynthesisAgent(config)
```

---

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**:
```bash
# Make sure you're in the ORCHEX-core directory
cd ORCHEX-core

# Install in development mode
pip install -e .
```

### Issue: Redis connection fails
**Solution**:
```bash
# Check if Redis is running
redis-cli ping

# Start Redis with Docker
docker-compose up -d redis

# Or use ORCHEX in mock mode (no Redis needed)
ORCHEX = ATLASEngine(libria_enabled=False)
```

### Issue: Tests fail
**Solution**:
```bash
# Install test dependencies
pip install pytest

# Run tests with verbose output
pytest tests/ -v -s

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

---

## 📖 Next Steps

### Week 1 (Current)
1. ✅ Run demo: `python demo.py`
2. ✅ Run tests: `pytest tests/ -v`
3. ✅ Read documentation: `README.md`

### Week 2 (Integration)
1. ⏳ Start Redis: `docker-compose up -d redis`
2. ⏳ Enable Libria: `ORCHEX = ATLASEngine(libria_enabled=True)`
3. ⏳ Test integration with Librex.QAP
4. ⏳ Expand to 12+ agents

### Week 3+ (Advanced)
1. ⏳ Implement custom agents
2. ⏳ Create custom workflows
3. ⏳ Performance optimization
4. ⏳ Production deployment

---

## 📞 Getting Help

### Documentation
- **README.md** - Complete usage guide
- **WEEK_1_COMPLETION_SUMMARY.md** - Implementation details
- **INTEGRATION_CHECKLIST.md** - Integration roadmap
- **AGENT_EXPANSION_GUIDE.md** - Adding more agents

### Code Examples
- **demo.py** - Working demonstration
- **tests/test_engine.py** - Engine usage examples
- **tests/test_agents.py** - Agent usage examples

---

## ✨ Key Features Quick Reference

```python
# 1. Initialize engine
ORCHEX = ATLASEngine(libria_enabled=False)

# 2. Register agents
agent = create_agent("synthesis", "agent_1")
ORCHEX.register_agent(agent)

# 3. Assign task
task = {"task_id": "t1", "task_type": "synthesis"}
agent_id = ORCHEX.assign_task(task)

# 4. Execute task
result = ORCHEX.execute_task(task, agent_id)

# 5. Execute workflow
result = ORCHEX.execute_workflow(
    "thesis_antithesis_synthesis",
    {"topic": "AI_safety"}
)

# 6. Get statistics
stats = ORCHEX.get_stats()
agent_stats = agent.get_stats()
```

---

## 🎯 Common Use Cases

### Use Case 1: Research Literature Review
```python
agent = create_agent("literature_review", "lit_1")
ORCHEX.register_agent(agent)

task = {
    "task_id": "review_001",
    "topic": "transformer_architectures",
    "query": "Recent advances in transformer models"
}

agent_id = ORCHEX.assign_task(task)
result = ORCHEX.execute_task(task, agent_id)
print(result["summary"])
```

### Use Case 2: Multi-Agent Collaboration
```python
# Register multiple complementary agents
agents = [
    create_agent("hypothesis_generation", "hyp_1"),
    create_agent("methodology", "meth_1"),
    create_agent("data_analysis", "data_1"),
    create_agent("validation", "val_1")
]

for agent in agents:
    ORCHEX.register_agent(agent)

# Execute collaborative workflow
# Each agent contributes their expertise
```

### Use Case 3: Quality Assurance Pipeline
```python
# Register QA agents
qa_agents = [
    create_agent("critical_analysis", "critic_1"),
    create_agent("validation", "validator_1"),
    create_agent("ethics_review", "ethics_1")
]

for agent in qa_agents:
    ORCHEX.register_agent(agent)

# Run QA on research output
# Ensures quality, validity, and ethics
```

---

**Ready to start?** Run `python demo.py` now! 🚀

**Questions?** Check `README.md` or the documentation files.

**Week 2?** See `INTEGRATION_CHECKLIST.md` for next steps.
