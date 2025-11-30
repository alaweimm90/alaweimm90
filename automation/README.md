# 🤖 Automation - Centralized AI Assets

> **Single source of truth** for all prompts, agents, workflows, orchestration, and tools.

This directory follows industry best practices from:
- **Anthropic**: [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- **CrewAI**: YAML-first agent configuration
- **LangChain**: Modular, composable architecture

---

## 📁 Directory Structure

```
automation/
├── README.md                    # This file
├── prompts/                     # All prompt templates
│   ├── CATALOG.md               # Master prompt index
│   ├── system/                  # System/orchestrator prompts
│   ├── project/                 # Project-specific superprompts
│   └── tasks/                   # Task prompts (audit, refactor, etc.)
│
├── agents/                      # Agent definitions
│   ├── config/
│   │   └── agents.yaml          # Agent registry (CrewAI-style)
│   └── templates/               # Agent templates
│
├── workflows/                   # Workflow definitions
│   ├── config/
│   │   └── workflows.yaml       # Workflow registry
│   └── templates/               # Workflow templates
│
├── orchestration/               # Orchestration logic
│   ├── config/
│   │   └── orchestration.yaml   # Routing & strategies
│   └── patterns/                # Orchestration patterns
│
└── tools/                       # Shared tools for agents
    └── config/
        └── tools.yaml           # Tool registry
```

---

## 🎯 Quick Reference

### Prompts (`prompts/`)

| Subdirectory | Purpose | Example |
|--------------|---------|---------|
| `system/` | System prompts for orchestrators | `orchestrator_system.md` |
| `project/` | Project-specific superprompts | `SIMCORE_SUPERPROMPT.md` |
| `tasks/` | Reusable task prompts | `PROMPT_OPTIMIZER.md` |

### Agents (`agents/config/agents.yaml`)

```yaml
agents:
  scientist_agent:
    role: "Research Scientist"
    goal: "Conduct rigorous scientific research"
    tools: [web_search, arxiv_search]
    llm_config:
      model: "claude-3-opus"
      temperature: 0.3
```

### Workflows (`workflows/config/workflows.yaml`)

```yaml
workflows:
  code_review:
    pattern: "evaluator_optimizer"  # Anthropic pattern
    stages:
      - name: "static_analysis"
        agent: "reviewer_agent"
      - name: "security_scan"
        agent: "reviewer_agent"
```

### Orchestration (`orchestration/config/orchestration.yaml`)

```yaml
patterns:
  prompt_chaining:
    use_when: "Task can be decomposed into fixed subtasks"
  
  routing:
    use_when: "Different categories require different handling"
  
  parallelization:
    use_when: "Tasks are independent, speed is important"
  
  orchestrator_workers:
    use_when: "Complex tasks requiring different expertise"
  
  evaluator_optimizer:
    use_when: "Clear evaluation criteria, iterative improvement"
```

---

## 🔑 Key Principles

### 1. YAML-First Configuration (CrewAI)
Define agents and tasks in YAML, not scattered code.

### 2. Separation of Concerns (LangChain)
- **Prompts**: What to say
- **Agents**: Who does it
- **Workflows**: How it flows
- **Orchestration**: When and where

### 3. Simple > Complex (Anthropic)
> "The most successful implementations use simple, composable patterns rather than complex frameworks."

### 4. Single Source of Truth
One location per asset type. No duplicates.

---

## 📊 Anthropic's Workflow Patterns

| Pattern | Description | Use When |
|---------|-------------|----------|
| **Prompt Chaining** | Sequential with gates | Fixed subtask decomposition |
| **Routing** | Classify → Route | Different categories need different handling |
| **Parallelization** | Concurrent execution | Independent tasks, speed matters |
| **Orchestrator-Workers** | Central LLM delegates | Complex, unpredictable subtasks |
| **Evaluator-Optimizer** | Iterative refinement | Clear criteria, improvement valuable |

---

## 🚀 Usage

### Load Agent Configuration
```python
import yaml

with open('automation/agents/config/agents.yaml') as f:
    agents = yaml.safe_load(f)

scientist = agents['agents']['scientist_agent']
print(f"Role: {scientist['role']}")
print(f"Tools: {scientist['tools']}")
```

### Load Workflow
```python
with open('automation/workflows/config/workflows.yaml') as f:
    workflows = yaml.safe_load(f)

code_review = workflows['workflows']['code_review']
for stage in code_review['stages']:
    print(f"Stage: {stage['name']} -> Agent: {stage['agent']}")
```

### Route Task
```python
with open('automation/orchestration/config/orchestration.yaml') as f:
    config = yaml.safe_load(f)

# Simple keyword-based routing
task = "refactor the authentication module"
keywords = config['tool_routing']['intent_extraction']['keywords']

for task_type, kws in keywords.items():
    if any(kw in task.lower() for kw in kws):
        print(f"Detected: {task_type}")
        print(f"Recommended tools: {config['tool_routing']['rules'][task_type]['tools']}")
        break
```

---

## 📝 Migration Notes

### Files Migrated From:
- `organizations/*/SUPERPROMPT.md` → `prompts/project/`
- `organizations/*/PROMPT_*.md` → `prompts/tasks/`
- `tools/legacy/orchestration/` → `orchestration/patterns/`
- `PROMPTS_CATALOG.md` → `prompts/CATALOG.md`

### Original Locations Preserved
Original files remain in place for backward compatibility. Future updates should be made in this centralized location.

---

## 🔗 Related Documentation

- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [CrewAI Documentation](https://docs.crewai.com/)
- [LangChain Architecture](https://python.langchain.com/docs/get_started/introduction)
- [`.metaHub/` Governance Policies](../.metaHub/)

---

## 📈 Roadmap

- [ ] Add prompt versioning
- [ ] Implement agent testing framework
- [ ] Add workflow visualization
- [ ] Create CLI for automation management
- [ ] Add prompt performance metrics
