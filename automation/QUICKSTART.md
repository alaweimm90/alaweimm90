# 🚀 Automation Quick Start

## CLI Commands

```bash
# List all assets
atlas prompts list
atlas agents list
atlas workflows list

# Show specific asset
atlas prompts show orchestrator
atlas agents show scientist_agent
atlas workflows show code_review

# Route a task
atlas route "refactor the authentication module"

# Search prompts
atlas automation prompts search "optimization"

# View orchestration patterns
atlas automation patterns
```

## Python API

```python
from automation import (
    load_agent,
    load_workflow,
    load_prompt,
    route_task,
    get_agents,
    get_workflows
)

# Load an agent
agent = load_agent("scientist_agent")
print(f"Role: {agent['role']}")
print(f"Tools: {agent['tools']}")

# Load a workflow
workflow = load_workflow("code_review")
for stage in workflow['stages']:
    print(f"Stage: {stage['name']}")

# Load a prompt
prompt = load_prompt("system", "orchestrator")
print(prompt[:500])

# Route a task
result = route_task("debug the payment processing error")
print(f"Type: {result['detected_type']}")
print(f"Tools: {result['recommended_tools']}")
```

## Directory Structure

```
automation/
├── prompts/
│   ├── system/      # orchestrator.md, router.md, evaluator.md
│   ├── project/     # 9 project superprompts
│   └── tasks/       # 13 task prompts
├── agents/
│   ├── config/agents.yaml    # 18 agent definitions
│   └── templates/            # 3 agent templates
├── workflows/
│   ├── config/workflows.yaml
│   └── templates/            # 3 workflow templates
├── orchestration/
│   ├── config/orchestration.yaml
│   └── patterns/             # 5 Anthropic patterns
└── tools/
    └── config/tools.yaml
```

## Agent Categories

| Category | Agents |
|----------|--------|
| **Research** | scientist, theory, scout, data_analyst |
| **Development** | coder, reviewer, debugger, architect, devops |
| **Content** | writer, critic, visualization, editor |
| **Security** | security, compliance |
| **Operations** | monitor, incident |

## Workflow Patterns

| Pattern | Use When |
|---------|----------|
| **Prompt Chaining** | Fixed sequential steps |
| **Routing** | Different inputs need different handling |
| **Parallelization** | Independent tasks, speed matters |
| **Orchestrator-Workers** | Complex, unpredictable subtasks |
| **Evaluator-Optimizer** | Iterative refinement needed |

## Task Routing Keywords

| Type | Keywords |
|------|----------|
| **Architecture** | design, architect, structure, plan |
| **Implementation** | implement, code, build, create |
| **Debugging** | debug, fix, error, bug, issue |
| **Refactoring** | refactor, clean, simplify, optimize |
| **Testing** | test, verify, validate, check |
| **Research** | research, analyze, investigate, study |

## Creating New Assets

### New Agent
```yaml
# automation/agents/templates/agent_template.yaml
name: "my_agent"
role: "My Role"
goal: "What this agent does"
backstory: |
  Background and expertise...
tools:
  - web_search
  - calculator
llm_config:
  model: "claude-3-sonnet"
  temperature: 0.3
```

### New Workflow
```yaml
# automation/workflows/templates/workflow_template.yaml
name: "my_workflow"
pattern: "prompt_chaining"
stages:
  - name: "step_1"
    agent: "agent_name"
    action: "What to do"
```

### New Prompt
```markdown
# automation/prompts/tasks/MY_PROMPT.md

# My Prompt Title

## Purpose
What this prompt does

## Instructions
The actual prompt content...
```
