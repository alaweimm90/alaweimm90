# Naming Convention for AI Knowledge

## Philosophy

Prompts should be named by **TASK/GOAL**, not by project or technology.

### Bad (Project-Specific)
- ❌ `REPZ_SUPERPROMPT.md`
- ❌ `TALAI_SUPERPROMPT.md`
- ❌ `PROMPT_Librex.Flow.md`

### Good (Task-Focused)
- ✅ `fitness-platform-development.md`
- ✅ `ai-research-platform.md`
- ✅ `flow-optimization.md`

## Naming Rules

### 1. Use Kebab-Case
```
good: multi-agent-coordination.md
bad:  MultiAgentCoordination.md
bad:  multi_agent_coordination.md
```

### 2. Start with Action/Domain
```
good: autonomous-code-review.md
good: workflow-orchestration.md
good: prompt-engineering-optimizer.md
```

### 3. Be Descriptive but Concise
```
good: quantum-circuit-simulator.md
bad:  qcs.md
bad:  quantum-circuit-simulation-system-for-research.md
```

### 4. Avoid Project Names
```
good: optimization-framework.md
bad:  mezan-optimization.md

good: fitness-platform-development.md
bad:  repz-platform.md
```

### 5. Group Related Prompts
```
optimization/
  ├── resource-allocation-optimization.md
  ├── flow-optimization.md
  ├── graph-optimization.md
  └── evolutionary-optimization.md

ai-agents/
  ├── multi-agent-coordination.md
  ├── autonomous-code-review.md
  └── workflow-orchestration.md
```

## Categories

### Development
- `api-design-development`
- `typescript-automation`
- `fullstack-saas-template`
- `cicd-pipeline-setup`

### Optimization
- `resource-allocation-optimization`
- `flow-optimization`
- `graph-optimization`
- `quadratic-assignment-problem`

### AI/ML
- `ai-ml-integration`
- `prompt-engineering-optimizer`
- `multi-agent-coordination`
- `chain-of-thought-reasoning`

### Architecture
- `monorepo-architecture`
- `platform-deployment`
- `repository-consolidation`

### Testing/QA
- `testing-qa-strategy`
- `automated-test-generation`
- `autonomous-code-review`

### Scientific Computing
- `quantum-materials-simulation`
- `physics-simulation-engine`
- `scientific-computing-library`

### Design
- `ui-ux-design`
- `design-system-creation`

### Governance
- `governance-compliance`
- `approval-gating-system`
- `security-implementation`

## Examples

### Before → After

| Before (Project-Specific) | After (Task-Focused) |
|---------------------------|----------------------|
| `REPZ_SUPERPROMPT` | `fitness-platform-development` |
| `TALAI_SUPERPROMPT` | `ai-research-platform` |
| `PROMPT_Librex.Flow` | `flow-optimization` |
| `KILO_CONSOLIDATION` | `repository-consolidation` |
| `ATLAS_PROMPT_OPTIMIZER` | `prompt-engineering-optimizer` |
| `crew_manager` | `multi-agent-coordination` |

## When to Create New Prompts

### Good Reasons
- ✅ New task/workflow pattern emerges
- ✅ Reusable across multiple projects
- ✅ Solves a general problem

### Bad Reasons
- ❌ Project-specific one-off
- ❌ Too similar to existing prompt
- ❌ Not reusable

## Prompt Template

```markdown
# [Task Name]

> **[One-line description of what this accomplishes]**

## Purpose

[Explain the general task/goal, not tied to any specific project]

## When to Use

- [Scenario 1]
- [Scenario 2]
- [Scenario 3]

## Prompt

[The actual prompt content]

## Examples

### Input
[Generic example input]

### Output
[Expected output]

## Related Prompts

- [Related prompt 1]
- [Related prompt 2]
```

## Maintenance

### Adding New Prompts
1. Choose task-focused name
2. Use kebab-case
3. Place in appropriate category
4. Update catalog: `python tools/update-catalog.py`

### Renaming Existing Prompts
1. Edit `tools/rename-prompts.py`
2. Add to `RENAME_MAP`
3. Run: `python tools/rename-prompts.py`
4. Update catalog: `python tools/update-catalog.py`

---

**Remember**: Prompts are tools for tasks, not monuments to projects.
