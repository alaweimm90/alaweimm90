# 🎯 Prompts Catalog

> **Centralized prompt library** - Single source of truth for all AI prompts.

---

## 📁 Organization

```
prompts/
├── CATALOG.md          # This file - Master index
├── system/             # System prompts for orchestrators
├── project/            # Project-specific superprompts (9 files)
└── tasks/              # Reusable task prompts (13 files)
```

---

## 📋 Project Superprompts (`project/`)

Project-specific prompts that define the context, goals, and constraints for each project.

| File | Project | Description |
|------|---------|-------------|
| `SIMCORE_CLAUDE_CODE_SUPERPROMPT.md` | SimCore | Simulation core development prompt |
| `TALAI_SUPERPROMPT.md` | TalAI | AI research assistant prompt |
| `SPIN_CIRC_SUPERPROMPT.md` | SpinCirc | Spintronics circuit design |
| `MAG_LOGIC_SUPERPROMPT.md` | MagLogic | Magnetic logic systems |
| `SCI_COMP_SUPERPROMPT.md` | SciComp | Scientific computing |
| `QUBE_ML_SUPERPROMPT.md` | QubeML | Quantum ML systems |
| `QMAT_SIM_SUPERPROMPT.md` | QMatSim | Quantum materials simulation |
| `REPZ_SUPERPROMPT.md` | Repz | Business/reputation system |
| `LOCAL_AI_ORCHESTRATION_SUPERPROMPT.md` | MEZAN | Local AI orchestration |

---

## 🔧 Task Prompts (`tasks/`)

Reusable prompts for common development and research tasks.

### Optimization & Cleanup

| File | Purpose |
|------|---------|
| `PROMPT_OPTIMIZER.md` | Optimize prompts for better AI responses |
| `ATLAS_PROMPT_OPTIMIZER.md` | Atlas-specific prompt optimization |
| `MASTER_CLEANUP_PROMPT.md` | Repository cleanup and organization |

### Design & Architecture

| File | Purpose |
|------|---------|
| `DESIGN_SYSTEM_PROMPTS.md` | UI/UX design system prompts |
| `BRAINSTORMING_PROMPTS.md` | Creative ideation and brainstorming |
| `CRAZY_IDEAS_MASTER_PROMPT.md` | Unconventional problem-solving |

### Libria Optimization Suite

Specialized prompts for optimization problem domains:

| File | Domain |
|------|--------|
| `PROMPT_QAPLibria.md` | Quadratic Assignment Problem |
| `PROMPT_FlowLibria.md` | Network Flow Optimization |
| `PROMPT_GraphLibria.md` | Graph Optimization |
| `PROMPT_AllocLibria.md` | Resource Allocation |
| `PROMPT_DualLibria.md` | Dual Optimization |
| `PROMPT_EvoLibria.md` | Evolutionary Algorithms |
| `PROMPT_MetaLibria.md` | Meta-heuristics |

---

## 🚀 Usage

### Load a Prompt

```python
from pathlib import Path

def load_prompt(category: str, name: str) -> str:
    """Load a prompt from the centralized library."""
    prompt_path = Path("automation/prompts") / category / f"{name}.md"
    return prompt_path.read_text(encoding="utf-8")

# Example
superprompt = load_prompt("project", "SIMCORE_CLAUDE_CODE_SUPERPROMPT")
task_prompt = load_prompt("tasks", "PROMPT_OPTIMIZER")
```

### Combine Prompts

```python
def build_context(project: str, task: str) -> str:
    """Build context by combining project and task prompts."""
    project_prompt = load_prompt("project", project)
    task_prompt = load_prompt("tasks", task)
    
    return f"""
# Project Context
{project_prompt}

# Task Instructions
{task_prompt}
"""
```

---

## 📝 Adding New Prompts

### 1. Choose the Right Category

- **`system/`**: Orchestrator/system-level prompts
- **`project/`**: Project-specific superprompts
- **`tasks/`**: Reusable task prompts

### 2. Follow Naming Convention

- Project prompts: `{PROJECT}_SUPERPROMPT.md`
- Task prompts: `PROMPT_{TaskName}.md` or `{TASK}_PROMPT.md`

### 3. Include Metadata

```markdown
---
name: "Prompt Name"
version: "1.0"
category: "project|tasks|system"
tags: ["optimization", "research", "development"]
created: "2024-11-29"
---

# Prompt Title

## Purpose
[What this prompt does]

## Usage
[How to use this prompt]

## Prompt Content
[The actual prompt]
```

### 4. Update This Catalog

Add your new prompt to the appropriate table above.

---

## 🔗 Related

- [Agents Registry](../agents/config/agents.yaml)
- [Workflows Registry](../workflows/config/workflows.yaml)
- [Orchestration Config](../orchestration/config/orchestration.yaml)
- [Automation README](../README.md)

---

## 📊 Statistics

| Category | Count | Total Size |
|----------|-------|------------|
| Project Superprompts | 9 | ~38 KB |
| Task Prompts | 13 | ~428 KB |
| System Prompts | 0 | - |
| **Total** | **22** | **~466 KB** |

---

*Last updated: 2024-11-29*
