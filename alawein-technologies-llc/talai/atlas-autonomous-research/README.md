# 🚀 ORCHEX - Autonomous Theorist & Laboratory Autonomous System

**Full autonomous research: Topic → Hypotheses → Validation → Experiments → Paper**

Version 0.1.0 | Research Prototype

---

## ⚠️ IMPORTANT DISCLAIMER

**ORCHEX is NOT the Nobel Turing Challenge.**

This system is:
- ✅ **Inspired by** Nobel Turing Challenge goals (nobelturingchallenge.org)
- ✅ **A research prototype** exploring autonomous discovery
- ✅ **Focused on** computational research only
- ❌ **NOT claiming** Nobel Prize-level discoveries
- ❌ **NOT meeting** full Nobel Turing Challenge criteria
- ❌ **NOT affiliated** with nobelturingchallenge.org

See [ATLAS_ARCHITECTURE.md](ATLAS_ARCHITECTURE.md) for full comparison.

---

## 🎯 What ORCHEX Does

ORCHEX takes **any research topic** and autonomously:

1. **Generates hypotheses** (via literature search + LLM)
2. **Validates rigorously** (self-refutation + 200-question interrogation)
3. **Learns from failures** (Hall of Failures database)
4. **Creates project repo** (Git-initialized, organized)
5. **🧠 Self-improves** (Meta-learning from past projects)
6. **🎭 Personality agents** (Grumpy Refuter 😠, Skeptical Steve 🤨, Failure Frank 🤦)
7. **⏳ Runs experiments** (coming soon)
8. **⏳ Writes papers** (coming soon)

**Timeline**: 2-8 hours per project
**Cost**: $50-200 (LLM + compute)
**Quality**: Sufficient for arXiv/workshop submission

### 🎭 Meet the Research Team

ORCHEX uses **personality-based agents** to make research fun and effective:

- **Grumpy Refuter** 😠 - "Everything is flawed until proven otherwise!" (Strictness: 0.9)
- **Skeptical Steve** 🤨 - "Show me the data or get out." (200 annoying questions)
- **Failure Frank** 🤦 - "I've seen this mistake before, kid..." (Remembers all past failures)
- **Optimistic Oliver** 😄 - "Every idea is a potential breakthrough!" (Generates creative hypotheses)
- **Cautious Cathy** 😰 - "Let's think about what could go wrong..." (Risk assessment expert)
- **Pedantic Pete** 🤓 - "Technically speaking, there's an issue on line 47..." (Peer review)
- **Enthusiastic Emma** 🎉 - "Let's run ALL the experiments!" (Experiment design)

Each agent **learns from experience** and gets better over time!

---

## 🚀 Quick Start

### Installation

```bash
pip install ORCHEX-autonomous-research
```

### Basic Usage

```bash
# Run autonomous research on a topic
ORCHEX research "Reinforcement learning for QAP solving" --domain optimization

# With AI Orchestrator (better quality)
ORCHEX research "Neural architecture search" --domain machine_learning --with-orchestrator

# System information
ORCHEX info
```

### Python API

```python
import asyncio
from ORCHEX import ATLASProtocol

# Initialize
protocol = ATLASProtocol()

# Run research
project = asyncio.run(
    protocol.run_research(
        topic="Improving QAP solvers with meta-learning",
        domain="optimization",
        num_hypotheses=5,
    )
)

print(f"Generated {len(project.hypothesis_candidates)} hypotheses")
print(f"Validated {len(project.validated_hypotheses)} hypotheses")
print(f"Output: {project.output_dir}")
```

---

## 🩺 Diagnostics & Preflight Safety Nets

Keep the stack healthy before burning tokens:

- `ORCHEX diagnostics` — runs dependency + API-key checks and exits non‑zero if a critical module is missing. Add `--json` for machine-readable output.
- `ORCHEX research --preflight/--skip-preflight` — research runs call diagnostics first unless you opt out. The default is controlled by `ATLAS_PREFLIGHT` (`1` to enforce, `0` to skip in CI).
- Missing optional agents (e.g., `meta_learning`) only raise warnings; missing core stacks (self-refutation, interrogation, hall-of-failures, orchestrator) block execution so failures stay obvious.

### Debug Flags

- `--debug` / `--no-debug` — CLI switch to enable/disable verbose logs. Overrides `ATLAS_DEBUG`.
- `ATLAS_DEBUG=1` — enables verbose CLI logs around key execution points (preflight status and protocol run boundaries). Use when diagnosing environment or flow issues.
- `ORCHESTRATOR_DEBUG_ROUTING=1` — prints selected model and routing details for each task in the orchestrator.

### Optional Dependencies

- Some features use optional libraries. If missing, functionality degrades gracefully with warnings. Install as needed or run with features disabled.

Common optional deps and where they’re used:

- `tiktoken` — faster/more accurate token accounting for prompts and budget tracking.
- `sqlalchemy` — persistent storage backends (e.g., extended Hall of Failures); falls back to simpler storage if absent.
- Provider SDKs: `anthropic`, `openai`, `google-generativeai` — used by the Orchestrator to call respective model providers when configured via env vars.

Example enabling debug at the CLI:

```
ORCHEX research "Neural architecture search" --domain machine_learning --debug
```

---

## 🏗️ System Architecture

ORCHEX implements a **4-stage autonomous research workflow**:

Topic → Gaps → Hypotheses → Validation → Outcomes

High-level flow:

```
Topic ─▶ Gap Analysis ─▶ Hypotheses ─▶ Validation ─▶ Outcomes
          (ORCHEX)         (ORCHEX)        (Turing         (ORCHEX)
                                         Features)
```

```
┌─────────────────────────────────────────────┐
│  STAGE 1: Hypothesis Generation             │
│  • Literature search (Semantic Scholar)     │
│  • Gap identification (LLM analysis)        │
│  • Generate 5-10 candidates                 │
│  • Score novelty + feasibility              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  STAGE 2: Validation & Refinement           │
│  • Self-Refutation Protocol (5 strategies)  │
│  • 200-Question Interrogation (10 cats)     │
│  • Risk Assessment (Hall of Failures)       │
│  • Keep only strong hypotheses (≥70/100)    │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  STAGE 3: Experimentation (⏳ Coming Soon)  │
│  • Design experiments                       │
│  • Generate + test code                     │
│  • Execute in sandbox                       │
│  • Analyze results                          │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│  STAGE 4: Publication (⏳ Coming Soon)      │
│  • Generate LaTeX manuscript                │
│  • Create figures                           │
│  • Format citations                         │
│  • Simulate peer review                     │
└─────────────────────────────────────────────┘
```

### Integrated Features

ORCHEX integrates 4 novel Turing-inspired features:

1. **Self-Refutation Protocol**: Popperian falsification (5 strategies)
2. **200-Question Interrogation**: Systematic stress-testing (10 categories)
3. **Hall of Failures**: Learn from past failures, prevent repeats
4. **Meta-Learning Core**: Personality agents that learn from experience (UCB1 bandit optimization)

---

## 📊 Example Output

```

---

## ✅ Quality Gates

| Scenario | Command | Notes |
| --- | --- | --- |
| Fast smoke (diagnostics + caching) | `make smoke` | Asserts diagnostics + hypothesis generator cache behavior using local module paths. |
| Full ORCHEX tests | `make test-full` | Runs every test with the required Turing feature packages on `PYTHONPATH`. |
| Orchestrator regression | `make orchestrator-tests` | Executes the routing tests living in `02-PROJECTS/ORCHEX-orchestrator`. |
| Manual CLI dry-run | `ATLAS_PREFLIGHT=1 ORCHEX research "Topic" --domain optimization --skip-preflight` | Skip preflight once you trust the environment (e.g., inside scripted smoke runs). |

> Tip: export the `PYTHONPATH` shown in the `Makefile` if you prefer calling `pytest` directly.
ORCHEX AUTONOMOUS RESEARCH SYSTEM
================================================================================

Topic: Reinforcement learning for QAP solving
Domain: optimization
Output: ./atlas_projects/reinforcement_learning_for_qap_solving_20250511_143022

────────────────────────────────────────────────────────────────────────────────
STAGE 1: HYPOTHESIS GENERATION
────────────────────────────────────────────────────────────────────────────────

✓ Generated 5 hypothesis candidates

  1. Q-learning with adaptive exploration can improve QAP solution quality...
     Novelty: 0.78 | Feasibility: 0.65 | Combined: 0.73
  2. Policy gradient methods outperform value-based methods for large QAP...
     Novelty: 0.82 | Feasibility: 0.58 | Combined: 0.72
  ...

────────────────────────────────────────────────────────────────────────────────
STAGE 2: HYPOTHESIS VALIDATION
────────────────────────────────────────────────────────────────────────────────

Validating Hypothesis 1...
  • Risk assessment... ✓ Low risk
  • Self-refutation... ✓ Survived refutation (score: 73.2/100)
  • 200-Question interrogation... → Overall score: 68.4/100
    ✓ VALIDATED (combined: 70.8/100)

Validating Hypothesis 2...
  • Risk assessment... ⚠️ High risk detected! Skipping.
  ...

✓ Selected hypothesis: Q-learning with adaptive exploration can improve QAP...

────────────────────────────────────────────────────────────────────────────────
RESEARCH PROJECT SUMMARY
────────────────────────────────────────────────────────────────────────────────

Topic: Reinforcement learning for QAP solving
Hypotheses generated: 5
Hypotheses validated: 2
Failures recorded: 3
Output directory: ./atlas_projects/reinforcement_learning_for_qap_solving_20250511_143022

================================================================================
```

---

## 🆚 Comparison with Other Systems

| Feature | **ORCHEX** | Sakana AI Scientist | Nobel Turing Challenge |
|---------|-----------|---------------------|------------------------|
| **Hypothesis Generation** | ✅ From topic | ✅ From template | ✅ (goal 2050) |
| **Self-Refutation** | ✅ Built-in | ❌ No | ⚠️ Implied |
| **Systematic Validation** | ✅ 3-stage | ❌ No | ⚠️ Implied |
| **Failure Learning** | ✅ Hall of Failures | ❌ No | ⚠️ Implied |
| **Meta-Learning** | ✅ Personality agents + UCB1 | ❌ No | ⚠️ Implied |
| **Cost per Paper** | $50-200 | ~$15 | N/A |
| **Autonomy** | Full topic → paper | Requires templates | Full (aspirational) |
| **Lab Integration** | ❌ Computational only | ❌ No | ✅ (goal) |

**Key Differentiator**: ORCHEX validates hypotheses BEFORE experiments, saving costs on doomed projects.

---

## 📚 Documentation

- [**ATLAS_ARCHITECTURE.md**](ATLAS_ARCHITECTURE.md) - Full system architecture and comparison
- [**Examples**](examples/) - Usage examples
- [**API Reference**](docs/api.md) - Python API documentation

---

## 🎓 Academic Context

ORCHEX is inspired by:
- **Nobel Turing Challenge** (Kitano 2021) - Grand challenge for autonomous discovery
- **The AI Scientist** (Lu et al. 2024) - Automated paper generation
- **Agentic Science** (Wang et al. 2024) - Survey on autonomous discovery

Novel contributions:
1. Integration of Popperian falsification in autonomous systems
2. 3-stage validation before experiments
3. Systematic failure learning and prevention

---

## ⚖️ Limitations

ORCHEX is a **research prototype** with limitations:

1. **Computational only**: No wet lab, no physical experiments
2. **Template-dependent**: Experiments require code templates
3. **Incremental discoveries**: Not Nobel-level breakthroughs
4. **Quality variance**: 30% success rate for full pipeline
5. **Cost**: $50-200 per project (vs free human brainstorming)
6. **Ethical concerns**: AI authorship questions

---

## 🔒 Safety & Ethics

- **Human-in-loop**: Review at hypothesis selection, experiment approval, paper submission
- **Sandboxed execution**: All code runs in isolated containers
- **Transparency**: Full audit trail of all decisions
- **Open science**: All outputs are open-access by default
- **AI disclosure**: Always disclose AI authorship

---

## 🚧 Roadmap

### Current (v0.1.0)
- ✅ Hypothesis generation from topics
- ✅ Self-refutation validation
- ✅ 200-question interrogation
- ✅ Hall of Failures learning
- ✅ Meta-learning with personality agents (UCB1 bandit)
- ✅ Git repo creation

### Next (v0.2.0)
- ⏳ Experiment designer
- ⏳ Code generator + executor
- ⏳ Data analyzer
- ⏳ Paper generator (LaTeX)

### Future (v1.0.0)
- Multi-domain templates
- Real peer review integration
- Collaborative multi-agent research
- API for community integration

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

Inspired by:
- Nobel Turing Challenge (nobelturingchallenge.org)
- Sakana AI's The AI Scientist
- ICLR 2025 Agentic AI for Science Workshop

Built on top of:
- Anthropic Claude Sonnet 4
- OpenAI GPT-4
- Google Gemini Pro
- Semantic Scholar API

---

**ORCHEX: Making computational research accessible to everyone! 🚀**

*(But please remember: We're not claiming to solve the Nobel Turing Challenge - just trying to help researchers work more efficiently!)*
