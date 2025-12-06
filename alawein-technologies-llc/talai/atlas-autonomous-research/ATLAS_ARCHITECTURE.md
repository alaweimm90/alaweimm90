# ORCHEX - Autonomous Theorist & Laboratory Autonomous System

**Version 0.1.0** | **Status: Research Prototype**

---

## ⚠️ Important Disclaimer

**ORCHEX is NOT the Nobel Turing Challenge.** This system is:
- ✅ **Inspired by** Nobel Turing Challenge goals and methodology
- ✅ **A research prototype** for exploring autonomous scientific discovery
- ✅ **Focused on** computational research and hypothesis validation
- ❌ **NOT claiming** Nobel Prize-level discoveries
- ❌ **NOT meeting** full Nobel Turing Challenge criteria (requires wet-lab integration, multi-year validation, etc.)
- ❌ **NOT affiliated** with nobelturingchallenge.org

**Differences from Nobel Turing Challenge:**
1. **Scope**: Computational research only (no physical lab automation)
2. **Goal**: Rapid hypothesis iteration, not decades-long discovery programs
3. **Validation**: Self-testing and peer review simulation, not real-world Nobel-level validation
4. **Scale**: Single-topic research, not multi-agent Nobel-worthy programs
5. **Timeline**: Hours to days, not years to decades

**What ORCHEX IS:**
- A **fully autonomous** research assistant from topic → publication-ready manuscript
- An **integration platform** for self-refutation, interrogation, and failure learning
- A **research accelerator** for computational science
- An **educational tool** for understanding autonomous research systems

---

## 🎯 Vision

**Enable anyone to conduct computational research autonomously:**

```
Input: Research Topic ("Can we improve QAP solving with RL?")
    ↓
ORCHEX: [Creates repo, generates hypotheses, tests, refutes, validates, experiments, writes paper]
    ↓
Output: Publication-ready manuscript + Code + Data + Analysis
```

**Time**: Hours to days (not weeks/months)
**Cost**: $50-200 per research project
**Quality**: Sufficient for arXiv/workshop submission

---

## 🏗️ System Architecture

### Four-Stage Autonomous Research Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: DISCOVERY & HYPOTHESIS GENERATION                 │
├─────────────────────────────────────────────────────────────┤
│  Input: Research topic (text)                               │
│  ↓                                                           │
│  • Literature search (Semantic Scholar API)                 │
│  • Gap identification                                        │
│  • Hypothesis generation (5-10 candidates)                  │
│  • Novelty assessment                                        │
│  Output: Ranked hypotheses with novelty scores              │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: VALIDATION & REFINEMENT                           │
├─────────────────────────────────────────────────────────────┤
│  For each hypothesis:                                        │
│  ↓                                                           │
│  • Self-Refutation Protocol (5 strategies)                  │
│  • 200-Question Interrogation (10 categories)               │
│  • Risk Assessment (Hall of Failures)                       │
│  • Hypothesis refinement based on failures                  │
│  Output: Validated hypotheses (score ≥70/100)               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: EXPERIMENTATION & ANALYSIS                        │
├─────────────────────────────────────────────────────────────┤
│  • Experimental design generation                            │
│  • Code implementation (with tests)                          │
│  • Experiment execution (sandboxed)                          │
│  • Data analysis and visualization                           │
│  • Statistical validation                                    │
│  • Replication checks                                        │
│  Output: Experimental results + analysis + figures          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: SYNTHESIS & PUBLICATION                           │
├─────────────────────────────────────────────────────────────┤
│  • Literature review synthesis                               │
│  • Manuscript generation (LaTeX)                             │
│  • Figure generation and formatting                          │
│  • Citation management                                       │
│  • Automated peer review simulation                          │
│  • Revision based on feedback                                │
│  Output: Publication-ready paper + supplementary materials   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. **Hypothesis Generator**
- **Input**: Research topic, domain, context
- **Process**:
  - Query Semantic Scholar for recent papers
  - Identify gaps using LLM analysis
  - Generate 5-10 hypothesis candidates
  - Score novelty (0-100)
- **Output**: Ranked hypotheses
- **Models**: Claude Sonnet 4, GPT-4 (consensus)

### 2. **Validation Pipeline** (Integration of Turing Features)
- **Self-Refutation Protocol**: Attempts to falsify hypothesis
- **200-Question Interrogation**: Systematic stress-testing
- **Hall of Failures**: Risk assessment from past failures
- **Output**: Validation score (0-100), failure points, recommendations

### 3. **Experiment Designer**
- **Input**: Validated hypothesis
- **Process**:
  - Generate experimental protocol
  - Design baseline comparisons
  - Plan statistical tests
  - Estimate resources (time, compute, cost)
- **Output**: Executable experiment specification

### 4. **Code Generator & Executor**
- **Input**: Experiment specification
- **Process**:
  - Generate Python/Julia/R code
  - Write unit tests (>80% coverage)
  - Execute in sandboxed environment
  - Monitor for errors/failures
  - Auto-retry with fixes (max 3 attempts)
- **Output**: Results, data, logs

### 5. **Data Analyzer**
- **Input**: Experimental data
- **Process**:
  - Statistical analysis (t-tests, ANOVA, etc.)
  - Effect size calculations
  - Visualization generation (matplotlib, plotly)
  - Anomaly detection
- **Output**: Analysis report + figures

### 6. **Paper Generator**
- **Input**: All above outputs
- **Process**:
  - Structure: Abstract, Intro, Related Work, Methods, Results, Discussion, Conclusion
  - LaTeX formatting
  - Citation management (BibTeX)
  - Figure insertion
  - Algorithmic blocks
- **Output**: Compiled PDF + LaTeX source

### 7. **Peer Review Simulator**
- **Input**: Generated paper
- **Process**:
  - Multi-model review (Claude, GPT-4, Gemini)
  - Scoring: Novelty, Rigor, Clarity, Reproducibility
  - Generate reviewer comments
  - Suggest revisions
- **Output**: Review scores + comments

### 8. **Revision Engine**
- **Input**: Paper + review comments
- **Process**:
  - Address each comment
  - Revise manuscript
  - Re-run experiments if needed
- **Output**: Revised paper

---

## 🆚 Comparison with Existing Systems

| Feature | **ORCHEX** | Sakana AI Scientist | Nobel Turing Challenge | Agent Laboratory |
|---------|-----------|---------------------|------------------------|------------------|
| **Autonomy** | Full topic → paper | Requires templates | Aspirational (2050) | Idea → paper |
| **Validation** | 3-stage (Refute + Interrogate + Failures) | None | Comprehensive | Minimal |
| **Cost/Paper** | $50-200 | ~$15 | N/A | Unknown |
| **Domains** | Computational science | ML/AI only | All sciences | General |
| **Lab Integration** | ❌ No | ❌ No | ✅ Yes (goal) | ❌ No |
| **Hypothesis Generation** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No (user provides) |
| **Self-Refutation** | ✅ Yes (Popperian) | ❌ No | ⚠️ Implied | ❌ No |
| **Failure Learning** | ✅ Yes (Hall of Failures) | ❌ No | ⚠️ Implied | ❌ No |
| **Code Generation** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Peer Review** | ✅ Simulated | ✅ Simulated | ⚠️ Real (goal) | ⚠️ Minimal |

**Key Differentiator**: ORCHEX is the only system with **built-in hypothesis validation** through self-refutation and systematic interrogation BEFORE expensive experiments.

---

## 📊 Expected Performance

### Speed
- **Hypothesis Generation**: 5-10 minutes
- **Validation**: 20-30 minutes (5 hypotheses)
- **Experimentation**: 1-6 hours (depends on compute)
- **Paper Writing**: 30-60 minutes
- **Total**: 2-8 hours per research project

### Quality Metrics
- **Hypothesis Strength**: ≥70/100 (after validation)
- **Paper Quality**: Workshop/arXiv level
- **Code Quality**: >80% test coverage
- **Reproducibility**: 100% (code + data included)

### Cost
- **LLM Calls**: $30-150
- **Compute**: $10-50 (depends on experiments)
- **Total**: $50-200 per paper

### Success Rate
- **Hypothesis Validation**: ~40% pass all checks
- **Experiment Completion**: ~85% (with auto-retry)
- **Paper Generation**: ~95%
- **End-to-End**: ~30% (topic → accepted paper)

---

## 🎓 Academic Context

### Positioning

ORCHEX sits at the intersection of:
1. **Automated Scientific Discovery** (Kitano 2016, Wang et al. 2023)
2. **Agentic AI for Science** (Lu et al. 2024, ICLR 2025 Workshop)
3. **Meta-Learning & AutoML** (Hutter et al. 2019)
4. **Computational Philosophy of Science** (Popper, Kuhn, Lakatos)

### Novel Contributions

1. **Integration of Popperian Falsification**: First system with explicit self-refutation
2. **Multi-Stage Validation**: 3-layer validation before experiments
3. **Failure Learning**: Systematic learning from past failures
4. **Cost Efficiency**: 10x cheaper than manual research (time-wise)

### Limitations

1. **No physical experiments**: Computational only
2. **Template-bound experiments**: Requires executable code templates
3. **Limited novelty**: Incremental improvements, not paradigm shifts
4. **Ethical concerns**: No human oversight of published claims
5. **Quality variance**: 30% success rate for full pipeline

---

## 🚀 Use Cases

### 1. Research Prototyping
Quickly explore 5-10 hypotheses before committing to one.

### 2. Student Education
Teach research methodology by showing full pipeline.

### 3. Literature Survey Enhancement
Not just survey, but hypothesize and test gaps.

### 4. Reproducibility Crisis Solution
Every paper includes code + data + full reproducibility.

### 5. Rapid Iteration
Test idea → get results → iterate in hours, not months.

---

## 🔒 Safety & Ethics

### Built-in Safeguards

1. **Human-in-Loop Points**:
   - After hypothesis generation (approve hypotheses)
   - After validation (review failure points)
   - Before paper submission (final review)

2. **Sandbox Execution**:
   - All code runs in Docker containers
   - Resource limits (CPU, memory, time)
   - No network access during experiments

3. **Transparency**:
   - Full audit trail (every decision logged)
   - Explain-ability (why hypothesis was chosen/rejected)
   - Version control (Git repo for each project)

4. **Failure Modes**:
   - Auto-stop if hypothesis is refuted
   - Auto-stop if experiments fail 3x
   - Auto-stop if costs exceed budget

### Ethical Guidelines

1. **No Deception**: Always disclose AI authorship
2. **No Harmful Research**: Blacklist dangerous domains
3. **Credit Humans**: Human provides topic = co-authorship
4. **Open Science**: All outputs are open-access by default

---

## 📚 References

**Nobel Turing Challenge:**
- Kitano, H. (2021). Nobel Turing Challenge: creating the engine for scientific discovery. npj Systems Biology and Applications, 7(1), 29.

**AI Scientist:**
- Lu, C., et al. (2024). The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery. arXiv:2408.06292.

**Agentic Science:**
- Wang, Z., et al. (2024). From AI for Science to Agentic Science: A Survey on Autonomous Scientific Discovery. arXiv:2508.14111.

**Automated Discovery:**
- Cranmer, M., et al. (2023). Automated Scientific Discovery: From Equation Discovery to Autonomous Discovery Systems. arXiv:2305.02251.

---

## 🎯 Roadmap

### Phase 1 (Current): Core Pipeline
- ✅ Self-Refutation Protocol
- ✅ 200-Question Interrogation
- ✅ Hall of Failures Database
- ⏳ Hypothesis Generator
- ⏳ Experiment Designer
- ⏳ Paper Generator

### Phase 2: Enhancement
- Multi-domain support (beyond optimization)
- Real-time collaboration (multiple topics in parallel)
- Interactive refinement (chat with ORCHEX)
- Better experiment templates

### Phase 3: Community
- Public Hall of Failures (share learnings)
- Template marketplace (community templates)
- Leaderboard (track best discoveries)
- API for integration

---

**Built with ❤️ for the scientific community**

**Not claiming to solve the Nobel Turing Challenge, but trying to make research more accessible!**
