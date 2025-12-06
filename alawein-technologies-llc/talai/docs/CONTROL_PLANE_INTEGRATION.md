# ORCHEX / Turingo / Librex Control Plane Integration Guide

> **Scope**: This document describes how the core tal‑ai systems fit together as a modular AI control stack:
>
> - ORCHEX Orchestrator
> - ORCHEX Autonomous Research
> - Turingo
> - Librex
>
> It focuses on **interfaces and extension points**, not implementation details.

---

## 1. High‑Level Architecture

```text
[User / Application]
      |
      v
+---------------------------+
| ORCHEX Autonomous Research |
|  - run_research(topic)    |
|  - workflow_orchestrator  |
|  - meta-learning agents   |
+---------------------------+
      |            \
      |             \  (optimization-heavy stages)
      v              \
+---------------------------+      +---------------------------+
| ORCHEX Orchestrator       |      | Turingo                   |
|  - router.select_model   |      |  - TuringoRodeo.solve()   |
|  - cost_tracker          |      |  - PuzzleProdigy, etc.    |
|  - provider adapters     |      |  - SOP_Stampede           |
+---------------------------+      |  - SOP_Validation         |
      |                            +---------------------------+
      |                                     |
      |   (LLM calls)                       |   (optimization engine calls)
      v                                     v
+---------------------------+      +---------------------------+
| LLM Providers             |      | Librex               |
|  - Claude / OpenAI / ... |      |  - optimize(...)         |
+---------------------------+      |  - adapters (QAP, etc.)  |
                                   +---------------------------+
```

- **ORCHEX Orchestrator**: LLM routing + budget enforcement + provider adapters.
- **ORCHEX Autonomous Research**: research protocol orchestrator, built on top of the Orchestrator and optimization engines.
- **Turingo**: multi‑paradigm optimization pipeline that can use Librex as its classical/engine layer.
- **Librex**: universal optimization framework with method registry and domain adapters.

Each layer can be evolved independently as long as it preserves its interfaces.

---

## 2. ORCHEX Orchestrator Control Plane

### 2.1 Role

ORCHEX Orchestrator is the **only place** that should talk to raw LLM providers.

Responsibilities:

- **Model routing**: choose `model_id` based on `task.task_type` and routing strategy (quality, balanced, fastest).
- **Fallback management**: build `fallback_chain` so failures don’t kill the request.
- **Cost control**: estimate cost, enforce per‑request and daily budgets, record actual usage.
- **Provider abstraction**: adapt Claude/OpenAI/Gemini/etc. behind a unified adapter interface.

### 2.2 Integration pattern

For higher‑level systems (ORCHEX Research, Turingo, other tools), define a small wrapper:

```python
# Pseudocode: LLM client wrapper

from atlas_orchestrator.core.types import Task, TaskType


async def llm_call(task_type: TaskType, prompt: str, **kwargs):
    """Unified entry point for all LLM calls in the tal-ai stack."""
    task = Task(task_type=task_type, prompt=prompt, **kwargs)
    return await orchestrator.execute(task)
```

Usage examples:

- Intent classification:

  ```python
  result = await llm_call(TaskType.RESEARCH_SUMMARY, prompt)
  ```

- Hypothesis generation:

  ```python
  result = await llm_call(TaskType.HYPOTHESIS_GENERATION, prompt)
  ```

**Rule**: other systems should not call providers directly; they must go through `llm_call` / Orchestrator.

### 2.3 Configuration

Central configuration lives in ORCHEX Orchestrator (code and/or config files):

- **Model catalog**: providers, model IDs, strengths per `TaskType`, cost functions.
- **Routing policies**: which strategy (balanced/quality/fastest) to use for each `TaskType`.
- **Budget policies**: per‑request and daily budgets; separate limits for "autonomous" vs "assistant" usage.

Extending this layer:

- Add new models by extending the catalog & adapters.
- Add new `TaskType` values and configure routing/budget rules per type.

---

## 3. ORCHEX Autonomous Research Control Plane

### 3.1 Role

ORCHEX Autonomous Research orchestrates a **full research pipeline**:

1. Adaptive workflow planning.
2. Hypothesis generation.
3. Aggressive validation (risk, refutation, interrogation).
4. Meta‑learning over personality agents.

It should treat ORCHEX Orchestrator as its **LLM substrate** and Turingo/Librex as **optimization substrates** when needed.

### 3.2 LLM integration

Where ORCHEX Research uses LLMs (examples):

- Intent classification.  
- Hypothesis generation.  
- Risk assessment queries.  
- Self‑refutation and interrogation prompts.

Pattern:

```python
# Example inside workflow_orchestrator / hypothesis_generator / protocols

from .llm_client import llm_call  # thin wrapper around ORCHEX Orchestrator


async def classify_intent(query: str, domain: str):
    prompt = build_intent_prompt(query, domain)
    result = await llm_call(TaskType.INTENT_CLASSIFICATION, prompt)
    return parse_intent_result(result)
```

Design guidelines:

- Keep **prompt construction** local to the component (e.g., `build_intent_prompt`).
- Use `TaskType` enums to describe the call semantics (classification, refutation, etc.).
- Avoid hard‑coding provider names or model IDs; those are Orchestrator concerns.

### 3.3 Optimization integration

When hypotheses involve optimization methods (especially QAP or related problems), ORCHEX Research can:

- Call **Turingo** for multi‑paradigm evaluation:

  ```python
  # Pseudocode inside a validation stage
  result = await turingo_client.solve_qap_benchmarks(hypothesis_config)
  # result includes per-instance objectives, validation scores, SOTA improvements
  ```

- Or call **Librex** directly for simple method evaluations:

  ```python
  from Librex import optimize_qap

  opt_result = optimize_qap(flow, distance, method="simulated_annealing", config=sa_config)
  ```

The outputs (objective, validity, SOTA improvement, validation scores) become **evidence** in the research protocol’s scoring.

### 3.4 Meta‑learning agents

ORCHEX Research’s meta‑learning system (UCB1 bandit) chooses personality agents for tasks like risk assessment and refutation.

Integration points:

- Keep **agent selection** and **reward updates** in the meta‑learning module.  
- Inside the protocol, treat agents as pluggable LLM personalities:

  ```python
  risk_agent = meta_learning.select_agent("risk_assessment", domain)
  result = await risk_agent.assess_risk(hypothesis, llm_call=llm_call)
  meta_learning.record_agent_action(risk_agent, result.score)
  ```

This keeps Orchestrator usage consistent while allowing new agent personas to be added without touching the core protocol.

---

## 4. Turingo Control Plane

### 4.1 Role

Turingo orchestrates **multi‑paradigm optimization**:

1. Problem analysis (PuzzleProdigy).
2. Ethics validation (EthicsEnforcer).
3. Architecture design (BlueprintBoss).
4. Parallel solution generation (SOP_Stampede: quantum/ML/classical).
5. Validation rodeo (SOP_Validation: benchmarks, novelty, adversarial checks).
6. Metrics and SOTA improvement.

It sits **on top of** optimization engines such as Librex.

### 4.2 Engine integration

For each paradigm:

- **Quantum / ML**: call their respective engines or APIs behind adapter classes.  
- **Classical**: use Librex for supported problem types (e.g., QAP).

Example classical integration:

```python
# Inside Gamma Gang / classical path

from Librex import optimize_qap

async def _solve_classical_qap(flow, distance, config):
    return optimize_qap(
        flow,
        distance,
        method=config.get("method", "simulated_annealing"),
        config=config.get("method_config", {}),
    )
```

### 4.3 Configuration

Turingo config should define:

- Supported `problem_type` values (qap, tsp, etc.).
- Enabled paradigms for each problem type.  
- Baseline SOTA objectives per benchmark instance.  
- Which Librex methods are valid per problem type.

This lets ORCHEX Research (or other callers) ask Turingo for **capability reports**, e.g.:

```python
capabilities = turingo_client.describe_capabilities()
```

---

## 5. Librex Control Plane

### 5.1 Role

Librex provides a **universal optimization interface** with:

- Method registry (simulated annealing, evolutionary methods, etc.).
- Domain adapters (QAP, and others as they are added).

### 5.2 Interfaces

Universal API:

```python
from Librex import optimize

result = optimize(problem, adapter, method="simulated_annealing", config={...})
```

QAP convenience API:

```python
from Librex import optimize_qap

result = optimize_qap(
    flow_matrix,
    distance_matrix,
    method="simulated_annealing",
    config={"max_iters": 1000},
)

print(result["solution"], result["objective"], result["is_valid"])
```

Adapter protocol (conceptual):

```python
class QAPAdapter:
    def validate(self, problem: dict) -> None: ...
    def to_internal(self, problem: dict) -> InternalProblem: ...
    def from_internal(self, solution: InternalSolution) -> dict: ...
```

### 5.3 Telemetry

Each optimization run should return, or be able to emit, metadata such as:

- Objective value.
- Validity flag.
- Iteration count / convergence status.
- Runtime statistics.

This makes downstream consumers (Turingo, ORCHEX Research) more robust and allows better benchmarking.

---

## 6. Cross‑Cutting Concerns

### 6.1 Observability

Recommended minimal telemetry at each layer:

- **ORCHEX Orchestrator**:
  - `task_type`, selected `model_id`, routing strategy, estimated vs actual cost, fallback usage.

- **ORCHEX Research**:
  - Topic, domain, workflow configuration.  
  - Per‑hypothesis scores (risk, refutation, interrogation).  
  - Which agents were selected and their rewards.

- **Turingo**:
  - Problem type, instance, chosen architecture, per‑paradigm objectives.  
  - Validation outcomes (benchmark, novelty, adversarial scores).  
  - SOTA improvement.

- **Librex**:
  - Method name, adapter, problem size, objective, is_valid, iterations.

Log formats can be JSON to make downstream analytics easier.

### 6.2 Safety & governance

- Use Orchestrator’s budgets to prevent runaway LLM spending, especially in **autonomous** modes.  
- Use Turingo’s EthicsEnforcer for optimization problems with possible dual-use risk.  
- Use ORCHEX Research’s risk/refutation/interrogation stages to avoid accepting poorly supported hypotheses.

---

## 7. Extensibility Guidelines

To extend the system cleanly:

1. **New LLM provider or model**:
   - Add to ORCHEX Orchestrator’s model catalog & adapters.
   - Configure strengths, costs, and routing strategy.

2. **New research pipeline variant**:
   - Add stages to ORCHEX Research’s workflow orchestrator.  
   - Use `llm_call` and optimization clients rather than direct provider calls.

3. **New optimization method**:
   - Add to Librex’s method registry.  
   - Ensure it works with existing adapters (or add new adapters).

4. **New problem type (optimization)**:
   - Add adapters & methods to Librex.  
   - Extend Turingo’s problem analysis, architecture design, and validation to support it.

5. **New personality agent**:
   - Register in ORCHEX Research’s meta‑learning agent catalog.  
   - Ensure it uses the shared `llm_call` interface internally.

By keeping each layer’s public interfaces stable (Orchestrator, Research, Turingo, Librex), you can safely evolve internals without breaking consumers.
