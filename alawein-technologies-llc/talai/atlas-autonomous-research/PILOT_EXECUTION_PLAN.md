# ORCHEX + UARO Private Pilot: Deep Test Suite

**2-Week Execution Plan for Universal Reasoning System**

**Date**: 2025-11-06
**Status**: Ready to Execute
**Foundation**: UARO v1.0 (63 tests, 100% pass rate)

---

## Executive Summary

This pilot tests UARO's universal reasoning capabilities across 5 diverse tracks:

1. **Track A: QAP** - Combinatorial optimization (Quadratic Assignment Problem)
2. **Track B: Scientific Refutation** - Conjecture testing and counterexample finding
3. **Track C: Auto-Notebook** - Reproducible research workflows
4. **Track D: Novel CLI** - Automated codebase analysis
5. **Track E: Breakthrough Ideation** - Creative problem solving with proof packs

**Goal**: Demonstrate problem-agnostic reasoning with complete explainability, marketplace viability, and enterprise-ready features.

---

## Foundation: What We Have (UARO v1.0)

### ✅ Existing Capabilities

| Component | Status | Lines | Coverage |
|-----------|--------|-------|----------|
| 12 Universal Primitives | ✅ Complete | 1,100 | 12/12 tests pass |
| Meta-Algorithm Solver | ✅ Complete | 500 | 13/13 tests pass |
| Explainability Engine | ✅ Complete | 650 | 17/17 tests pass |
| Primitive Marketplace | ✅ Complete | 700 | 21/21 tests pass |
| ORCHEX Integration | ✅ Complete | 600 | - |
| **Total** | **100% pass rate** | **3,550** | **63/63 tests** |

### Primitives Available

**Decomposition**: DivideAndConquer, HierarchicalDecomposition
**Search**: BFS, DFS, BestFirst, BeamSearch
**Constraints**: ConstraintPropagation, BacktrackingSearch
**Logic**: ForwardChaining, BackwardChaining
**Optimization**: LocalSearch, SimulatedAnnealing

### Proven Working Example

✅ **Robot Path Planning**: 8x8 grid solved in 2 iterations using BFS

---

## Pilot Architecture: 5 Tracks × 2 Weeks

```
Week 1: Foundation + Track A + Track D
├── Days 1-2: Setup, metrics, QAP formulation
├── Days 3-4: QAP runs, CLI v0.1 prototype
└── Days 5-7: Ablations, proof packs, initial results

Week 2: Tracks B, C, E + Integration
├── Days 8-9: Scientific refutation, auto-notebook
├── Days 10-11: Breakthrough ideation, marketplace features
├── Days 12-13: Integration testing, calibration
└── Day 14: Live demo, proof pack publishing, results
```

---

## Track A: QAP (Quadratic Assignment Problem)

### Problem Statement

Assign N facilities to N locations minimizing flow × distance. NP-hard, real-world benchmark.

### Why QAP?

- **Standard benchmark**: Publicly available instances (QAPLIB)
- **Proven baseline**: Robust Tabu Search achieves known bounds
- **Clear metrics**: Optimality gap = (your_solution - best_known) / best_known
- **Meta-scheduler test**: Perfect for testing primitive switching

### UARO Mapping

```python
class QAPProblem(Problem):
    """
    State: Assignment of facilities to locations
    Actions: Swap two facility assignments
    Cost: Sum of flow[i][j] * distance[assign[i]][assign[j]]
    Goal: Minimize total cost
    """

    def initial_state(self):
        return random_permutation(n)

    def goal_test(self, state):
        return False  # Optimization problem, no explicit goal

    def actions(self, state):
        # All pairwise swaps
        return [(i, j) for i in range(n) for j in range(i+1, n)]

    def result(self, state, action):
        i, j = action
        new_state = state.copy()
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def cost(self, state, action):
        return self.evaluate(self.result(state, action))
```

### Primitives to Use

1. **LocalSearch**: 2-opt neighborhood (swap pairs)
2. **SimulatedAnnealing**: Accept worse moves with probability
3. **BeamSearch**: Keep top-k candidates
4. **DivideAndConquer**: Partition into subproblems

### Meta-Scheduler Strategy

```
IF plateau detected (no improvement in N steps):
    → Switch from LocalSearch to SimulatedAnnealing

IF oscillation detected (bouncing between similar states):
    → Increase tabu tenure or add diversification

IF high variance in recent moves:
    → Switch to BeamSearch to explore multiple paths

IF budget < 20% remaining:
    → Switch to greedy hill climbing for fast finish
```

### Deliverables

**Day 1-2**: QAP problem formulation, 10 QAPLIB instances loaded
**Day 3-4**: Meta-scheduler with plateau/oscillation detectors
**Day 5-6**: Full runs on all 10 instances with ablations
**Day 7**: Proof packs generated for all runs

### Success Criteria

- ✅ **Median gap ≤ 2%** vs robust Tabu Search baseline
- ✅ **≥15% lift** from meta-scheduler over single primitive
- ✅ **Complete proof docs** showing primitive switches and rationales
- ✅ **Calibrated confidence**: Predicted quality matches actual

### Metrics

```python
{
    "optimality_gap": (solution_cost - best_known) / best_known,
    "time_to_first_improvement": seconds,
    "primitive_switches": count,
    "plateau_detections": count,
    "final_confidence": 0.0-1.0,
    "calibration_error": |predicted_gap - actual_gap|
}
```

---

## Track B: Scientific Refutation

### Problem Statement

Given a conjecture or paper claim, find counterexamples or prove limitations.

### Example Target

**Collatz Conjecture** (simplified): "For n < 10,000, the sequence always reaches 1"

**Goldbach's Conjecture** (bounded): "Every even number 4-1000 is sum of two primes"

**Or**: Pick a recent arXiv preprint with testable claims

### UARO Mapping

```python
class ConjectureProblem(Problem):
    """
    State: Candidate counterexample
    Actions: Modify candidate parameters
    Goal: Find input that violates conjecture
    """

    def initial_state(self):
        return random_candidate()

    def goal_test(self, state):
        return violates_conjecture(state) and verify(state)

    def actions(self, state):
        # Perturbations guided by property structure
        return generate_perturbations(state)
```

### Primitives to Use

1. **BackwardChaining**: Start from negation of conjecture, find premises
2. **ForwardChaining**: Apply known theorems to derive contradictions
3. **BestFirstSearch**: Search parameter space for counterexamples
4. **BeamSearch**: Keep multiple candidate counterexamples

### Meta-Scheduler Strategy

```
IF proof tactics yielding new lemmas:
    → Stay with ForwardChaining/BackwardChaining

IF search finding promising candidates:
    → Switch to BestFirstSearch with heuristic = distance from violation

IF stuck in proof:
    → Try RandomWalk to generate test cases

IF found candidate counterexample:
    → Switch to verification mode (exhaustive checking)
```

### Deliverables

**Day 8**: Conjecture selected, formulated as UARO problem
**Day 9**: Proof tactics implemented, search strategies tested
**Day 10**: Full run with self-refutation transcript

### Success Criteria

- ✅ **Machine-checkable fragments** where feasible
- ✅ **Minimal counterexample** if found, or proof sketch if not
- ✅ **Self-refutation transcript**: "Tried X, failed because Y"
- ✅ **Verified steps**: Every claim backed by test/proof

### Example Output

```markdown
# Goldbach Conjecture (n ≤ 1000) - Refutation Attempt

## Result: VERIFIED (no counterexample found)

## Search Strategy
1. BackwardChaining: Assume even n has no prime pair sum
2. ForwardChaining: Enumerate all primes < n, check pairs
3. BestFirstSearch: Test edge cases (n = 4, 6, 8, ...)

## Proof Trace
- Step 1: Tested n=4: 2+2=4 ✓
- Step 2: Tested n=6: 3+3=6 ✓
- Step 3: Tested n=8: 3+5=8 ✓
...
- Step 499: Tested n=1000: 3+997=1000 ✓

## Confidence: 99.9% (exhaustive search completed)
## Limitations: Only tested n ≤ 1000, not general proof
```

---

## Track C: Auto-Notebook

### Problem Statement

Jupyter notebooks that are fully reproducible with cell contracts and one-click proof pack export.

### Why This Matters

- **Reproducibility crisis**: Most notebooks fail to rerun
- **Black box**: Can't tell what each cell does
- **No provenance**: Where did this data come from?

### UARO Integration

```python
class NotebookCell:
    """
    Each cell is a mini-problem with:
    - Preconditions (input contracts)
    - Postconditions (output contracts)
    - Reasoning trace (what this cell does and why)
    """

    def __init__(self, code, inputs, outputs):
        self.code = code
        self.inputs = inputs  # {"df": "pandas.DataFrame"}
        self.outputs = outputs  # {"result": "float"}
        self.trace = []

    def verify_contracts(self):
        # Check input types match expectations
        # Check output types match specifications
        # Record in trace
        pass

    def to_proof_pack(self):
        return {
            "cell_id": self.id,
            "code": self.code,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "trace": self.trace,
            "timestamp": now(),
            "environment": get_env_snapshot()
        }
```

### Features

1. **Cell Contracts**: Explicit input/output types and invariants
2. **Deterministic Reruns**: Pin versions, seeds, data snapshots
3. **Proof Pack Export**: HTML/PDF with complete provenance
4. **One-Click Reproduce**: Others can rerun with same results

### Deliverables

**Day 10**: Auto-notebook extension prototype
**Day 11**: Example notebook with contracts and proof pack
**Day 12**: Reproducibility test (rerun on fresh environment)

### Success Criteria

- ✅ **Deterministic reruns**: Same inputs → same outputs
- ✅ **Contract validation**: All cells pass input/output checks
- ✅ **Proof pack export**: HTML with embedded run signature
- ✅ **No manual steps**: Fully automated from start to finish

---

## Track D: Novel CLI

### Problem Statement

Point CLI at any Git repo → automated analysis plan with hotspots, test stubs, and proof pack.

### Command Interface

```bash
# Install
pip install uaro-cli

# Analyze any repo
uaro analyze https://github.com/user/repo

# Output:
# - hotspots.md: Complex functions that need attention
# - plan.md: Suggested improvements and test coverage
# - test_stubs.py: Generated test templates
# - proof_pack.html: Complete analysis with reasoning trace
```

### UARO Mapping

```python
class CodebaseAnalysisProblem(Problem):
    """
    State: Current understanding of codebase
    Actions: Analyze file, trace dependencies, identify patterns
    Goal: Complete analysis plan with test coverage
    """

    def initial_state(self):
        return {"files": [], "dependencies": {}, "hotspots": []}

    def goal_test(self, state):
        return state["coverage"] > 0.8 and state["hotspots_analyzed"]

    def actions(self, state):
        return [
            "analyze_next_file",
            "trace_dependencies",
            "identify_hotspots",
            "generate_tests"
        ]
```

### Analysis Pipeline

```
1. Clone repo → AST parse
2. Build dependency graph → Identify hotspots
3. Complexity analysis → Rank by risk
4. Generate test stubs → Coverage gaps
5. Create analysis plan → Proof pack
```

### Primitives to Use

1. **HierarchicalDecomposition**: Break codebase into modules
2. **DFS**: Traverse dependency graph
3. **ConstraintPropagation**: Infer types and contracts
4. **ForwardChaining**: Apply code quality rules

### Deliverables

**Day 3-4**: CLI v0.1 prototype
**Day 5**: Tested on 3 public repos
**Day 7**: Published analysis plans to marketplace

### Success Criteria

- ✅ **Valid proof pack**: CLI generates complete analysis
- ✅ **Actionable insights**: Hotspots are real complexity issues
- ✅ **Test stubs work**: Generated tests are runnable
- ✅ **Marketplace ready**: Published and discoverable

### Example Output

```markdown
# Codebase Analysis: flask (v2.3.0)

## Hotspots (by cyclomatic complexity)
1. `flask/app.py:Flask.dispatch_request()` - CC: 42
2. `flask/blueprints.py:Blueprint.route()` - CC: 28
3. `flask/helpers.py:send_file()` - CC: 24

## Test Coverage Gaps
- `flask/app.py`: 67% coverage (missing error paths)
- `flask/config.py`: 82% coverage (missing edge cases)

## Suggested Improvements
1. Refactor `Flask.dispatch_request()` - extract 3 functions
2. Add error handling tests for edge cases
3. Document complex algorithms with proof traces

## Generated Test Stubs
See `test_stubs.py` for 15 test templates

## Confidence: 85%
## Analysis Time: 47 seconds
```

---

## Track E: Breakthrough Ideation

### Problem Statement

Creative problem solving with structured brainstorming and proof packs for novel ideas.

### ORCHEX Brainstorming Integration

Leverage ORCHEX's existing brainstorming engine (8 strategies):

1. **SCAMPER**: Substitute, Combine, Adapt, Modify, Put to other uses, Eliminate, Reverse
2. **Mind Mapping**: Visual idea organization
3. **Six Thinking Hats**: Multiple perspectives
4. **Random Word Association**: Forced connections
5. **Analogical Thinking**: Transfer from other domains
6. **Constraint Relaxation**: Remove assumptions
7. **Reverse Brainstorming**: Solve opposite problem
8. **Morphological Analysis**: Systematic combinations

### UARO Enhancement

Add **Proof Packs** for each idea:

```python
class IdeaProofPack:
    """
    Every generated idea comes with:
    - Reasoning trace (how we got here)
    - Feasibility analysis (can this work?)
    - Risk register (what could go wrong?)
    - Validation plan (how to test it?)
    - Prior art search (has this been done?)
    """

    def __init__(self, idea):
        self.idea = idea
        self.reasoning_trace = []
        self.feasibility = None
        self.risks = []
        self.validation_plan = []
        self.prior_art = []

    def generate(self):
        # Use UARO solver to validate idea
        result = solve_with_uaro(IdeaValidationProblem(self.idea))

        # Extract reasoning trace
        self.reasoning_trace = result.reasoning_trace

        # Generate proof pack
        return explain_solution(result, format="html")
```

### Deliverables

**Day 11**: Integrate ORCHEX brainstorming with UARO proof packs
**Day 12**: Generate 10 breakthrough ideas with full proof packs
**Day 13**: Validation testing, feasibility scoring

### Success Criteria

- ✅ **10 novel ideas** generated across different strategies
- ✅ **Complete proof packs** for each idea
- ✅ **Feasibility scores**: Calibrated predictions
- ✅ **Prior art check**: Automated search results

---

## Enhanced Features (Weeks 1-2)

### 1. Reasoning Marketplace v2

**New Features**:
- **Reasoning Insurance**: Pay premium, get payout if wrong
- **Red-Team as a Service**: Rent "Destroyer" agents
- **Challenge Arena**: Live leaderboard, public replays
- **Proof-pack sharing**: Deep-link "Replicate" button

**Implementation**:

```python
class ReasoningInsurance:
    """
    Insure your solution against being wrong

    Premium = risk_score * coverage_amount
    Payout = coverage_amount if solution fails verification
    """

    def calculate_premium(self, solution_result):
        # Based on calibrated confidence
        risk = 1.0 - solution_result.confidence
        premium = risk * self.coverage_amount * 0.1  # 10% loading
        return premium

    def verify_and_payout(self, solution_result, test_cases):
        # Run verification tests
        passed = all(test(solution_result.solution) for test in test_cases)

        if not passed:
            return self.coverage_amount  # Payout
        return 0  # No payout, solution was correct
```

### 2. Calibration Engine

**Problem**: Confidence scores are often miscalibrated ("90% confident but wrong 50% of the time")

**Solution**: Track historical accuracy by confidence band

```python
class CalibrationEngine:
    """
    Tracks: "When I said 90% confident, was I right 90% of the time?"
    """

    def __init__(self):
        self.history = []  # (predicted_confidence, actual_outcome)

    def record(self, predicted_confidence, actual_outcome):
        self.history.append((predicted_confidence, actual_outcome))

    def calibration_error(self):
        # Expected Calibration Error (ECE)
        bins = np.linspace(0, 1, 11)  # 10 bins
        errors = []

        for i in range(len(bins)-1):
            bin_predictions = [
                actual for pred, actual in self.history
                if bins[i] <= pred < bins[i+1]
            ]
            if bin_predictions:
                bin_accuracy = np.mean(bin_predictions)
                bin_confidence = (bins[i] + bins[i+1]) / 2
                errors.append(abs(bin_accuracy - bin_confidence))

        return np.mean(errors)

    def recalibrate(self, raw_confidence):
        # Platt scaling or isotonic regression
        return self.calibration_model.predict(raw_confidence)
```

### 3. Signal Detection & Auto-Correct Loop

**Signals**:

```python
class SignalDetector:
    """
    Monitors solver progress and triggers interventions
    """

    def detect_plateau(self, history, window=10):
        """No improvement in last N steps"""
        recent = history[-window:]
        return max(recent) == min(recent)

    def detect_oscillation(self, history, window=10):
        """Bouncing between similar states"""
        recent = history[-window:]
        return len(set(recent)) < len(recent) / 2

    def detect_overconfidence(self, predicted, actual):
        """Predicted confidence >> actual accuracy"""
        return predicted - actual > 0.2

    def detect_budget_pressure(self, elapsed, budget):
        """Running out of time"""
        return elapsed / budget > 0.8
```

**Actions**:

```python
class AutoCorrectLoop:
    """
    Responds to detected signals with interventions
    """

    def on_plateau(self, solver_state):
        # Diversify: perturb solution, widen beam
        return {
            "action": "perturb",
            "magnitude": 0.1,
            "primitive": "SimulatedAnnealing"
        }

    def on_oscillation(self, solver_state):
        # Add tabu or symmetry-breaking
        return {
            "action": "add_constraint",
            "type": "tabu",
            "tenure": 10
        }

    def on_overconfidence(self, solver_state):
        # Down-weight aggressive moves
        return {
            "action": "adjust_weights",
            "aggressive": 0.5,
            "conservative": 1.5
        }

    def on_budget_pressure(self, solver_state):
        # Switch to greedy finalize
        return {
            "action": "switch_primitive",
            "primitive": "LocalSearch",
            "mode": "greedy"
        }
```

---

## 14-Day Execution Timeline

### Week 1: Foundation + QAP + CLI

| Day | Track | Activities | Deliverables | Owner |
|-----|-------|-----------|--------------|-------|
| **1** | Setup | Pre-register metrics, freeze seeds, load QAPLIB | Metrics doc, 10 QAP instances | Lead |
| **2** | A+D | QAP formulation, CLI prototype, repo structure | QAPProblem class, CLI scaffold | Dev1+Dev2 |
| **3** | A | Meta-scheduler with signal detection | Plateau/oscillation detectors | Dev1 |
| **4** | A+D | QAP runs on 5 instances, CLI tested on 2 repos | Initial results, CLI v0.1 | Dev1+Dev2 |
| **5** | A | Ablation studies, primitive comparison | Performance comparison | Dev1 |
| **6** | A+D | Full QAP runs, CLI on 3rd repo | Complete QAP results, CLI demo | Dev1+Dev2 |
| **7** | A+D | Proof packs generation, marketplace publish | 10 QAP proof packs, 1 CLI pack | All |

### Week 2: Science + Notebook + Ideas + Integration

| Day | Track | Activities | Deliverables | Owner |
|-----|-------|-----------|--------------|-------|
| **8** | B | Select conjecture, formulate problem | Conjecture problem class | Dev1 |
| **9** | B+C | Proof tactics, auto-notebook prototype | Refutation transcript, notebook extension | Dev1+Dev2 |
| **10** | C+E | Notebook with contracts, idea generation | Example notebook, 5 ideas | Dev2+Dev3 |
| **11** | E | ORCHEX brainstorming + UARO proof packs | 10 ideas with proof packs | Dev3 |
| **12** | All | Calibration engine, insurance module | Calibration plots, insurance quotes | Dev1 |
| **13** | All | Integration testing, replication scripts | End-to-end tests | All |
| **14** | All | Live challenge, demo, publish proof packs | 15+ proof packs, demo video | All |

---

## Success Metrics (Pre-Registered)

### Track A: QAP

```python
success_criteria = {
    "median_gap": lambda x: x <= 0.02,  # ≤2% gap
    "meta_lift": lambda x: x >= 0.15,   # ≥15% improvement
    "proof_quality": lambda x: x >= 0.9, # Complete traces
    "calibration": lambda x: x <= 0.1    # ≤10% error
}
```

### Track B: Scientific Refutation

```python
success_criteria = {
    "machine_checkable": lambda x: x > 0.5,  # >50% checkable
    "minimal_counterexample": lambda x: x if found else "proof sketch",
    "transcript_complete": lambda x: x == True,
    "verified_steps": lambda x: x == 1.0  # All steps verified
}
```

### Track C: Auto-Notebook

```python
success_criteria = {
    "deterministic": lambda x: x == 1.0,  # 100% reproducible
    "contracts_pass": lambda x: x >= 0.95, # ≥95% cells pass
    "proof_pack": lambda x: x == "generated",
    "manual_steps": lambda x: x == 0  # Zero manual steps
}
```

### Track D: Novel CLI

```python
success_criteria = {
    "valid_pack": lambda x: x == True,
    "actionable_insights": lambda x: x >= 0.8,  # ≥80% useful
    "runnable_tests": lambda x: x >= 0.9,  # ≥90% work
    "marketplace_ready": lambda x: x == True
}
```

### Track E: Breakthrough Ideation

```python
success_criteria = {
    "novel_ideas": lambda x: x >= 10,
    "proof_packs": lambda x: x == x["novel_ideas"],  # All have packs
    "feasibility_calibrated": lambda x: x <= 0.15,  # ≤15% error
    "prior_art_complete": lambda x: x >= 0.9  # ≥90% checked
}
```

---

## Risk Register & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Benchmark overfitting** | Medium | High | Pre-register metrics, freeze seeds, hidden test set |
| **"Magic box" skepticism** | High | Medium | Lead every demo with replay and counterfactuals |
| **Marketplace spam** | Medium | Medium | Verification tests required, documentation minimum |
| **Calibration drift** | Medium | High | Continuous monitoring, monthly recalibration |
| **Insurance fraud** | Low | High | Multi-party verification, attestation required |
| **Timeline slippage** | Medium | High | Daily standups, aggressive scope cuts if needed |

---

## Monetization Strategy

### Tier 1: Free (Growth)
- 100 solver runs/month
- Basic proof packs (Markdown)
- Community marketplace access
- Public challenge participation

### Tier 2: Pro ($49/month)
- Unlimited solver runs
- All proof formats (HTML, LaTeX, PDF)
- Priority support
- Custom primitive publishing

### Tier 3: Team ($199/month)
- Everything in Pro
- Shared workspace
- Audit logs (90 days)
- Red-team credits (10/month)

### Tier 4: Enterprise (Custom)
- Everything in Team
- Unlimited seats
- SLA guarantees
- Dedicated support
- Custom integrations
- Audit logs (forever)
- Insurance SKU

### Add-Ons
- **Insurance**: $0.10 - $10 per solution (risk-based pricing)
- **Red-team**: $5 per attack attempt
- **Marketplace rake**: 15% of all transactions
- **Featured listings**: $50/month
- **Challenge sponsorship**: Custom pricing

---

## Go-to-Market Stunts (Launch Week)

### Day 14: Launch Events

**1. Research Battle Royale** (Live Stream)
- 3 problems, 3 teams, 30 minutes each
- Viewers vote on most interesting approach
- Winner gets $1,000 bounty

**2. Proof-Bounty Drop**
- 10 hard instances, $100 each for best solution
- On-chain attestation
- 48-hour deadline

**3. Replication Swarm**
- Community runs our proof packs
- Verification badges for successful replications
- Prize for finding discrepancy

**4. Hall of Failures Launch**
- Curate 20 interesting failures
- Document what went wrong
- Turn failures into learning data

**5. Office Hours with the Trace**
- Founders walk through live proof doc
- Take questions on reasoning decisions
- Show counterfactual replays

---

## Key Artifacts (End of Week 2)

1. **QAP Proof Packs** (10): Complete optimization runs with reasoning traces
2. **Conjecture Report** (1): Scientific refutation or verification attempt
3. **Auto-Notebook** (1): Fully reproducible notebook with contracts
4. **CLI Analysis** (3): Codebase analysis plans for public repos
5. **Breakthrough Ideas** (10): Novel ideas with feasibility proof packs
6. **Calibration Report** (1): Historical accuracy vs confidence
7. **Insurance Quotes** (5): Sample risk assessments
8. **Demo Video** (1): 10-minute highlight reel
9. **Launch Blog** (1): Complete pilot results write-up
10. **Marketplace Listings** (15+): Published proof packs and primitives

---

## Resource Requirements

### Team
- **Dev1**: QAP, scientific refutation, calibration (senior)
- **Dev2**: CLI, auto-notebook (mid)
- **Dev3**: Breakthrough ideation, marketplace features (mid)
- **Lead**: Metrics, demos, go-to-market (architect)

### Infrastructure
- **Compute**: 100 CPU hours for QAP runs
- **Storage**: 10GB for proof packs, replays
- **CI/CD**: GitHub Actions for auto-testing
- **Monitoring**: DataDog or similar for live metrics

### Budget
- **Bounties**: $5,000 (battle royale + proof bounties)
- **Infrastructure**: $500 (cloud compute)
- **Marketing**: $1,000 (social ads, influencer posts)
- **Contingency**: $1,500
- **Total**: $8,000

---

## Next Steps (Day 0 - Tomorrow)

### Immediate Actions

1. **Freeze scope**: Confirm 5 tracks, no additions
2. **Pre-register metrics**: Commit to GitHub with timestamp
3. **Load datasets**: QAPLIB instances, target repos
4. **Assign owners**: Dev1/2/3 + Lead confirmed
5. **Kickoff meeting**: 1 hour, align on timeline

### Pre-Flight Checklist

- [ ] QAPLIB instances downloaded (10 files)
- [ ] Baseline results documented (Tabu Search)
- [ ] Target conjecture selected (Goldbach or Collatz)
- [ ] 3 public repos identified for CLI testing
- [ ] ORCHEX brainstorming engine reviewed
- [ ] Marketplace infrastructure tested
- [ ] Insurance pricing model designed
- [ ] Calibration engine scaffolded
- [ ] Demo environment provisioned
- [ ] Git repo created with tracked issues

---

## Open Questions for User

1. **Target buyer personas**: Who are the two main buyers? (e.g., Ops Research Lead vs Data Science Manager)
2. **Benchmark suite**: QAPLIB confirmed, or prefer different benchmark?
3. **Public repo for CLI**: Suggest 3 repos for analysis (e.g., Flask, FastAPI, Transformers?)
4. **Insurance in pilot**: Include insurance module? (Yes/No)
5. **Challenge prize budget**: $5K confirmed or adjust?
6. **Launch partner**: Any early adopters lined up for pilot testing?

---

**Status**: Ready to begin Day 1
**Next Review**: End of Week 1 (Day 7)
**Final Demo**: Day 14 (Live stream + proof pack publishing)

Let's ship this! 🚀
