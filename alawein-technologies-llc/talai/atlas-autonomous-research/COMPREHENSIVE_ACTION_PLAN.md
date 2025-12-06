# ORCHEX + UARO: Comprehensive Action Plan (Option D)

**For**: Solo Founder, Private Testing Phase
**Budget**: <$5K/month for 3 months
**Timeline**: 14-day sprint + 3-month roadmap
**Goal**: Validate on 7 hard problems + build 5 killer features + create viral loops

---

## 🚀 START HERE (Do This First)

### Day 0 Pre-Flight (2 hours)

**Right now, in order:**

```bash
# 1. Clone and verify foundation (5 min)
cd 02-PROJECTS/ORCHEX-autonomous-research
python -m pytest tests/ -v  # Should see 63/63 pass

# 2. Set up tracking (5 min)
mkdir -p pilot/{qap,theorem,paper,notebook,cli,ideas,meta}
mkdir -p pilot/results
git checkout -b pilot/sprint-1

# 3. Pre-register metrics (10 min)
cat > pilot/METRICS.md << 'EOF'
# Pre-Registered Metrics (2025-11-06)

## QAP Track
- median_gap_target: 0.02  # ≤2% vs baseline
- meta_lift_target: 0.15   # ≥15% improvement
- runs_required: 10        # QAPLIB instances

## Theorem Track
- novel_conjectures: 3     # Generate 3 new
- proof_attempts: 3        # Attempt proofs
- machine_checkable: 0.5   # ≥50% verifiable

## Paper Track
- critiques_generated: 3   # 3 paper analyses
- flaws_found: ≥1         # Find at least 1 flaw
- counterexamples: ≥1     # Find at least 1 counterexample

## Notebook Track
- deterministic_reruns: 1.0  # 100% reproducible
- cells_with_contracts: 0.95 # ≥95% have contracts

## CLI Track
- repos_analyzed: 3
- actionable_insights: 0.8  # ≥80% useful
- test_coverage: 0.9        # ≥90% tests run

## Ideas Track
- breakthrough_ideas: 10
- feasibility_scored: 10
- prior_art_checked: 0.9

## Meta Track
- calibration_error: <0.1
- self_improvements: ≥5
EOF

# 4. Download benchmarks (15 min)
cd pilot/qap
wget http://anjos.mgi.polymtl.ca/qaplib/inst.html
# Download first 10 instances (tai12a through tai20a)

# 5. Create work tracker (5 min)
cat > pilot/TODO.md << 'EOF'
# Sprint 1 TODO

## Week 1: QAP + CLI + Foundation
- [ ] Day 1: QAP formulation + signal detectors
- [ ] Day 2: Meta-scheduler implementation
- [ ] Day 3: QAP runs (5 instances)
- [ ] Day 4: CLI prototype
- [ ] Day 5: QAP runs (5 more) + CLI test
- [ ] Day 6: Ablation studies
- [ ] Day 7: Proof packs + review

## Week 2: Theorem + Paper + Notebook + Ideas
- [ ] Day 8: Theorem generation
- [ ] Day 9: Paper critique + notebook
- [ ] Day 10: Paper disproval attempts
- [ ] Day 11: Ideas + meta improvements
- [ ] Day 12: Integration testing
- [ ] Day 13: Calibration + refinement
- [ ] Day 14: Results + demo prep
EOF

# 6. Set budget tracking (5 min)
cat > pilot/BUDGET.md << 'EOF'
# Budget Tracker (Month 1)

## Allocated: $5,000
- Compute (AWS/GCP): $2,000
- APIs (OpenAI/Anthropic): $1,500
- Tools/Services: $500
- Bounties/Testing: $500
- Contingency: $500

## Spent: $0
EOF

# 7. Ready to start!
echo "✅ Pre-flight complete. Start Day 1 now."
```

**What you just did**:
- Verified UARO foundation works (63 tests passing)
- Pre-registered metrics (prevents p-hacking)
- Downloaded benchmarks
- Set up tracking
- Defined budget

**Next**: Start Day 1 (see below)

---

## 📅 14-Day Sprint: Day-by-Day Execution

### **Day 1: QAP Formulation + Signal Detectors**

**Time**: 6-8 hours
**Goal**: QAP problem class + plateau/oscillation detectors working

**Morning (3 hours): QAP Problem Class**

```python
# pilot/qap/qap_problem.py

import numpy as np
from typing import List, Tuple
from uaro import Problem

class QAPProblem(Problem):
    """
    Quadratic Assignment Problem

    Given n facilities and n locations:
    - Flow matrix F[i][j] = flow between facility i and j
    - Distance matrix D[i][j] = distance between location i and j

    Goal: Find assignment π minimizing:
    cost(π) = Σ F[i][j] × D[π[i]][π[j]]

    Example:
        n=4 facilities, 4 locations
        F = [[0,5,2,4], [5,0,3,0], [2,3,0,0], [4,0,0,0]]
        D = [[0,8,15,1], [8,0,13,6], [15,13,0,12], [1,6,12,0]]

        Optimal: [1,2,3,0] with cost = ...
    """

    def __init__(self, flow: np.ndarray, distance: np.ndarray, name: str = "qap"):
        """
        Args:
            flow: n×n flow matrix
            distance: n×n distance matrix
            name: Instance name (e.g., "tai12a")
        """
        assert flow.shape == distance.shape
        assert flow.shape[0] == flow.shape[1]

        self.n = flow.shape[0]
        self.flow = flow
        self.distance = distance
        self.name = name

        # Best known solution (if available)
        self.best_known = None
        self.best_known_cost = None

    def initial_state(self) -> np.ndarray:
        """Random initial assignment"""
        return np.random.permutation(self.n)

    def goal_test(self, state: np.ndarray) -> bool:
        """No explicit goal for optimization problems"""
        return False

    def actions(self, state: np.ndarray) -> List[Tuple[int, int]]:
        """All pairwise swaps"""
        actions = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                actions.append((i, j))
        return actions

    def result(self, state: np.ndarray, action: Tuple[int, int]) -> np.ndarray:
        """Swap two facilities"""
        i, j = action
        new_state = state.copy()
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state

    def cost(self, state: np.ndarray, action: Tuple[int, int]) -> float:
        """Evaluate assignment quality"""
        new_state = self.result(state, action)
        return self.evaluate(new_state)

    def evaluate(self, assignment: np.ndarray) -> float:
        """
        Calculate total cost of assignment

        cost = Σ F[i][j] × D[assignment[i]][assignment[j]]
        """
        total = 0.0
        for i in range(self.n):
            for j in range(self.n):
                total += self.flow[i][j] * self.distance[assignment[i]][assignment[j]]
        return total

    def optimality_gap(self, assignment: np.ndarray) -> float:
        """
        Gap from best known solution

        gap = (current_cost - best_known_cost) / best_known_cost
        """
        if self.best_known_cost is None:
            return float('inf')

        current_cost = self.evaluate(assignment)
        return (current_cost - self.best_known_cost) / self.best_known_cost

    @classmethod
    def from_qaplib(cls, filename: str):
        """Load from QAPLIB format"""
        with open(filename) as f:
            lines = f.readlines()

        n = int(lines[0].strip())

        # Parse flow matrix
        flow_lines = lines[1:n+1]
        flow = np.array([[float(x) for x in line.split()] for line in flow_lines])

        # Parse distance matrix
        dist_lines = lines[n+1:2*n+1]
        distance = np.array([[float(x) for x in line.split()] for line in dist_lines])

        problem = cls(flow, distance, name=filename.split('/')[-1])

        # Try to load best known solution
        try:
            with open(filename + '.sln') as f:
                problem.best_known_cost = float(f.readline().strip())
        except:
            pass

        return problem


# Test it
if __name__ == "__main__":
    # Create tiny example
    n = 4
    flow = np.array([
        [0, 5, 2, 4],
        [5, 0, 3, 0],
        [2, 3, 0, 0],
        [4, 0, 0, 0]
    ])

    distance = np.array([
        [0, 8, 15, 1],
        [8, 0, 13, 6],
        [15, 13, 0, 12],
        [1, 6, 12, 0]
    ])

    problem = QAPProblem(flow, distance, name="tiny4")

    # Test methods
    state = problem.initial_state()
    print(f"Initial state: {state}")
    print(f"Initial cost: {problem.evaluate(state)}")

    actions = problem.actions(state)
    print(f"Number of actions: {len(actions)}")  # Should be n*(n-1)/2 = 6

    # Try a swap
    new_state = problem.result(state, actions[0])
    new_cost = problem.evaluate(new_state)
    print(f"After swap {actions[0]}: {new_state}, cost: {new_cost}")

    print("✅ QAPProblem working!")
```

**Afternoon (3 hours): Signal Detectors**

```python
# pilot/qap/signal_detectors.py

from typing import List, Dict, Any
import numpy as np

class SignalDetector:
    """
    Monitors solver progress and detects when to intervene

    Signals:
    - Plateau: No improvement for N steps
    - Oscillation: Cycling between similar states
    - Overconfidence: Predicted quality >> actual
    - Budget pressure: Running out of time
    """

    def __init__(self):
        self.history = []  # List of (iteration, cost, state)

    def record(self, iteration: int, cost: float, state: Any):
        """Add observation to history"""
        self.history.append((iteration, cost, state))

    def detect_plateau(self, window: int = 10, threshold: float = 0.001) -> bool:
        """
        No improvement in last N steps

        Args:
            window: Number of recent steps to check
            threshold: Minimum improvement to not be plateau

        Returns:
            True if plateau detected
        """
        if len(self.history) < window:
            return False

        recent_costs = [cost for _, cost, _ in self.history[-window:]]
        best_recent = min(recent_costs)
        worst_recent = max(recent_costs)

        improvement = (worst_recent - best_recent) / worst_recent if worst_recent > 0 else 0

        return improvement < threshold

    def detect_oscillation(self, window: int = 10, threshold: float = 0.5) -> bool:
        """
        Bouncing between similar states

        Check if cost variance is high relative to mean improvement

        Returns:
            True if oscillating
        """
        if len(self.history) < window:
            return False

        recent_costs = [cost for _, cost, _ in self.history[-window:]]

        # High variance relative to mean suggests oscillation
        mean_cost = np.mean(recent_costs)
        std_cost = np.std(recent_costs)

        if mean_cost == 0:
            return False

        coefficient_of_variation = std_cost / mean_cost

        return coefficient_of_variation > threshold

    def detect_overconfidence(
        self,
        predicted_quality: float,
        actual_quality: float,
        threshold: float = 0.2
    ) -> bool:
        """
        Predicted confidence >> actual accuracy

        Args:
            predicted_quality: Solver's confidence in solution (0-1)
            actual_quality: Actual solution quality (0-1)
            threshold: Max acceptable gap

        Returns:
            True if overconfident
        """
        return predicted_quality - actual_quality > threshold

    def detect_budget_pressure(
        self,
        elapsed_iterations: int,
        max_iterations: int,
        threshold: float = 0.8
    ) -> bool:
        """
        Running out of time

        Args:
            elapsed_iterations: Iterations used so far
            max_iterations: Total iteration budget
            threshold: Fraction used before pressure

        Returns:
            True if budget pressure detected
        """
        return elapsed_iterations / max_iterations > threshold

    def get_signal_summary(self) -> Dict[str, Any]:
        """Get all current signals"""
        return {
            "plateau": self.detect_plateau(),
            "oscillation": self.detect_oscillation(),
            "history_length": len(self.history),
            "best_cost": min(cost for _, cost, _ in self.history) if self.history else None,
            "recent_improvement": self._recent_improvement_rate()
        }

    def _recent_improvement_rate(self, window: int = 10) -> float:
        """Calculate improvement rate over recent window"""
        if len(self.history) < 2:
            return 0.0

        recent = self.history[-min(window, len(self.history)):]
        if len(recent) < 2:
            return 0.0

        first_cost = recent[0][1]
        last_cost = recent[-1][1]

        if first_cost == 0:
            return 0.0

        return (first_cost - last_cost) / first_cost


# Test it
if __name__ == "__main__":
    detector = SignalDetector()

    # Simulate plateau
    for i in range(15):
        cost = 100.0 if i < 10 else 100.0 - i * 0.0001  # Very slow improvement
        detector.record(i, cost, f"state_{i}")

    print("Plateau test:", detector.detect_plateau())  # Should be True

    # Simulate oscillation
    detector2 = SignalDetector()
    for i in range(15):
        cost = 100.0 + (10.0 if i % 2 == 0 else -10.0)  # Bouncing
        detector2.record(i, cost, f"state_{i}")

    print("Oscillation test:", detector2.detect_oscillation())  # Should be True

    # Check summary
    print("Summary:", detector.get_signal_summary())

    print("✅ SignalDetector working!")
```

**End of Day 1 Checklist**:
- [ ] `QAPProblem` class passes tests
- [ ] Can load QAPLIB instance
- [ ] `SignalDetector` detects plateau
- [ ] `SignalDetector` detects oscillation
- [ ] Committed code to `pilot/sprint-1` branch

**Time Check**: If you're behind, skip the tests and come back later. Priority is getting the class working.

---

### **Day 2: Meta-Scheduler Implementation**

**Time**: 6-8 hours
**Goal**: Smart primitive selection based on signals

**Morning (4 hours): Meta-Scheduler Core**

```python
# pilot/qap/meta_scheduler.py

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from uaro import ReasoningPrimitive
from pilot.qap.signal_detectors import SignalDetector

@dataclass
class SchedulerState:
    """Current solver state"""
    iteration: int
    cost: float
    confidence: float
    active_primitive: Optional[str]
    signals: Dict[str, bool]

class MetaScheduler:
    """
    Chooses which primitive to use next based on:
    - Current state
    - Signal detections
    - Historical performance
    - Compute budget

    Strategy:
    - Start with LocalSearch (fast, greedy)
    - If plateau → SimulatedAnnealing (escape local optima)
    - If oscillation → Add tabu/constraints
    - If budget low → Greedy finish
    - If high variance → BeamSearch (explore multiple paths)
    """

    def __init__(self, primitives: List[ReasoningPrimitive], max_iterations: int = 1000):
        """
        Args:
            primitives: Available reasoning primitives
            max_iterations: Total iteration budget
        """
        self.primitives = {p.name: p for p in primitives}
        self.max_iterations = max_iterations

        # Track performance
        self.primitive_stats = {
            name: {"uses": 0, "improvements": 0, "time": 0.0}
            for name in self.primitives.keys()
        }

        # Current state
        self.current_primitive = None
        self.iterations_since_switch = 0

    def select_primitive(
        self,
        state: SchedulerState,
        detector: SignalDetector
    ) -> ReasoningPrimitive:
        """
        Choose next primitive based on signals

        Decision tree:
        1. Budget pressure? → LocalSearch (greedy finish)
        2. Plateau? → SimulatedAnnealing (exploration)
        3. Oscillation? → Add constraints or tabu
        4. High variance? → BeamSearch (multiple paths)
        5. Default: Continue current or LocalSearch
        """

        # Check signals
        signals = detector.get_signal_summary()

        # Decision logic
        if detector.detect_budget_pressure(state.iteration, self.max_iterations):
            return self._switch_to("LocalSearch", "budget_pressure")

        if signals["plateau"]:
            return self._switch_to("SimulatedAnnealing", "plateau_escape")

        if signals["oscillation"]:
            # For QAP, this means adding tabu or switching to beam
            return self._switch_to("BeamSearch", "oscillation_damping")

        # Check if current primitive is making progress
        if self.iterations_since_switch > 50:
            recent_improvement = signals["recent_improvement"]
            if recent_improvement < 0.001:  # Less than 0.1% improvement
                # Try something different
                return self._switch_to_best_alternative()

        # Continue with current if making progress
        if self.current_primitive and state.confidence > 0.7:
            self.iterations_since_switch += 1
            return self.primitives[self.current_primitive]

        # Default: Start with or return to LocalSearch
        return self._switch_to("LocalSearch", "default")

    def _switch_to(self, primitive_name: str, reason: str) -> ReasoningPrimitive:
        """Switch to a specific primitive"""
        if primitive_name not in self.primitives:
            # Fallback to LocalSearch if requested primitive not available
            primitive_name = "LocalSearch"

        self.current_primitive = primitive_name
        self.iterations_since_switch = 0
        self.primitive_stats[primitive_name]["uses"] += 1

        print(f"  🔀 Switching to {primitive_name} (reason: {reason})")

        return self.primitives[primitive_name]

    def _switch_to_best_alternative(self) -> ReasoningPrimitive:
        """Switch to primitive with best historical performance"""
        # Calculate success rate for each primitive
        scores = {}
        for name, stats in self.primitive_stats.items():
            if stats["uses"] > 0:
                success_rate = stats["improvements"] / stats["uses"]
                scores[name] = success_rate
            else:
                scores[name] = 0.5  # Unknown, assume average

        # Pick best (excluding current)
        alternatives = {k: v for k, v in scores.items() if k != self.current_primitive}
        if not alternatives:
            return self._switch_to("LocalSearch", "no_alternatives")

        best_name = max(alternatives, key=alternatives.get)
        return self._switch_to(best_name, "best_alternative")

    def record_result(self, improved: bool, time_taken: float):
        """Record outcome of current primitive"""
        if self.current_primitive:
            stats = self.primitive_stats[self.current_primitive]
            if improved:
                stats["improvements"] += 1
            stats["time"] += time_taken

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = {}
        for name, data in self.primitive_stats.items():
            success_rate = data["improvements"] / data["uses"] if data["uses"] > 0 else 0
            avg_time = data["time"] / data["uses"] if data["uses"] > 0 else 0

            stats[name] = {
                "uses": data["uses"],
                "success_rate": success_rate,
                "avg_time_per_use": avg_time
            }

        return stats


# Test it
if __name__ == "__main__":
    from uaro import LocalSearch, SimulatedAnnealing, BeamSearch

    # Create scheduler
    primitives = [LocalSearch(), SimulatedAnnealing(), BeamSearch()]
    scheduler = MetaScheduler(primitives, max_iterations=1000)
    detector = SignalDetector()

    # Simulate decision making
    state = SchedulerState(
        iteration=0,
        cost=100.0,
        confidence=0.5,
        active_primitive=None,
        signals={}
    )

    # Should start with LocalSearch
    primitive = scheduler.select_primitive(state, detector)
    print(f"Selected: {primitive.name}")  # Should be LocalSearch

    # Simulate plateau
    for i in range(15):
        detector.record(i, 100.0, f"state_{i}")  # No improvement

    state.iteration = 15
    primitive = scheduler.select_primitive(state, detector)
    print(f"After plateau: {primitive.name}")  # Should switch to SimulatedAnnealing

    # Get stats
    print("Stats:", scheduler.get_statistics())

    print("✅ MetaScheduler working!")
```

**Afternoon (3 hours): Integration with UARO Solver**

```python
# pilot/qap/qap_solver.py

from typing import Optional
import time
from uaro import UniversalSolver, SolutionResult
from pilot.qap.qap_problem import QAPProblem
from pilot.qap.signal_detectors import SignalDetector
from pilot.qap.meta_scheduler import MetaScheduler, SchedulerState

def solve_qap_with_meta_scheduler(
    problem: QAPProblem,
    max_iterations: int = 1000,
    time_limit: float = 300.0  # 5 minutes
) -> SolutionResult:
    """
    Solve QAP using UARO with meta-scheduler

    Args:
        problem: QAP instance
        max_iterations: Maximum iterations
        time_limit: Maximum time in seconds

    Returns:
        SolutionResult with complete trace
    """

    # Initialize components
    from uaro import LocalSearch, SimulatedAnnealing, BeamSearch
    primitives = [LocalSearch(), SimulatedAnnealing(), BeamSearch()]

    scheduler = MetaScheduler(primitives, max_iterations=max_iterations)
    detector = SignalDetector()

    # Initial state
    current_solution = problem.initial_state()
    current_cost = problem.evaluate(current_solution)
    best_solution = current_solution.copy()
    best_cost = current_cost

    # Tracking
    start_time = time.time()
    reasoning_trace = []

    print(f"\n🎯 Solving {problem.name}")
    print(f"   Problem size: n={problem.n}")
    print(f"   Initial cost: {current_cost:.2f}")
    if problem.best_known_cost:
        gap = problem.optimality_gap(current_solution)
        print(f"   Initial gap: {gap*100:.2f}%")
    print()

    # Main loop
    for iteration in range(max_iterations):
        elapsed = time.time() - start_time

        # Check time limit
        if elapsed > time_limit:
            print(f"  ⏱️  Time limit reached")
            break

        # Record state
        detector.record(iteration, current_cost, current_solution)

        # Create scheduler state
        confidence = 1.0 - (current_cost - best_cost) / best_cost if best_cost > 0 else 0.5
        state = SchedulerState(
            iteration=iteration,
            cost=current_cost,
            confidence=confidence,
            active_primitive=scheduler.current_primitive,
            signals=detector.get_signal_summary()
        )

        # Select primitive
        primitive = scheduler.select_primitive(state, detector)

        # Apply primitive (simplified for demo)
        # In real implementation, this would call primitive.apply(problem)
        # For now, simulate with a simple move

        # Try random swap (2-opt)
        import random
        i, j = random.randint(0, problem.n-1), random.randint(0, problem.n-1)
        if i != j:
            candidate = current_solution.copy()
            candidate[i], candidate[j] = candidate[j], candidate[i]
            candidate_cost = problem.evaluate(candidate)

            # Accept based on primitive strategy
            if primitive.name == "LocalSearch":
                # Greedy: only accept improvements
                if candidate_cost < current_cost:
                    current_solution = candidate
                    current_cost = candidate_cost
                    improved = True
                else:
                    improved = False

            elif primitive.name == "SimulatedAnnealing":
                # Accept worse moves with probability
                delta = candidate_cost - current_cost
                temperature = 100.0 * (1.0 - iteration / max_iterations)
                accept_prob = np.exp(-delta / temperature) if delta > 0 else 1.0

                if random.random() < accept_prob:
                    current_solution = candidate
                    current_cost = candidate_cost
                    improved = candidate_cost < current_cost
                else:
                    improved = False

            else:  # BeamSearch or other
                # For simplicity, act like LocalSearch
                if candidate_cost < current_cost:
                    current_solution = candidate
                    current_cost = candidate_cost
                    improved = True
                else:
                    improved = False

            # Update best
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
                print(f"  ✨ New best at iteration {iteration}: {best_cost:.2f}")
                if problem.best_known_cost:
                    gap = (best_cost - problem.best_known_cost) / problem.best_known_cost
                    print(f"     Gap: {gap*100:.2f}%")

            # Record result
            scheduler.record_result(improved, time.time() - start_time)

        # Progress update every 100 iterations
        if iteration % 100 == 0:
            print(f"  Iteration {iteration}: cost={current_cost:.2f}, primitive={primitive.name}")

    # Final results
    final_gap = problem.optimality_gap(best_solution) if problem.best_known_cost else None

    print(f"\n📊 Final Results:")
    print(f"   Best cost: {best_cost:.2f}")
    if final_gap is not None:
        print(f"   Final gap: {final_gap*100:.2f}%")
    print(f"   Iterations: {iteration+1}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"\n📈 Primitive Statistics:")
    for name, stats in scheduler.get_statistics().items():
        print(f"   {name}: {stats['uses']} uses, {stats['success_rate']*100:.1f}% success")

    # TODO: Create proper SolutionResult
    # For now, return basic info
    return {
        "problem": problem.name,
        "best_solution": best_solution,
        "best_cost": best_cost,
        "optimality_gap": final_gap,
        "iterations": iteration + 1,
        "time": elapsed,
        "scheduler_stats": scheduler.get_statistics()
    }


# Test it
if __name__ == "__main__":
    # Load a small QAPLIB instance
    # For testing, create a tiny synthetic instance
    n = 8
    np.random.seed(42)
    flow = np.random.randint(0, 100, (n, n))
    distance = np.random.randint(1, 50, (n, n))
    np.fill_diagonal(flow, 0)
    np.fill_diagonal(distance, 0)

    problem = QAPProblem(flow, distance, name="test8")

    result = solve_qap_with_meta_scheduler(problem, max_iterations=500, time_limit=30)

    print("\n✅ QAP solver with meta-scheduler working!")
```

**End of Day 2 Checklist**:
- [ ] `MetaScheduler` makes intelligent decisions
- [ ] Signal-based primitive switching works
- [ ] Can solve tiny QAP instance
- [ ] Tracks primitive performance
- [ ] Committed all code

---

### **Day 3: QAP Runs (5 instances)**

**Time**: 6-8 hours
**Goal**: Run on 5 QAPLIB instances, collect results

I'll stop here with the detailed day-by-day because this document is getting massive. Let me create a different format that's more usable...
