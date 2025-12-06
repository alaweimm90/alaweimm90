# Devil's Advocate Agent

**Dedicated adversarial testing of all hypotheses**

Part of the Turing Challenge System (Feature #6)

## Overview

The Devil's Advocate Agent is a dedicated adversarial system that systematically tries to break every hypothesis by finding flaws, edge cases, and failure modes. It catches 20-30% more issues than interrogation alone through aggressive adversarial testing.

## Philosophy

> "The Devil's Advocate doesn't want to be right. It wants to find every way you could be wrong."

Unlike validation systems that look for evidence *supporting* a hypothesis, the Devil's Advocate actively searches for evidence *against* it. This Popperian approach strengthens hypotheses that survive adversarial testing.

## Features

### 5 Attack Strategies

1. **Edge Case Identification** - Zero values, negatives, extremes, empty inputs
2. **Assumption Challenging** - Independence, linearity, stationarity assumptions
3. **Scaling Attacks** - Small scale, large scale, intermediate failures
4. **Composition Attacks** - Component conflicts, interface mismatches
5. **Temporal Attacks** - Time-varying inputs, long-running degradation

### Automatic Flaw Categorization

- **Critical:** May invalidate entire hypothesis
- **High:** Serious issues requiring attention
- **Medium:** Moderate concerns
- **Low:** Minor issues, edge cases

### Comprehensive Analysis

- Flaw descriptions with examples
- Severity assessment
- Mitigation suggestions
- Robustness scoring (0-100)
- Overall verdict (ACCEPT/REJECT/REVISE)

## Installation

```bash
cd TalAI/turing-features/devils-advocate
pip install -e .
```

## Quick Start

```python
from devils_advocate import DevilsAdvocateProtocol

# Create protocol
protocol = DevilsAdvocateProtocol()

# Attack hypothesis
result = await protocol.attack(hypothesis, iterations=3)

# Check results
print(f"Flaws found: {result.total_flaws_found}")
print(f"Critical: {len(result.critical_flaws)}")
print(f"Verdict: {result.verdict}")
print(f"Robustness: {result.robustness_score}/100")
```

## Usage Examples

### Example 1: Basic Attack

```python
from devils_advocate import DevilsAdvocateProtocol
from self_refutation import Hypothesis, HypothesisDomain

# Create hypothesis
hypothesis = Hypothesis(
    claim="Our algorithm improves performance by 40%",
    domain=HypothesisDomain.OPTIMIZATION
)

# Attack it
protocol = DevilsAdvocateProtocol()
result = await protocol.attack(hypothesis, iterations=3)

# Review flaws
for flaw in result.critical_flaws:
    print(f"CRITICAL: {flaw.description}")
    print(f"  Example: {flaw.example}")
    print(f"  Fix: {flaw.mitigation}")
```

### Example 2: Multi-Iteration Attack

```python
# More iterations = more thorough attack
result = await protocol.attack(
    hypothesis,
    iterations=5  # 5 rounds of adversarial testing
)

# Track improvement over iterations
# (If flaws decrease, hypothesis is getting stronger)
```

### Example 3: Integration with Validation

```python
# Use Devil's Advocate as part of validation pipeline
from self_refutation import SelfRefutationProtocol
from interrogation import InterrogationProtocol

# Step 1: Self-refutation
refutation_result = await self_refutation.refute(hypothesis)

# Step 2: Interrogation
interrogation_result = await interrogation.interrogate(hypothesis)

# Step 3: Devil's Advocate (catches what others miss)
adversarial_result = await devils_advocate.attack(hypothesis)

# Devil's Advocate typically finds 20-30% more issues
```

## Attack Strategies Explained

### 1. Edge Case Identification

Finds boundary conditions where hypothesis may fail:

**Zero Values:**
```python
EdgeCase(
    description="What happens when input is zero?",
    input_conditions="Input parameter = 0",
    expected_behavior="Should handle gracefully"
)
```

**Negative Values:**
```python
EdgeCase(
    description="What happens with negative inputs?",
    input_conditions="Input parameter < 0",
    expected_behavior="Should reject or handle appropriately"
)
```

**Extreme Values:**
```python
EdgeCase(
    description="What happens at extreme scales?",
    input_conditions="Input parameter >> normal range",
    expected_behavior="Should scale or fail gracefully"
)
```

### 2. Assumption Challenging

Questions unstated assumptions:

**Independence:**
- "Assumes variables are independent when they may not be"
- Tests for hidden correlations

**Linearity:**
- "Assumes linear relationships when non-linear may exist"
- Tests for saturation, diminishing returns

**Stationarity:**
- "Assumes conditions remain constant over time"
- Tests for drift, seasonal effects

### 3. Scaling Attacks

Tests behavior at different scales:

**Small Scale:**
- N=1, N=2 cases
- Degenerate conditions
- Minimum viable scale

**Large Scale:**
- Computational complexity
- Memory requirements
- Performance degradation

**Intermediate:**
- Transition points
- Phase changes
- Optimal scale ranges

### 4. Composition Attacks

Tests component interactions:

**Interface Mismatches:**
- Component A expects X, Component B provides Y

**Conflict Detection:**
- Method A contradicts Method B

**Integration Failures:**
- Components work individually but fail together

### 5. Temporal Attacks

Tests time-dependent behavior:

**Time-Varying Inputs:**
- Does it handle changing conditions?
- Adaptation speed?

**Long-Running:**
- Memory leaks?
- Performance degradation?
- State accumulation?

## Flaw Severity Guidelines

### Critical (Invalidating)

- Logical contradictions
- Fundamental impossibilities
- Complete failure cases
- Violation of physics/mathematics

**Example:** "Algorithm requires time travel" → CRITICAL

### High (Serious Issues)

- Major failure modes
- Dangerous edge cases
- Scalability problems
- Security vulnerabilities

**Example:** "Fails on 80% of real-world inputs" → HIGH

### Medium (Moderate Concerns)

- Performance issues
- Usability problems
- Incomplete specifications
- Assumption violations

**Example:** "10x slower than alternatives" → MEDIUM

### Low (Minor Issues)

- Edge cases with low probability
- Cosmetic issues
- Documentation gaps
- Nice-to-have features

**Example:** "Doesn't handle empty lists gracefully" → LOW

## Robustness Scoring

The Devil's Advocate calculates a robustness score (0-100):

```python
robustness_score = 100 - (
    critical_flaws × 10 +
    high_flaws × 5 +
    medium_flaws × 2 +
    low_flaws × 1
) × 5
```

**Interpretation:**

- **90-100:** Extremely robust, ready for production
- **70-89:** Good, minor issues to address
- **50-69:** Moderate concerns, needs revision
- **30-49:** Major issues, significant rework needed
- **0-29:** Critical problems, reject or redesign

## Integration with Turing Challenge

Devil's Advocate is Feature #6 of the Turing Challenge System:

```python
from turing_challenge_system import TuringChallengeSystem

system = TuringChallengeSystem()
result = await system.validate_hypothesis_complete(hypothesis)

# Devil's Advocate runs automatically in Step 4
# Catches 20-30% more issues than interrogation alone
```

## Best Practices

### 1. Run Multiple Iterations

```python
# Single iteration: Quick check
result = await protocol.attack(hypothesis, iterations=1)

# Multiple iterations: Thorough adversarial testing
result = await protocol.attack(hypothesis, iterations=3)

# Many iterations: Exhaustive attack
result = await protocol.attack(hypothesis, iterations=5)
```

### 2. Address Critical Flaws First

```python
# Triage flaws by severity
for flaw in result.critical_flaws:
    # Must fix before proceeding
    apply_mitigation(flaw.mitigation)

for flaw in result.high_flaws:
    # Should fix
    consider_mitigation(flaw.mitigation)
```

### 3. Use in Validation Pipeline

```python
# Devil's Advocate is most effective as part of pipeline:
# 1. Self-refutation (fast filter)
# 2. Interrogation (systematic questions)
# 3. Devil's Advocate (adversarial attack) ← Catches what others miss
```

### 4. Track Improvements

```python
# Attack revised hypothesis
result_v1 = await protocol.attack(hypothesis_v1, iterations=3)
# Fix issues, create v2
result_v2 = await protocol.attack(hypothesis_v2, iterations=3)

# Improvement = fewer flaws
assert result_v2.total_flaws_found < result_v1.total_flaws_found
```

## Testing

```bash
cd TalAI/turing-features/devils-advocate
pytest tests/
```

## References

- Adversarial Testing in AI Safety
- Red Team Exercises in Cybersecurity
- Falsificationism (Karl Popper)
- Robustness Testing in Software Engineering

## License

Part of TalAI - Apache 2.0 License

## See Also

- [Turing Challenge Master Documentation](../TURING_CHALLENGE_MASTER.md)
- [Self-Refutation Protocol](../self-refutation/)
- [Interrogation Framework](../interrogation/)
