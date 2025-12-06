# Swarm Intelligence Voting

**Democratic consensus from 100+ agents**

Part of the Turing Challenge System (Feature #7)

## Overview

Swarm Intelligence Voting enables collective decision-making from large numbers of AI agents (typically 100+). Like a swarm of bees deciding on a new hive location, multiple agents vote democratically to reach consensus on research questions.

## Philosophy

> "The wisdom of crowds: 100 agents voting together are smarter than 1 expert deciding alone."

Collective intelligence emerges from aggregating diverse perspectives. This system harnesses that intelligence while detecting and preventing groupthink.

## Features

### Weighted Voting

- **Expertise-based weights:** Agents with relevant expertise get higher weight
- **Performance-based weights:** Historical accuracy increases voting power
- **Automatic weight calculation:** Based on expertise match to question

### Groupthink Detection

- **Variance analysis:** Low opinion variance triggers warnings
- **Dominance detection:** 90%+ agreement flagged as potential groupthink
- **Automatic prevention:** Inject diversity when needed

### Consensus Measurement

- **Strong (>80%):** High agreement, proceed with confidence
- **Moderate (60-80%):** Good agreement, reasonable confidence
- **Weak (40-60%):** Split opinions, caution advised
- **No Consensus (<40%):** Major disagreement, revisit question

### Comprehensive Analytics

- **Vote distribution:** How votes spread across options
- **Diversity scoring:** How much opinions differ (0-100)
- **Confidence metrics:** Average agent confidence
- **Reliability scoring:** Overall result trustworthiness

## Installation

```bash
cd TalAI/turing-features/swarm-voting
pip install -e .
```

## Quick Start

```python
from swarm_voting import SwarmVotingProtocol

# Create protocol with 100 agents
protocol = SwarmVotingProtocol(num_agents=100)

# Vote on a question
result = await protocol.vote(
    question="Should we proceed with this hypothesis?",
    options=["Yes", "No", "Needs revision"]
)

# Check results
print(f"Verdict: {result.verdict}")
print(f"Winner: {result.winning_option} ({result.vote_percentage:.0f}%)")
print(f"Consensus: {result.consensus_level}")
print(f"Reliability: {result.reliability:.0f}/100")
```

## Usage Examples

### Example 1: Binary Decision

```python
from swarm_voting import SwarmVotingProtocol

protocol = SwarmVotingProtocol(num_agents=100)

result = await protocol.vote(
    question="Is this hypothesis scientifically valid?",
    options=["Valid", "Invalid"]
)

if result.consensus_level == "strong":
    print(f"STRONG CONSENSUS: {result.winning_option}")
else:
    print("Divided opinions - need more analysis")
```

### Example 2: Multiple Options

```python
result = await protocol.vote(
    question="Which research direction should we pursue?",
    options=[
        "Quantum approach",
        "Machine learning approach",
        "Classical optimization",
        "Hybrid method"
    ]
)

# Get ranked preferences
for option, count in sorted(
    result.vote_distribution.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{option}: {count} votes")
```

### Example 3: With Context

```python
result = await protocol.vote(
    question="Should we publish these results?",
    options=["Publish now", "Revise first", "More experiments needed"],
    context="""
    Results show 15% improvement over state-of-the-art.
    Statistical significance: p < 0.01
    Sample size: n=100
    """
)

# Context helps agents make informed decisions
```

### Example 4: Groupthink Check

```python
result = await protocol.vote(
    question="Is this the best approach?",
    options=["Yes", "No"]
)

if result.groupthink_detected:
    print("⚠️  WARNING: Groupthink detected!")
    print("   Opinions too uniform - consider:")
    print("   - Seeking devil's advocate perspective")
    print("   - Injecting diverse viewpoints")
    print("   - Re-voting after discussion")
```

## Voting Mechanism

### Weight Calculation

```python
weight = base_weight × expertise_multiplier × accuracy_multiplier

where:
  base_weight = 1.0
  expertise_multiplier = 1.5 if expertise matches, else 1.0
  accuracy_multiplier = 0.5 + past_accuracy (0-1)
```

**Example:**
- Agent with optimization expertise voting on optimization question: 1.0 × 1.5 × 1.2 = 1.8
- Agent with biology expertise voting on optimization question: 1.0 × 1.0 × 0.8 = 0.8

### Consensus Calculation

```python
consensus_level = determine_level(vote_percentage)

Strong:       vote_percentage >= 80%
Moderate:     60% <= vote_percentage < 80%
Weak:         40% <= vote_percentage < 60%
No Consensus: vote_percentage < 40%
```

### Groupthink Detection

Two detection methods:

**1. Dominance Check:**
```python
if winning_option_votes / total_votes > 0.90:
    flag_groupthink()
```

**2. Variance Check:**
```python
confidence_variance = variance([v.confidence for v in votes])
if confidence_variance < 0.15:
    flag_groupthink()
```

### Diversity Scoring

```python
diversity = 100 × (1 - total_deviation / max_deviation)

where:
  total_deviation = sum of deviations from perfect distribution
  max_deviation = maximum possible deviation
```

**Perfect diversity (100):** All options equally voted
**No diversity (0):** All votes for one option

## Consensus Levels Explained

### Strong Consensus (>80%)

- **Interpretation:** Clear agreement, proceed with confidence
- **Action:** Move forward
- **Reliability:** 90/100
- **Example:** 85 agents vote "Yes", 15 vote "No"

### Moderate Consensus (60-80%)

- **Interpretation:** Good agreement, reasonable confidence
- **Action:** Proceed with minor caution
- **Reliability:** 70/100
- **Example:** 70 agents vote "Yes", 30 vote "No"

### Weak Consensus (40-60%)

- **Interpretation:** Split opinions, significant disagreement
- **Action:** Investigate further before deciding
- **Reliability:** 50/100
- **Example:** 55 agents vote "Yes", 45 vote "No"

### No Consensus (<40%)

- **Interpretation:** Major disagreement, no clear direction
- **Action:** Revisit question, gather more information
- **Reliability:** 30/100
- **Example:** 35% Yes, 35% No, 30% Undecided

## Reliability Scoring

```python
reliability = min(100, max(0,
    consensus_score + diversity_bonus - groupthink_penalty
))

where:
  consensus_score = 90 (strong) | 70 (moderate) | 50 (weak) | 30 (none)
  diversity_bonus = diversity_score × 0.1
  groupthink_penalty = 20 if groupthink_detected else 0
```

**High reliability (80+):** Trust the result
**Medium reliability (50-79):** Result is useful but not definitive
**Low reliability (<50):** Result is questionable, reconsider

## Integration with Turing Challenge

Swarm Voting is Feature #7 of the Turing Challenge System:

```python
from turing_challenge_system import TuringChallengeSystem

system = TuringChallengeSystem()
result = await system.validate_hypothesis_complete(hypothesis)

# Swarm Voting used in Step 5 for democratic consensus
# 100+ agents vote on whether to proceed
```

## Best Practices

### 1. Use Appropriate Swarm Size

```python
# Small swarm (10-30): Quick decisions, less reliable
protocol = SwarmVotingProtocol(num_agents=20)

# Medium swarm (50-100): Balanced
protocol = SwarmVotingProtocol(num_agents=75)

# Large swarm (100+): Slower but more reliable
protocol = SwarmVotingProtocol(num_agents=200)
```

### 2. Provide Context

```python
# Better decisions with context
result = await protocol.vote(
    question="Should we proceed?",
    options=["Yes", "No"],
    context="Previous experiments showed 20% improvement with p<0.05"
)
```

### 3. Check for Groupthink

```python
if result.groupthink_detected:
    # Inject diversity
    # Re-vote after discussion
    # Seek dissenting opinions
```

### 4. Interpret Consensus Levels

```python
if result.consensus_level == "strong":
    proceed_with_confidence()
elif result.consensus_level == "moderate":
    proceed_with_caution()
elif result.consensus_level == "weak":
    investigate_further()
else:  # no consensus
    revisit_question()
```

## Comparison to Other Methods

### Swarm Voting vs Single Expert

| Aspect | Swarm Voting | Single Expert |
|--------|--------------|---------------|
| Accuracy | Higher (aggregated wisdom) | Variable |
| Bias | Lower (diverse perspectives) | Higher |
| Coverage | Comprehensive | Limited |
| Confidence | Measurable | Subjective |
| Scalability | High | Low |

### Swarm Voting vs Tournament

| Aspect | Swarm Voting | Tournament |
|--------|--------------|------------|
| Purpose | Democratic decision | Find best solution |
| Output | Consensus opinion | Single winner |
| Process | Parallel voting | Sequential competition |
| Diversity | Preserved | Reduced (elimination) |

## Testing

```bash
cd TalAI/turing-features/swarm-voting
pytest tests/
```

## References

- Wisdom of Crowds (James Surowiecki)
- Swarm Intelligence (Kennedy & Eberhart)
- Collective Decision-Making in Distributed Systems
- Consensus Algorithms in Distributed Computing

## License

Part of TalAI - Apache 2.0 License

## See Also

- [Turing Challenge Master Documentation](../TURING_CHALLENGE_MASTER.md)
- [Agent Tournaments](../agent-tournaments/)
- [Emergent Behavior Monitoring](../emergent-behavior/)
