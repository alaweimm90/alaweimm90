# Self-Refutation Protocol - Examples

This directory contains example scripts demonstrating how to use the Self-Refutation Protocol.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Simple example showing how to test a single hypothesis.

```bash
python basic_usage.py
```

**What you'll learn:**
- Creating a Hypothesis object
- Running the refutation protocol
- Interpreting the strength score
- Understanding strategy results

### 2. QAP Hypothesis Testing (`qap_hypothesis_example.py`)

Tests 4 hypotheses with different strength levels, focused on QAP optimization.

```bash
python qap_hypothesis_example.py
```

**What you'll learn:**
- How to write strong vs weak hypotheses
- Why specific, bounded claims are better
- The importance of stating assumptions
- How absolute terms ("always", "all") hurt hypothesis strength

### 3. Batch Testing (`batch_testing.py`)

Tests multiple hypotheses in parallel across different domains.

```bash
python batch_testing.py
```

**What you'll learn:**
- Parallel hypothesis testing
- Comparing hypotheses across domains
- Batch statistics and analysis
- Resource prioritization based on scores

### 4. JSON-based Testing

Load hypotheses from JSON file:

```bash
# Using CLI
self-refute-batch hypotheses.json

# Or using Python
python -c "
import asyncio
import json
from self_refutation import SelfRefutationProtocol, Hypothesis

async def main():
    with open('hypotheses.json') as f:
        data = json.load(f)
    hypotheses = [Hypothesis(**h) for h in data]

    protocol = SelfRefutationProtocol()
    results = await protocol.refute_batch(hypotheses)

    for h, r in zip(hypotheses, results):
        print(f'{r.strength_score:.1f}/100 - {h.claim[:60]}...')

asyncio.run(main())
"
```

## CLI Examples

The package also provides a CLI for quick testing:

### Test a single hypothesis

```bash
self-refute "Our algorithm improves performance by 30%" --domain optimization
```

### Test from file

```bash
self-refute-file hypothesis.json --output results.json
```

### Batch testing

```bash
self-refute-batch hypotheses.json --min-score 60 --output results.json
```

### Run built-in example

```bash
self-refute example
```

## Tips for Writing Strong Hypotheses

Based on these examples, here are key insights:

### ✅ DO:
- Be **specific** ("8% improvement" not "better")
- State **assumptions** explicitly
- Include **boundary conditions** ("on tai instances", "for sparse matrices")
- Make **testable predictions**
- Acknowledge **limitations**

### ❌ DON'T:
- Use absolute terms ("always", "all", "never", "guaranteed")
- Make vague claims ("improves performance")
- Omit context ("tested on some examples")
- Claim impossibilities ("polynomial-time NP-complete solver")
- Ignore boundary conditions

## Expected Outputs

Running these examples will show:

1. **Strength Scores (0-100)**
   - 81-100: Strong, proceed to experiment
   - 61-80: Minor concerns
   - 41-60: Moderate concerns, revision recommended
   - 21-40: Major issues
   - 0-20: Critically flawed

2. **Strategy Results**
   - Which strategies passed/failed
   - Evidence found by each strategy
   - Severity of failures

3. **Recommendations**
   - Actionable next steps
   - Specific improvements to make
   - Whether to proceed, revise, or reject

## Integration with AI Orchestrator

For enhanced analysis with multi-model AI:

```python
from atlas_orchestrator import Orchestrator
from self_refutation import SelfRefutationProtocol

orchestrator = Orchestrator()
protocol = SelfRefutationProtocol(orchestrator=orchestrator)

# Now strategies can use AI for deeper analysis
result = await protocol.refute(hypothesis)
```

Or via CLI:

```bash
self-refute "Your hypothesis" --with-orchestrator
```

## Next Steps

After running these examples:

1. Try testing your own hypotheses
2. Experiment with different domains
3. Compare parallel vs sequential execution
4. Integrate with your research workflow
5. Use as pre-experiment validation gate

## Questions?

See the main [README.md](../README.md) for full documentation.
