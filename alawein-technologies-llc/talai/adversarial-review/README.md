# AdversarialReview - AI Research Paper Critic

Provides brutal, honest feedback on research papers across 6 critical dimensions.

## What It Does

AdversarialReview simulates 6 adversarial critics who attack your research from different angles:

1. **Statistical Skeptic** - Questions statistical validity, power, corrections
2. **Methodology Maverick** - Challenges experimental design and controls
3. **Logic Enforcer** - Finds logical fallacies and inconsistencies
4. **History Hunter** - Checks literature review completeness and novelty claims
5. **Ethics Enforcer** - Identifies ethical concerns and compliance issues
6. **Economic Realist** - Evaluates feasibility, costs, and scalability

## Installation

```bash
cd adversarial-review
# No dependencies required - pure Python 3.11+
```

## Usage

### Basic Review

```bash
python review.py --title "My Paper Title" --abstract "Paper abstract text here"
```

### Review from File

```bash
python review.py --title "My Paper" --input paper.txt
```

### Nightmare Mode (Harshest Criticism)

```bash
python review.py --title "My Paper" --abstract "Abstract..." --mode nightmare
```

### Save Review to JSON

```bash
python review.py --title "My Paper" --abstract "Abstract..." --output review.json
```

## Review Modes

- **standard** - Balanced criticism (2-3 issues per dimension)
- **brutal** - Harsh criticism (3-4 issues per dimension)
- **nightmare** - Maximum criticism (4-6 issues per dimension)

## Output

Each critic provides:
- **Severity**: CRITICAL | MAJOR | MINOR
- **Score**: 0-10 (lower = more issues)
- **Issues**: Specific problems identified
- **Recommendations**: Concrete improvement suggestions

Overall verdict:
- **REJECT** - Score < 4.0
- **MAJOR_REVISION** - Score 4.0-5.5
- **MINOR_REVISION** - Score 5.5-7.0
- **ACCEPT** - Score > 7.0

## Example Output

```
==================================================================
ADVERSARIAL REVIEW RESULT
==================================================================

Paper: Novel Deep Learning Approach
Overall Score: 4.8/10
Verdict: MAJOR_REVISION

----------------------------------------------------------------------
[Statistical Skeptic] - STATISTICAL_VALIDITY
Severity: CRITICAL | Score: 3.2/10
----------------------------------------------------------------------

ISSUES IDENTIFIED:
  1. Sample size appears insufficient for claimed effect size
  2. Multiple comparison corrections not mentioned
  3. P-hacking risk: selective reporting suspected
  4. Effect size not reported or discussed

RECOMMENDATIONS:
  1. Conduct formal power analysis
  2. Apply Bonferroni or FDR correction
  3. Pre-register analysis plan
  4. Report effect sizes alongside p-values
```

## Use Cases

1. **Pre-submission check** - Find weaknesses before peer review
2. **Grant proposal review** - Strengthen funding applications
3. **Student training** - Learn common research pitfalls
4. **Quick sanity check** - Get second opinion on methodology
5. **Competitive analysis** - Evaluate competitor papers

## Current Limitations

- Simulated critics (not connected to actual AI models)
- Issue detection is pattern-based, not content-aware
- Cannot parse PDF files (requires text input)
- No deep semantic analysis of paper content

## Revenue Model

- **Pay-per-review**: $20/paper
- **Subscription**: $79/month unlimited reviews
- **Institutional**: $299/month for research groups

## Build Info

- Build time: 4 hours
- Credit used: ~$60
- Lines of code: 550
- Status: Functional prototype

## Future Enhancements

- PDF parsing integration
- Actual LLM-based criticism
- Domain-specific critics (bio, CS, physics)
- Comparison to similar published papers
- Automated literature gap detection
