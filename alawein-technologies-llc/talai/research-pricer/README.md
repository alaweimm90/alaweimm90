# ResearchPricer - Grant ROI Calculator

Predict research ROI before applying for grants. Calculate expected publications, citations, career impact, and economic returns.

## Concept

Before investing months writing a grant proposal, calculate if it's worth it. Get data-driven predictions on:
- **Publications** - How many papers will result?
- **Citations** - What's the citation trajectory?
- **Career impact** - Tenure/promotion probability?
- **Economic ROI** - Total return on investment?

## Features

### For Researchers
- Submit grant proposals for ROI analysis
- Get predictions on publications, citations, career outcomes
- Compare multiple proposals side-by-side
- Assess risk vs reward before applying

### For Grant Reviewers
- Objective metrics for proposal evaluation
- Domain-wide statistics
- Historical ROI data
- Risk assessment

### For Institutions
- Portfolio optimization
- Resource allocation decisions
- Track grant performance predictions
- Domain benchmarking

## Usage

### Submit a Proposal

```bash
python pricer.py submit \
  --title "Novel quantum error correction using topological codes" \
  --domain "physics" \
  --funding 500000 \
  --duration 36 \
  --team-size 4 \
  --institution "MIT" \
  --pi-name "Dr. Alice Chen" \
  --pi-h-index 28 \
  --novelty 0.85 \
  --feasibility 0.70 \
  --impact transformative \
  --methodology "Theoretical and experimental validation" \
  --prior-pubs 42
```

### Calculate ROI

```bash
python pricer.py calculate --proposal-id 1
```

### Compare Proposals

```bash
# Compare proposals 1, 2, and 3
python pricer.py compare --proposal-ids 1,2,3
```

### Domain Statistics

```bash
python pricer.py domain-stats --domain physics
```

## Output Examples

### ROI Calculation

```
======================================================================
ROI PREDICTION - Novel quantum error correction using topological codes
======================================================================

Funding: $500,000.00 over 36 months
PI: Dr. Alice Chen (h-index: 28)
Domain: physics | Impact: transformative

📚 PUBLICATION METRICS
  Expected publications: 8.4
  Average journal IF: 12.3
  Conference presentations: 14
  Time to first publication: 8 months
  Time to breakthrough: 22 months

📊 CITATION METRICS
  1-year citations: 124
  3-year citations: 331
  5-year citations: 496
  H-index increase: +6.7

🎓 CAREER IMPACT
  Tenure probability: 87.3%
  Promotion probability: 78.5%
  Future grants expected: 8.2
  Collaborations formed: 6

💰 ECONOMIC IMPACT
  Total ROI: +347.8%
  Patent value: $284,532.00
  Commercialization potential: $1,247,893.00
  Industry partnerships: 3

⚠️  RISK ASSESSMENT
  Failure probability: 27.5%
  Risk level: MEDIUM

📈 OVERALL SCORES
  Academic value: 86.2/100
  Commercial value: 73.5/100
  Career value: 89.7/100

🎯 RECOMMENDATION: STRONGLY RECOMMEND

======================================================================
ANALYSIS
======================================================================
This transformative-impact physics project with $500,000 funding over
36 months is expected to produce 8.4 publications and 496 citations
within 5 years.

The PI's h-index of 28 and 42 prior publications suggest strong research
capability.

With a novelty score of 0.85 and feasibility of 0.70, this project
carries medium risk.

Expected ROI: 347.8% with academic value 86/100, commercial value 74/100,
and career value 90/100.

📋 COMPARABLE GRANTS:
  • Similar physics grant at Stanford
  • Comparable transformative impact project
  • Related Theoretical and experimental validation methodology study

📊 CONFIDENCE INTERVALS:
  Publications: 5.9 - 10.9
  Citations (5yr): 248 - 744
  ROI: 208.7% - 486.9%
```

### Proposal Comparison

```
======================================================================
PROPOSAL COMPARISON (sorted by ROI)
======================================================================

#1. Novel quantum error correction using topological codes
    Funding: $500,000
    ROI: +347.8%
    Publications: 8.4
    Citations (5yr): 496
    Academic: 86 | Commercial: 74 | Career: 90
    Risk: medium | Recommendation: strongly_recommend

#2. Machine learning for protein folding prediction
    Funding: $300,000
    ROI: +215.3%
    Publications: 5.2
    Citations (5yr): 287
    Academic: 72 | Commercial: 45 | Career: 68
    Risk: low | Recommendation: recommend

#3. Social network analysis of scientific collaboration
    Funding: $150,000
    ROI: +89.4%
    Publications: 3.8
    Citations (5yr): 94
    Academic: 58 | Commercial: 12 | Career: 52
    Risk: low | Recommendation: neutral
```

### Domain Statistics

```
======================================================================
DOMAIN STATISTICS - PHYSICS
======================================================================

Total proposals analyzed: 12
Average ROI: 245.7%
Average publications: 6.8
Average citations (5yr): 342.5
Average academic value: 78.3/100
Average commercial value: 52.1/100
High-risk proposals: 33.3%
Strongly recommended: 7
```

## Metrics Explained

### Input Scores

**Novelty Score (0-1):**
- 0.0-0.3: Incremental improvement
- 0.3-0.6: Significant advance
- 0.6-0.8: Breakthrough potential
- 0.8-1.0: Paradigm-shifting

**Feasibility Score (0-1):**
- 0.0-0.3: High technical risk
- 0.3-0.6: Moderate challenges
- 0.6-0.8: Well-defined path
- 0.8-1.0: Straightforward execution

**Impact Potential:**
- **low**: Narrow specialist interest
- **medium**: Broad field impact
- **high**: Cross-domain influence
- **transformative**: Industry/society impact

### Prediction Methodology

**Publications:**
- Base rate from funding level
- Adjusted by PI h-index, team size, duration
- Multiplied by novelty × feasibility

**Citations:**
- Function of journal quality and impact potential
- Time-decay model (30% @ 1yr, 80% @ 3yr, 120% @ 5yr)
- PI reputation factor

**Career Impact:**
- Tenure: Base rate + publications + impact
- Promotion: Similar model with different weights
- Future grants: Proportional to current success + PI factor

**Economic ROI:**
- Publication value: $20k per paper
- Citation value: $1k per citation
- Patent probability based on impact potential
- Commercialization for high-impact projects
- Career value from tenure/promotion
- Future grant value estimated

**Risk Assessment:**
- Inverse of feasibility
- Weighted by novelty (high novelty = higher variance)
- Classified: low (<20%), medium (20-50%), high (>50%)

## Use Cases

### For PhD Students/Postdocs
- "Should I apply for this fellowship?"
- "Which proposal has better ROI?"
- "Is the funding level adequate?"

### For PIs
- "Which of my 5 ideas should I write up?"
- "Compare NIH vs NSF funding outcomes"
- "Portfolio diversification strategy"

### For Department Chairs
- "Allocate internal seed funding"
- "Assess junior faculty proposals"
- "Benchmark against peer institutions"

### For Grant Reviewers
- "Data-driven evaluation metrics"
- "Historical performance comparison"
- "Risk-adjusted rankings"

## Pricing Model

### Individual Researchers
- **Free tier:** 3 proposals/month
- **Pro:** $49/month - Unlimited proposals, comparisons
- **Academic:** $99/month - + Domain stats, API access

### Institutions
- **Department:** $499/month - 10 users, all features
- **University:** $1,999/month - Unlimited users, custom models
- **Enterprise:** Custom - Training on your historical data

## Revenue Projection

**Target market:**
- 1M+ researchers worldwide applying for grants
- 10,000+ universities and research institutions
- Grant agencies and foundations

**Conservative estimates:**
- 1% adoption = 10,000 researchers @ $49/month = $490k/month
- 100 institutions @ $499/month = $50k/month
- **Total:** $540k/month = $6.5M/year

## Build Info

- Build time: 5 hours
- Credit used: ~$60
- Lines of code: 650
- Status: Functional prototype

## Future Enhancements

### Better Predictions
- Train on historical grant data
- Field-specific models
- PI track record analysis
- Institution prestige factors

### Advanced Features
- Collaboration network analysis
- Optimal team composition
- Timeline optimization
- Budget allocation suggestions
- Grant writing assistance integration
- Success probability by funding agency

### Integration
- Connect to NIH Reporter, NSF Awards
- Import CV data automatically
- Link to citation databases
- Export to grant writing tools

## Validation

To validate predictions, need:
1. Historical grant data (funding, PI, domain)
2. Outcome data (publications, citations, career progression)
3. Train ML model on correlations
4. Backtest on recent grants
5. Calibrate confidence intervals

**Accuracy targets:**
- Publications: ±30% (achievable with PI track record)
- Citations: ±50% (high variance, field-dependent)
- ROI: ±40% (depends on economic assumptions)

## Ethical Considerations

### Positive
- Reduces wasted effort on low-probability grants
- Democratizes access to success metrics
- Evidence-based decision making
- Resource optimization

### Concerns
- Could discourage high-risk/high-reward research
- Bias against junior researchers
- Gaming the metrics
- Self-fulfilling prophecies

### Mitigations
- Emphasize predictions are probabilistic
- Include "transformative" category for moonshots
- Normalize for career stage
- Audit for bias in recommendations
- Human judgment remains critical

---

**Status:** Prototype - Needs validation with real grant outcome data
