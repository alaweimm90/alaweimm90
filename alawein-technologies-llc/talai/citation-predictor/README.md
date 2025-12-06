# CitationPredictor - Predict Future Citation Counts

Estimate how many citations a research paper will receive based on author reputation, venue prestige, topic novelty, and other factors.

## What It Does

Predicts citation counts at 1, 3, and 5 years using a statistical model that considers:

1. **Author reputation** - Average h-index of authors
2. **Venue prestige** - Journal/conference impact factor
3. **Topic novelty** - Presence of novelty indicators in abstract
4. **Abstract quality** - Length and structure
5. **Reference count** - Number of cited works
6. **Publication timing** - Age of paper
7. **Field growth** - Citation dynamics of research area
8. **Collaboration size** - Number of co-authors

## Installation

```bash
cd citation-predictor
# No dependencies - pure Python 3.11+
```

## Usage

### Predict for Single Paper

Create metadata file (paper.json):
```json
{
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", "Shazeer", "Parmar", "Uszkoreit", "Jones", "Gomez", "Kaiser", "Polosukhin"],
  "venue": "NIPS",
  "year": 2017,
  "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
  "num_references": 42,
  "field": "NLP",
  "author_h_index_avg": 35.0,
  "venue_impact_factor": 9.2
}
```

Run prediction:
```bash
python predictor.py predict --input paper.json --current 50000
```

### Batch Prediction

```bash
python predictor.py batch --input papers.json --output predictions.json
```

## Output Example

```
Citation Prediction for: Attention Is All You Need
======================================================================

Current citations: 50000
Trajectory: exponential

Predicted citations:
  1 year:   65000  (range: 45500-84500)
  3 years: 112000  (range: 67200-156800)
  5 years: 193000  (range: 96500-289500)

Expected percentile: 99.0% (vs similar papers)

Influence factors:
  author_reputation        [####################] 0.70
  venue_prestige          [##################  ] 0.92
  topic_novelty           [################    ] 0.80
  field_growth            [###############     ] 0.75
  abstract_quality        [##############      ] 0.70
  reference_count         [##############      ] 0.70
  collaboration_size      [############        ] 0.60
  publication_timing      [############        ] 0.60
```

## Growth Trajectories

**Exponential** - High-impact papers with sustained growth
- Top-tier venues + high author reputation + novel topic
- Example: Breakthrough methods (Transformers, ResNet, BERT)

**Power Law** - Strong initial growth that gradually slows
- Good venues + solid contributions
- Example: Solid incremental improvements

**Linear** - Steady citation accumulation
- Mid-tier venues + established topics
- Example: Application papers, datasets

**Plateau** - Limited impact, citations level off
- Lower-tier venues + incremental work
- Example: Niche applications

## Confidence Intervals

Uncertainty increases with prediction horizon:
- **1 year**: ±30% (narrow band)
- **3 years**: ±40% (medium band)
- **5 years**: ±50% (wide band)

## Use Cases

1. **Career planning** - Estimate research impact
2. **Hiring decisions** - Evaluate candidate productivity
3. **Grant writing** - Project paper impact for proposals
4. **Journal selection** - Choose venue based on expected citations
5. **Collaboration** - Identify high-impact co-authors
6. **Tenure review** - Predict citation trajectories

## Venue Impact Factors

Built-in impact factors for major venues:

| Venue | Impact Factor |
|-------|---------------|
| Nature | 42.8 |
| Science | 41.8 |
| Cell | 38.6 |
| NeurIPS | 9.2 |
| ICML | 8.5 |
| CVPR | 7.8 |
| ICLR | 7.2 |
| AAAI | 4.8 |
| ACL | 6.1 |

## Field Growth Rates

Annual citation growth by field:

| Field | Growth Rate |
|-------|-------------|
| AI | 1.35 (35% annual) |
| ML | 1.32 |
| CV | 1.28 |
| NLP | 1.25 |
| Robotics | 1.18 |
| Biology | 1.15 |
| Physics | 1.10 |

## Accuracy

Based on historical data (simulated):
- **1-year predictions**: ±25% error
- **3-year predictions**: ±35% error
- **5-year predictions**: ±45% error

Best predictions for:
- Established authors (h-index > 20)
- Top-tier venues (impact > 7.0)
- Active research areas (high field growth)

## Current Limitations

- Model trained on simulated data (not real historical citations)
- Does not account for social media amplification
- Simplified venue categorization
- No content-based semantic analysis
- Does not model paper-paper citation networks

## Revenue Model

- **Individual**: $49/month (50 predictions)
- **Lab/Group**: $149/month (500 predictions)
- **Institution**: $499/month (unlimited)
- **API access**: $0.10/prediction

## Build Info

- Build time: 4 hours
- Credit used: ~$60
- Lines of code: 480
- Status: Functional prototype

## Future Enhancements

- Train on real citation datasets (Microsoft Academic, OpenCitations)
- Add paper-paper network effects
- Include social media signals (Twitter mentions, Reddit upvotes)
- Temporal dynamics (citation bursts, decay patterns)
- Domain transfer learning
- Calibrated uncertainty quantification
