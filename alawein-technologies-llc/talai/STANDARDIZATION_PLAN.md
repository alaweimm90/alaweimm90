# Repository Standardization & Expansion Plan

**Date:** 2025-11-15
**Goal:** Standardize all products + build missing ideas from inventory

---

## Phase 1: Standardize Existing 7 Products

### Current Structure (Inconsistent)
```
product-name/
├── script.py
├── README.md
└── example_data.*
```

### Target Structure (Golden Template)
```
product-name/
├── src/
│   └── product_name/
│       ├── __init__.py
│       ├── cli.py          # Command-line interface
│       ├── core.py         # Core logic
│       └── models.py       # Data models
├── tests/
│   ├── __init__.py
│   └── test_core.py
├── examples/
│   ├── input_example.json
│   └── output_example.json
├── docs/
│   ├── USER_GUIDE.md
│   └── API.md
├── pyproject.toml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

### Products to Standardize

1. **adversarial-review** → `adversarial_review`
2. **promptforge-lite** → `promptforge_lite`
3. **abstract-writer** → `abstract_writer`
4. **citation-predictor** → `citation_predictor`
5. **hypothesis-match** → `hypothesis_match`
6. **paper-miner** → `paper_miner`
7. **data-cleaner** → `data_cleaner`

---

## Phase 2: Missing Critical Ideas (Build Next)

### Tier 1: Must-Build (High Impact)

#### 1. **FailureDB** - Failure Futures Market
**Concept:** Bet on which research ideas will fail (prediction market)
**Unique Value:** Learn from failures, avoid duplicate mistakes
**Build Time:** 8 hours
**Revenue:** Trading fees + $99/month data access

#### 2. **IdeaCalculus** - Mathematical Framework for Ideas
**Concept:** Formal calculus for idea manipulation (derivatives, integrals, limits)
**Operations:**
- d/dt(idea) = rate of novelty decay
- ∫(ideas) = synthesis of concepts
- lim(idea→∞) = ultimate implication
**Build Time:** 12 hours
**Revenue:** $149/month for researchers + academics

#### 3. **PromptMarketplace** - Buy/Sell Prompts
**Concept:** Darwinian evolution of prompts through economic selection
**Features:**
- Prompt listings with performance metrics
- Usage-based royalties (creator gets % of each use)
- A/B testing leaderboards
- Version control for prompts
**Build Time:** 10 hours
**Revenue:** 15% marketplace commission

#### 4. **ResearchPricer** - Grant ROI Calculator
**Concept:** Predict research ROI before applying for grants
**Features:**
- Expected publication count
- Citation trajectory
- Career advancement probability
- Economic impact estimation
**Build Time:** 5 hours
**Revenue:** $199/month institutions

#### 5. **ExperimentDesigner** - Automated Protocol Generator
**Concept:** Input hypothesis → complete experiment protocol
**Features:**
- Power analysis for sample size
- Control variable identification
- Equipment/reagent lists
- Timeline generation
- Cost estimation
**Build Time:** 10 hours
**Revenue:** $149/month

### Tier 2: Quick Wins (3-4 hours each)

6. **ReferenceFinder** - Auto-find relevant citations
7. **DatasetMatcher** - Find similar public datasets
8. **MethodologyMiner** - Extract methods from papers
9. **StatisticsValidator** - Check statistical validity
10. **IntroGenerator** - Write paper introductions

---

## Phase 3: Fusion Products (Combine Similar Tools)

### Fusion Idea 1: **ResearchOS**
**Combine:**
- PaperMiner + CitationPredictor + AdversarialReview + AbstractWriter

**Value:** Complete research workflow platform
- Analyze literature → Predict impact → Get critique → Write paper

### Fusion Idea 2: **PromptStudio**
**Combine:**
- PromptForge Lite + PromptMarketplace + (new) PromptRouter

**Value:** Complete prompt engineering suite
- Extract → Share/Sell → Auto-select best prompts

### Fusion Idea 3: **DataWorkbench**
**Combine:**
- DataCleaner + DataSynthesizer + StatisticsValidator + VisualizationEngine

**Value:** End-to-end data pipeline
- Clean → Validate → Augment → Visualize

### Fusion Idea 4: **CollabHub**
**Combine:**
- HypothesisMatch + SkillsGraph + MeetingScheduler + ProjectTracker

**Value:** Complete collaboration platform
- Find partners → Map skills → Schedule → Track progress

---

## Phase 4: Missing Unique Ideas from Inventory

### From CRAZY_IDEAS_MASTER_PROMPT.md

11. **GhostResearcher** - AI resurrects dead scientists
    - Simulate Einstein, Feynman, Curie based on their papers
    - "What would Einstein say about quantum computing?"
    - Build time: 6 hours

12. **ResearchBattleRoyale** - 100 hypotheses compete
    - Tournament bracket for research ideas
    - AI judges evaluate each round
    - Crowd voting + expert panels
    - Build time: 8 hours

13. **ChaosEngine** - Force random domain collisions
    - "What if biology used blockchain?"
    - "What if physics used game theory?"
    - Random domain mashups for novelty
    - Build time: 5 hours

14. **MultiverseResearch** - Test across parallel realities
    - Simulate hypothesis in different physical constants
    - Alternate history simulations
    - Counterfactual analysis engine
    - Build time: 12 hours

15. **TimeTravelCitations** - Predict future citations (RENAME: we have CitationPredictor)
    - Already built, but enhance with temporal dynamics

### New Ideas (Not in Inventory)

16. **IdeaMutation** - Genetic algorithm for ideas
    - Crossover: Combine two ideas
    - Mutation: Random variation
    - Selection: Based on novelty + feasibility scores
    - Build time: 8 hours

17. **ConceptEmbedding** - Vector space for ideas
    - Every idea gets a position in high-dim space
    - Distance = conceptual similarity
    - Clustering reveals research themes
    - Build time: 6 hours

18. **AutoReviewer** - Fully automated peer review
    - Checks methodology, statistics, novelty
    - Compares to existing literature
    - Generates review report (accept/revise/reject)
    - Build time: 10 hours

19. **SkepticBot** - Professional idea critic
    - Adversarial AI that tears apart every hypothesis
    - Red team for research ideas
    - Finds edge cases and failure modes
    - Build time: 4 hours

20. **IdeaGenealogy** - Track idea evolution
    - Citation graph for ideas (not papers)
    - Who thought of X first?
    - How did concept Y evolve?
    - Build time: 8 hours

---

## Recommended Build Order (Next 10 Products)

### Sprint 2: High-Value Core (5 products, ~40 hours)

1. **FailureDB** (8h) - Unique, high viral potential
2. **IdeaCalculus** (12h) - Novel mathematical framework
3. **PromptMarketplace** (10h) - Strong revenue model
4. **ResearchPricer** (5h) - Quick win, institutional sales
5. **ExperimentDesigner** (10h) - High value for researchers

**Total:** 45 hours, ~$675 credit

### Sprint 3: Quick Wins (5 products, ~20 hours)

6. **ReferenceFinder** (4h)
7. **DatasetMatcher** (4h)
8. **MethodologyMiner** (5h)
9. **IntroGenerator** (4h)
10. **ChaosEngine** (5h)

**Total:** 22 hours, ~$330 credit

### Sprint 4: Fusion Products (2-3 platforms, ~30 hours)

11. **ResearchOS** (15h) - Mega-platform combining 4 tools
12. **PromptStudio** (10h) - Complete prompt suite
13. **DataWorkbench** (12h) - Data pipeline platform

**Total:** 37 hours, ~$555 credit

---

## Total Plan Summary

**Products after all sprints:** 10 existing + 13 new = **23 total products**

**Credit allocation:**
- Sprint 1 (completed): $480
- Sprint 2: $675
- Sprint 3: $330
- Sprint 4: $555 (may need to optimize)
**Total:** $2,040 (over $1000 budget)

**Optimized plan for $1000 budget:**
- Use remaining $520 for Sprint 2 (5 products)
- Focus on highest-value: FailureDB, IdeaCalculus, PromptMarketplace, ResearchPricer, ExperimentDesigner
- Leave fusion products for later (require existing products to be polished first)

---

## Implementation Strategy

### For Each New Product:

1. **Core implementation** (60% of time)
   - Single Python file with CLI
   - Core logic + data models
   - Example data

2. **Testing** (20% of time)
   - Manual CLI testing
   - Example workflows

3. **Documentation** (20% of time)
   - README with features, usage, examples
   - Revenue model
   - Technical specs

### Standardization Process:

1. **Restructure** (per product, ~1 hour)
   - Create src/ directory structure
   - Move code to proper modules
   - Add __init__.py files
   - Create pyproject.toml

2. **Add missing components**
   - LICENSE file (MIT)
   - CHANGELOG.md
   - Basic tests/

3. **Update documentation**
   - Consistent README format
   - Installation instructions
   - API documentation

---

## Success Metrics

**By End of All Sprints:**

- ✅ 23 functional products
- ✅ All standardized to golden template
- ✅ All tested with examples
- ✅ All documented
- ✅ Total code: ~10,000+ LOC
- ✅ Revenue potential: $3,000-10,000/month MRR

**Immediate Next Steps:**

1. Standardize existing 7 products (7 hours)
2. Build FailureDB (8 hours)
3. Build IdeaCalculus (12 hours)
4. Build PromptMarketplace (10 hours)
5. Build ResearchPricer (5 hours)

**Timeline:** ~42 hours of focused work

---

## File Structure After Standardization

```
IDEAS/
├── products/
│   ├── adversarial_review/
│   ├── promptforge_lite/
│   ├── abstract_writer/
│   ├── citation_predictor/
│   ├── hypothesis_match/
│   ├── paper_miner/
│   ├── data_cleaner/
│   ├── failure_db/          # NEW
│   ├── idea_calculus/       # NEW
│   ├── prompt_marketplace/  # NEW
│   ├── research_pricer/     # NEW
│   └── experiment_designer/ # NEW
│
├── platforms/               # Fusion products
│   ├── research_os/
│   ├── prompt_studio/
│   └── data_workbench/
│
├── core/                    # Pre-existing
│   ├── ideaforge/
│   ├── buildforge/
│   └── turingo/
│
├── docs/
│   ├── PRODUCTS_INDEX.md
│   ├── STANDARDIZATION_PLAN.md (this file)
│   ├── SPRINT_COMPLETION_REPORT.md
│   └── MASTER_IDEA_INVENTORY.md
│
└── archive/
```

---

**Status:** Plan created, ready for execution
**Next Action:** Begin standardization of existing products
