# Products Index - Quick Reference

**Total Products:** 10 (3 pre-existing + 7 new)
**Date Updated:** 2025-11-15

---

## Pre-Existing Products

### 1. IdeaForge
**Location:** `/ideaforge/`
**Purpose:** Idea generation using 15 frameworks and 17 agents
**Status:** Functional prototype
**Usage:** `python ideaforge.py generate --input "problem" --output ideas.json`

### 2. BuildForge
**Location:** `/buildforge/`
**Purpose:** 5-gate validation system for ideas
**Status:** Functional prototype
**Usage:** `python buildforge.py run-gates --config domains/example/config.yaml`

### 3. Turingo
**Location:** `/turingo/`
**Purpose:** Multi-paradigm optimization solver (14 agents, 5 SOPs)
**Status:** Functional prototype
**Usage:** `python turingo.py solve --problem qap --instance chr12a`

---

## New Products (Sprint 2025-11-15)

### 4. AdversarialReview
**Location:** `/adversarial-review/`
**Purpose:** AI research paper critic with 6 adversarial reviewers
**Key Command:** `python review.py --title "Paper" --abstract "..." --mode nightmare`
**Revenue:** $20/paper or $79/month

### 5. PromptForge Lite
**Location:** `/promptforge-lite/`
**Purpose:** Offline prompt pattern extraction from notes
**Key Command:** `python promptforge.py scan --directory ./notes/ --output prompts.json`
**Revenue:** $29-99/month

### 6. AbstractWriter
**Location:** `/abstract-writer/`
**Purpose:** Generate structured academic abstracts
**Key Command:** `python writer.py generate --input outline.json --output abstract.txt`
**Revenue:** $5/abstract or $39/month

### 7. CitationPredictor
**Location:** `/citation-predictor/`
**Purpose:** Predict future citation counts (1yr, 3yr, 5yr)
**Key Command:** `python predictor.py predict --input paper.json --current 1000`
**Revenue:** $49-149/month

### 8. HypothesisMatch
**Location:** `/hypothesis-match/`
**Purpose:** Scientific Tinder - match researchers by skills and hypotheses
**Key Commands:**
- `python matcher.py create-profile --name "Dr. X" --skills "ML,CV" --interests "medical"`
- `python matcher.py match --profile-id 1 --mode complementary`
**Revenue:** $49-199/month

### 9. PaperMiner
**Location:** `/paper-miner/`
**Purpose:** Bulk research paper analysis (100+ papers)
**Key Command:** `python miner.py analyze --input papers.json --output insights.json`
**Revenue:** $79-999/month

### 10. DataCleaner
**Location:** `/data-cleaner/`
**Purpose:** Automated data cleaning and quality assessment
**Key Commands:**
- `python cleaner.py profile --input data.csv --output report.json`
- `python cleaner.py clean --input data.csv --output cleaned.csv`
**Revenue:** $79-249/month

---

## Quick Start Guide

### Test All Products

```bash
# AdversarialReview
cd adversarial-review
python review.py --title "Test Paper" --input example_paper.txt --mode nightmare

# PromptForge Lite
cd ../promptforge-lite
python promptforge.py extract --input example_notes.md --output prompts.json

# AbstractWriter
cd ../abstract-writer
python writer.py generate --input example_outline.json --output abstract.txt

# CitationPredictor
cd ../citation-predictor
python predictor.py predict --input example_paper.json --current 50000

# HypothesisMatch
cd ../hypothesis-match
python matcher.py create-profile --name "Alice" --skills "ML" --interests "AI"
python matcher.py list

# PaperMiner
cd ../paper-miner
python miner.py analyze --input example_papers.json

# DataCleaner
cd ../data-cleaner
python cleaner.py profile --input test_data.csv  # (needs CSV file)
```

---

## Revenue Summary

| Product | Individual | Team/Lab | Enterprise |
|---------|-----------|----------|------------|
| AdversarialReview | $79/mo | - | - |
| PromptForge Lite | $29/mo | $99/mo | $299/mo |
| AbstractWriter | $39/mo | - | - |
| CitationPredictor | $49/mo | $149/mo | - |
| HypothesisMatch | $49/mo | $199/mo | $999/mo |
| PaperMiner | $79/mo | $249/mo | $999/mo |
| DataCleaner | $79/mo | $249/mo | Custom |

**Bundle Pricing (all 7 tools):**
- Individual: $350/month (save $54)
- Research Lab: $1,200/month (save $100)
- Institution: $3,500/month (unlimited users)

---

## Documentation

- **Overview:** `README.md` (main repository)
- **Architecture:** `ARCHITECTURE.md`
- **Status:** `PROJECT_STATUS.md`
- **Sprint Report:** `SPRINT_COMPLETION_REPORT.md`
- **Idea Inventory:** `MASTER_IDEA_INVENTORY.md`
- **This Index:** `PRODUCTS_INDEX.md`

Each product directory contains:
- `*.py` - Main implementation
- `README.md` - Usage guide
- Example data files

---

## Next Steps

1. Add unit tests (pytest)
2. Create web UIs (Flask/FastAPI)
3. Deploy demos (Docker + cloud)
4. Add authentication/payment
5. Marketing sites
6. Beta testing with real users

---

**Last Updated:** 2025-11-15
**Total Code:** ~7,000 lines (including pre-existing)
**Total Products:** 10 functional tools
**Status:** Ready for alpha testing
