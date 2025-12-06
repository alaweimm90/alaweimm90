# IDEAS Suite - Quick Reference

**Last Updated:** 2025-11-16
**Total Products:** 17 | **Refactored:** 5 | **Quality:** 81/100

---

## 📋 Quick Commands

```bash
# Navigate the suite
cat MASTER_INDEX.md

# View standards
cat STANDARDS_AND_CONVENTIONS.md

# Refactor a product
python refactor_agents.py --product failure-db

# Refactor all
python refactor_agents.py --all --report

# Check quality
python refactor_agents.py --product chaos-engine --quality
```

---

## 📦 Session 2 Products (Refactored ✅)

| Product | What It Does | Quality | LOC |
|---------|--------------|---------|-----|
| **failure-db** | Bet on research failures | 81/100 | 1,297 |
| **research-pricer** | Calculate grant ROI | 81/100 | 1,155 |
| **experiment-designer** | Generate protocols | 81/100 | 1,809 |
| **chaos-engine** | Domain collisions | 81/100 | 1,049 |
| **ghost-researcher** | Consult dead scientists | 81/100 | 1,255 |

---

## 🎯 Find a Product

**Need:** Research help → **Use:** PaperMiner, CitationPredictor
**Need:** Novel ideas → **Use:** ChaosEngine, IdeaCalculus
**Need:** Grant writing → **Use:** ResearchPricer, ExperimentDesigner
**Need:** Fresh perspective → **Use:** GhostResearcher, AdversarialReview
**Need:** Prompts → **Use:** PromptForge, PromptMarketplace

---

## 🛠️ Refactoring Agents

1. **StructureAgent** - Creates golden template
2. **CodeStyleAgent** - Enforces Python style
3. **DocAgent** - Consolidates docs
4. **NamingAgent** - Checks conventions
5. **QualityAgent** - Measures quality
6. **ConsolidationAgent** - Removes clutter

---

## 📁 Golden Template

```
product-name/
├── src/product_name/
│   ├── __init__.py
│   └── main.py
├── tests/
├── examples/
├── docs/
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## 📊 Quality Scores

**Target:** 80/100 minimum
**Session 2:** 81/100 average ✅
**Session 1:** Not yet measured

---

## 🚀 Quick Start

```bash
# Install product
cd failure-db && pip install -e .

# Run example
cd examples && python example_basic.py

# Run tests
pytest tests/

# View docs
cat docs/API.md
```

---

## 📖 Documentation

- **MASTER_INDEX.md** - Central hub (start here)
- **STANDARDS_AND_CONVENTIONS.md** - Coding standards
- **REFACTOR_REPORT.md** - Quality metrics
- **COMPLETE_PRODUCT_CATALOG.md** - All 60+ ideas

---

## ✅ Naming Rules

- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case()`
- Constants: `SCREAMING_SNAKE`
- Private: `_snake_case()`

---

**For full details:** See MASTER_INDEX.md
