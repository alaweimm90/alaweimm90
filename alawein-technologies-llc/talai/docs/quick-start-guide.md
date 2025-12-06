# Quick Start Guide - IDEAS Innovation Stack

**Get started with IdeaForge + BuildForge in 5 minutes**

---

## ⚡ 5-Minute Quickstart

### Step 1: Generate Ideas (2 minutes)

```bash
cd /mnt/c/Users/mesha/Documents/IDEAS/ideaforge

# Generate 50 ideas from a research paper
python ideaforge.py generate \
  --input ~/Downloads/paper.pdf \
  --mode crazy \
  --count 50 \
  --output ideas.json
```

**Output**: `ideas.json` with 50 ranked ideas

---

### Step 2: Review & Pick (2 minutes)

```bash
# View ranked ideas
python ideaforge.py rank ideas.json --format markdown

# Pick top idea
python ideaforge.py rank ideas.json --top 1 > top_idea.json
```

**Manual review**: Open `ideas.json`, read through top 10, pick your favorite

---

### Step 3: Next Steps (1 minute decision)

**Option A: Validate First** (Recommended for products)
```bash
# Follow 4-week validation sprint
cd ../validation-framework
cat sprint-guide.md

# Week 1: Interview 20 users
# Week 2: Show mockups
# Week 3: Test pricing
# Week 4: Get 10 pre-sales

# If you get 10+ pre-sales → BUILD IT
```

**Option B: Research Track** (For academic/research projects)
```bash
# Run BuildForge gates
cd ../buildforge

# Convert IdeaForge idea to BuildForge config
python buildforge.py convert \
  --agis-idea ../ideaforge/top_idea.json \
  --domain my_research \
  --output domains/my_research/config.yaml

# Run novelty check (G1)
python buildforge.py gate1 \
  --config domains/my_research/config.yaml \
  --output novelty_report.md

# Review novelty_report.md
# If novelty_score >= 65 → PROCEED to G2-G5
```

---

## 📚 Full Workflows

### Workflow 1: Startup Validation (Product Track)

```
Day 1: IdeaForge generates 50 ideas (6 min)
Day 1: Pick top 3 (1 hour)
Weeks 1-4: Validate all 3 simultaneously
  - Interview 60 people (20 per idea)
  - Test 3 different landing pages
  - Run 3 pre-sale campaigns
Week 4: Pick the one with 10+ pre-sales
Weeks 5-12: Build MVP with BuildForge
Week 13-14: Benchmark & deploy
```

**Total**: 14 weeks from idea to deployed product

---

### Workflow 2: Research Paper (Academic Track)

```
Hour 1: IdeaForge generates 50 research ideas (6 min)
Hour 1: Pick most novel idea (rest of hour)
Week 1: BuildForge G1 (novelty assessment)
Week 2: BuildForge G2 (theory + proof)
Week 3: BuildForge G3 (feasibility)
Weeks 4-11: BuildForge G4 (build + benchmark)
Week 12: BuildForge G5 (write paper)
Week 13: Submit to ArXiv + conference
```

**Total**: 13 weeks from idea to submitted paper

---

### Workflow 3: PromptForge Extraction

```
Hour 1: Point PromptForge at your Obsidian vault
Hour 1: Review extracted prompts/workflows/tools
Hour 2: Organize and tag extractions
Hour 3: Use extracted artifacts in IdeaForge/BuildForge
Week 2+: MetaForge generates ideas to improve your workflow
```

---

## 🎯 Example Commands

### Generate Ideas from Different Sources

```bash
# From PDF
python ideaforge.py generate --input paper.pdf --output ideas.json

# From URL
python ideaforge.py generate --input "https://arxiv.org/abs/2024.12345" --output ideas.json

# From text
python ideaforge.py generate --input "Problem: researchers can't find collaborators" --output ideas.json

# From multiple sources
python ideaforge.py generate \
  --input paper1.pdf \
  --input paper2.pdf \
  --input "problem statement here" \
  --output combined_ideas.json
```

---

### Filter & Rank Ideas

```bash
# Rank by impact
python ideaforge.py rank ideas.json --by impact --top 10

# Rank by feasibility
python ideaforge.py rank ideas.json --by feasibility --top 10

# Rank by novelty
python ideaforge.py rank ideas.json --by novelty --top 10

# Rank by combined score (impact * novelty * feasibility)
python ideaforge.py rank ideas.json --by score --top 10

# Filter by domain
python ideaforge.py filter ideas.json --domain "machine learning" --output ml_ideas.json
```

---

### BuildForge Commands

```bash
# Run all gates automatically
python buildforge.py run-gates \
  --config domains/my_domain/config.yaml \
  --mode autonomous

# Run gates one by one
python buildforge.py gate1 --config domains/my_domain/config.yaml  # Novelty
python buildforge.py gate2 --config domains/my_domain/config.yaml  # Theory
python buildforge.py gate3 --config domains/my_domain/config.yaml  # Feasibility
python buildforge.py gate4 --config domains/my_domain/config.yaml  # Validation
python buildforge.py gate5 --config domains/my_domain/config.yaml  # Publication

# Deploy
python buildforge.py deploy \
  --config domains/my_domain/config.yaml \
  --track both  # research + product
```

---

### PromptForge Commands

```bash
# Extract from Obsidian vault
python promptforge.py extract \
  --vault ~/Documents/Obsidian \
  --output extracted/

# Extract from single note
python promptforge.py extract \
  --note my_note.md \
  --output extracted/

# Search extracted prompts
python promptforge.py search \
  --query "summarization" \
  --vault extracted/

# Convert to different formats
python promptforge.py convert \
  --input extracted/prompt.yaml \
  --format gpt4 \
  --output prompt_gpt4.json
```

---

## 🔧 Configuration

### IdeaForge Configuration

Edit `ideaforge/config.yaml`:

```yaml
# Thinking frameworks to use (pick 1-15)
frameworks:
  - first_principles
  - inversion
  - cross_domain_analogy
  # ... add more

# Agents to deploy (pick 1-17)
agents:
  - meta_orchestrator
  - planner
  - researcher
  # ... add more

# Output settings
output:
  format: json  # json | markdown | yaml
  max_ideas: 50
  min_score: 6.0  # 1-10 scale
```

---

### BuildForge Configuration

Create `buildforge/domains/my_domain/config.yaml`:

```yaml
domain: "My Research Area"
keywords: ["keyword1", "keyword2"]

novelty_threshold: 65  # 0-100 score

sota_baselines:
  - name: "Existing Method 1"
    metric_value: 12.5
    runtime_sec: 45.2

benchmarks:
  dataset: "Standard Benchmark"
  instances: ["test1", "test2"]

evaluation_metrics:
  - accuracy
  - runtime
  - memory

release_criteria:
  - "beats SOTA on 3+ instances by >= 2%"
  - "runtime parity (+/- 10%)"

price_point:  # if product track
  api_per_use: 0.10
  enterprise_monthly: 5000
```

---

## ❓ Troubleshooting

### "No ideas generated"
- Check input file exists
- Verify API keys are set (OPENAI_API_KEY, etc.)
- Try increasing `--count` parameter

### "Novelty score too low"
- Review prior art identified by G1
- Consider pivoting the approach
- Check if idea is truly novel vs. existing work

### "Benchmarks failing"
- Review failure modes in G3 output
- Check if baselines are correct
- Verify dataset is properly loaded

### "Pre-sales not converting"
- Review pricing (might be too high)
- Check if problem is urgent enough
- Validate solution actually solves the problem

---

## 📖 Next Steps

1. **Read the master doc**: `docs/IdeaForge_MIND_BENDING_DOCUMENTATION_SUITE.md`
2. **Study examples**: `docs/examples/Librex.QAP-example.md`
3. **Follow validation sprint**: `validation-framework/sprint-guide.md`
4. **Run your first pipeline**: Generate ideas → Validate → Build

---

## 🎯 Success Checklist

- [ ] Generated 50 ideas with IdeaForge
- [ ] Picked top 3 ideas
- [ ] Validated at least 1 idea (10+ pre-sales OR novelty >= 65)
- [ ] Built MVP (8 weeks)
- [ ] Passed benchmarks (G4)
- [ ] Deployed product or published paper

**When you hit all 6 checkboxes, you've successfully shipped!** 🚀

---

**Questions?** Check `docs/faq.md` or review examples in `docs/examples/`
