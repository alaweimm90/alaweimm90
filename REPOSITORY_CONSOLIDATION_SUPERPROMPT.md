# Repository Consolidation Superprompt

> **Purpose:** Complete context for AI agents to understand and work with the entire Alawein ecosystem
> **Created:** December 5, 2025
> **Status:** ACTIVE - Use this for all consolidation work

---

## Mission Statement

Consolidate the scattered monorepo into a clean multi-repo structure while **preserving ALL features** developed over weeks of work. Nothing gets deleted - everything gets properly organized.

---

## Current State Analysis

### What Happened (Refactor History)

1. **Original Structure:** Multiple organizations merged into single monorepo
2. **Dec 5, 2025 Refactor Started:**
   - Moved planning docs to `docs/planning/`
   - Moved `.ai/` → `.config/ai/`
   - Moved `.claude/` → `.config/claude/`
   - All project code moved to `.archive/organizations/`
3. **Problem:** Projects are now buried in archive, structure unclear

### Current Repository Layout

```
GitHub/                              # Root: alaweimm90/alaweimm90
├── .archive/                        # 47,805 archived files
│   └── organizations/               # ALL PROJECT CODE IS HERE
│       ├── AlaweinOS/               # Alawein Technologies products
│       │   ├── Attributa/           # Attribution analysis (React+Supabase)
│       │   ├── Foundry/             # Product incubator (14 subdirs)
│       │   ├── HELIOS/              # Autonomous research AI
│       │   ├── LLMWorks/            # LLM experimentation tools
│       │   ├── Librex/              # Optimization framework
│       │   ├── Librex.QAP/          # QAP solver (separate)
│       │   ├── MEZAN/               # Meta-solver orchestrator
│       │   │   ├── ATLAS/           # Atlas core (renamed to Orchex)
│       │   │   ├── Libria/          # Solver library (7 solvers)
│       │   │   └── MEZAN/           # MEZAN core
│       │   ├── QMLab/               # Quantum mechanics lab
│       │   ├── SimCore/             # Simulation framework
│       │   └── TalAI/               # AI research platform (50 modules!)
│       ├── alaweimm90-business/     # Business products
│       │   ├── LiveItIconic/        # Luxury e-commerce
│       │   ├── MarketingAutomation/ # Marketing tools
│       │   └── Repz/                # Fitness app (also at Desktop/REPZ)
│       ├── alaweimm90-science/      # Research projects
│       │   ├── MagLogic/            # Magnetic logic circuits
│       │   ├── QMatSim/             # Quantum material simulation
│       │   ├── QubeML/              # Quantum ML framework
│       │   ├── SciComp/             # Scientific computing
│       │   └── SpinCirc/            # Spintronics circuits
│       └── MeatheadPhysicist/       # Physics education (EMPTY?)
├── automation/                      # Python automation (ACTIVE - KEEP)
│   ├── prompts/                     # 53 prompts (9 system, 28 project, 16 tasks)
│   ├── agents/                      # DevOps agents
│   ├── workflows/                   # Workflow definitions
│   ├── deployment/                  # Deployment registry
│   └── orchestration/               # Multi-agent patterns
├── tools/                           # TypeScript toolkit (ACTIVE - KEEP)
│   ├── orchex/                      # ORCHEX CLI (76 files)
│   ├── ai/                          # AI orchestration (36 files)
│   ├── cli/                         # CLI tools
│   └── security/                    # Security scanning
├── docs/                            # Documentation
│   ├── pages/                       # GitHub Pages (landing pages)
│   │   ├── brands/                  # Product landing pages
│   │   │   ├── talai/
│   │   │   ├── librex/
│   │   │   ├── mezan/
│   │   │   └── repz/
│   │   └── personas/
│   ├── planning/                    # Planning docs (7 files)
│   └── developer/                   # Developer docs
├── business/                        # Business docs
├── tests/                           # Tests (25 items)
└── [config files]                   # package.json, tsconfig, etc.
```

---

## Complete Project Inventory

### Tier 1: Revenue-Ready Products (P0)

| Project | Location | Modules | Status | Domain |
|---------|----------|---------|--------|--------|
| **TalAI** | `.archive/organizations/AlaweinOS/TalAI/` | 50 | Development | talai.dev |
| **Repz** | `C:\Users\mesha\Desktop\REPZ` + archive | Full app | Development | getrepz.app |
| **Librex** | `.archive/organizations/AlaweinOS/Librex/` | 10 | Development | librex.dev |

### Tier 2: Platform Products (P1)

| Project | Location | Modules | Status | Domain |
|---------|----------|---------|--------|--------|
| **MEZAN** | `.archive/organizations/AlaweinOS/MEZAN/` | 30 | Development | - |
| **Attributa** | `.archive/organizations/AlaweinOS/Attributa/` | 6 | Development | attributa.dev |
| **LLMWorks** | `.archive/organizations/AlaweinOS/LLMWorks/` | 7 | Development | llmworks.dev |
| **LiveItIconic** | `.archive/organizations/alaweimm90-business/LiveItIconic/` | 13 | Development | liveiticonic.com |

### Tier 3: Research/Future (P2)

| Project | Location | Modules | Status |
|---------|----------|---------|--------|
| **SimCore** | `.archive/organizations/AlaweinOS/SimCore/` | - | Research |
| **QMLab** | `.archive/organizations/AlaweinOS/QMLab/` | - | Research |
| **HELIOS** | `.archive/organizations/AlaweinOS/HELIOS/` | 4 | Research |
| **Foundry** | `.archive/organizations/AlaweinOS/Foundry/` | 14 | Incubator |

### Tier 4: Science Projects (P3)

| Project | Location | Focus |
|---------|----------|-------|
| **MagLogic** | `.archive/organizations/alaweimm90-science/MagLogic/` | Magnetic logic circuits |
| **SpinCirc** | `.archive/organizations/alaweimm90-science/SpinCirc/` | Spintronics circuits |
| **QMatSim** | `.archive/organizations/alaweimm90-science/QMatSim/` | Quantum materials |
| **QubeML** | `.archive/organizations/alaweimm90-science/QubeML/` | Quantum ML |
| **SciComp** | `.archive/organizations/alaweimm90-science/SciComp/` | Scientific computing |

---

## TalAI Complete Module Inventory (50 Modules)

### Core Tools (Revenue-Ready)
| Module | Purpose | Priority |
|--------|---------|----------|
| `adversarial-review/` | AI peer review simulation | P0 |
| `lit-review-bot/` | Literature review automation | P0 |
| `grant-writer/` | Grant proposal assistance | P0 |
| `paper-miner/` | Paper data extraction | P0 |
| `hypothesis-match/` | Hypothesis generation | P1 |

### Research Tools
| Module | Purpose |
|--------|---------|
| `abstract-writer/` | Publication abstract generation |
| `active-learning/` | Active learning pipelines |
| `analytics/` | Research analytics dashboard |
| `causal-inference/` | Causal analysis tools |
| `citation-predictor/` | Citation impact prediction |
| `data-cleaner/` | Automated data cleaning |
| `experiment-designer/` | Optimal experiment design |
| `failure-db/` | Database of failed experiments |
| `ghost-researcher/` | Autonomous exploration agent |
| `idea-calculus/` | Idea combination engine |
| `ideaforge/` | Creative ideation tools |
| `knowledge-graph/` | Research knowledge graphs |
| `research-pricer/` | Research cost estimation |

### Prompt Engineering
| Module | Purpose |
|--------|---------|
| `prompt-marketplace/` | Prompt trading platform |
| `promptforge/` | Prompt engineering tools |
| `promptforge-lite/` | Lightweight prompt tools |

### Infrastructure
| Module | Purpose |
|--------|---------|
| `atlas-autonomous-research/` | Autonomous research agent |
| `atlas-orchestrator/` | Multi-agent orchestration |
| `buildforge/` | Build automation |
| `chaos-engine/` | Stress-testing research methods |
| `data-pipeline/` | Data processing pipelines |
| `edge/` | Edge deployment |
| `enterprise/` | Enterprise features |
| `federated/` | Federated learning |
| `k8s/` | Kubernetes configs |
| `monitoring/` | Monitoring & observability |
| `nginx/` | Web server configs |
| `performance/` | Performance optimization |
| `validation/` | Research validation |

### Domain-Specific
| Module | Purpose |
|--------|---------|
| `materials-science/` | Materials research tools |
| `meta-science/` | Meta-science analysis |
| `multimodal/` | Multimodal AI tools |
| `neuroscience/` | Neuroscience tools |
| `synthetic-biology/` | Synthetic bio tools |
| `turing-features/` | AI testing (8 sub-modules) |
| `turingo/` | Turing test tools |

### Other
| Module | Purpose |
|--------|---------|
| `alaweinos/` | AlaweinOS integration |
| `brand/` | Branding assets |
| `consortia/` | Research consortia tools |
| `docs/` | Documentation |
| `examples/` | Example code |
| `PEDs-Playbook/` | Research performance guide |
| `src/` | Core source code |
| `tests/` | Test suites |

---

## MEZAN/Libria Solver Inventory

### MEZAN Core
| Component | Purpose |
|-----------|---------|
| `ATLAS/` | Orchestration core (→ Orchex) |
| `MEZAN/` | Meta-solver core |
| `core/` | Shared core utilities |
| `visualization/` | Visualization tools |

### Libria Solvers (7 Solvers)
| Solver | Problem Domain |
|--------|----------------|
| `libria-qap/` | Quadratic Assignment Problem |
| `libria-flow/` | Network Flow |
| `libria-alloc/` | Resource Allocation |
| `libria-evo/` | Evolutionary Optimization |
| `libria-graph/` | Graph Optimization |
| `libria-dual/` | Dual Problems |
| `libria-meta/` | Meta-optimization |

---

## Active Infrastructure (DO NOT ARCHIVE)

### automation/ (Python)
```
automation/
├── prompts/
│   ├── system/              # 9 system prompts
│   ├── project/             # 28 project superprompts
│   │   ├── TALAI_SUPERPROMPT.md
│   │   ├── REPZ_SUPERPROMPT.md
│   │   ├── SIMCORE_CLAUDE_CODE_SUPERPROMPT.md
│   │   ├── MAG_LOGIC_SUPERPROMPT.md
│   │   ├── SPIN_CIRC_SUPERPROMPT.md
│   │   ├── QMAT_SIM_SUPERPROMPT.md
│   │   ├── QUBE_ML_SUPERPROMPT.md
│   │   ├── SCI_COMP_SUPERPROMPT.md
│   │   ├── LOVABLE_FULLSTACK_TEMPLATE_SYSTEM.md (81KB!)
│   │   └── ... (20 more)
│   └── tasks/               # 16 task prompts
├── agents/devops/           # 6 DevOps agent configs
├── workflows/               # Workflow definitions
├── deployment/              # 16 deployment configs
├── orchestration/           # Multi-agent patterns
├── cli.py                   # Python CLI
├── executor.py              # Workflow executor
└── validation.py            # Asset validation
```

### tools/orchex/ (TypeScript)
```
tools/orchex/
├── cli/                     # 11 CLI modules
├── orchestration/           # 11 orchestration modules
├── analysis/                # 7 analysis modules
├── services/                # 8 service modules
├── adapters/                # 5 adapters
├── api/                     # 5 API modules
├── storage/                 # 6 storage modules
├── refactoring/             # 4 refactoring modules
├── core/                    # 3 core modules
├── agents/                  # 2 agent modules
├── integrations/            # 3 integrations
├── config/                  # 3 config files
├── types/                   # Type definitions
├── utils/                   # Utilities
└── index.ts                 # Entry point
```

---

## Existing Superprompts (28 Total)

### Project Superprompts
1. `TALAI_SUPERPROMPT.md` - TalAI platform
2. `REPZ_SUPERPROMPT.md` - Repz fitness app
3. `SIMCORE_CLAUDE_CODE_SUPERPROMPT.md` - SimCore simulation
4. `MAG_LOGIC_SUPERPROMPT.md` - Magnetic logic
5. `SPIN_CIRC_SUPERPROMPT.md` - Spintronics
6. `QMAT_SIM_SUPERPROMPT.md` - Quantum materials
7. `QUBE_ML_SUPERPROMPT.md` - Quantum ML
8. `SCI_COMP_SUPERPROMPT.md` - Scientific computing

### Architecture Superprompts
9. `MONOREPO_ARCHITECTURE_SUPERPROMPT.md`
10. `PLATFORM_DEPLOYMENT_SUPERPROMPT.md`
11. `GOVERNANCE_COMPLIANCE_SUPERPROMPT.md`
12. `SECURITY_CYBERSECURITY_SUPERPROMPT.md`

### Development Superprompts
13. `AI_ML_INTEGRATION_SUPERPROMPT.md`
14. `CICD_PIPELINE_SUPERPROMPT.md`
15. `TESTING_QA_SUPERPROMPT.md`
16. `UI_UX_DESIGN_SUPERPROMPT.md`
17. `GATING_APPROVAL_SUPERPROMPT.md`
18. `PROMPT_OPTIMIZATION_SUPERPROMPT.md`

### Template Superprompts
19. `LOVABLE_FULLSTACK_TEMPLATE_SYSTEM.md` (81KB - comprehensive!)
20. `LOVABLE_TEMPLATE_SUPERPROMPT.md` (52KB)
21. `KILO_CONSOLIDATION_SUPERPROMPT.md`
22. `LOCAL_AI_ORCHESTRATION_SUPERPROMPT.md`
23. `ENTERPRISE_AGENTIC_AI_SUPERPROMPT.md`

### Technical Superprompts
24. `api-development.md`
25. `automation-ts-implementation.md`
26. `data-engineering-pipeline.md`
27. `ml-pipeline-development.md`
28. `session-summary-2024-11-30.md`

---

## Domain Inventory

### Owned Domains (11)
| Domain | Product | Status |
|--------|---------|--------|
| getrepz.app | Repz | ✅ Active |
| attributa.dev | Attributa | ✅ Active |
| simcore.dev | SimCore | ✅ Owned |
| qmlab.online | QMLab | ✅ Owned |
| llmworks.dev | LLMWorks | ✅ Owned |
| malawein.com | Portfolio | 🔴 Setup needed |
| malawein.info | Redirect | 🔴 Configure |
| meshal.ai | Professional | 🔴 Setup needed |
| repzapp.com | Redirect | 🔴 Configure |
| repzcoach.com | Repz Coach | 🔴 Future |
| aiclarity.com | (Unused) | Consider selling |

### Needed Domains (3)
| Domain | Product | Priority |
|--------|---------|----------|
| talai.dev | TalAI | P0 |
| librex.dev | Librex | P0 |
| alawein.tech | Parent company | P0 |

---

## Consolidation Rules

### NEVER DELETE
1. Any code in `.archive/organizations/`
2. Any file in `automation/`
3. Any file in `tools/`
4. Any superprompt in `automation/prompts/`
5. Any landing page in `docs/pages/`

### SAFE TO CONSOLIDATE
1. Duplicate configs across projects
2. Redundant documentation
3. Empty placeholder directories
4. Build artifacts (dist/, node_modules/)

### MIGRATION PATTERN
```
1. Identify source in .archive/organizations/
2. Create target repo or folder
3. COPY (not move) files
4. Verify functionality
5. Update references
6. Only then consider cleanup
```

---

## Recommended Final Structure

### Option A: Multi-Repo (Recommended)
```
GitHub Organizations:
├── alaweimm90/alaweimm90     # Hub: automation, tools, docs
├── alaweimm90/talai          # TalAI platform
├── alaweimm90/librex         # Librex framework
├── alaweimm90/repz           # Already at Desktop/REPZ
├── alaweimm90/mezan          # MEZAN orchestrator
└── alaweimm90/[others]       # As needed
```

### Option B: Organized Monorepo
```
GitHub/
├── products/                 # All products (restored from archive)
│   ├── talai/
│   ├── librex/
│   ├── mezan/
│   ├── attributa/
│   └── ...
├── research/                 # Science projects
│   ├── maglogic/
│   ├── spincirc/
│   └── ...
├── automation/               # Keep as-is
├── tools/                    # Keep as-is
└── docs/                     # Keep as-is
```

---

## Action Items for AI Agents

### Immediate (Before Any Work)
1. Read this superprompt completely
2. Check `.archive/organizations/` for source files
3. Never delete without explicit confirmation
4. Preserve all 50 TalAI modules
5. Preserve all 7 Libria solvers

### When Consolidating
1. Create codemaps for each project
2. Document dependencies between modules
3. Identify shared code that can be extracted
4. Maintain test coverage
5. Update superprompts as needed

### When Deploying
1. Use existing landing pages in `docs/pages/brands/`
2. Reference deployment configs in `automation/deployment/`
3. Follow patterns in existing superprompts
4. Test on staging before production

---

## Contact & Ownership

- **Owner:** Meshal Alawein
- **GitHub:** @alaweimm90
- **Email:** meshal@berkeley.edu
- **LLCs:** Alawein Technologies LLC, REPZ LLC, Live It Iconic LLC

---

_Last Updated: December 5, 2025_
_Version: 1.0_
