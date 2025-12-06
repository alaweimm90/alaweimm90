# Final Repository Structure

> **Decision:** Organized monorepo with GitHub Pages  
> **Date:** December 5, 2025  
> **Status:** PROPOSED - Awaiting approval

---

## Philosophy

- **ALL projects preserved** - nothing deleted
- **Organized structure** - projects restored from archive
- **GitHub Pages** for landing pages & docs
- **Shared automation** stays active
- **No orphaned code** - everything properly organized

---

## Final Structure

### 1. Hub Repository: `alawein` (this repo)

**Purpose:** Landing pages, automation, shared tooling

```
GitHub/                              # alawein/alawein
├── .github/                         # GitHub config, workflows
├── automation/                      # Python automation system (KEEP)
│   ├── prompts/                     # 49 prompts
│   ├── agents/                      # 24 agents
│   ├── workflows/                   # 11 workflows
│   └── deployment/                  # Deployment registry
├── tools/                           # Shared CLI tools (KEEP)
│   ├── orchex/                      # Orchestration CLI
│   └── ai/                          # AI tools
├── docs/                            # GitHub Pages source
│   └── pages/                       # Landing pages (served at alawein.github.io)
│       ├── index.html               # Main hub page
│       └── brands/                  # Product landing pages
│           ├── talai/               # talai.dev landing
│           ├── librex/              # librex.dev landing
│           ├── mezan/               # mezan landing
│           └── repz/                # getrepz.app landing
├── templates/                       # Project templates (NEW)
│   ├── nextjs-saas/                 # SaaS starter
│   ├── python-api/                  # FastAPI starter
│   └── landing-page/                # Static landing page
├── .archive/                        # Historical reference only
├── README.md
├── CLAUDE.md
└── package.json
```

**What to DELETE from root:**
- All planning docs (move to `docs/planning/` or delete)
- Duplicate configs
- Old status files

---

### 2. Product Repositories (Separate Repos)

Each product gets its own repo on GitHub:

#### Active Products (Create Now)

| Repo Name | Domain | Source Location | Priority |
|-----------|--------|-----------------|----------|
| `talai` | talai.dev | `.archive/organizations/AlaweinOS/TalAI/` | P0 |
| `librex` | librex.dev | `.archive/organizations/AlaweinOS/Librex/` | P1 |
| `repz` | getrepz.app | `C:\Users\mesha\Desktop\REPZ` (already exists!) | P0 |

#### Future Products (Create When Ready)

| Repo Name | Domain | Source Location | Priority |
|-----------|--------|-----------------|----------|
| `mezan` | - | `.archive/organizations/AlaweinOS/MEZAN/` | P2 |
| `simcore` | simcore.dev | `.archive/organizations/AlaweinOS/SimCore/` | P2 |
| `qmlab` | qmlab.online | `.archive/organizations/AlaweinOS/QMLab/` | P2 |
| `attributa` | attributa.dev | `.archive/organizations/AlaweinOS/Attributa/` | P2 |
| `liveiticonic` | liveiticonic.com | `.archive/organizations/alawein-business/LiveItIconic/` | P3 |

#### Research/Personal (Keep Archived)

| Project | Status | Notes |
|---------|--------|-------|
| MagLogic, SpinCirc, QMatSim, QubeML, SciComp | Archive | Activate when research resumes |
| MeatheadPhysicist | Archive | Physics education content |
| Portfolio, DrMalawein, Rounaq | `.personal/` | Personal sites |

---

### 3. TalAI Repo Structure (Priority P0)

**Repo:** `github.com/alawein/talai`

```
talai/
├── apps/                            # Deployable applications
│   ├── web/                         # Main web app (Next.js)
│   ├── api/                         # FastAPI backend
│   └── docs/                        # Documentation site
├── packages/                        # Shared packages
│   ├── core/                        # Core library
│   ├── ui/                          # UI components
│   └── types/                       # TypeScript types
├── tools/                           # TalAI-specific tools
│   ├── adversarial-review/          # ⭐ PRIORITY - Revenue ready
│   ├── lit-review-bot/              # Literature review
│   ├── grant-writer/                # Grant writing
│   ├── hypothesis-match/            # Hypothesis generation
│   └── paper-miner/                 # Paper extraction
├── research/                        # Research modules (not deployed)
│   ├── chaos-engine/
│   ├── ghost-researcher/
│   ├── idea-calculus/
│   └── ...
├── .github/
├── package.json                     # Turborepo config
├── turbo.json
└── README.md
```

**Modules to consolidate from archive:**
- `adversarial-review/` → `tools/adversarial-review/`
- `lit-review-bot/` → `tools/lit-review-bot/`
- `grant-writer/` → `tools/grant-writer/`
- `promptforge/` → `packages/promptforge/`
- All others → `research/` or delete if duplicate

---

### 4. Librex Repo Structure (Priority P1)

**Repo:** `github.com/alawein/librex`

```
librex/
├── core/                            # Core optimization library
├── solvers/                         # Solver implementations
│   ├── qap/                         # Quadratic Assignment
│   ├── flow/                        # Network Flow
│   └── alloc/                       # Resource Allocation
├── bindings/                        # Language bindings
│   ├── python/
│   └── javascript/
├── examples/
├── docs/
└── README.md
```

---

### 5. Domain Mapping

| Domain | Points To | Type |
|--------|-----------|------|
| `alawein.github.io` | Hub landing pages | GitHub Pages |
| `talai.dev` | TalAI web app | Vercel |
| `librex.dev` | Librex docs | GitHub Pages |
| `getrepz.app` | Repz app | Vercel |
| `malawein.com` | Personal portfolio | Vercel |

---

## Migration Plan

### Phase 1: Clean Hub (Today)

1. ✅ Keep `automation/`, `tools/`, `docs/pages/`
2. Delete or archive everything else at root
3. Update `docs/pages/` to be the main content

### Phase 2: Create TalAI Repo (This Week)

1. Create `github.com/alawein/talai`
2. Copy from `.archive/organizations/AlaweinOS/TalAI/`
3. Consolidate 50 modules into clean structure
4. Deploy to Vercel

### Phase 3: Create Librex Repo (Next Week)

1. Create `github.com/alawein/librex`
2. Copy from `.archive/organizations/AlaweinOS/Librex/`
3. Merge `Librex.QAP` into main repo

### Phase 4: Cleanup (After Migration)

1. Delete `.archive/` contents (after verifying migration)
2. Simplify hub repo
3. Update all documentation

---

## What Gets Deleted

### From Root (Safe to Delete)
- `REFACTOR_STATUS.md` - temporary
- `START_HERE.md` - move to docs
- `projects/` - just a README, info in docs
- `business/` - move to docs or separate repo
- `tests/` - move to respective product repos
- `scripts/` - move to tools/

### From Archive (After Migration)
- Everything in `.archive/organizations/` after copying to new repos

---

## Approval Checklist

- [ ] Agree with multi-repo approach (not monorepo)
- [ ] Agree with TalAI as P0 priority
- [ ] Agree with keeping automation in hub
- [ ] Agree with GitHub Pages for landing pages
- [ ] Ready to start migration

---

**Next Step:** Approve this structure, then I'll start Phase 1 cleanup.
