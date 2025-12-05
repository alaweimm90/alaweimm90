# Infrastructure Decision Framework

**Purpose:** Systematic mapping of products → GitHub → Websites → Delivery  
**Last Updated:** December 5, 2025

---

## Part 1: MEZAN / Libria / QAP Architecture (Your Questions Answered)

### Q: Is "Libria" used within MEZAN?

**YES.** Here's the hierarchy:

```
MEZAN (Meta-Equilibrium Zero-regret Assignment Network)
│
├── = META-SOLVER that routes to domain-specific Libria solvers
│
└── Libria Solvers (the "worker" algorithms):
    ├── Librex.QAP      ← Quadratic Assignment Problem (YOUR FOCUS)
    ├── Librex.Flow     ← Network flow/routing
    ├── Librex.Alloc    ← Resource allocation
    ├── Librex.Graph    ← Graph topology optimization
    ├── Librex.Evo      ← Evolutionary optimization (MAP-Elites)
    ├── Librex.Dual     ← Dual decomposition
    └── Librex.Meta     ← Tournament-based solver selection (Swiss system)
```

### Q: What's MEZAN's novelty?

**MEZAN's Core Innovation:**

```
┌─────────────────────────────────────────────┐
│            MEZAN ENGINE                     │
│   Dual-solver balancing with bandit-style   │
│   adaptive trust allocation                 │
├─────────────────────────────────────────────┤
│                                             │
│    Solver_L          ←→        Solver_R     │
│  (continuous)                  (discrete)   │
│       │                            │        │
│       └────── BALANCING ───────────┘        │
│              (UCB, Thompson, ε-greedy)      │
│                                             │
│   "Same problem, two approaches,            │
│    learns which to trust per-context"       │
│                                             │
│   = 1664× speedup claim comes from          │
│     smart routing + layer optimization      │
└─────────────────────────────────────────────┘
```

### Q: Would this be a CLI tool with "split" or "2-sided" optimization?

**YES.** MEZAN + Libria could be exposed as CLI:

```bash
# MEZAN routes to the best solver automatically
orchex solve --problem qap --input data.json --method auto

# Or force a specific Libria solver
orchex solve --solver Librex.QAP --input data.json
```

### Q: Focus on QAP first, name it separately?

**RECOMMENDED:** Ship Librex.QAP as a standalone tool first.

| Name           | Pros                                     | Cons                                |
| -------------- | ---------------------------------------- | ----------------------------------- |
| **Librex.QAP** | Clear what it does (QAP + Libria family) | Ties to Libria name                 |
| **Librex-QAP** | Uses your preferred Librex name          | Compound name                       |
| **Librex**     | Clean, short                             | Too generic (need domain qualifier) |
| **QAPSolve**   | Generic, searchable                      | Boring                              |

**My recommendation:** Keep **Librex.QAP** as the technical name, release under the **Librex** brand:

```
Librex™ QAP Solver (Librex.QAP)
```

---

## Part 2: Trademark Check - Libria vs Librex

| Name       | Status                | Conflicts                                                          |
| ---------- | --------------------- | ------------------------------------------------------------------ |
| **Libria** | ⚠️ POTENTIAL CONFLICT | "Libria" is the fictional city from the movie "Equilibrium" (2002) |
| **Librex** | ✅ APPEARS SAFE       | No significant trademark conflicts found                           |

**Recommendation:** Use **Librex** as the brand, avoid **Libria**.

---

## Part 3: GitHub Organization Strategy

### Current State (4 Orgs)

```
organizations/
├── AlaweinOS/           ← Main tech products
│   ├── Librex       (rename to Librex?)
│   ├── MEZAN
│   ├── TalAI
│   ├── HELIOS
│   ├── SimCore
│   ├── Foundry
│   ├── Librex.QAP
│   └── ...
│
├── MeatheadPhysicist/   ← Physics persona/brand
│
├── alaweimm90-science/  ← Needs rename (scientific tools)
│   ├── MagLogic
│   ├── QMatSim
│   ├── QubeML
│   ├── SpinCirc
│   └── SciComp
│
└── alaweimm90-business/ ← Needs rename (consumer products)
    ├── Repz
    ├── LiveItIconic
    └── MarketingAutomation
```

### Recommended Consolidation

**Option A: Personal Account + Pages (SIMPLEST)**

```
github.com/alawein           ← All repos under personal account
├── alawein.github.io        ← GitHub Pages for landing pages
├── librex                   ← Optimization framework (fka Librex)
├── mezan                    ← Meta-solver
├── talai                    ← AI research tools
├── helios                   ← Research platform
├── orchex                   ← Automation CLI
├── repz                     ← Fitness app
└── [scientific repos...]
```

**Pros:** No org management, simple, free  
**Cons:** Less "professional" appearance

**Option B: One Organization (RECOMMENDED)**

```
github.com/alawein-labs      ← Single org for all products
├── librex
├── mezan
├── talai
├── helios
├── orchex
├── repz
└── [scientific repos...]

github.com/alawein           ← Personal profile points to org
```

**Pros:** Professional, unified branding  
**Cons:** Need to maintain org

**Option C: Keep Separate Orgs (OVERHEAD)**

```
github.com/alawein-labs     ← Tech products
github.com/alawein-science  ← Scientific tools (if separate brand)
github.com/repz-fitness     ← Consumer app (later, when 10K users)
```

**My recommendation:** **Option B** - Single org (alawein-labs) with everything.

---

## Part 4: Product → Infrastructure Decision Matrix

### Revenue Products (Priority)

| Product                     | GitHub Location       | Website Type                  | Delivery         | Priority | Notes                    |
| --------------------------- | --------------------- | ----------------------------- | ---------------- | -------- | ------------------------ |
| **TalAI AdversarialReview** | `alawein-labs/talai`  | SaaS Platform (talai.dev)     | Web App          | 🔥 P0    | Ship first               |
| **Librex QAP**              | `alawein-labs/librex` | Docs + Landing (GitHub Pages) | CLI + Python lib | P1       | Research paper potential |
| **MEZAN**                   | `alawein-labs/mezan`  | Docs (GitHub Pages)           | Python lib + API | P2       | After Librex proven      |
| **Repz**                    | `alawein-labs/repz`   | Mobile App (App Store)        | iOS App          | P4       | Defer to 10K users       |

### Research/Platform Products

| Product     | GitHub Location        | Website Type        | Delivery          | Priority | Notes           |
| ----------- | ---------------------- | ------------------- | ----------------- | -------- | --------------- |
| **HELIOS**  | `alawein-labs/helios`  | None (README only)  | Research platform | P3       | Internal first  |
| **Foundry** | `alawein-labs/foundry` | None (internal)     | Idea incubator    | P5       | Not user-facing |
| **Orchex**  | `alawein-labs/orchex`  | Docs (GitHub Pages) | CLI               | P2       | Dev tool        |

### Scientific Tools (Lower Priority)

| Product  | GitHub Location         | Website Type | Delivery   | Priority |
| -------- | ----------------------- | ------------ | ---------- | -------- |
| MagLogic | `alawein-labs/maglogic` | README only  | Python lib | P5       |
| QMatSim  | `alawein-labs/qmatsim`  | README only  | Python lib | P5       |
| QubeML   | `alawein-labs/qubeml`   | README only  | Python lib | P5       |
| SpinCirc | `alawein-labs/spincirc` | README only  | Python lib | P5       |
| SciComp  | `alawein-labs/scicomp`  | README only  | Python lib | P5       |

---

## Part 5: Website Types Explained

| Type                      | What It Is                   | Cost           | When to Use                     |
| ------------------------- | ---------------------------- | -------------- | ------------------------------- |
| **None (README only)**    | Just GitHub README           | Free           | Internal tools, libraries       |
| **GitHub Pages**          | Static site from repo        | Free           | Docs, landing pages, portfolios |
| **Custom Domain + Pages** | GitHub Pages + custom domain | ~$15/yr        | Products with branding          |
| **SaaS Platform**         | Full web application         | Hosting costs  | Revenue products (TalAI)        |
| **Mobile App**            | iOS/Android app              | $99/yr (Apple) | Consumer apps (Repz)            |

### Products That Need Custom Domains

| Domain         | Purpose                   | Type           | Priority |
| -------------- | ------------------------- | -------------- | -------- |
| `alawein.tech` | Parent company landing    | Static + Links | P1       |
| `talai.dev`    | TalAI SaaS platform       | Full web app   | 🔥 P0    |
| `orchex.dev`   | Orchex CLI docs           | GitHub Pages   | P2       |
| `librex.dev`   | Librex optimization suite | GitHub Pages   | P2       |
| `repz.app`     | Repz mobile app landing   | Static         | P4       |

---

## Part 6: Template Gap Analysis

### What You Have ✅

| Template              | Location                                                     | Purpose           |
| --------------------- | ------------------------------------------------------------ | ----------------- |
| Profile README        | `.metaHub/templates/profiles/`                               | GitHub profile    |
| Org README            | `.metaHub/templates/organizations/`                          | Org landing pages |
| Consumer Repo         | `.metaHub/templates/consumer-repos/`                         | Individual repos  |
| Lovable Design System | `automation/prompts/project/LOVABLE_TEMPLATE_SUPERPROMPT.md` | UI/UX tokens      |
| Repo Structure        | `.metaHub/templates/structures/portfolio-structure.yaml`     | Folder structures |

### What's Missing ❌

| Template Needed     | Purpose                          | Priority |
| ------------------- | -------------------------------- | -------- |
| **Backend Service** | FastAPI/Python backend structure | P1       |
| **SaaS Web App**    | React + Next.js + Auth + Billing | P0       |
| **iOS App**         | SwiftUI app structure            | P3       |
| **CLI Tool**        | Click/Typer Python CLI structure | P1       |
| **Python Library**  | PyPI-publishable package         | P1       |

### Templates to Extract from Lovable

Based on `LOVABLE_TEMPLATE_SUPERPROMPT.md`, extract:

1. **SaaS Dashboard Template** → `templates/saas-dashboard/`
2. **Landing Page Template** → `templates/landing-page/`
3. **Documentation Site** → `templates/docs-site/`
4. **Admin Panel** → `templates/admin-panel/`

---

## Part 7: Recommended Action Plan

### Phase 1: Consolidate (Today)

1. **Create** `alawein-labs` GitHub organization
2. **Move** repos from AlaweinOS, alaweimm90-science, alaweimm90-business
3. **Rename** Librex → Librex
4. **Archive** MeatheadPhysicist (keep as redirect)

### Phase 2: Domain Setup (This Week)

1. Register `alawein.tech` → parent company
2. Register `talai.dev` → SaaS product
3. Register `librex.dev` → optimization suite
4. Set up GitHub Pages for docs

### Phase 3: Templates (This Week)

1. Create backend template (FastAPI + Docker)
2. Create CLI template (Typer + PyPI)
3. Create SaaS template (Next.js + Supabase + Stripe)

### Phase 4: Ship (This Month)

1. Deploy TalAI AdversarialReview to `talai.dev`
2. Publish Librex.QAP to PyPI under Librex brand
3. Set up `alawein.tech` landing page

---

## Part 8: Answers to Your Specific Questions

### Q1: What should github.com/alawein showcase?

**Pinned repositories (top 6):**

1. `alawein-labs/talai` - AI research tools
2. `alawein-labs/librex` - Optimization framework
3. `alawein-labs/mezan` - Meta-solver
4. `alawein-labs/orchex` - Automation CLI
5. `alawein` - This meta-governance repo (renamed)
6. One scientific tool (e.g., `qubeml`)

### Q2: How many GitHub organizations do you need?

**Answer: 1 (or 0)**

- **Simplest:** Everything under `github.com/alawein` (personal account)
- **Professional:** Everything under `github.com/alawein-labs` (one org)

Don't create multiple orgs. It's overhead with no benefit at your current stage.

### Q3: Which products need full websites?

| Product          | Website?     | Type                 |
| ---------------- | ------------ | -------------------- |
| TalAI            | ✅ YES       | Full SaaS platform   |
| Librex/MEZAN     | ⚠️ Docs only | GitHub Pages         |
| Orchex           | ⚠️ Docs only | GitHub Pages         |
| Scientific tools | ❌ NO        | README only          |
| Repz             | ✅ Later     | Mobile app + landing |

### Q4: Is Repz the only mobile app?

**Yes.** Everything else is:

- CLI tools (Orchex, Librex)
- Python libraries (MEZAN, scientific tools)
- Web apps (TalAI)

### Q5: Should we use flat repo + pages?

**Yes, for now.** Use:

- Personal account OR single org
- GitHub Pages for docs/landing pages
- Deploy SaaS apps (TalAI) to Vercel/Railway

---

## Summary Decision Tree

```
                    ┌─────────────────┐
                    │  Is it a SaaS   │
                    │    product?     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
          YES ▼                         NO  ▼
    ┌─────────────────┐           ┌─────────────────┐
    │ Full domain +   │           │  Is it user-    │
    │ web app hosting │           │    facing?      │
    │ (talai.dev)     │           └────────┬────────┘
    └─────────────────┘                    │
                                  ┌────────┴────────┐
                                  │                 │
                              YES ▼             NO  ▼
                        ┌─────────────────┐  ┌─────────────────┐
                        │ GitHub Pages +  │  │ README only     │
                        │ custom domain   │  │ (no website)    │
                        │ (librex.dev)    │  └─────────────────┘
                        └─────────────────┘
```

---

_This framework should be referenced alongside [MASTER_PLAN.md](./MASTER_PLAN.md) for complete context._
