# GitHub Repository Structure

**Owner:** Meshal Alawein (PhD Physics, UC Berkeley)
**Last Updated:** December 5, 2025
**Status:** ✅ Restructured & Ready to Ship

> **📋 For comprehensive planning, checklists, and business details, see [MASTER_PLAN.md](./MASTER_PLAN.md)**

---

## Legal Entities (Planned)

```text
ALAWEIN TECHNOLOGIES LLC (California)   REPZ LLC (Delaware - when 10K users)
├── Librex (optimization)           └── Repz (AI Fitness)
├── MEZAN (meta-solver)
├── Orchex (automation, fka Atlas)
├── TalAI (25+ products) ← SHIP FIRST
├── HELIOS (research platform)
├── Foundry (innovation, fka CrazyIdeas)
└── SciLab (5 physics tools)
```

---

## GitHub Organizations (Current → Planned)

| Current               | Planned                | Status         |
| --------------------- | ---------------------- | -------------- |
| `AlaweinOS`           | `AlaweinLabs`          | Rename pending |
| `alaweimm90-science`  | Merge into AlaweinLabs | Pending        |
| `alaweimm90-business` | Keep for Repz          | Active         |
| `MeatheadPhysicist`   | Archive                | Pending        |

---

## Root Directory Structure

```text
GitHub/
├── .archive/               # Historical files (47,805+ files preserved)
│   └── organizations/      # ⚠️ ARCHIVED - All project code preserved here
│       ├── AlaweinOS/      # Librex, MEZAN, TalAI, HELIOS, SimCore, Foundry
│       ├── alawein-science/# MagLogic, QMatSim, QubeML, SpinCirc, SciComp
│       ├── alawein-business/# Repz
│       └── MeatheadPhysicist/# Quantum research
│
├── .personal/              # Personal projects (portfolio, drmalawein, rounaq)
│
├── automation/             # AI orchestration system
│   ├── prompts/            # 49 prompts
│   ├── agents/             # 24 agents
│   ├── workflows/          # 11 workflows
│   └── orchestration/      # Anthropic patterns
│
├── tools/
│   └── orchex/             # Automation CLI (fka atlas)
├── scripts/                # Build/deploy scripts
├── projects/               # Project registry (85+ projects documented)
├── business/               # LLC & strategy docs
├── docs/                   # Documentation
│   └── pages/              # GitHub Pages (LLC landing pages)
└── .ai/                    # AI orchestration hub
```

> **Note:** The `organizations/` folder has been archived to `.archive/organizations/` as of December 5, 2025. All 47,805+ files are preserved and accessible.

---

## Key Products by Revenue Potential

| Tier | Product                 | Est. Revenue  | Status          |
| ---- | ----------------------- | ------------- | --------------- |
| 🥇   | TalAI AdversarialReview | $79/mo        | Ready to launch |
| 🥇   | Librex Enterprise       | $10K+/license | Beta            |
| 🥈   | TalAI GrantWriter       | $199/mo       | Ready           |
| 🥈   | HELIOS                  | Enterprise    | Alpha           |
| 🥉   | Repz                    | $9.99/mo      | Development     |

---

## Tech Stack

- **Languages:** Python (core), TypeScript (web)
- **Frameworks:** FastAPI, Next.js, React
- **Infrastructure:** Docker, Kubernetes, Terraform
- **AI/ML:** PyTorch, JAX, LangChain
- **Databases:** PostgreSQL, Supabase

---

## Automation CLI (Orchex)

```bash
# Python CLI
orchex prompts list
orchex agents list
orchex workflows list
orchex route "task description"

# TypeScript CLI (automation-ts/)
npx automation deploy list
npx automation validate
```

---

## Immediate Priorities

1. [ ] 🔥 Ship TalAI AdversarialReview
2. [ ] File Alawein Technologies LLC (California)
3. [ ] Get EIN (free, instant)
4. [ ] Register domains: alawein.tech, Librex.dev, talai.dev
5. [ ] Open business bank account

---

## Recent Changes (Dec 5, 2025)

- ✅ Renamed Optilibria → Librex
- ✅ Renamed Atlas → Orchex
- ✅ Renamed CrazyIdeas → Foundry
- ✅ Archived governance docs
- ✅ Cleaned folder structure

---

## Contact

- **Email:** `meshal@berkeley.edu`
- **GitHub:** [@alawein](https://github.com/alawein)
