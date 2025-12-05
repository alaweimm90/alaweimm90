# Execution Plan

> **Created:** December 5, 2025  
> **Status:** Ready for Execution  
> **Total Projects:** 85+ (including sub-modules)

---

## Phase 1: Immediate Actions (This Week)

### 1.1 Domain Registration (Priority: P0)

| Domain           | Purpose                | Registrar | Est. Cost |
| ---------------- | ---------------------- | --------- | --------- |
| **librex.dev**   | Optimization framework | Namecheap | ~$15/yr   |
| **talai.dev**    | AI research platform   | Namecheap | ~$15/yr   |
| **alawein.tech** | Parent company landing | Namecheap | ~$12/yr   |

**Commands:**

```bash
# After registration, configure DNS:
# A Record: @ → GitHub Pages IP (185.199.108.153)
# CNAME: www → alawein.github.io
```

### 1.2 LLC Formation (Priority: P0)

| LLC                          | State      | Filing Fee | Timeline  |
| ---------------------------- | ---------- | ---------- | --------- |
| **Alawein Technologies LLC** | California | $70        | 1-2 weeks |

**Steps:**

1. File Articles of Organization with CA Secretary of State
2. Get EIN from IRS (free, instant online)
3. Open business bank account
4. Register domains under LLC

---

## Phase 2: Project Activation Priority

### Tier 1: Launch Ready (Domains Owned, Code Exists)

| Project       | Domain           | LLC          | Action           |
| ------------- | ---------------- | ------------ | ---------------- |
| **Repz**      | getrepz.app ✅   | REPZ LLC     | Deploy to Vercel |
| **Attributa** | attributa.dev ✅ | Alawein Tech | Deploy to Vercel |

**Activation Commands:**

```powershell
# Activate Repz
Copy-Item -Recurse ".archive/organizations/alaweimm90-business/Repz" "C:/projects/repz"
cd C:/projects/repz
git init
git remote add origin git@github.com:alawein/repz.git
npm install
vercel deploy --prod
```

### Tier 2: Domain Needed (Code Exists)

| Project    | Needed Domain | LLC          | Priority |
| ---------- | ------------- | ------------ | -------- |
| **TalAI**  | talai.dev     | Alawein Tech | High     |
| **Librex** | librex.dev    | Alawein Tech | High     |
| **MEZAN**  | mezan.dev     | Alawein Tech | Medium   |

### Tier 3: Research Phase

| Project  | Domain          | Status   |
| -------- | --------------- | -------- |
| SimCore  | simcore.dev ✅  | Research |
| QMLab    | qmlab.online ✅ | Research |
| LLMWorks | llmworks.dev ✅ | Research |

---

## Phase 3: GitHub Pages Deployment

### 3.1 Enable GitHub Pages

```bash
# In GitHub repo settings:
# Settings → Pages → Source: Deploy from branch
# Branch: main, Folder: /docs/pages
```

### 3.2 Custom Domain Setup

```bash
# After enabling Pages, add CNAME:
echo "alawein.tech" > docs/pages/CNAME
git add docs/pages/CNAME
git commit -m "chore: add custom domain for GitHub Pages"
git push
```

### 3.3 Pages Structure Created

```
docs/pages/
├── index.html                    # Main landing (3 LLCs)
├── CNAME                         # alawein.tech
├── brands/
│   ├── librex/index.html        # Librex framework
│   ├── talai/index.html         # TalAI platform
│   ├── mezan/index.html         # MEZAN orchestrator
│   └── repz/index.html          # Repz fitness
└── personas/
    └── meathead-physicist/index.html
```

---

## Phase 4: Commit & Push

### 4.1 Stage All Changes

```powershell
git add -A
```

### 4.2 Create Comprehensive Commit

```powershell
git commit -m "feat: major repository restructure and brand consolidation

BREAKING CHANGES:
- Archived organizations/ to .archive/organizations/ (47,805 files)
- Renamed Equilibria → Librex (optimization framework)
- Updated username alaweimm90 → alawein

NEW STRUCTURE:
- .personal/ - Personal projects (portfolio, drmalawein, rounaq)
- .metaHub/templates/saas-fullstack/ - Full-stack SaaS template
- business/ - Brand & infrastructure documentation
- docs/pages/ - GitHub Pages for LLC landing pages
- projects/README.md - Complete project registry (85+ projects)

GITHUB PAGES:
- Main landing page for 3 LLCs
- Brand pages: Librex, TalAI, MEZAN, Repz
- MeatheadPhysicist persona page
- CNAME configured for alawein.tech

DOCUMENTATION:
- MASTER_PLAN.md - Single source of truth
- EXECUTION_PLAN.md - Prioritized action items
- COMPLETE_BRAND_AND_INFRASTRUCTURE_AUDIT.md

Closes #brand-consolidation"
```

### 4.3 Push to Remote

```powershell
git push origin main
```

---

## Timeline Summary

| Week       | Actions                                                |
| ---------- | ------------------------------------------------------ |
| **Week 1** | Register domains (librex.dev, talai.dev, alawein.tech) |
| **Week 1** | File Alawein Technologies LLC                          |
| **Week 2** | Deploy Repz to getrepz.app                             |
| **Week 2** | Enable GitHub Pages, configure alawein.tech            |
| **Week 3** | Deploy Attributa to attributa.dev                      |
| **Week 4** | Begin TalAI activation                                 |

---

## Project-to-Domain-to-LLC Mapping

| Project      | Domain           | LLC            | Archive Path                      |
| ------------ | ---------------- | -------------- | --------------------------------- |
| TalAI        | talai.dev        | Alawein Tech   | AlaweinOS/TalAI/                  |
| Librex       | librex.dev       | Alawein Tech   | AlaweinOS/Librex/                 |
| MEZAN        | mezan.dev        | Alawein Tech   | AlaweinOS/MEZAN/                  |
| Attributa    | attributa.dev ✅ | Alawein Tech   | AlaweinOS/Attributa/              |
| SimCore      | simcore.dev ✅   | Alawein Tech   | AlaweinOS/SimCore/                |
| QMLab        | qmlab.online ✅  | Alawein Tech   | AlaweinOS/QMLab/                  |
| LLMWorks     | llmworks.dev ✅  | Alawein Tech   | AlaweinOS/LLMWorks/               |
| Repz         | getrepz.app ✅   | REPZ LLC       | alaweimm90-business/Repz/         |
| LiveItIconic | liveiticonic.com | Live It Iconic | alaweimm90-business/LiveItIconic/ |
| Portfolio    | malawein.com ✅  | Personal       | .personal/portfolio/              |

---

_Plan created: December 5, 2025_
