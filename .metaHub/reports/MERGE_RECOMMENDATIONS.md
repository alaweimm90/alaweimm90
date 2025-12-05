# Project Merge Recommendations

## Overview

**Current**: 64 projects across 4 orgs  
**Proposed**: ~20 projects (69% reduction)  
**Benefit**: Reduced maintenance, unified codebases, clearer architecture

## Merge Strategy

### alawein-business (9 → 3 projects)

#### Merge 1: Business Apps (5 → 1)
**Target**: `business-apps`  
**Merge**: BenchBarrier, CallaLilyCouture, DrAloweinPortfolio, LiveItIconic, Repz  
**Reason**: All React apps, similar tech stack  
**Structure**:
```
business-apps/
├── apps/
│   ├── bench-barrier/
│   ├── calla-lily/
│   ├── dr-alowein/
│   ├── live-it-iconic/
│   └── repz/
└── shared/
    ├── components/
    └── utils/
```

#### Keep Separate (2)
- MarketingAutomation (backend services)
- templates (scaffolding)

**Result**: 9 → 3 projects

---

### alawein-science (6 → 2 projects)

#### Merge 1: Physics Simulations (5 → 1)
**Target**: `physics-sim`  
**Merge**: MagLogic, QMatSim, QubeML, SciComp, SpinCirc  
**Reason**: All scientific computing libraries  
**Structure**:
```
physics-sim/
├── magnetics/      # MagLogic
├── quantum/        # QMatSim, QubeML
├── scientific/     # SciComp
├── spin/          # SpinCirc
└── core/          # Shared utilities
```

#### Keep Separate (1)
- TalAI (AI research, different domain)

**Result**: 6 → 2 projects

---

### AlaweinOS (19 → 8 projects)

#### Merge 1: Web Apps (4 → 1)
**Target**: `web-apps`  
**Merge**: Attributa, LLMWorks, QMLab, SimCore  
**Reason**: All React/TypeScript apps  

#### Merge 2: Optimization Suite (3 → 1)
**Target**: `libria` (already named!)  
**Merge**: Optilibria, Librex.QAP, MEZAN  
**Reason**: Core optimization platform  

#### Merge 3: Research Tools (3 → 1)
**Target**: `research-tools`  
**Merge**: Benchmarks, HELIOS, TalAI  
**Reason**: Research automation  

#### Keep Separate (5)
- FitnessApp (consumer product)
- Foundry (ideation/prototypes)
- docker (infrastructure)
- k8s (infrastructure)
- docs (documentation)

**Result**: 19 → 8 projects

---

### MeatheadPhysicist (30 → 10 projects)

#### Merge 1: Core Library (8 → 1)
**Target**: `meathead-core`  
**Merge**: Benchmarks, CLI, Deployment, Notebooks, Quantum, src, tests, Tools  
**Reason**: Core scientific computing library  

#### Merge 2: Services (2 → 1)
**Target**: `services`  
**Merge**: API, Dashboard  
**Reason**: Backend services  

#### Merge 3: Infrastructure (6 → 1)
**Target**: `infra`  
**Merge**: Cloud, Database, k8s, MLOps, Monitoring, Terraform  
**Reason**: DevOps/infrastructure  

#### Keep Separate (7)
- Automation (research automation)
- Config (shared configs)
- docs (documentation)
- Education (teaching materials)
- Examples (demos)
- Frontend (web UI)
- Visualizations (3D/VR)
- GHPages (static site)
- Integrations (external APIs)
- Nginx (web server)
- Notes (research notes)
- Papers (publications)
- Projects (active research)
- scripts (utilities)

**Result**: 30 → 10 projects

---

## Summary Table

| Organization | Before | After | Reduction | Key Merges |
|--------------|--------|-------|-----------|------------|
| alawein-business | 9 | 3 | 67% | 5 React apps → 1 |
| alawein-science | 6 | 2 | 67% | 5 physics libs → 1 |
| AlaweinOS | 19 | 8 | 58% | 3 optimization → libria |
| MeatheadPhysicist | 30 | 10 | 67% | 8 core libs → 1 |
| **Total** | **64** | **23** | **64%** | **21 merges** |

## Implementation Plan

### Phase 1: Low Risk (1 week)
- Merge alawein-science physics libs
- Merge AlaweinOS optimization (Optilibria + Librex.QAP + MEZAN → Libria)

### Phase 2: Medium Risk (2 weeks)
- Merge alawein-business React apps
- Merge AlaweinOS web apps

### Phase 3: High Risk (3 weeks)
- Merge MeatheadPhysicist core library
- Merge MeatheadPhysicist infrastructure

## Benefits

### Maintenance
- **64 → 23 projects** (64% reduction)
- **64 CI pipelines** → 23
- **64 dependency files** → 23
- **64 README files** → 23

### Code Reuse
- Shared components across apps
- Unified utilities
- Single source of truth

### Architecture
- Clearer boundaries
- Monorepo benefits
- Easier refactoring

## Risks

### Low Risk
- Physics libs (independent, well-tested)
- Optimization suite (already related)

### Medium Risk
- React apps (different domains, but similar tech)
- Web apps (different purposes)

### High Risk
- MeatheadPhysicist core (30 projects, complex dependencies)

## Rollback Plan

All merges use git subtree:
```bash
# Merge
git subtree add --prefix=apps/project1 ../project1 main

# Rollback
git revert <merge-commit>
```

## Next Steps

1. **Review recommendations** - Approve/modify merge groups
2. **Pilot merge** - Start with alawein-science (lowest risk)
3. **Validate** - Ensure tests pass, builds work
4. **Iterate** - Apply pattern to remaining orgs
5. **Cleanup** - Archive old repos

## Decision Needed

**Which org to start with?**
- alawein-science (6 → 2, lowest risk)
- AlaweinOS optimization (3 → 1, high value)
- alawein-business (9 → 3, medium risk)
