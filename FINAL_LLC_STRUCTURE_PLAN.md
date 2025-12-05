# Final LLC & Brand Structure Plan

## Complete Portfolio Inventory

**Date:** December 4, 2025  
**Total Projects Discovered:** 50+  
**Organizations:** 4  
**Infrastructure Systems:** 2

---

## Complete Project Tree

```
ALAWEIN TECHNOLOGIES LLC (Parent Entity)
│
├─── GITHUB ORGANIZATIONS ─────────────────────────────────────────────────────
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  AlaweinOS (12 Projects)                                                │
│   │  "Enterprise Software & Optimization Systems"                           │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │                                                                         │
│   │  OPTIMIZATION CORE                                                      │
│   │  ├── Optilibria      31+ GPU-accelerated algorithms    [CORE IP]       │
│   │  ├── MEZAN           Meta-solver engine (1664× speedup) [CORE IP]      │
│   │  └── QAPlibria       Quadratic Assignment solver        [CORE IP]       │
│   │                                                                         │
│   │  AI & RESEARCH PLATFORMS                                                │
│   │  ├── TalAI           19 research products (~21K LOC)    [REVENUE]      │
│   │  ├── HELIOS          Autonomous research (20K+ LOC)     [PLATFORM]     │
│   │  └── CrazyIdeas      22+ frameworks, 3 pitch decks      [R&D]          │
│   │                                                                         │
│   │  SIMULATION & TOOLS                                                     │
│   │  ├── SimCore         HPC simulation engine              [RESEARCH]     │
│   │  ├── QMLab           Quantum materials lab              [RESEARCH]     │
│   │  ├── LLMWorks        LLM utilities                      [TOOLS]        │
│   │  ├── Attributa       Attribution system                 [TOOLS]        │
│   │  └── Benchmarks      Performance testing                [INFRA]        │
│   │                                                                         │
│   │  DEPRECATED                                                             │
│   │  └── FitnessApp      Duplicate of Repz                  [DELETE]       │
│   │                                                                         │
│   └─────────────────────────────────────────────────────────────────────────┘
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  alaweimm90-business (6 Projects)                                       │
│   │  "Consumer Applications & Commercial Products"                          │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │                                                                         │
│   │  ACTIVE PRODUCTS                                                        │
│   │  ├── Repz              AI fitness platform              [CONSUMER]     │
│   │  ├── LiveItIconic      E-commerce platform              [CONSUMER]     │
│   │  └── MarketingAutomation Marketing tools                [B2B]          │
│   │                                                                         │
│   │  PLACEHOLDERS (Archive)                                                 │
│   │  ├── CallaLilyCouture  Fashion e-commerce               [ARCHIVE]      │
│   │  ├── BenchBarrier      Fitness equipment                [ARCHIVE]      │
│   │  └── DrAloweinPortfolio Personal portfolio              [MOVE]         │
│   │                                                                         │
│   └─────────────────────────────────────────────────────────────────────────┘
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  alaweimm90-science (6 Projects)                                        │
│   │  "Scientific Computing & Physics Research"                              │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │                                                                         │
│   │  PHYSICS SIMULATION                                                     │
│   │  ├── MagLogic          Magnetic systems modeling        [RESEARCH]     │
│   │  ├── QMatSim           Quantum materials simulation     [RESEARCH]     │
│   │  ├── SpinCirc          Spintronics circuits             [RESEARCH]     │
│   │  └── SciComp           HPC computing toolkit            [RESEARCH]     │
│   │                                                                         │
│   │  QUANTUM ML                                                             │
│   │  └── QubeML            Quantum machine learning         [RESEARCH]     │
│   │                                                                         │
│   │  DUPLICATE                                                              │
│   │  └── TalAI             Cross-linked (delete)            [DELETE]       │
│   │                                                                         │
│   └─────────────────────────────────────────────────────────────────────────┘
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  MeatheadPhysicist (1 Platform, 30 Components)                          │
│   │  "Autonomous Multi-Agent Quantum Research System"                       │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │                                                                         │
│   │  CORE PLATFORM                                                          │
│   │  ├── src/agents/       9 specialized research agents    [CORE]         │
│   │  ├── Quantum/          Quantum physics research         [RESEARCH]     │
│   │  ├── Projects/         Bell inequality analysis         [RESEARCH]     │
│   │  └── Papers/           Research publications            [ACADEMIC]     │
│   │                                                                         │
│   │  INFRASTRUCTURE                                                         │
│   │  ├── API/              REST API (FastAPI)               [INFRA]        │
│   │  ├── Dashboard/        Web dashboard                    [INFRA]        │
│   │  ├── Database/         Data persistence                 [INFRA]        │
│   │  ├── MLOps/            ML operations                    [INFRA]        │
│   │  ├── Monitoring/       Prometheus/Grafana               [INFRA]        │
│   │  ├── k8s/              Kubernetes manifests             [DEPLOY]       │
│   │  └── Terraform/        Infrastructure as code           [DEPLOY]       │
│   │                                                                         │
│   │  TOOLS & DOCS                                                           │
│   │  ├── CLI/              Command-line interface           [TOOLS]        │
│   │  ├── Notebooks/        Jupyter notebooks                [DOCS]         │
│   │  ├── Education/        Learning materials               [DOCS]         │
│   │  └── Visualizations/   Data visualization               [TOOLS]        │
│   │                                                                         │
│   └─────────────────────────────────────────────────────────────────────────┘
│
├─── INFRASTRUCTURE SYSTEMS ───────────────────────────────────────────────────
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  automation/ (Python CLI)                                               │
│   │  "Unified AI Automation System"                                         │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │  ├── prompts/          49 prompts (system, project, task)              │
│   │  ├── agents/           24 agent definitions                            │
│   │  ├── workflows/        11 workflow definitions                         │
│   │  ├── orchestration/    5 Anthropic patterns, 4 crews                   │
│   │  └── deployment/       Deployment registry                             │
│   └─────────────────────────────────────────────────────────────────────────┘
│
│   ┌─────────────────────────────────────────────────────────────────────────┐
│   │  tools/ (DevOps & AI)                                                   │
│   │  "Development & Operations Toolkit"                                     │
│   ├─────────────────────────────────────────────────────────────────────────┤
│   │  ├── ai/               AI tools (MCP server, compliance)               │
│   │  ├── atlas/            ATLAS orchestration CLI                         │
│   │  ├── devops/           20 DevOps agents                                │
│   │  ├── security/         Security scanning tools                         │
│   │  └── cli/              Command-line utilities                          │
│   └─────────────────────────────────────────────────────────────────────────┘
│
└───────────────────────────────────────────────────────────────────────────────
```

---

## Project Categorization Matrix

### By Business Domain

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           BUSINESS DOMAINS                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   ENTERPRISE    │  │    CONSUMER     │  │    RESEARCH     │            │
│  │   B2B SaaS      │  │    B2C Apps     │  │    Academic     │            │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤            │
│  │ • Optilibria    │  │ • Repz          │  │ • MagLogic      │            │
│  │ • MEZAN         │  │ • LiveItIconic  │  │ • QMatSim       │            │
│  │ • TalAI Suite   │  │                 │  │ • QubeML        │            │
│  │ • HELIOS        │  │                 │  │ • SpinCirc      │            │
│  │ • SimCore       │  │                 │  │ • SciComp       │            │
│  │                 │  │                 │  │ • MeatheadPhys. │            │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤            │
│  │ Revenue: High   │  │ Revenue: Medium │  │ Revenue: Grants │            │
│  │ Risk: Low       │  │ Risk: HIGH      │  │ Risk: Low       │            │
│  │ LLC: Main       │  │ LLC: Separate?  │  │ LLC: Main       │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### By Technology Stack

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          TECHNOLOGY STACKS                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  PYTHON (Core)           TYPESCRIPT (Web)        INFRASTRUCTURE            │
│  ══════════════          ════════════════        ══════════════            │
│  • Optilibria            • Repz (React)          • Docker                  │
│  • MEZAN                 • LiveItIconic          • Kubernetes              │
│  • TalAI (19 products)   • LLMWorks              • Terraform               │
│  • HELIOS                • automation-ts         • Prometheus              │
│  • SimCore               • Dashboard             • Grafana                 │
│  • QAPlibria                                                               │
│  • Scientific Suite                                                        │
│  • MeatheadPhysicist                                                       │
│                                                                            │
│  SHARED: FastAPI, JAX, PyTorch, NumPy, Pandas                              │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### By Revenue Potential

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         REVENUE TIERS                                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  TIER 1: $1M+ POTENTIAL (Enterprise)                                       │
│  ════════════════════════════════════                                       │
│  ┌──────────────┬─────────────────────────────────────────────────────┐    │
│  │ Optilibria   │ Enterprise optimization licensing ($50-500K/yr)     │    │
│  │ MEZAN        │ Meta-solver SaaS ($10-100K/yr per customer)         │    │
│  │ HELIOS       │ Research automation platform ($100K+/yr)            │    │
│  └──────────────┴─────────────────────────────────────────────────────┘    │
│                                                                            │
│  TIER 2: $100K-1M POTENTIAL (SaaS)                                         │
│  ═════════════════════════════════                                          │
│  ┌──────────────┬─────────────────────────────────────────────────────┐    │
│  │ TalAI Suite  │ 19 products × $29-199/mo = $500K-2M ARR potential   │    │
│  │ Repz         │ Consumer fitness ($10-50/mo × users)                │    │
│  │ CrazyIdeas   │ 5+ products ready to launch                         │    │
│  └──────────────┴─────────────────────────────────────────────────────┘    │
│                                                                            │
│  TIER 3: RESEARCH/GRANTS                                                   │
│  ═══════════════════════                                                    │
│  ┌──────────────┬─────────────────────────────────────────────────────┐    │
│  │ Scientific   │ SBIR/STTR grants ($50-500K)                         │    │
│  │ MeatheadPhys │ Research partnerships, academic licensing           │    │
│  └──────────────┴─────────────────────────────────────────────────────┘    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Recommended LLC & Brand Structure

### Phase 1: Single LLC (NOW)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                      ALAWEIN TECHNOLOGIES LLC                               │
│                         (Delaware C-Corp or LLC)                            │
│                                                                             │
│                    ┌─────────────────────────────┐                          │
│                    │      PARENT ENTITY          │                          │
│                    │  • Owns all IP              │                          │
│                    │  • Holds trademarks         │                          │
│                    │  • Single tax return        │                          │
│                    │  • ~$1,500/year cost        │                          │
│                    └─────────────┬───────────────┘                          │
│                                  │                                          │
│         ┌────────────────────────┼────────────────────────┐                 │
│         │                        │                        │                 │
│         ▼                        ▼                        ▼                 │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │   BRAND:    │         │   BRAND:    │         │   BRAND:    │           │
│  │   LIBRIA    │         │   TALAIR    │         │   REPZ      │           │
│  │             │         │             │         │             │           │
│  │ Optilibria  │         │ 19 Research │         │ Fitness     │           │
│  │ MEZAN       │         │ Products    │         │ Platform    │           │
│  │ QAPlibria   │         │ HELIOS      │         │             │           │
│  │ SimCore     │         │ CrazyIdeas  │         │             │           │
│  └─────────────┘         └─────────────┘         └─────────────┘           │
│                                                                             │
│  ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│  │   BRAND:    │         │   BRAND:    │         │   BRAND:    │           │
│  │   SCILAB    │         │  QUANTUM    │         │  ALAWEIN    │           │
│  │             │         │   SCOUT     │         │   LABS      │           │
│  │ MagLogic    │         │             │         │             │           │
│  │ QMatSim     │         │ MeatheadPhy │         │ Consulting  │           │
│  │ QubeML      │         │ (rebrand)   │         │ Services    │           │
│  │ SpinCirc    │         │             │         │             │           │
│  │ SciComp     │         │             │         │             │           │
│  └─────────────┘         └─────────────┘         └─────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Spin Off Consumer (When Repz ARR > $50K)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  FOUNDER (Meshal Alawein)                                                   │
│         │                                                                   │
│         ├──────────────────────────────────────────┐                        │
│         │ 100% owns                                │ 100% owns              │
│         ▼                                          ▼                        │
│  ┌─────────────────────────────┐    ┌─────────────────────────────┐        │
│  │  ALAWEIN TECHNOLOGIES LLC   │    │       REPZ LLC              │        │
│  │  (Enterprise + Research)    │    │    (Consumer Fitness)       │        │
│  ├─────────────────────────────┤    ├─────────────────────────────┤        │
│  │                             │    │                             │        │
│  │  • Libria (Optimization)    │    │  • Repz App                 │        │
│  │  • TalAI (Research Tools)   │    │  • Future consumer apps     │        │
│  │  • HELIOS (Platform)        │    │  • Separate liability       │        │
│  │  • SciLab (Scientific)      │    │  • Clean cap table for VC   │        │
│  │  • Quantum Scout            │    │                             │        │
│  │  • Consulting               │    │                             │        │
│  │                             │    │                             │        │
│  └─────────────────────────────┘    └─────────────────────────────┘        │
│                                                                             │
│  WHY SEPARATE REPZ:                                                         │
│  • Consumer liability (fitness claims, health data)                         │
│  • Different investor profile (consumer VC vs enterprise)                   │
│  • Potential acquisition target                                             │
│  • Different growth trajectory                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Brand Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BRAND HIERARCHY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         ALAWEIN TECHNOLOGIES                                │
│                    "Intelligent Systems for Complex Problems"               │
│                                  │                                          │
│     ┌──────────────┬─────────────┼─────────────┬──────────────┐            │
│     │              │             │             │              │            │
│     ▼              ▼             ▼             ▼              ▼            │
│  ┌──────┐     ┌──────┐     ┌──────┐     ┌──────┐     ┌──────────┐         │
│  │LIBRIA│     │TALAIR│     │SCILAB│     │ REPZ │     │ QUANTUM  │         │
│  │      │     │      │     │      │     │      │     │  SCOUT   │         │
│  └──┬───┘     └──┬───┘     └──┬───┘     └──┬───┘     └────┬─────┘         │
│     │            │            │            │              │               │
│  Optimization  Research    Scientific   Fitness       Quantum             │
│  & Decision    Tools &     Computing    & Health      Research            │
│  Systems       AI          & Physics                  Platform            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BRAND DETAILS:                                                             │
│                                                                             │
│  LIBRIA (libria.dev)                                                        │
│  ├── Tagline: "Optimize Everything"                                        │
│  ├── Products: Optilibria, MEZAN, QAPlibria, SimCore                        │
│  └── Audience: Enterprise, Operations Research, Data Science               │
│                                                                             │
│  TALAIR (talair.io)                                                         │
│  ├── Tagline: "AI-Powered Research Tools"                                   │
│  ├── Products: 19 TalAI tools, HELIOS, CrazyIdeas products                  │
│  └── Audience: Researchers, Scientists, Academics                          │
│                                                                             │
│  SCILAB (scilab.alawein.tech)                                               │
│  ├── Tagline: "Physics-Grade Scientific Computing"                          │
│  ├── Products: MagLogic, QMatSim, QubeML, SpinCirc, SciComp                 │
│  └── Audience: Physics researchers, Materials scientists                   │
│                                                                             │
│  REPZ (repz.app)                                                            │
│  ├── Tagline: "Train Smarter, Not Harder"                                   │
│  ├── Products: Repz fitness app                                             │
│  └── Audience: Fitness enthusiasts, Athletes                               │
│                                                                             │
│  QUANTUM SCOUT (quantumscout.io)                                            │
│  ├── Tagline: "Autonomous Quantum Research"                                 │
│  ├── Products: MeatheadPhysicist platform (rebranded)                       │
│  └── Audience: Quantum researchers, Physics labs                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## GitHub Organization Restructure

### Current vs Recommended

```
CURRENT STRUCTURE                    RECOMMENDED STRUCTURE
═══════════════════                  ═════════════════════

organizations/                       organizations/
├── AlaweinOS/                       ├── AlaweinOS/           [KEEP - Core IP]
│   ├── Optilibria                   │   ├── Optilibria       ✓
│   ├── MEZAN                        │   ├── MEZAN            ✓
│   ├── QAPlibria                    │   ├── QAPlibria        ✓
│   ├── TalAI                        │   ├── TalAI            ✓
│   ├── HELIOS                       │   ├── HELIOS           ✓
│   ├── CrazyIdeas                   │   ├── CrazyIdeas       ✓
│   ├── SimCore                      │   ├── SimCore          ✓
│   ├── QMLab                        │   ├── QMLab            ✓
│   ├── LLMWorks                     │   ├── LLMWorks         ✓
│   ├── Attributa                    │   ├── Attributa        ✓
│   ├── Benchmarks                   │   └── Benchmarks       ✓
│   └── FitnessApp        ──────────────────────────────────► DELETE
│                                    │
├── alaweimm90-business/             ├── alaweimm90-business/ [KEEP - Consumer]
│   ├── Repz                         │   ├── Repz             ✓
│   ├── LiveItIconic                 │   └── LiveItIconic     ✓
│   ├── CallaLilyCouture  ──────────────────────────────────► ARCHIVE
│   ├── BenchBarrier      ──────────────────────────────────► ARCHIVE
│   ├── DrAloweinPortfolio ─────────────────────────────────► MOVE (personal)
│   └── MarketingAutomation          │
│                                    │
├── alaweimm90-science/              ├── alaweimm90-science/  [KEEP - Research]
│   ├── MagLogic                     │   ├── MagLogic         ✓
│   ├── QMatSim                      │   ├── QMatSim          ✓
│   ├── QubeML                       │   ├── QubeML           ✓
│   ├── SpinCirc                     │   ├── SpinCirc         ✓
│   ├── SciComp                      │   └── SciComp          ✓
│   └── TalAI             ──────────────────────────────────► DELETE (duplicate)
│                                    │
└── MeatheadPhysicist/               └── MeatheadPhysicist/   [KEEP - Quantum]
    └── [30 components]                  └── [30 components]  ✓ (rebrand later)
```

---

## Domain & Trademark Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DOMAINS TO REGISTER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PRIORITY 1 (Register Today)                                                │
│  ═══════════════════════════                                                │
│  □ alawein.tech          Parent company                                     │
│  □ alawein.io            Alternative                                        │
│  □ libria.dev            Optimization brand                                 │
│  □ talair.io             Research tools brand                               │
│  □ repz.app              Fitness app                                        │
│                                                                             │
│  PRIORITY 2 (This Week)                                                     │
│  ══════════════════════                                                     │
│  □ mezan.dev             Meta-solver                                        │
│  □ optilibria.com        Optimization framework                             │
│  □ helios-research.io    Research platform                                  │
│  □ quantumscout.io       Quantum research                                   │
│                                                                             │
│  TRADEMARKS TO FILE                                                         │
│  ══════════════════                                                         │
│  □ LIBRIA                Class 42 (Software)                                │
│  □ MEZAN                 Class 42 (Software)                                │
│  □ TALAIR                Class 42 (Software)                                │
│  □ REPZ                  Class 9, 42 (App, Software)                        │
│  □ OPTILIBRIA            Class 42 (Software)                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION TIMELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  WEEK 1: LEGAL FOUNDATION                                                   │
│  ════════════════════════                                                   │
│  □ Day 1: Check LLC name availability "Alawein Technologies LLC"            │
│  □ Day 1: File Delaware LLC ($90 + registered agent ~$100/yr)               │
│  □ Day 2: Get EIN from IRS (free, immediate)                                │
│  □ Day 3: Open business bank account (Mercury, Brex, or Chase)              │
│  □ Day 4: Register priority domains                                         │
│  □ Day 5: Set up Stripe for payments                                        │
│                                                                             │
│  WEEK 2: CLEANUP & LAUNCH PREP                                              │
│  ═════════════════════════════                                              │
│  □ Delete: FitnessApp (AlaweinOS), TalAI (science)                          │
│  □ Archive: CallaLilyCouture, BenchBarrier                                  │
│  □ Move: DrAloweinPortfolio to personal                                     │
│  □ Create landing page: talair.io (TalAI products)                          │
│  □ Prepare: AdversarialReview for launch                                    │
│                                                                             │
│  WEEK 3-4: FIRST REVENUE                                                    │
│  ════════════════════════                                                   │
│  □ Launch: TalAI AdversarialReview ($79/mo)                                 │
│  □ Launch: TalAI GrantWriter ($199/mo)                                      │
│  □ Outreach: 100 researchers (email, Twitter, LinkedIn)                     │
│  □ Content: Blog posts, demo videos                                         │
│                                                                             │
│  MONTH 2-3: SCALE                                                           │
│  ═══════════════════                                                        │
│  □ Launch: 3 more TalAI products                                            │
│  □ Apply: SBIR/STTR grants                                                  │
│  □ Begin: HELIOS beta with select labs                                      │
│  □ Consider: Repz separate LLC (if traction)                                │
│                                                                             │
│  MONTH 4-6: ENTERPRISE                                                      │
│  ═════════════════════                                                      │
│  □ Launch: Optilibria enterprise licensing                                  │
│  □ Launch: MEZAN SaaS                                                       │
│  □ Hire: First contractor/employee                                          │
│  □ Trademark: File for key brands                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Category            | Count | Action                                     |
| ------------------- | ----- | ------------------------------------------ |
| **GitHub Orgs**     | 4     | Keep all                                   |
| **Active Projects** | 35+   | Keep                                       |
| **Duplicates**      | 2     | Delete (FitnessApp, TalAI-science)         |
| **Placeholders**    | 3     | Archive                                    |
| **Brands**          | 5     | Libria, TalAI, SciLab, Repz, Quantum Scout |
| **LLCs (Phase 1)**  | 1     | Alawein Technologies LLC                   |
| **LLCs (Phase 2)**  | 2     | + Repz LLC when ready                      |

**Total Estimated IP Value:** $12-27M  
**First Revenue Target:** TalAI AdversarialReview ($79/mo) in Week 3  
**Annual LLC Cost:** ~$1,500 (Phase 1)

---

_Plan created December 4, 2025_
