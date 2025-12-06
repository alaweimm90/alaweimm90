# 🚀 Comprehensive Project Discovery & Integration Prompt

## 🚀 Context

I'm building a unified studio platform with two hubs:
1. **Templates Studio** (/studio/templates) - Reusable design patterns and components
2. **Platforms Studio** (/studio/platforms) - Functional platform projects organized by tier

Currently, I have scaffolding for these platforms, but need to know which ones actually exist and where their source code is located.

## 🎯 Critical Questions - Please Answer Thoroughly

### Tier 1: Scientific/Technical Platforms

1. **SimCore** - Scientific computing platform
   - Does this exist? If yes, where is the source code?
   - What's the folder path? (e.g., /path/to/simcore or C:\path\to\simcore)
   - Key files/components to integrate?
   - Color scheme/design theme (if different from briefs)?

2. **QMLab** - Quantum mechanics laboratory

   - Does this exist? Where?
   - Folder path?
   - Key components?
   - Design theme details?

3. **OptiLibria** - Optimization algorithms platform
   - Exists? Path?
   - Key files?
   - Design details?

### Tier 2: AI/ML Platforms

1. **TalAI** - AI research platform
   - Exists? Path?
   - Components?
   - Design?

2. **LLMWorks** - Language model development

   - Exists? Path?
   - Components?

### Tier 3: Cultural/Themed

1. **MEZAN** - Arabic/Middle Eastern platform
   - Exists? Path?
   - RTL support? Special considerations?

### Tier 4: Business/Professional

1. **Attributa** - Attribution analytics
   - Exists? Path?

2. **REPZ** - Agent network platform

   - Exists? Path?

### Tier 5: Lifestyle/E-commerce

1. **LiveItIconic** - Lifestyle brand
   - Exists? Path?

2. **PeptideVault** - Biotech e-commerce

   - Exists? Path?

### Tier 6: Family Projects

1. **Mom's Business** - Family business website
   - Exists? Path?

2. **Dad's Website** - Academic portfolio

   - Exists? Path?

### Portfolio Projects

1. **Portfolio #2** - Secondary portfolio design
   - Does this exist separately from Portfolio #1?
   - If yes, where? What's different from the current portfolio?

## 📋 What I Need Per Project (if it exists)

For EACH existing project, please provide:

```text
Project Name: [Name]
Path: [Absolute or relative path]
Status: [Exists/Scaffolded/To-be-created]
Main Component File: [e.g., SimCoreDashboard.tsx]
Key Sub-components: [List of important components]
Design Theme:
- Primary color: [hex]
- Secondary color: [hex]
- Design system used: [Tailwind/Custom/Other]
- Special features: [RTL, 3D, etc.]
Dependencies: [Special libraries or integrations]
Database/API Integration: [If any]
Authentication Required: [Yes/No]
Folder Structure:
- Root path
- Components folder
- Styles/CSS
- Utils/Helpers
- Any other important directories
```

## 🏗️ Architecture Clarifications

1. Are the platforms meant to be:
   - Separate React components integrated into the main app?
   - Standalone applications?
   - Hybrid (some standalone, some integrated)?

2. Should they use:
   - Local Tailwind styling?
   - Imported design system?
   - Custom CSS?

3. Are they protected routes (require authentication)?

4. Do they have their own state management or use the main app's Zustand stores?

## 📁 Folder Organization Expected

```text
src/studios/platforms/
├── portfolio-1/         (existing)
├── portfolio-2/         (if exists)
├── simcore/             (if exists)
├── qmlab/               (if exists)
├── mezan/               (if exists)
├── talai/               (if exists)
├── optilibria/          (if exists)
├── llmworks/            (if exists)
├── attributa/           (if exists)
├── repz/                (if exists)
├── liveiticonic/        (if exists)
└── peptidevault/        (if exists)

src/studios/templates/
├── DashboardTemplate/   (to be created)
├── LandingTemplate/     (to be created)
├── PortfolioTemplate/   (to be created)
└── EcommerceTemplate/   (to be created)
```

## 📊 Summary Table Needed

| Platform | Exists? | Path | Status | Primary Color | Notes |
|----------|---------|------|--------|---------------|-------|
| Portfolio #1 | Yes | src/pages/Portfolio.tsx | Active | Varies | |
| Portfolio #2 | ? | ? | ? | ? | |
| SimCore | ? | ? | ? | ? | |
| QMLab | ? | ? | ? | ? | |
| OptiLibria | ? | ? | ? | ? | |
| TalAI | ? | ? | ? | ? | |
| LLMWorks | ? | ? | ? | ? | |
| MEZAN | ? | ? | ? | ? | |
| Attributa | ? | ? | ? | ? | |
| REPZ | ? | ? | ? | ? | |
| LiveItIconic | ? | ? | ? | ? | |
| PeptideVault | ? | ? | ? | ? | |
| Mom's Business | ? | ? | ? | ? | |
| Dad's Website | ? | ? | ? | ? | |

## 🔍 Final Ask

Please scan the codebase and:

1. **Confirm which platforms actually exist** (not just scaffolded folders)
2. **Provide exact file paths** for each existing project
3. **List the main dashboard/component file** for each platform
4. **Confirm folder structure** and key files
5. **Identify any missing or incorrectly documented projects**
6. **Check for any additional projects** not listed here

## 🎯 Expected Output Format

Please respond with:
1. **A filled summary table** showing existence status
2. **Detailed breakdown** for each existing project using the template above
3. **Any discrepancies** found between documentation and actual code
4. **Recommendations** for integration approach

This will help me accurately integrate the platforms into the studio hubs without making assumptions about what actually exists vs. what's just planned.

---

## 🚀 Usage Instructions

1. Copy this entire prompt into Windsurf's chat
2. Ensure your codebase context is loaded
3. Ask Windsurf to scan and provide the information
4. Share the results with me for proper integration

**Goal**: Get accurate, actionable data about which projects exist and where they're located so I can integrate them properly into the studio platform! 🎯
