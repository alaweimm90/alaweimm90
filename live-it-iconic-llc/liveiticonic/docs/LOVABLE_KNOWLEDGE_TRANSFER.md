# Transferring Project Knowledge Between Lovable Projects

**Purpose:** Guide for sharing architecture, types, and configuration across multiple Lovable projects  
**Last Updated:** 2025-11-08

---

## Overview

When managing multiple Lovable projects (e.g., separate website and e-commerce apps, or creating a new project with similar architecture), you need a way to share:
- Architecture decisions
- Type definitions
- Design system tokens
- Configuration files
- AI assistant instructions

This guide covers **5 methods** for knowledge transfer, ranked by ease of use.

---

## Method 1: Lovable Custom Knowledge (⭐ Recommended)

**Best for:** AI assistant instructions, architecture docs, key type definitions

### How It Works
Lovable has a built-in **Custom Knowledge** feature that lets you upload documentation that the AI assistant will automatically reference.

### Steps

1. **In Source Project:**
   - Prepare key documentation files:
     ```
     MONOREPO_STRUCTURE.md
     docs/CONTROL_HUB_GUIDE.md
     src/theme/tokens.ts
     src/types/product.ts
     tailwind.config.ts
     ```

2. **In Target Project:**
   - Go to **Project Settings** (gear icon in top right)
   - Click **Manage Knowledge**
   - Click **Add Knowledge**
   - Upload or paste the prepared files
   - Optionally, add a description for each file

3. **Result:**
   - AI assistant automatically references these files when generating code
   - Knowledge persists across all sessions
   - Easy to update as your architecture evolves

### What to Upload

**Essential Files:**
```
✅ MONOREPO_STRUCTURE.md          # Architecture overview
✅ docs/CONTROL_HUB_GUIDE.md      # AI instructions
✅ src/theme/tokens.ts             # Design tokens
✅ src/types/product.ts            # Type definitions
✅ .eslintrc.json                  # Code standards
✅ README.md                       # Project overview
```

**Optional but Helpful:**
```
⭕ tailwind.config.ts              # Styling configuration
⭕ vite.config.ts                  # Build config
⭕ package.json (workspace config)  # Dependencies
⭕ Key feature documentation
```

### Pros
✅ Built into Lovable, no external tools  
✅ AI automatically uses this knowledge  
✅ Easy to update  
✅ Persists across sessions  

### Cons
❌ Files must be re-uploaded if updated in source project  
❌ No automatic sync between projects  

---

## Method 2: GitHub Repository as Source of Truth

**Best for:** Full codebase sharing, version control, team collaboration

### How It Works
Connect both Lovable projects to GitHub repositories, then use Git to share code and documentation.

### Architecture

```
Option A: Monorepo (One GitHub Repo)
live-it-iconic-monorepo/
├── apps/
│   ├── website/        → Lovable Project 1
│   ├── shop/           → Lovable Project 2
│   └── admin/          → Lovable Project 3 (future)
├── packages/           → Shared by all
└── docs/              → Shared by all

Option B: Separate Repos with Shared Docs
live-it-iconic-website/      → Lovable Project 1
live-it-iconic-shop/         → Lovable Project 2
live-it-iconic-shared/       → Documentation + types (Git submodule)
```

### Steps

#### Option A: Monorepo Setup

1. **Connect Source Project to GitHub:**
   - In Lovable: Click **GitHub** → **Connect to GitHub**
   - Authorize and create repository
   - Lovable auto-syncs changes

2. **Set Up Monorepo Structure:**
   - Follow `MONOREPO_STRUCTURE.md`
   - Organize into `apps/` and `packages/`

3. **Create New Lovable Projects:**
   - Create new Lovable project for each app
   - Connect to same GitHub repo
   - Each project works on different `/apps/[app-name]` folder

4. **Configure Build Paths:**
   - Update each Lovable project's `vite.config.ts` to point to correct app folder

#### Option B: Separate Repos with Git Submodules

1. **Create Shared Documentation Repo:**
   ```bash
   # On GitHub, create: live-it-iconic-shared
   mkdir live-it-iconic-shared
   cd live-it-iconic-shared
   git init
   
   # Add shared files
   mkdir -p docs types design-tokens
   cp MONOREPO_STRUCTURE.md .
   cp docs/CONTROL_HUB_GUIDE.md docs/
   cp -r src/types/* types/
   cp src/theme/tokens.ts design-tokens/
   
   git add .
   git commit -m "Initial shared knowledge"
   git push origin main
   ```

2. **Add Submodule to Each Project:**
   ```bash
   # In live-it-iconic-website
   git submodule add https://github.com/org/live-it-iconic-shared shared
   
   # In live-it-iconic-shop
   git submodule add https://github.com/org/live-it-iconic-shared shared
   ```

3. **Update Shared Knowledge:**
   ```bash
   # In live-it-iconic-shared
   git pull origin main
   git add .
   git commit -m "Update types"
   git push
   
   # In each project
   git submodule update --remote shared
   ```

### Pros
✅ Full version control  
✅ True code sharing (not just docs)  
✅ Team collaboration with Git workflows  
✅ Lovable auto-syncs with GitHub  

### Cons
❌ More complex setup  
❌ Requires Git knowledge  
❌ Lovable has experimental branch support  

---

## Method 3: npm Package Publishing

**Best for:** Mature shared packages (UI library, utilities) used across multiple projects

### How It Works
Publish shared packages to npm (or GitHub Packages), then install them in each Lovable project.

### Steps

1. **Prepare Shared Package:**
   ```bash
   # In packages/types
   cat > package.json << EOF
   {
     "name": "@live-it-iconic/types",
     "version": "1.0.0",
     "main": "./dist/index.js",
     "types": "./dist/index.d.ts",
     "scripts": {
       "build": "tsc"
     }
   }
   EOF
   ```

2. **Build and Publish:**
   ```bash
   npm run build
   npm publish --access=public
   ```

3. **Install in Lovable Projects:**
   ```bash
   # In each Lovable project
   npm install @live-it-iconic/types
   ```

4. **Use in Code:**
   ```typescript
   import { Product, Collection } from '@live-it-iconic/types';
   ```

### Pros
✅ Standard package management  
✅ Version control for shared code  
✅ Easy to install in any project  
✅ Works outside Lovable too  

### Cons
❌ Requires publishing workflow  
❌ Updates need new versions  

---

## Method 4: Documentation Website

**Best for:** Team onboarding, external documentation, searchable knowledge base

### How It Works
Create a separate documentation website (e.g., using VitePress or Docusaurus) that centralizes all architecture, types, and guidelines.

### Quick Setup with VitePress

```bash
# Create docs site
npx vitepress init

# Structure
docs/
├── .vitepress/
│   └── config.ts
├── architecture/
│   └── monorepo.md
├── types/
│   └── product.md
├── design-system/
│   └── tokens.md
└── index.md
```

**Deploy:** Vercel, Netlify, GitHub Pages (free)

### Link in Lovable Projects

```markdown
<!-- README.md in each Lovable project -->
📚 **Documentation:** https://docs.liveiticonic.com
```

### Use Custom Knowledge to Point to It

Upload a file to Lovable Custom Knowledge:

```markdown
# Project Documentation

All architecture, types, and guidelines are maintained at:
https://docs.liveiticonic.com

Key sections:
- Architecture: https://docs.liveiticonic.com/architecture/monorepo
- Types: https://docs.liveiticonic.com/types/product
- Design System: https://docs.liveiticonic.com/design-system/tokens
```

### Pros
✅ Central source of truth  
✅ Searchable, versioned  
✅ Great for team onboarding  
✅ Accessible outside Lovable  

### Cons
❌ Requires separate hosting  
❌ Must manually keep in sync  

---

## Method 5: Copy-Paste Key Files

**Best for:** Quick one-time setup, small projects, prototyping

### Essential Files Checklist

```
Core Documentation:
├── MONOREPO_STRUCTURE.md
├── docs/CONTROL_HUB_GUIDE.md
├── README.md
└── CONTRIBUTING.md

Type Definitions:
├── src/types/product.ts
├── src/types/order.ts
└── src/types/index.ts

Design System:
├── src/theme/tokens.ts
├── tailwind.config.ts
└── src/index.css

Configuration:
├── .eslintrc.json
├── prettier.config.js
└── tsconfig.json
```

### Steps

1. **In Source Project:**
   - Identify files to copy (use checklist above)
   - Copy to temporary folder or cloud storage

2. **In Target Lovable Project:**
   - Create necessary directories
   - Paste files into correct locations
   - Update any project-specific paths

3. **Adapt Imports:**
   ```typescript
   // Update imports if folder structure differs
   // Source: import { Product } from '@lii/types';
   // Target: import { Product } from '@/types/product';
   ```

### Pros
✅ Fastest for one-time setup  
✅ No external tools required  

### Cons
❌ No automatic sync  
❌ Manual updates required  

---

## Comparison Matrix

| Method | Setup | Sync | Best Use Case |
|--------|-------|------|---------------|
| **Custom Knowledge** | 🟢 Easy | 🟡 Manual | AI instructions |
| **GitHub Monorepo** | 🟡 Medium | 🟢 Auto | Full codebase sharing |
| **npm Packages** | 🔴 Complex | 🟢 Versioned | Mature libraries |
| **Docs Website** | 🟡 Medium | 🟡 Manual | Team onboarding |
| **Copy-Paste** | 🟢 Easy | 🔴 None | Quick prototyping |

---

## Recommended Workflow

### For Small Teams (1-3 developers)
1. **Use Lovable Custom Knowledge** for AI instructions
2. **Use GitHub Monorepo** for code sharing
3. Copy-paste for quick prototypes

### For Medium Teams (4-10 developers)
1. **GitHub Monorepo** with separate `apps/`
2. **Documentation Website** for onboarding
3. **Custom Knowledge** to link to docs

### For Large Teams (10+ developers)
1. **Separate Repos** with Git submodules
2. **npm Packages** for mature libraries
3. **Docs Website** as central hub
4. **CI/CD** to auto-publish packages

---

## Example: Two Lovable Projects

**Scenario:** Marketing website + e-commerce shop need to share types and design tokens.

### Step 1: Create Shared Package

```bash
mkdir -p shared-knowledge
cd shared-knowledge

cp ../MONOREPO_STRUCTURE.md .
cp ../docs/CONTROL_HUB_GUIDE.md .
mkdir types design-tokens
cp ../src/types/*.ts types/
cp ../src/theme/tokens.ts design-tokens/
```

### Step 2: Upload to Both Projects

**Website Project:**
1. Settings → Manage Knowledge
2. Upload all files from `shared-knowledge/`

**Shop Project:**
1. Settings → Manage Knowledge
2. Upload same files

### Step 3: Keep in Sync

When updating types:
1. Copy to `shared-knowledge/`
2. Re-upload to both projects
3. Or use GitHub for version control

---

## Best Practices

### DO ✅
- Use Custom Knowledge for AI instructions
- Version shared knowledge in Git
- Document changes with changelogs
- Keep types in single source of truth

### DON'T ❌
- Don't duplicate type definitions
- Don't hardcode values (use tokens)
- Don't forget to update both projects

---

## Troubleshooting

### "AI isn't using my custom knowledge"
- Check files are uploaded in Settings → Manage Knowledge
- Ensure file formats are readable (Markdown, TypeScript, JSON)
- Try re-uploading with descriptions

### "Types are out of sync"
- Use GitHub for automatic sync
- Set weekly reminder to re-upload
- Consider npm packages for version control

---

## Resources

- [Lovable Documentation](https://docs.lovable.dev/)
- [Custom Knowledge Guide](https://docs.lovable.dev/features/custom-knowledge)
- [Git Submodules Tutorial](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [npm Publishing](https://docs.npmjs.com/creating-and-publishing-scoped-public-packages)

---

**Last Updated:** 2025-11-08
