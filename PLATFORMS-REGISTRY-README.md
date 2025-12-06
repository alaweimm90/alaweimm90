# Platforms Registry & Brand Pages

> **Central registry for all Alawein platforms, products, and research projects.**

This document describes the architecture connecting the **meta-monorepo** (`Desktop\GitHub`), the **Studios app** (`quantum-dev-profile`), and the **brand landing pages** (`docs/pages/brands/`).

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DEPLOYMENT TOPOLOGY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐     ┌──────────────────────┐                     │
│  │  quantum-dev-profile │     │   Meta-Monorepo      │                     │
│  │  (Portfolio + Studio)│     │   (Desktop\GitHub)   │                     │
│  │                      │     │                      │                     │
│  │  /                   │     │  PROJECT-PLATFORMS-  │                     │
│  │  /studio             │────▶│  CONFIG.ts           │                     │
│  │  /studio/templates   │     │                      │                     │
│  │  /studio/platforms   │     │  docs/pages/brands/  │                     │
│  │                      │     │  ├── index.html      │                     │
│  │  Deployed to:        │     │  ├── simcore/        │                     │
│  │  malawein.com        │     │  ├── qmlab/          │                     │
│  └──────────────────────┘     │  ├── llmworks/       │                     │
│                               │  ├── attributa/      │                     │
│  ┌──────────────────────┐     │  ├── repz/           │                     │
│  │  Individual Apps     │     │  ├── liveiticonic/   │                     │
│  │                      │     │  ├── talai/          │                     │
│  │  SimCore → simcore.dev    │  ├── mezan/           │                     │
│  │  QMLab → qmlab.online     │  ├── librex/          │                     │
│  │  LLMWorks → llmworks.dev  │  └── ...              │                     │
│  │  Attributa → attributa.dev│                       │                     │
│  │  REPZ → getrepz.app       │  Deployed to:         │                     │
│  │  LiveItIconic → liveiticonic.com  GitHub Pages    │                     │
│  └──────────────────────┘     └──────────────────────┘                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```text
Desktop\GitHub/
├── PROJECT-PLATFORMS-CONFIG.ts    # Central registry (TypeScript)
├── PLATFORMS-REGISTRY-README.md   # This file
├── PROJECT-DISCOVERY-PROMPT.md    # Discovery prompt for LLMs
├── docs/
│   ├── PROJECT_REGISTRY.md        # Human-readable project list
│   ├── LLC_PROJECT_REGISTRY.md    # LLC ownership mapping
│   └── pages/
│       └── brands/
│           ├── index.html         # Brand directory hub
│           ├── simcore/index.html
│           ├── qmlab/index.html
│           ├── llmworks/index.html
│           ├── attributa/index.html
│           ├── repz/index.html
│           ├── liveiticonic/index.html
│           ├── talai/index.html
│           ├── mezan/index.html
│           ├── librex/index.html
│           ├── helios/index.html
│           ├── foundry/index.html
│           ├── maglogic/index.html
│           ├── qmatsim/index.html
│           ├── qubeml/index.html
│           ├── scicomp/index.html
│           ├── spincirc/index.html
│           ├── meatheadphysicist/index.html
│           ├── portfolio/index.html
│           ├── drmalawein/index.html
│           ├── rounaq/index.html
│           └── marketing-automation/index.html
└── [product repos...]
```

---

## PROJECT-PLATFORMS-CONFIG.ts

### Types

```typescript
type PlatformStatus = "active" | "backend" | "planned";
type PlatformTier = "scientific" | "ai-ml" | "cultural" | "business" | "lifestyle" | "family" | "portfolio";
type PlatformAppType = "spa" | "backend";

interface PlatformDefinition {
  id: string;                      // Stable ID for routing
  name: string;                    // Display name
  tier: PlatformTier;              // Category
  status: PlatformStatus;          // Development status
  repoPath?: string;               // Path in monorepo
  appType?: PlatformAppType;       // SPA or backend
  primaryColorToken?: string;      // Design token
  hasAuth?: boolean;               // Requires authentication
  hasAdminArea?: boolean;          // Has admin dashboard
  tags?: string[];                 // Searchable tags
  notes?: string;                  // Integration notes
  domainUrl?: string;              // Live domain (e.g. https://simcore.dev)
  brandPageUrl?: string;           // Brand page path
  githubUrl?: string;              // GitHub repository
  tagline?: string;                // Short description
  gradientColors?: [string, string]; // Card gradient
}
```

### Exports

```typescript
// Data
export const PLATFORMS: PlatformDefinition[];

// Helpers
export const getPlatformsByTier: (tier: PlatformTier) => PlatformDefinition[];
export const getActivePlatformsByTier: (tier: PlatformTier) => PlatformDefinition[];
export const getAllActiveSpaPlatforms: () => PlatformDefinition[];
export const getPlatformById: (id: string) => PlatformDefinition | undefined;
export const getPlatformsGroupedByTier: () => Record<PlatformTier, PlatformDefinition[]>;
export const getPlatformPrimaryUrl: (platform: PlatformDefinition) => string | undefined;

// Constants
export const TIER_LABELS: Record<PlatformTier, string>;
export const TIER_ORDER: PlatformTier[];
```

---

## Integration with Studios App (quantum-dev-profile)

### Option 1: Direct Import (Recommended for Development)

Copy `PROJECT-PLATFORMS-CONFIG.ts` to `quantum-dev-profile/src/data/platforms.ts`:

```typescript
// src/studios/platforms/PlatformsHub.tsx
import {
  PLATFORMS,
  getPlatformsGroupedByTier,
  TIER_ORDER,
  TIER_LABELS,
  type PlatformDefinition,
} from "@/data/platforms";

export function PlatformsHub() {
  const grouped = getPlatformsGroupedByTier();

  return (
    <div className="space-y-8">
      {TIER_ORDER.map((tier) => (
        <section key={tier}>
          <h2 className="text-lg font-semibold mb-4">{TIER_LABELS[tier]}</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {grouped[tier].map((platform) => (
              <PlatformCard key={platform.id} platform={platform} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function PlatformCard({ platform }: { platform: PlatformDefinition }) {
  const [from, to] = platform.gradientColors || ["#38bdf8", "#0284c7"];

  return (
    <div
      className="rounded-xl border p-4 hover:border-primary transition-colors"
      style={{ background: `linear-gradient(135deg, ${from}10, ${to}10)` }}
    >
      <div className="flex items-start justify-between mb-2">
        <h3 className="font-semibold">{platform.name}</h3>
        <StatusBadge status={platform.status} />
      </div>
      <p className="text-sm text-muted-foreground mb-3">{platform.tagline}</p>
      <div className="flex gap-2">
        {platform.domainUrl && (
          <a href={platform.domainUrl} className="btn btn-primary btn-sm">
            Open App
          </a>
        )}
        {platform.brandPageUrl && (
          <a href={platform.brandPageUrl} className="btn btn-secondary btn-sm">
            Learn More
          </a>
        )}
        {platform.githubUrl && (
          <a href={platform.githubUrl} className="btn btn-ghost btn-sm">
            GitHub
          </a>
        )}
      </div>
    </div>
  );
}
```

### Option 2: NPM Package (Production)

1. Create `packages/platforms-registry/` in this monorepo
2. Add `package.json` with name `@alawein/platforms-registry`
3. Publish to GitHub Packages or npm
4. Install in `quantum-dev-profile`: `npm install @alawein/platforms-registry`

---

## Deployment Strategy

### Brand Pages (GitHub Pages)

1. Enable GitHub Pages for this repo with source = `docs/`
2. URLs become: `https://<org>.github.io/<repo>/pages/brands/<id>/`

### Studios App (Vercel/Netlify)

1. Deploy `quantum-dev-profile` to Vercel or Netlify
2. Configure custom domain: `malawein.com`
3. Routes:
   - `/` → Portfolio
   - `/studio` → StudioSelector
   - `/studio/templates` → TemplatesHub
   - `/studio/platforms` → PlatformsHub

### Individual Apps

| Platform | Domain | Hosting |
|----------|--------|---------|
| SimCore | simcore.dev | Vercel |
| QMLab | qmlab.online | Vercel |
| LLMWorks | llmworks.dev | Vercel |
| Attributa | attributa.dev | Vercel |
| REPZ | getrepz.app | Vercel |
| LiveItIconic | liveiticonic.com | Vercel |

---

## Domain Routing

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOMAIN ROUTING                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  malawein.com ─────────────▶ quantum-dev-profile (Portfolio)              │
│  meshal.ai ────────────────▶ quantum-dev-profile (Portfolio)              │
│                                                                           │
│  simcore.dev ──────────────▶ SimCore SPA                                  │
│  qmlab.online ─────────────▶ QMLab SPA                                    │
│  llmworks.dev ─────────────▶ LLMWorks SPA                                 │
│  attributa.dev ────────────▶ Attributa SPA                                │
│  getrepz.app ──────────────▶ REPZ SPA                                     │
│  liveiticonic.com ─────────▶ LiveItIconic SPA                             │
│                                                                           │
│  talai.dev ────────────────▶ Brand Page (until SPA ready)                 │
│  mezan.dev ────────────────▶ Brand Page (until SPA ready)                 │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Adding a New Platform

1. **Add to `PROJECT-PLATFORMS-CONFIG.ts`:**

   ```typescript
   {
     id: "newplatform",
     name: "New Platform",
     tier: "business",
     status: "planned",
     repoPath: "alawein-technologies-llc/newplatform",
     appType: "spa",
     tags: ["new", "platform"],
     tagline: "Description of the new platform",
     gradientColors: ["#22c55e", "#15803d"],
   }
   ```

2. **Create brand page:**

   ```bash
   mkdir docs/pages/brands/newplatform
   # Copy template from existing brand page
   cp docs/pages/brands/simcore/index.html docs/pages/brands/newplatform/index.html
   # Edit with platform-specific content
   ```

3. **Update `docs/pages/brands/index.html`** to include the new card

4. **Sync to Studios app** (copy updated config or publish package)

---

## CLI Integration

The `automation-ts` CLI can query this registry:

```bash
# List all platforms
npx automation deploy list

# Get platform info
npx automation deploy info simcore

# Deploy a platform
npx automation deploy portfolio simcore --platform vercel
```

---

## Maintenance

- **Registry updates:** Edit `PROJECT-PLATFORMS-CONFIG.ts`
- **Brand page updates:** Edit `docs/pages/brands/<id>/index.html`
- **Domain changes:** Update `domainUrl` in config and DNS records
- **Status changes:** Update `status` field when platforms go live

---

Last updated: December 6, 2025
