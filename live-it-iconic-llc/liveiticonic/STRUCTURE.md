# Live It Iconic - Directory Structure Guide

**Navigation guide for the Live It Iconic wellness platform codebase.**

---

## 📁 Directory Overview

```
live-it-iconic/
├── 📄 Root Documentation (10 files)
├── 📚 docs/ - Organized documentation
├── 💻 src/ - Source code
├── 🎨 public/ - Public assets
├── 🧪 tests/ - Test suites
├── 🏗️ infrastructure/ - Deployment configs
├── 📦 supabase/ - Database configs
└── ⚙️ Configuration files
```

---

## 📄 Root Level Files

### Essential Documentation (Max 10)

| File | Purpose |
|------|---------|
| `README.md` | Main entry point - "What is this?" |
| `PROJECT.md` | Complete project overview |
| `STRUCTURE.md` | This file - Directory guide |
| `QUICK_START.md` | Getting started in 5 minutes |
| `WELLNESS_PLATFORM_DOCUMENTATION.md` | Comprehensive feature documentation |
| `DEPLOYMENT_INFRASTRUCTURE_PLAN.md` | Deployment & infrastructure guide |
| `CHANGELOG.md` | Version history |
| `CONTRIBUTING.md` | Contribution guidelines |
| `SECURITY.md` | Security policies |
| `LICENSE` | Legal information |

### Configuration Files

| File | Purpose |
|------|---------|
| `package.json` | Dependencies & scripts |
| `vite.config.ts` | Vite configuration |
| `tailwind.config.ts` | TailwindCSS configuration |
| `tsconfig.json` | TypeScript configuration |
| `components.json` | shadcn/ui configuration |
| `.gitignore` | Git ignore rules |
| `.env.example` | Environment variable template |

---

## 📚 Documentation Hub (`docs/`)

### Structure

```
docs/
├── README.md                    # Documentation index & navigation
│
├── getting-started/             # New user guides
│   ├── installation.md
│   ├── configuration.md
│   └── first-steps.md
│
├── guides/                      # Feature guides
│   ├── wellness-features.md
│   ├── i18n-guide.md
│   ├── testing-guide.md
│   ├── storybook.md
│   ├── stripe-integration.md
│   ├── social-media.md
│   └── admin-dashboard.md
│
├── api/                         # API documentation
│   ├── README.md
│   ├── endpoints.md
│   ├── authentication.md
│   └── webhooks.md
│
├── architecture/                # System design
│   ├── system-design.md
│   ├── database-schema.md
│   ├── security.md
│   └── performance.md
│
├── deployment/                  # Deployment guides
│   ├── infrastructure.md
│   ├── monitoring.md
│   └── launch-checklist.md
│
├── reference/                   # Quick reference
│   ├── commit-messages.md
│   ├── pull-requests.md
│   ├── governance.md
│   ├── reference-card.md
│   └── environment-variables.md
│
├── ai/                          # AI & automation
│   ├── claude-prompts.md
│   └── superprompt.md
│
├── planning/                    # Business & planning
│   └── business-plan.md
│
└── archive/                     # Historical documents
    └── implementation-reports/
```

### Finding Documentation

| Need | Location |
|------|----------|
| "How do I start?" | `docs/getting-started/installation.md` |
| "How does X feature work?" | `docs/guides/[feature].md` |
| "API endpoints?" | `docs/api/endpoints.md` |
| "System architecture?" | `docs/architecture/system-design.md` |
| "How to deploy?" | `docs/deployment/infrastructure.md` |
| "Quick reference?" | `docs/reference/reference-card.md` |

---

## 💻 Source Code (`src/`)

### Directory Structure

```
src/
├── main.tsx                     # Application entry point
├── App.tsx                      # Root component
│
├── components/                  # React components
│   ├── wellness/                # Wellness-specific components
│   │   └── WellnessDashboard.tsx
│   ├── ui/                      # shadcn/ui components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── [40+ components]
│   ├── admin/                   # Admin dashboard components
│   ├── brandmarks/              # Brand assets
│   ├── checkout/                # Checkout flow
│   ├── icons/                   # Custom icons
│   ├── logo/                    # Logo variations
│   ├── product/                 # Product components
│   ├── seo/                     # SEO components
│   ├── showcase/                # Showcase components
│   └── utils/                   # Utility components
│
├── services/                    # Business logic (Singleton pattern)
│   ├── wearableIntegration.ts   # Wearable device integration
│   ├── mentalHealthService.ts   # Mental health tracking
│   ├── habitTrackingService.ts  # Habit tracking & streaks
│   ├── gamificationService.ts   # Points, levels, achievements
│   ├── socialFeaturesService.ts # Social & community
│   ├── nutritionService.ts      # Nutrition & meal logging
│   ├── aiRecommendationsService.ts # AI-powered recommendations
│   └── __tests__/               # Service tests
│       └── habitTrackingService.test.ts
│
├── middleware/                  # Request middleware
│   ├── rateLimiting.ts          # API rate limiting
│   └── privacyEnhancement.ts    # Differential privacy
│
├── types/                       # TypeScript definitions
│   └── wellness.ts              # Wellness platform types (30+ interfaces)
│
├── hooks/                       # Custom React hooks
│   ├── use-mobile.tsx
│   ├── use-toast.ts
│   └── [custom hooks]
│
├── contexts/                    # React contexts
│   ├── AuthContext.tsx
│   ├── CartContext.tsx
│   ├── CurrencyContext.tsx
│   └── WishlistContext.tsx
│
├── pages/                       # Page components
│   ├── Index.tsx                # Home page
│   ├── Shop.tsx                 # Shop page
│   ├── Profile.tsx              # User profile
│   ├── Wellness.tsx             # Wellness dashboard (planned)
│   └── [other pages]
│
├── integrations/                # Third-party integrations
│   └── supabase/
│       ├── client.ts            # Supabase client
│       └── types.ts             # Generated types
│
├── utils/                       # Utility functions
│   ├── cn.ts                    # Class name utility
│   ├── formatters.ts            # Data formatters
│   └── [utilities]
│
├── constants/                   # Application constants
│   ├── products.tsx
│   ├── podcastPerspectives.tsx
│   └── [constants]
│
├── data/                        # Static data
│   └── [data files]
│
├── i18n/                        # Internationalization
│   ├── config.ts
│   └── locales/
│       ├── en/
│       └── [other languages]
│
├── emails/                      # Email templates
│   ├── components/
│   ├── templates/
│   └── utils/
│
├── styles/                      # Global styles
│   └── globals.css
│
└── lib/                         # Shared utilities
    └── utils.ts
```

### Component Organization

**Naming Convention:**
- **Components:** PascalCase (`WellnessDashboard.tsx`)
- **Utilities:** camelCase (`formatDate.ts`)
- **Hooks:** `use-` prefix (`use-mobile.tsx`)
- **Contexts:** `Context` suffix (`AuthContext.tsx`)

**File Structure Pattern:**
```typescript
// 1. Imports
import React from 'react';
import { Button } from '@/components/ui/button';
import type { User } from '@/types';

// 2. Types
interface Props {
  user: User;
}

// 3. Component
export const UserProfile: React.FC<Props> = ({ user }) => {
  // Implementation
};
```

### Service Pattern (Singleton)

All services follow this pattern:

```typescript
export class ExampleService {
  private static instance: ExampleService;

  private constructor() {}

  static getInstance(): ExampleService {
    if (!ExampleService.instance) {
      ExampleService.instance = new ExampleService();
    }
    return ExampleService.instance;
  }

  // Methods...
}

export const exampleService = ExampleService.getInstance();
```

**Services:**
- ✅ Centralized business logic
- ✅ Testable
- ✅ Reusable
- ✅ Type-safe

---

## 🎨 Public Assets (`public/`)

```
public/
├── service-worker.js            # PWA service worker
├── offline.html                 # Offline fallback page
├── manifest.json                # PWA manifest
├── favicon.ico                  # Favicon
├── icons/                       # App icons
└── [static assets]
```

**Service Worker Features:**
- Offline functionality
- Background sync
- Push notifications
- Cache strategies

---

## 🧪 Tests (`tests/`)

```
tests/
├── setup.ts                     # Test setup
├── e2e/                         # End-to-end tests (Playwright)
│   ├── auth.spec.ts
│   ├── wellness.spec.ts
│   └── [e2e tests]
└── [test utilities]

src/**/__tests__/                # Unit tests (co-located)
├── services/__tests__/
│   └── habitTrackingService.test.ts
└── components/__tests__/
    └── WellnessDashboard.test.tsx
```

**Testing Strategy:**
- **Unit Tests:** Co-located with source (80% coverage goal)
- **Integration Tests:** `tests/integration/`
- **E2E Tests:** `tests/e2e/` (Playwright)

---

## 🏗️ Infrastructure (`infrastructure/`)

```
infrastructure/
├── docker/
│   └── Dockerfile
├── terraform/                   # Infrastructure as Code (planned)
└── scripts/                     # Deployment scripts
```

---

## 📦 Supabase (`supabase/`)

```
supabase/
├── config.toml                  # Supabase configuration
├── migrations/                  # Database migrations
│   ├── 001_initial_schema.sql
│   ├── 002_wellness_tables.sql
│   └── [migrations]
├── functions/                   # Edge functions
└── seed.sql                     # Seed data
```

**Database Tables:**
- `health_metrics` - Wearable data
- `mental_health_entries` - Mood tracking
- `habits` - Habit definitions
- `habit_logs` - Completions
- `habit_streaks` - Streak tracking
- `meal_logs` - Nutrition data
- `nutrition_plans` - Meal plans
- `community_posts` - Social posts
- `user_achievements` - Unlocked achievements
- `recommendations` - AI recommendations
- `privacy_settings` - User privacy controls

---

## ⚙️ Configuration Files

### TypeScript (`tsconfig.json`)
- Path aliases (`@/*` → `src/*`)
- Strict mode enabled
- Target: ES2020

### Vite (`vite.config.ts`)
- React plugin with SWC
- Path resolution
- Build optimization
- Dev server settings

### Tailwind (`tailwind.config.ts`)
- Custom color palette
- shadcn/ui integration
- Animations
- Typography plugin

### ESLint (`eslint.config.js`)
- TypeScript rules
- React hooks rules
- Import organization

---

## 🗺️ Path Aliases

Configured in `tsconfig.json`:

```typescript
{
  "@/*": "./src/*",
  "@/components/*": "./src/components/*",
  "@/services/*": "./src/services/*",
  "@/hooks/*": "./src/hooks/*",
  "@/types/*": "./src/types/*",
  "@/utils/*": "./src/utils/*"
}
```

**Usage:**
```typescript
// Instead of:
import { Button } from '../../../components/ui/button';

// Use:
import { Button } from '@/components/ui/button';
```

---

## 📊 File Count & Statistics

| Directory | File Count | Lines of Code |
|-----------|------------|---------------|
| `src/` | 200+ | ~15,000 |
| `src/services/` | 7 | ~3,500 |
| `src/components/` | 100+ | ~8,000 |
| `src/types/` | 1 | ~350 |
| `docs/` | ~30 (after consolidation) | ~5,000 |
| `tests/` | 20+ | ~2,000 |

**Total:** ~450 files, ~25,000 lines of code

---

## 🔍 Finding Things

### "Where is...?"

| Looking for | Location |
|------------|----------|
| Habit tracking logic | `src/services/habitTrackingService.ts` |
| Wellness dashboard | `src/components/wellness/WellnessDashboard.tsx` |
| Type definitions | `src/types/wellness.ts` |
| Rate limiting | `src/middleware/rateLimiting.ts` |
| Privacy controls | `src/middleware/privacyEnhancement.ts` |
| Service worker | `public/service-worker.js` |
| Database schema | `supabase/migrations/` |
| Tests | `src/**/__tests__/` or `tests/` |
| Documentation | `docs/` |
| UI components | `src/components/ui/` |

### "How do I...?"

| Task | Location |
|------|----------|
| Add a new service | Create in `src/services/` with singleton pattern |
| Add a UI component | Add to `src/components/ui/` |
| Add a page | Create in `src/pages/` |
| Add a route | Update `src/App.tsx` |
| Add a database table | Create migration in `supabase/migrations/` |
| Add a test | Co-locate in `__tests__/` folder |
| Add documentation | Add to appropriate `docs/` subdirectory |

---

## 🚀 Common Workflows

### Adding a New Feature

1. **Plan:** Document in `docs/planning/`
2. **Types:** Add to `src/types/`
3. **Service:** Create in `src/services/`
4. **Components:** Add to `src/components/`
5. **Tests:** Add to `__tests__/`
6. **Docs:** Update `docs/guides/`

### Debugging

1. **Check logs:** Browser console
2. **Check network:** Dev tools Network tab
3. **Check database:** Supabase dashboard
4. **Check service:** Add console.log in service methods
5. **Check types:** TypeScript errors

### Deploying

1. **Build:** `npm run build`
2. **Test:** `npm test`
3. **Preview:** `npm run preview`
4. **Deploy:** Push to Vercel

---

## 📚 Related Documentation

- [PROJECT.md](./PROJECT.md) - Project overview
- [QUICK_START.md](./QUICK_START.md) - Getting started
- [WELLNESS_PLATFORM_DOCUMENTATION.md](./WELLNESS_PLATFORM_DOCUMENTATION.md) - Features
- [DEPLOYMENT_INFRASTRUCTURE_PLAN.md](./DEPLOYMENT_INFRASTRUCTURE_PLAN.md) - Deployment
- [docs/README.md](./docs/README.md) - Documentation index

---

## ✅ Best Practices

### Organization
- ✅ Keep components small (< 200 lines)
- ✅ Co-locate related files
- ✅ Use path aliases
- ✅ Follow naming conventions

### Code
- ✅ TypeScript everywhere
- ✅ Singleton pattern for services
- ✅ Functional components with hooks
- ✅ Test coverage 80%+

### Documentation
- ✅ README in every major directory
- ✅ JSDoc comments for complex functions
- ✅ Update docs with code changes
- ✅ Examples in documentation

---

**Last Updated:** 2025-11-19
**Version:** 1.0.0
**Maintained By:** alawein-business
