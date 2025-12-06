# Full-Stack SaaS Template

> Production-ready template for React + Supabase applications.
> Based on patterns from Repz and LiveItIconic projects.

## Stack

| Layer | Technology |
|-------|------------|
| **Framework** | Vite + React 18 |
| **Language** | TypeScript 5.x |
| **Styling** | TailwindCSS + shadcn/ui |
| **State** | Zustand / React Context |
| **Backend** | Supabase (Auth, DB, Storage, Edge Functions) |
| **Payments** | Stripe |
| **Email** | Resend / Supabase |
| **Deployment** | Vercel / GitHub Pages |

## Directory Structure

```
project-name/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── deploy.yml
├── public/
│   ├── robots.txt
│   └── sitemap.xml
├── src/
│   ├── api/                    # API route handlers
│   │   ├── auth/
│   │   ├── payments/
│   │   └── webhooks/
│   ├── assets/                 # Static assets
│   ├── components/             # React components
│   │   └── ui/                 # shadcn/ui components
│   ├── config/                 # App configuration
│   │   └── analytics.ts
│   ├── contexts/               # React contexts
│   │   ├── AuthContext.tsx
│   │   └── CartContext.tsx
│   ├── hooks/                  # Custom hooks
│   │   ├── useAnalytics.ts
│   │   └── useLocalStorage.ts
│   ├── integrations/           # External service integrations
│   │   └── supabase/
│   ├── pages/                  # Page components
│   ├── services/               # Business logic
│   │   ├── authService.ts
│   │   ├── paymentService.ts
│   │   └── stripeService.ts
│   ├── styles/                 # Global styles
│   ├── types/                  # TypeScript types
│   └── utils/                  # Utility functions
├── supabase/
│   ├── config.toml
│   ├── functions/              # Edge functions
│   ├── migrations/             # Database migrations
│   └── seed.sql
├── tests/
│   ├── e2e/
│   └── integration/
├── .env.example
├── eslint.config.js
├── index.html
├── package.json
├── postcss.config.js
├── tailwind.config.ts
├── tsconfig.json
├── vercel.json
└── vite.config.ts
```

## Quick Start

```bash
# 1. Create new project
npx degit alawein/alawein/.metaHub/templates/saas-fullstack my-saas-app
cd my-saas-app

# 2. Install dependencies
npm install

# 3. Set up environment
cp .env.example .env.local
# Edit .env.local with your Supabase/Stripe keys

# 4. Start development
npm run dev
```

## Key Features

### Authentication
- Email/password login
- OAuth providers (Google, GitHub)
- Protected routes
- Session management

### Payments (Stripe)
- Subscription management
- One-time payments
- Webhook handling
- Customer portal

### Database (Supabase)
- Row-level security
- Real-time subscriptions
- Storage for files
- Edge functions for serverless logic

### UI/UX
- Responsive design
- Dark mode support
- Accessible components (shadcn/ui)
- Loading states
- Error handling

## Configuration Files

See individual template files for:
- `package.json.template`
- `vite.config.ts.template`
- `tailwind.config.ts.template`
- `tsconfig.json.template`

## Lovable.dev Integration

When exporting from lovable.dev:

1. Download ZIP from lovable.dev
2. Extract to project folder
3. Compare with this template structure
4. Add missing configurations:
   - `.github/workflows/`
   - `supabase/` migrations
   - `.env.example`
   - Test files
5. Apply governance (CLAUDE.md, CONTRIBUTING.md)

## Related Templates

| Template | Use Case |
|----------|----------|
| `saas-fullstack/` | Full React + Supabase app |
| `landing-page/` | Marketing/landing pages |
| `api-backend/` | API-only backend |
| `python-cli/` | Python command-line tools |

---

*Template version: 1.0.0*
*Based on: Repz, LiveItIconic project patterns*

