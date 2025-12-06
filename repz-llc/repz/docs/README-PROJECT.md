# REPZ Coach Pro Platform

> **Elite Fitness Coaching Platform** - Connecting certified personal trainers with clients through personalized training programs, nutrition plans, and progress tracking.

[![Production Ready](https://img.shields.io/badge/Status-Production%20Ready-green)](https://repzcoach.com)
[![React](https://img.shields.io/badge/React-18.3-blue)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.5-blue)](https://www.typescriptlang.org/)
[![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-green)](https://supabase.com/)
[![Stripe](https://img.shields.io/badge/Payments-Stripe-purple)](https://stripe.com/)

---

## 🎯 Overview

REPZ Coach Pro is a comprehensive fitness coaching SaaS platform that provides:

- **Personalized Training Programs** - AI-powered workout recommendations
- **Nutrition Planning** - Macro tracking, meal plans, auto grocery lists
- **Progress Tracking** - Body measurements, strength benchmarks, photos
- **Wearable Integration** - Whoop, Strava, Apple Health, Google Fit
- **AI Coaching Assistant** - Form analysis, predictive analytics
- **Tiered Subscriptions** - 4 tiers from $89-$349/month

---

## 🏗️ Architecture

### Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 18 + TypeScript + Vite 7 |
| **UI Framework** | Radix UI + Tailwind CSS + shadcn/ui |
| **Backend** | Supabase (PostgreSQL + Edge Functions) |
| **State Management** | TanStack React Query + React Context |
| **Authentication** | Supabase Auth (PKCE flow) |
| **Payments** | Stripe (subscriptions + webhooks) |
| **Mobile** | Capacitor (iOS/Android) |
| **Testing** | Vitest + Playwright + MSW |
| **Deployment** | Vercel |

### Directory Structure

```
repz/
├── src/
│   ├── components/     # 60+ component directories
│   ├── pages/          # 25+ page components
│   ├── hooks/          # 26 custom hooks
│   ├── contexts/       # Auth context
│   ├── services/       # Business logic services
│   ├── constants/      # Tier configs, routes, templates
│   ├── types/          # TypeScript definitions
│   └── integrations/   # Third-party integrations
├── supabase/
│   ├── functions/      # 40 Edge Functions
│   └── migrations/     # 115+ database migrations
├── tests/
│   ├── e2e/            # Playwright tests
│   └── unit/           # Vitest tests
├── docs/               # Documentation
└── scripts/            # Automation scripts
```

---

## 💎 Subscription Tiers

| Tier | Price | Key Features |
|------|-------|--------------|
| **Core** | $89/mo | Training program, nutrition plan, 72h response |
| **Adaptive** | $149/mo | + Biomarkers, wearables, weekly check-ins, 48h response |
| **Performance** | $229/mo | + AI assistant, form analysis, PEDs protocols, 24h response |
| **Longevity** | $349/mo | + In-person training, concierge service, 12h response |

### Billing Options
- Monthly (base price)
- Quarterly (5% discount)
- Semi-annual (10% discount)
- Annual (20% discount)

---

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- npm or yarn
- Supabase account
- Stripe account

### Installation

```bash
# Clone the repository
git clone https://github.com/repz-llc/repz.git
cd repz

# Install dependencies
npm install

# Copy environment variables
cp .env.example .env.local

# Start development server
npm run dev
```

### Environment Variables

```bash
# Supabase
VITE_SUPABASE_URL=your_supabase_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Stripe
STRIPE_SECRET_KEY=your_stripe_secret_key
STRIPE_WEBHOOK_SECRET=your_webhook_secret

# See .env.example for complete list
```

---

## 📋 Available Scripts

### Development

```bash
npm run dev              # Start dev server (port 8080)
npm run build            # Production build
npm run preview          # Preview production build
npm run type-check       # TypeScript validation
npm run lint             # ESLint check
```

### Testing

```bash
npm test                 # Run Vitest in watch mode
npm run test:run         # Run tests once
npm run test:coverage    # Coverage report
npm run test:e2e         # Playwright E2E tests
npm run test:a11y        # Accessibility tests
```

### Database

```bash
npm run db:migrate       # Apply migrations
npm run db:seed          # Seed database
npm run db:reset         # Reset database
```

### Deployment

```bash
npm run build:production # Production build
npm run validate:production # Full validation
npm run vercel:deploy    # Deploy to Vercel
```

---

## 🔐 Authentication Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  Supabase   │────▶│  Database   │
│   (React)   │     │    Auth     │     │  (Postgres) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │
       │                   ▼
       │           ┌─────────────┐
       └──────────▶│   Stripe    │
                   │  (Payments) │
                   └─────────────┘
```

### Security Features
- PKCE OAuth flow
- JWT token validation
- Role-based access control (client/coach/admin)
- Row Level Security on all tables
- Stripe webhook signature verification

---

## 📊 Database Schema

### Core Tables

| Table | Purpose |
|-------|---------|
| `client_profiles` | Client user data and tier info |
| `coach_profiles` | Coach credentials and specializations |
| `subscriptions` | Stripe subscription management |
| `peds_protocols` | PEDs tracking with medical oversight |
| `bioregulators_protocols` | Bioregulator protocols |
| `food_items` | Nutrition database |
| `meal_plans` | Meal planning system |
| `grocery_lists` | Auto-generated grocery lists |
| `medical_consultations` | Medical oversight records |

---

## 🔌 Integrations

### Wearable Devices
- **Whoop** - Recovery, strain, sleep data
- **Strava** - Activity tracking
- **Apple Health** - iOS health data
- **Google Fit** - Android health data

### Third-Party Services
- **Stripe** - Payment processing
- **Calendly** - Booking integration
- **OpenAI** - AI coaching features
- **ElevenLabs** - Voice synthesis
- **Sentry** - Error tracking
- **Google Analytics** - Analytics

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `src/constants/tiers.ts` | Tier configuration (single source of truth) |
| `src/constants/trainingTemplates.ts` | Training program templates |
| `src/contexts/AuthContext.tsx` | Authentication state |
| `src/App.tsx` | Main routing configuration |
| `CLAUDE.md` | AI assistant guidelines |
| `PRODUCTION-CONFIG.md` | Production configuration |
| `DEPLOYMENT_GUIDE.md` | Deployment instructions |

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](./CLAUDE.md) | AI assistant development guide |
| [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) | Complete deployment instructions |
| [PRODUCTION-CONFIG.md](./PRODUCTION-CONFIG.md) | Stripe, Calendly, Analytics setup |
| [DEVELOPER-ONBOARDING-GUIDE.md](./DEVELOPER-ONBOARDING-GUIDE.md) | New developer setup |
| [docs/COMPREHENSIVE-AUDIT-REPORT.md](./docs/COMPREHENSIVE-AUDIT-REPORT.md) | Full codebase audit |
| [docs/CLIENT-INTAKE-FORM-PDF.md](./docs/CLIENT-INTAKE-FORM-PDF.md) | Client intake form template |

---

## 🧪 Testing Strategy

### Test Types
- **Unit Tests** - Vitest with React Testing Library
- **E2E Tests** - Playwright for user journeys
- **Accessibility** - axe-core integration
- **Performance** - k6 load testing

### Coverage Targets
- Statements: 80%
- Branches: 80%
- Functions: 80%
- Lines: 80%

---

## 🚢 Deployment

### Vercel Configuration
- Build command: `npm run build:production`
- Output directory: `dist`
- Framework: Vite
- Region: IAD1 (US East)

### Security Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Referrer-Policy: strict-origin-when-cross-origin

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- TypeScript strict mode
- ESLint + Prettier formatting
- Conventional commits
- PR reviews required

---

## 📄 License

This project is proprietary software owned by REPZ LLC.

---

## 📞 Support

- **Website:** [repzcoach.com](https://repzcoach.com)
- **Email:** support@repzcoach.com
- **Documentation:** See `/docs` folder

---

**Built with ❤️ by the REPZ Team**

*Last Updated: December 2025*
