# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REPZ is a comprehensive fitness coaching platform built with React, TypeScript, and Supabase. It connects certified personal trainers with clients through a tiered subscription model, offering personalized training programs, nutrition plans, and progress tracking.

## Development Commands

### Core Development
- `npm run dev` - Start development server (port 8080 configured in vite.config.ts)
- `npm run build` - Production build
- `npm run build:dev` - Development build
- `npm run build:staging` - Staging environment build  
- `npm run build:production` - Production environment build
- `npm run preview` - Preview production build
- `npm run preview:staging` - Preview staging build

### Testing & Quality
- `npm test` - Run tests with Vitest in watch mode
- `npm run test:run` - Run tests once without watch mode
- `npm run test:coverage` - Run tests with coverage report (80% threshold)
- `npm run test:ui` - Run tests with Vitest UI interface
- `npm run test:unit` - Run unit tests only
- `npm run test:component` - Run component tests only
- `npm run test:integration` - Run integration tests
- `npm run test:e2e` - Run Playwright end-to-end tests
- `npm run test:e2e:headed` - Run E2E tests with browser UI
- `npm run test:e2e:debug` - Debug E2E tests
- `npm run test:a11y` - Run accessibility tests
- `npm run test:performance:smoke` - Run smoke performance tests with k6
- `npm run test:auth` - Run auth-specific tests (pattern matching)
- `npm run test:stripe` - Run Stripe-specific tests
- `npm run type-check` - TypeScript type checking without emit
- `npm run lint` - ESLint with automatic fixes
- Test infrastructure: Vitest configured with jsdom, React Testing Library, MSW for mocking
- Test setup file: `src/test/setup.ts`
- Test files located in `__tests__/` directories within component folders
- Run single test file: `npm test -- path/to/test.tsx`

### Database & Backend
- `npm run db:migrate` - Apply Supabase migrations
- `npm run db:reset` - Reset database
- `npm run db:seed` - Seed database
- Supabase client configured at `src/integrations/supabase/client.ts`
- Edge functions in `supabase/functions/` (30+ functions)
- Database migrations in `supabase/migrations/`
- Generate types: `npx supabase gen types typescript --local > src/integrations/supabase/types.ts`

### Analysis & Monitoring
- `npm run analyze:bundle` - Analyze bundle size with vite-bundle-analyzer
- `npm run analyze:deps` - Analyze dependencies
- `npm run audit:quick` - Quick project audit
- `npm run audit:routes` - Audit dead pages/routes
- `npm run validate:production` - Full production validation (type-check + audits + build)
- `npm run refactor:analyze` - Analyze refactoring opportunities

## Architecture Overview

### Tech Stack
- **Frontend**: React 18 + TypeScript + Vite
- **UI Framework**: Radix UI + Tailwind CSS + shadcn/ui (50+ components)
- **Backend**: Supabase (PostgreSQL + Edge Functions)
- **State Management**: React Query (@tanstack/react-query) + React Context
- **Authentication**: Supabase Auth with PKCE flow
- **Payments**: Stripe integration with webhook handling
- **Mobile**: Capacitor for native features
- **Forms**: react-hook-form + Zod validation
- **Charts**: Recharts for data visualization
- **Testing**: Vitest + Playwright + MSW

### Key Directories
- `src/components/` - Reusable UI components organized by feature
- `src/pages/` - Route-level page components  
- `src/hooks/` - Custom React hooks for data fetching and state
- `src/contexts/` - React context providers (AuthContext, etc.)
- `src/lib/` - Utility functions and configurations
- `src/constants/` - Application constants (routes, tiers)
- `src/integrations/supabase/` - Supabase client and types
- `src/test/` - Test setup and fixtures
- `supabase/functions/` - Edge Functions (30+ serverless functions)
- `supabase/migrations/` - Database migrations

### Component Organization
Components are organized by feature area with extensive subdirectories:
- `ui/` - shadcn/ui components and design system (50+ components)
- `auth/` - Authentication forms and flows
- `dashboard/` - Dashboard layouts with role-specific content
- `intake/` - Multi-step onboarding forms (7 steps: Account → Personal → Health → Training → Nutrition → Goals → Payment)  
- `analytics/` - Business intelligence, performance metrics, predictive insights
- `pricing/` - Tier-based subscription displays and comparison tools
- `client/` - Client-specific features (workout tracking, progress, check-ins)
- `mobile/` - Mobile-optimized components and Capacitor integration
- `ai/` - AI coaching features and form analysis
- `testing/` - Development and QA testing infrastructure
- `shared/` - Cross-feature shared components and utilities

## Tier System Architecture

The platform uses a 4-tier subscription model with correct tier IDs:

1. **Core Program** (`core`) - $89/month - Essential training and nutrition
2. **Adaptive Engine** (`adaptive`) - $149/month - Interactive coaching with tracking  
3. **Performance Suite** (`performance`) - $229/month - Advanced biohacking and AI
4. **Longevity Concierge** (`longevity`) - $349/month - Premium concierge service

### Tier Configuration
- **IMPORTANT**: Always use correct tier IDs: `core`, `adaptive`, `performance`, `longevity`
- Tier definitions: `src/constants/tiers.ts`
- Database enum: `tier_enum` in Supabase uses these exact values
- Feature mapping with extensive tier-based access control
- Pricing includes monthly/quarterly/semi-annual/annual options with discounts
- Stripe price IDs configured via environment variables for each tier and billing period

### Deprecated Tier Names (DO NOT USE)
- ❌ `baseline`, `prime`, `precision` (old system)
- ❌ `foundational`, `essentials`, `deluxe` (intermediate system)
- ✅ Always use: `core`, `adaptive`, `performance`, `longevity`

## Authentication Flow

- Supabase Auth with localStorage persistence
- PKCE flow for security
- Route protection via `ProtectedRoute` and `AdminRoute` components
- Role-based access control (client/coach/admin)
- Session debugging available via `AuthDebugger` component

## Key Features by Component

### Multi-Step Intake Form (`src/components/intake/`)
- 7-step onboarding process
- Account → Personal → Health → Training → Nutrition → Goals → Payment
- Progress tracking and validation
- Tier selection integration

### Dashboard System (`src/components/dashboard/`)
- Role-specific dashboards (client/coach/admin)
- Tier-based feature gating
- Real-time progress tracking
- Analytics integration

### Analytics Platform (`src/components/analytics/`)
- Business intelligence dashboard
- Performance metrics and charts
- Predictive insights (Performance+ tiers)
- Data visualization with Recharts

### Testing Infrastructure (`src/components/testing/`)
- Comprehensive testing dashboard
- Stripe test environment
- Development debug panels
- System health monitoring

## Styling System

### Design System
- Tailwind CSS with extensive custom configuration
- REPZ brand colors (orange #F15B23, black #000000)
- Premium color palette with HSL semantic tokens
- Custom animations and transitions
- Responsive breakpoints with mobile-first approach

### Component Theming
- shadcn/ui components with dark mode support
- CSS custom properties for theme switching
- Elegant animations (`animate-elegant-float`, `animate-premium-glow`)
- Professional gradients and glassmorphism effects

## Database Schema

### Key Tables (via Supabase)
- `client_profiles` - Client user profiles with role and tier information
- `coach_profiles` - Coach profiles with certifications and specializations
- `subscriptions` - Stripe subscription management with tier tracking
- `client_data` - Client fitness data and progress tracking
- `coach_assignments` - Coach-client relationships
- `messages` - Communication between coaches and clients
- `achievements` - Gamification and milestone tracking
- `supplement_protocols` - Tier-based supplement recommendations
- `ai_analysis_results` - AI coaching analysis data

### Row Level Security (RLS)
- All tables use RLS policies for data protection
- Role-based access control enforced at database level
- Coach permissions for client data access

## Development Guidelines

### Code Patterns
- TypeScript with relaxed strict mode for rapid development
- React hooks for state management
- Custom hooks for data fetching (`useClientData`, `useAnalytics`)
- Error boundaries for graceful error handling
- Path aliases: `@/` maps to `src/`

### File Conventions
- Components use PascalCase (e.g., `ClientDashboard.tsx`)
- Hooks use camelCase with `use` prefix (e.g., `useAuth.tsx`)
- Constants use UPPER_SNAKE_CASE
- Test files in `__tests__/` directories or `*.test.tsx`

### State Management
- React Query for server state and caching
- React Context for global client state (auth, theme)
- Local state with useState/useReducer for component state
- Form state managed with react-hook-form + zod validation

## Testing Strategy

### Test Configuration
- Vitest with jsdom environment
- React Testing Library for component tests
- MSW for API mocking
- Playwright for E2E tests
- Test fixtures in `src/test/fixtures/`
- Setup file configures global test utilities

### Coverage Areas
- Component rendering and user interactions
- Custom hooks behavior
- Authentication flows
- Form validation
- Tier access control
- Coverage thresholds: 80% for all metrics

## Build Configuration

### Vite Configuration
- Port 8080 for development server
- Code splitting with manual chunks (vendor, ui, charts, utils, supabase, stripe)
- Source maps enabled for development/staging
- Chunk size warning limit: 1000kb
- ESBuild minification for production

### TypeScript Configuration
- Path aliases configured (`@/` for `src/`)
- Relaxed type checking (`noImplicitAny: false`, `strictNullChecks: false`)
- Multiple tsconfig files for app and node environments

## Edge Functions

The project includes 30+ Supabase Edge Functions for various backend operations:
- **AI Functions**: `ai-fitness-analysis`, `form-analysis-ai`, `nutrition-ai-assistant`, `predictive-health-analytics`
- **Stripe Integration**: `create-checkout`, `stripe-webhook`, `customer-portal`, `subscription-management`
- **Analytics**: `business-intelligence`, `cohort-analysis`, `ab-testing-engine`
- **Coaching**: `live-workout-coaching`, `video-analysis`, `elevenlabs-tts`
- **Monitoring**: `production-monitoring`, `security-monitor`, `error-handler`

## Environment Configuration

### Configuration Files
- `.env.example` - Template with required environment variables
- `.env.production` - Production environment settings
- `.env.staging` - Staging environment settings
- **`PRODUCTION-CONFIG.md`** - Complete production configuration guide with all integration details

### Required Environment Variables
- `VITE_SUPABASE_URL` - Supabase project URL
- `VITE_SUPABASE_ANON_KEY` - Supabase anonymous key
- `VITE_STRIPE_PUBLIC_KEY` - Stripe publishable key
- `VITE_STRIPE_SECRET_KEY` - Stripe secret key (Edge Functions only)
- `VITE_OPENAI_API_KEY` - OpenAI API key for AI features
- Stripe price IDs for each tier and billing period (see PRODUCTION-CONFIG.md)
- Calendly integration URLs for booking systems
- Google Analytics and Tag Manager IDs
- Feature flags and tier-specific configuration variables

### Production Integration Setup
**IMPORTANT:** Always reference `PRODUCTION-CONFIG.md` for:
- Complete Stripe product and price IDs for all tiers and billing cycles
- Calendly booking URLs for in-person training and consultations
- Google Analytics and Tag Manager installation codes
- Environment variable templates and TypeScript integration examples
- Security best practices and deployment guidelines

## Critical Architecture Patterns

### Data Fetching Pattern
- All data fetching uses React Query hooks in `src/hooks/`
- Queries are organized by feature domain (e.g., `useClientData`, `useCoachAssignments`)
- Real-time subscriptions integrate with React Query cache invalidation
- Error handling via React Query's built-in retry logic

### Route-Based Code Splitting
- Routes defined in `src/App.tsx` with lazy loading
- Role-based route protection in `src/components/auth/ProtectedRoute.tsx`
- Tier-based feature access via `useTierAccess` hook

### Form Handling Architecture
- Multi-step forms use shared state via React Context
- Form validation with Zod schemas in component files
- react-hook-form integration for performance
- Progress persistence in localStorage for intake forms

### Supabase Integration Pattern
- Database queries through typed client in `src/integrations/supabase/client.ts`
- Edge Functions called via `supabase.functions.invoke()`
- Real-time subscriptions with automatic cleanup
- RLS policies require authenticated user context

## Performance Optimizations

### Bundle Optimization
- Manual code splitting for vendor libraries
- Separate chunks for UI components, charts, utilities
- Dynamic imports for route-level components
- Tree shaking enabled

### Development Workflow
- Fast refresh with Vite HMR
- Pre-bundled dependencies for faster dev server
- TypeScript transpilation with SWC

## Important Notes

- The codebase uses a modular architecture with feature-based organization
- Extensive tier-based access control throughout the application
- Real-time features via Supabase subscriptions
- Comprehensive error handling and user feedback
- Professional design system with premium aesthetics
- Mobile-responsive with native app capabilities
- **CRITICAL**: Always run `npm run lint` and `npm run type-check` before commits
- Database migrations available in `supabase/migrations/` directory
- **Production Configuration:** See `PRODUCTION-CONFIG.md` for complete Stripe, Calendly, and analytics setup