# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Project**: LLM Works - Open-source LLM evaluation platform  
**Domain**: llmworks.dev (planned)  
**Status**: Complete rebrand from Aegis AI implemented January 12, 2025

## Commands

### Development
- `npm run dev` - Start development server on port 8080
- `npm run build` - Build for production
- `npm run build:dev` - Build for development mode
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Installation
- `npm install` - Install dependencies

## Architecture Overview

**LLM Works** is a **Vite + React + TypeScript** application with **shadcn/ui** components and **Supabase** backend integration.

### Product Overview
Open-source LLM evaluation platform featuring:
- **The Arena**: Interactive model testing (debates, creative challenges, explanations)
- **The Bench**: Rigorous benchmarking (MMLU, TruthfulQA, custom tests)  
- **Arbiter Framework**: Consistent evaluation protocols
- **Verifier System**: Cryptographic audit trails
- **Dynamic Elo Rankings**: Performance tracking over time

### Tech Stack
- **Build Tool**: Vite
- **Framework**: React 18 with TypeScript
- **Routing**: React Router v6
- **UI Components**: shadcn/ui (Radix UI primitives with Tailwind CSS)
- **Styling**: Tailwind CSS with custom configuration
- **State Management**: React Query (TanStack Query)
- **Backend**: Supabase (authentication and database)
- **Forms**: React Hook Form with Zod validation

### Project Structure
- `/src/pages/` - Route components (Index, Arena, Bench, NotFound)
- `/src/components/` - Reusable React components
- `/src/components/ui/` - shadcn/ui component library (auto-generated, don't edit directly)
- `/src/integrations/supabase/` - Supabase client and types (auto-generated)
- `/src/lib/` - Utility functions
- `/src/hooks/` - Custom React hooks

### Key Configuration
- **Path Aliases**: `@/` maps to `./src/` directory
- **TypeScript**: Relaxed settings with no strict null checks and unused parameter warnings disabled
- **Routes**: All routes defined in `src/App.tsx` - add new routes above the catch-all route

### Important Notes
- **Brand Identity**: Recently rebranded from "Aegis AI" to "LLM Works" (January 2025)
- **Color Palette**: Analytical Blue (#4F83F0) primary, Insight Orange (#FF7A2A) accent  
- **Domain Migration**: Planned migration to llmworks.dev domain
- This is a Lovable.dev project - changes pushed to GitHub sync with the Lovable platform
- Supabase integration files in `/src/integrations/supabase/` are auto-generated - do not edit directly
- The UI components in `/src/components/ui/` follow shadcn/ui patterns - prefer using existing components over creating new ones
- Development server runs on port 8080 with IPv6 host binding (`::`)

### Brand Guidelines
- Use "LLM Works" for all branding (not "Aegis AI")
- Maintain Arena/Bench product naming (core metaphors)
- Preserve Arbiter/Verifier framework terminology
- Copy should emphasize: evidence-first, transparency, auditability, trust
- CTAs should be open-source friendly: "Try in Browser", "View Examples", "Browse Docs"

### Documentation References
- Brand guidelines: `/docs/brand/LLM_WORKS_BRAND_GUIDELINES.md`
- Implementation guide: `/docs/REBRAND_IMPLEMENTATION_GUIDE.md`
- Domain migration: `/docs/DOMAIN_MIGRATION_GUIDE.md`