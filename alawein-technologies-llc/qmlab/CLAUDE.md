# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QMLab - An interactive quantum machine learning playground for building quantum circuits, visualizing quantum states, and training quantum ML models. Built with React, TypeScript, Vite, and shadcn/ui components.

**Production URL**: https://qmlab.online/  
**Lovable Project**: https://lovable.dev/projects/3c1db199-322a-4441-ad0f-dc0bd0865f78

## Essential Commands

```bash
# Install dependencies
npm install

# Start development server (runs on port 8080)
npm run dev

# Build for production
npm run build

# Build for development mode
npm run build:dev

# Run linter
npm run lint

# Preview production build (port 4173)
npm run preview

# Run tests
npm run test                 # Standard Playwright tests
npm run test:a11y            # Accessibility tests
npm run test:mobile          # Mobile device tests
```

## Tech Stack

### Core
- **Build Tool**: Vite with React SWC plugin
- **Framework**: React 18 with TypeScript
- **Routing**: React Router v6
- **UI Components**: shadcn/ui (Radix UI primitives + Tailwind CSS)
- **State Management**: React Query (TanStack Query)
- **Forms**: React Hook Form with Zod validation
- **3D Visualization**: Three.js for quantum state visualizations

### Testing
- **E2E Testing**: Playwright with multiple configurations
  - Standard tests in `/tests`
  - Accessibility tests in `/tests/accessibility`
  - Multiple browser profiles including reduced motion and high contrast modes
  - WebGL disabled testing for fallback scenarios

## Architecture

### Component Structure
1. **Quantum Components** (`/src/components/`)
   - `BlochSphere.tsx` - 3D quantum state visualization
   - `CircuitBuilder.tsx` - Drag-and-drop circuit constructor
   - `TrainingDashboard.tsx` - ML training metrics and charts
   - `MiniBlochSphere.tsx` - Compact state visualization

2. **UI Library** (`/src/components/ui/`)
   - 40+ shadcn/ui components (copied, not npm packages)
   - Custom `StatusChip.tsx` for quantum state indicators
   - All components follow Radix UI + Tailwind pattern

3. **Utility Components**
   - `AccessibilityProvider.tsx` - Screen reader announcements
   - `ErrorBoundary.tsx` - Error handling
   - `TutorialOverlay.tsx` - Interactive tutorials
   - `QuantumBackground.tsx` - Animated particle effects

### Import Patterns
- **Path Alias**: `@/` maps to `./src/`
- **TypeScript**: Relaxed strictness for rapid development
  - `noImplicitAny: false`
  - `strictNullChecks: false`
  - `noUnusedParameters: false`

### Styling System
- **Tailwind CSS** with quantum-themed customizations
- **Custom Colors**: `state-pure`, `state-mixed`, `state-entangled`
- **Fonts**: Space Grotesk (display), Inter (body), JetBrains Mono (code)
- **Animations**: `quantum-pulse`, `quantum-glow`, `float`
- **Accessibility**: Respects `prefers-reduced-motion`

## Analytics & Monitoring

### Google Analytics 4
- **GTM Container**: GTM-P5V49HTH
- **GA4 Property**: G-7810TS77ND
- **Event Tracking**: Specialized quantum interaction events in `/src/lib/analytics.ts`
  - Circuit building events
  - Training metrics
  - Bloch sphere interactions
  - Tutorial progression

### Event Functions (`/src/lib/analytics.ts`)
```typescript
trackQuantumEvents.circuitGateAdded()
trackQuantumEvents.trainingStarted()
trackQuantumEvents.blochSphereRotated()
// 20+ specialized event types
```

## Development Environment

### Vite Configuration
- Development: All interfaces (`::`) on port 8080
- Preview: Port 4173 for production builds
- Hot module replacement with React SWC
- Component tagger for Lovable.dev integration

### Deployment
- **Primary**: Lovable.dev with automated Git syncing
- **Repository**: GitHub with main branch deployment
- **Domain**: qmlab.online (configured through Lovable)

## Accessibility Requirements

### Implementation Checklist
- Screen reader support via AccessibilityProvider
- Skip navigation links for keyboard users
- ARIA labels on all interactive elements
- Focus management in modals and overlays
- Reduced motion alternatives for animations

### Testing Requirements
- Test with screen readers for quantum visualizations
- Verify keyboard navigation for all interactions
- Check reduced motion doesn't break functionality
- Run `npm run test:a11y` for automated checks

## Performance Considerations

- Lazy load quantum visualization components
- Three.js optimizations for 3D rendering
- Component-level performance monitoring
- Reduced motion CSS alternatives
- WebGL fallback support