# QMLab Architecture

## Project Structure

```
qmlab/
├── src/
│   ├── components/          # React components
│   │   ├── ui/             # Reusable UI components (shadcn/ui)
│   │   └── ...             # Feature components
│   ├── config/             # Application configuration
│   │   └── theme.config.ts # Theme customization
│   ├── contexts/           # React contexts
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utilities and libraries
│   │   ├── logger.ts       # Centralized logging
│   │   └── utils.ts        # Common utilities
│   ├── pages/              # Page components
│   ├── styles/             # Global styles
│   │   ├── design-system/  # Design tokens and system
│   │   └── ...             # Feature-specific styles
│   └── utils/              # Utility functions
├── public/                 # Static assets
├── docs/                   # Documentation
└── ...                     # Config files
```

## Key Technologies

- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS + CSS Variables
- **UI Components**: shadcn/ui (Radix UI primitives)
- **State Management**: React Query (TanStack Query)
- **Routing**: React Router v6
- **3D Graphics**: Three.js
- **Animation**: CSS animations + React Spring

## Design System

The design system is centralized in `src/styles/design-system/` and `src/config/theme.config.ts`:

- **Colors**: Quantum-inspired palette (blue, purple, slate)
- **Typography**: Inter (sans-serif) + Fira Code (monospace)
- **Spacing**: 4px base unit with consistent scale
- **Effects**: Glass morphism, gradients, shadows
- **Animations**: Quantum-themed with reduced motion support

## Code Organization

### Components
- **UI Components** (`src/components/ui/`): Reusable, accessible primitives
- **Feature Components**: Domain-specific, composed from UI components
- **Page Components** (`src/pages/`): Top-level route components

### Hooks
- Custom hooks follow `use*` naming convention
- Located in `src/hooks/`
- Focused, single-responsibility design

### Styling
- CSS Modules for component-specific styles
- Tailwind utilities for layout and spacing
- CSS variables for theming (defined in design-system/)
- Semantic tokens over direct color values

### Utilities
- Pure functions in `src/lib/` and `src/utils/`
- Well-tested, reusable logic
- Type-safe with TypeScript

## Performance Considerations

- Lazy loading for routes and heavy components
- Code splitting at route level
- Optimized Three.js rendering
- Web Workers for intensive computations
- Service Worker for offline support

## Accessibility

- WCAG 2.1 AA compliant
- Semantic HTML
- ARIA labels and roles
- Keyboard navigation
- Screen reader support
- Focus management
- Color contrast compliance

## Testing Strategy

- Unit tests for utilities and hooks
- Component tests for UI components
- E2E tests for critical user flows (Playwright)
- Accessibility tests (jest-axe, Playwright)
- Visual regression tests (Playwright)

## Deployment

- Static site deployment
- CDN for assets
- Environment-based configuration
- CI/CD via GitHub Actions
