# SimCore Development Guide

**Interactive Scientific Computing Laboratory** - Development Documentation

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `npm install` | Install dependencies |
| `npm run dev` | Start development server (http://localhost:5173) |
| `npm run build` | Production build |
| `npm run preview` | Preview production build |
| `npm test` | Run tests |
| `npm run typecheck` | TypeScript validation |
| `npm run lint` | ESLint check |
| `npm run audit:full` | Complete quality audit |

---

## Development Setup

### Prerequisites

- **Node.js**: 18+ (20 LTS recommended)
- **npm**: 9+ (comes with Node.js)
- **Git**: 2.x+

### Installation

```bash
# Clone AlaweinOS monorepo
git clone https://github.com/AlaweinOS/AlaweinOS.git
cd AlaweinOS/SimCore

# Install dependencies
npm install

# Start development server
npm run dev
```

Development server runs at: **http://localhost:5173**

### First Time Setup

1. **Install Playwright browsers** (for E2E testing):
   ```bash
   npx playwright install --with-deps
   ```

2. **Verify installation**:
   ```bash
   npm run typecheck  # Should show no errors
   npm run build      # Should build successfully
   ```

---

## Project Structure

```
SimCore/
├── src/                          # Source code
│   ├── components/               # React components
│   │   ├── ui/                   # shadcn/ui base components
│   │   ├── graphene/             # Graphene simulation components
│   │   ├── ising/                # Ising model components
│   │   ├── llg/                  # LLG dynamics components
│   │   ├── mos2/                 # MoS2 valley physics components
│   │   └── ...                   # Other simulation components
│   ├── pages/                    # Page components (routes)
│   │   ├── Index.tsx             # Homepage
│   │   ├── GrapheneBandStructure.tsx
│   │   ├── IsingModel.tsx
│   │   └── ...                   # 30+ simulation pages
│   ├── lib/                      # Core libraries
│   │   ├── webgpu-*.ts           # WebGPU acceleration
│   │   ├── graphene-physics-*.ts # Physics calculations
│   │   └── utils.ts              # Utilities
│   ├── hooks/                    # Custom React hooks
│   ├── styles/                   # CSS and styling
│   ├── types/                    # TypeScript definitions
│   ├── data/                     # Module metadata
│   ├── App.tsx                   # Main application component
│   └── main.tsx                  # Entry point
├── public/                       # Static assets
│   ├── sw.js                     # Service worker (PWA)
│   ├── manifest.json             # PWA manifest
│   └── workers/                  # Web workers
├── docs/                         # Documentation
├── scripts/                      # Build and audit scripts
├── index.html                    # HTML entry point
├── package.json                  # Dependencies and scripts
├── tsconfig.json                 # TypeScript configuration
├── vite.config.ts                # Vite build configuration
└── tailwind.config.ts            # Tailwind CSS configuration
```

---

## Development Workflow

### Adding a New Simulation

1. **Create page component**:
   ```bash
   # Create new file: src/pages/YourSimulation.tsx
   ```

2. **Implement simulation**:
   ```typescript
   import React, { useState } from 'react';
   import { PhysicsModuleLayout } from '@/components/PhysicsModuleLayout';

   export default function YourSimulation() {
     const [param, setParam] = useState(1.0);

     return (
       <PhysicsModuleLayout
         title="Your Simulation"
         category="quantum"
         description="Description of your simulation"
       >
         {/* Simulation UI and visualization */}
       </PhysicsModuleLayout>
     );
   }
   ```

3. **Add to modules data**:
   ```typescript
   // src/data/modules.ts
   export const physicsModules: PhysicsModule[] = [
     // ... existing modules
     {
       id: 'your-simulation',
       title: 'Your Simulation',
       description: '...',
       category: 'Quantum Mechanics',
       difficulty: 'Intermediate',
       tags: ['Physics', 'Simulation'],
       isImplemented: true,
       route: '/modules/your-simulation',
       theory: { /* ... */ }
     }
   ];
   ```

4. **Add route**:
   ```typescript
   // src/App.tsx
   import YourSimulation from '@/pages/YourSimulation';

   // In Routes component:
   <Route path="/modules/your-simulation" element={<YourSimulation />} />
   ```

5. **Test**:
   ```bash
   npm run dev
   # Navigate to http://localhost:5173/modules/your-simulation
   ```

### Code Style

**TypeScript**:
- Use strict mode
- Explicit return types for functions
- Interfaces for props and state
- No `any` types without justification

**React**:
- Functional components with hooks
- Props interfaces defined above component
- Comprehensive JSDoc comments

**Formatting**:
- ESLint for linting
- Prettier for formatting (via ESLint integration)
- 2-space indentation
- Single quotes for strings

### Testing

```bash
# Run unit tests
npm test

# Run E2E tests
npm run test:e2e  # (if configured)

# Type checking
npm run typecheck

# Linting
npm run lint
```

---

## WebGPU Development

### Using WebGPU for Simulations

SimCore includes a WebGPU acceleration framework:

```typescript
import { initializeWebGPU, createComputePipeline } from '@/lib/webgpu-physics';

// Initialize WebGPU
const gpu = await initializeWebGPU();

// Create compute shader
const shader = `
  @group(0) @binding(0) var<storage, read_write> data: array<f32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    data[index] = data[index] * 2.0; // Example computation
  }
`;

// Run computation
const result = await runWebGPUComputation(gpu, shader, inputData);
```

**Features**:
- Automatic CPU fallback
- Memory management
- Compute shader pipelines
- Performance profiling

---

## Performance Optimization

### Targets

- **Lighthouse Desktop**: 90+ score
- **Lighthouse Mobile**: 85+ score
- **LCP** (Largest Contentful Paint): <2.5s
- **FID** (First Input Delay): <100ms
- **CLS** (Cumulative Layout Shift): <0.1

### Optimization Checklist

- [ ] Code splitting with lazy loading
- [ ] Image optimization (WebP, responsive)
- [ ] Tree shaking (Vite handles automatically)
- [ ] Bundle size analysis
- [ ] Service worker caching
- [ ] WebGPU for intensive calculations
- [ ] Memoization for expensive computations

### Running Performance Audits

```bash
# Build production version
npm run build

# Preview build
npm run preview

# Run full audit suite
npm run audit:full

# Check specific audits
npm run audit:lh      # Lighthouse
npm run audit:axe     # Accessibility
npm run audit:pw      # Playwright tests
```

Results saved in `.audit/` directory.

---

## Accessibility

SimCore is WCAG 2.1 AA compliant.

### Guidelines

- **Keyboard navigation**: All interactive elements accessible via keyboard
- **Screen readers**: Proper ARIA labels and semantic HTML
- **Color contrast**: Minimum 4.5:1 for text
- **Focus indicators**: Visible focus states
- **Responsive text**: Scalable font sizes

### Testing Accessibility

```bash
# Run accessibility audit
npm run audit:axe

# Run Pa11y crawler
npm run audit:pa11y
```

---

## Progressive Web App (PWA)

SimCore is a full PWA with offline capability.

### Features

- **Service Worker**: Smart caching (`public/sw.js`)
- **Manifest**: App metadata (`public/manifest.json`)
- **Offline mode**: Fallback page (`public/offline.html`)
- **Installable**: Add to homescreen

### Testing PWA

1. Build production version: `npm run build`
2. Serve: `npm run preview`
3. Open Chrome DevTools → Application
4. Check:
   - Service Worker registered
   - Manifest valid
   - Offline mode works

---

## Deployment

### Production Build

```bash
# Create optimized build
npm run build

# Output in dist/ directory
# Serve static files from dist/
```

### Build Optimization

Vite automatically:
- Minifies JavaScript/CSS
- Tree-shakes unused code
- Optimizes images
- Generates source maps
- Code splits by route

### Deployment Targets

**Static Hosting**:
- Netlify
- Vercel
- GitHub Pages
- AWS S3 + CloudFront

**Configuration**:
```bash
# Build
npm run build

# Deploy dist/ directory to your hosting provider
```

---

## Troubleshooting

### Common Issues

**Issue**: `npm install` fails
- **Solution**: Clear npm cache: `npm cache clean --force`
- Use Node.js 18+ LTS

**Issue**: TypeScript errors
- **Solution**: Run `npm run typecheck` to see all errors
- Check `tsconfig.json` configuration

**Issue**: WebGPU not working
- **Solution**: Check browser support (Chrome 113+, Edge 113+)
- CPU fallback will activate automatically

**Issue**: Build fails
- **Solution**: Clear cache: `rm -rf dist node_modules .vite`
- Reinstall: `npm install`
- Rebuild: `npm run build`

**Issue**: Slow development server
- **Solution**: Vite uses esbuild for fast HMR
- Check for large files in `public/`
- Reduce number of open modules

---

## Contributing

See root [CONTRIBUTING.md](../CONTRIBUTING.md) for:
- Code standards
- Git workflow
- PR process
- Testing requirements

### SimCore-Specific Guidelines

**Scientific Accuracy**:
- Verify physics equations against literature
- Include references in theory panels
- Test numerical accuracy
- Document assumptions

**Performance**:
- Profile before optimizing
- Use WebGPU for intensive calculations
- Lazy load simulation components
- Memoize expensive computations

**Documentation**:
- Add KaTeX equations for theory
- Include usage examples
- Document parameters
- Link to research papers

---

## Resources

### Internal Documentation

- [README.md](README.md) - Project overview
- [SIMCORE_CLAUDE_CODE_DOCUMENTATION.md](SIMCORE_CLAUDE_CODE_DOCUMENTATION.md) - Complete documentation
- [SIMCORE_CLAUDE_CODE_SUPERPROMPT.md](SIMCORE_CLAUDE_CODE_SUPERPROMPT.md) - Implementation guide
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) - Additional development docs

### External Resources

- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Vite Guide](https://vitejs.dev/guide/)
- [shadcn/ui](https://ui.shadcn.com)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [Tailwind CSS](https://tailwindcss.com)

---

## Contact

**Maintainer**: Dr. Meshal Alawein
**Email**: meshal@berkeley.edu
**Organization**: AlaweinOS

---

**Last Updated**: 2025-11-19
