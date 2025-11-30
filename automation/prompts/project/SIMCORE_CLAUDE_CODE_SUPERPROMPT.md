# SimCore Claude Code Superprompt
*Ultimate implementation guide for the SimCore Interactive Scientific Computing Laboratory*

## 🚀 Master Superprompt

```
Implement the complete SimCore Interactive Scientific Computing Laboratory as a world-class scientific computing platform. Create a comprehensive web-based laboratory with 25+ interactive scientific simulations across physics, mathematics, and scientific machine learning.

CORE REQUIREMENTS:
- React 18.3 + TypeScript 5.5 + Vite architecture
- 25+ interactive scientific modules across 7 physics domains
- WebGPU acceleration framework for intensive calculations
- Research-grade accuracy with literature-verified implementations
- Progressive Web App with offline capability and service workers
- Comprehensive audit system (Lighthouse + axe + Playwright + CSS stats + Pa11y)
- Mobile-first responsive design with touch optimization
- Publication-quality visualizations using plotly.js and Three.js
- shadcn/ui components with physics domain theming
- KaTeX mathematical typesetting integration
- Complete accessibility compliance (WCAG 2.1 AA)

SCIENTIFIC MODULES TO IMPLEMENT:
Quantum Mechanics: Graphene Band Structure, MoS₂ Valley Physics, TDSE Solver, Quantum Tunneling
Statistical Physics: 2D Ising Model, Boltzmann Distribution, LLG Dynamics
Advanced Methods: Physics-Informed Neural Networks, Machine Learning Analysis, WebGPU Simulations
Mathematical Modeling: Advanced Mathematical Methods, Scientific ML, Numerical Analysis

TECHNICAL SPECIFICATIONS:
- WebGPU compute pipelines for parallel calculations
- Three-tier design token system with physics domain theming
- Lazy-loaded modules with code splitting optimization
- Memory-optimized GPU algorithms with cleanup utilities
- Real-time simulations at 60fps with hardware acceleration
- Comprehensive error boundaries and fallback systems
- Service workers with smart caching strategies
- Cross-platform compatibility with progressive enhancement

QUALITY ASSURANCE:
- Automated audit pipeline with multiple tools
- Performance targets: 90+ Lighthouse score, <2.5s LCP
- 95%+ test coverage for physics calculations
- Literature verification for all scientific implementations
- Accessibility compliance with comprehensive testing

Reference the complete documentation in SIMCORE_CLAUDE_CODE_DOCUMENTATION.md for detailed implementation guidance, component architecture, and scientific accuracy requirements.
```

---

## 🎯 Focused Implementation Prompts

### 1. Scientific Computing Core
```
Implement the SimCore scientific computing core with WebGPU acceleration framework. Create the physics calculation libraries, mathematical algorithms, and GPU compute pipelines. Include the tight-binding calculations for graphene band structure, Monte Carlo algorithms for Ising model, and TDSE solver with wave packet dynamics. Ensure research-grade accuracy with literature verification and comprehensive error handling.
```

### 2. Interactive Visualization System
```
Build the SimCore visualization system with publication-quality scientific plots using plotly.js and Three.js 3D graphics. Implement the crystal structure viewer, band structure plots, wave function visualization, and lattice displays. Include interactive parameter controls, real-time updates, and export capabilities for research use. Ensure responsive design and touch optimization.
```

### 3. Physics Module Architecture
```
Create the SimCore physics module system with 25+ interactive simulations across quantum mechanics, statistical physics, and advanced computational methods. Implement the PhysicsModuleLayout component, domain-specific theming, theory integration panels, and parameter validation. Include lazy loading, code splitting, and memory management for optimal performance.
```

### 4. Progressive Web App Implementation
```
Implement SimCore as a comprehensive Progressive Web App with offline capability, service workers, and smart caching. Include the audit system with Lighthouse, axe, Playwright, and Pa11y integration. Ensure mobile-first responsive design, touch targets, and cross-platform compatibility with progressive enhancement strategies.
```

### 5. Educational Integration System
```
Build the SimCore educational framework with theory integration, LaTeX equation rendering using KaTeX, scientific references, and interactive derivations. Implement the tutorial system, difficulty progression, and comprehensive documentation. Include shareable URLs, session save/load, and collaboration features for educational use.
```

---

## 🛠️ Technical Implementation Guide

### Required Technology Stack
```json
{
  "core": {
    "react": "^18.3.1",
    "typescript": "^5.5.3",
    "vite": "^5.4.1"
  },
  "ui": {
    "@radix-ui/react-*": "latest",
    "tailwindcss": "^3.4.11",
    "lucide-react": "^0.462.0",
    "class-variance-authority": "^0.7.1"
  },
  "scientific": {
    "plotly.js": "^3.0.1",
    "react-plotly.js": "^2.6.0",
    "@react-three/fiber": "^8.18.0",
    "@react-three/drei": "^9.122.0",
    "three": "^0.160.1",
    "mathjs": "^14.5.3",
    "katex": "^0.16.22",
    "react-katex": "^3.1.0"
  },
  "performance": {
    "framer-motion": "^10.18.0",
    "zustand": "latest",
    "web-vitals": "^4.2.0"
  },
  "testing": {
    "@lhci/cli": "^0.14.0",
    "@axe-core/playwright": "^4.10.0",
    "playwright": "^1.48.2",
    "vitest": "^3.2.4"
  }
}
```

### Essential Component Patterns
```typescript
// Physics Module Layout Pattern
interface PhysicsModuleProps {
  title: string;
  category: 'quantum' | 'statistical' | 'energy' | 'fields';
  description: string;
  children: React.ReactNode;
}

// WebGPU Computation Hook Pattern
const useWebGPUCalculation = <T>(
  computeShader: string,
  params: ComputeParams
): { result: T | null; isCalculating: boolean } => {
  // Implementation with GPU acceleration and CPU fallback
};

// Scientific Visualization Pattern
interface ScientificPlotProps {
  data: PlotData[];
  layout: Partial<Plotly.Layout>;
  interactive?: boolean;
  exportable?: boolean;
}
```

---

## ✅ Success Validation Checklist

### Core Functionality
- [ ] 25+ interactive scientific modules implemented
- [ ] WebGPU acceleration framework functional
- [ ] Research-grade accuracy verified against literature
- [ ] Progressive Web App with offline capability
- [ ] Mobile-first responsive design implemented

### Performance Metrics
- [ ] Lighthouse Performance Score: 90+ (Desktop), 85+ (Mobile)
- [ ] Core Web Vitals: LCP <2.5s, FID <100ms, CLS <0.1
- [ ] Bundle Size: <500KB gzipped for initial load
- [ ] Real-time simulations at 60fps achieved

### Quality Assurance
- [ ] Comprehensive audit pipeline functional
- [ ] WCAG 2.1 AA accessibility compliance achieved
- [ ] 95%+ test coverage for physics calculations
- [ ] Cross-browser compatibility verified
- [ ] Error boundaries and fallback systems implemented

### Scientific Accuracy
- [ ] All equations verified against literature
- [ ] Numerical benchmarks match published results
- [ ] Unit consistency maintained throughout
- [ ] Proper error handling for edge cases
- [ ] Academic citations and references included

### User Experience
- [ ] Intuitive navigation and module discovery
- [ ] Responsive design across all devices
- [ ] Touch-optimized interactions for mobile
- [ ] Theory integration with LaTeX equations
- [ ] Export capabilities for research use

---

## 🚀 Deployment Commands

### Development Setup
```bash
# Clone and setup
git clone https://github.com/alaweimm90/SimCore.git
cd SimCore
npm install

# Start development server
npm run dev
# → http://localhost:5173

# Run comprehensive audit suite
npm run audit:full
```

### Production Build
```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to production
npm run deploy
```

### Quality Assurance
```bash
# Run full audit suite
npm run audit:full

# Individual audit tools
npm run audit:lh      # Lighthouse
npm run audit:axe     # Accessibility
npm run audit:pw      # Playwright tests
npm run audit:css     # CSS analysis
npm run audit:pa11y   # Pa11y accessibility

# Run tests
npm run test
npm run test:coverage
```

---

## 📋 Quick Reference

### Essential Imports
```typescript
// Core React and utilities
import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { cn } from '@/lib/utils';

// UI Components
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';

// Scientific visualization
import Plot from 'react-plotly.js';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { BlockMath, InlineMath } from 'react-katex';

// Physics calculations
import { calculateBandStructure } from '@/lib/physics/graphene';
import { runMonteCarloIsing } from '@/lib/physics/ising';
import { solveTDSE } from '@/lib/physics/quantum';
```

### Key File Locations
```
src/
├── components/ui/           # shadcn/ui components
├── components/physics/      # Physics-specific components
├── lib/physics/            # Physics calculation libraries
├── lib/webgpu/             # WebGPU acceleration framework
├── modules/                # Scientific modules
├── hooks/                  # Custom React hooks
└── styles/                 # CSS and styling
```

---

## 🎯 Implementation Priority

### Phase 1: Foundation (Week 1)
1. Set up React + TypeScript + Vite architecture
2. Implement shadcn/ui component system
3. Create physics domain theming system
4. Build WebGPU acceleration framework

### Phase 2: Core Modules (Week 2-3)
1. Implement quantum mechanics modules
2. Build statistical physics simulations
3. Create advanced computational methods
4. Add scientific visualization system

### Phase 3: Enhancement (Week 4)
1. Implement Progressive Web App features
2. Add comprehensive audit system
3. Optimize performance and accessibility
4. Complete educational integration

### Phase 4: Polish (Week 5)
1. Conduct thorough testing and validation
2. Optimize for production deployment
3. Complete documentation and examples
4. Prepare for academic and research use

---

**SimCore represents the future of web-based scientific computing education, combining research-grade accuracy with modern web technologies and universal accessibility.**