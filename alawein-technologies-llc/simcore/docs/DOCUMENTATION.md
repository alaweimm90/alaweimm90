# SimCore: Comprehensive User & Developer Documentation

*Complete guide to the interactive scientific computing platform for physics, mathematics, and scientific machine learning*

## Table of Contents

1. [Platform Overview](#platform-overview)
2. [Scientific Module Guide](#scientific-module-guide)
3. [User Interface](#user-interface)
4. [API Reference](#api-reference)
5. [Technical Architecture](#technical-architecture)
6. [Scientific Implementation](#scientific-implementation)
7. [Development Guide](#development-guide)
8. [Performance & Optimization](#performance--optimization)
9. [Accessibility](#accessibility)
10. [Troubleshooting](#troubleshooting)

---

## Platform Overview

SimCore is a research-grade scientific computing platform featuring WebGPU-accelerated simulations, real-time solvers, and machine learning integration. The platform provides 25+ interactive modules across physics, mathematics, and scientific machine learning. SimCore refers to the platform's core computational engine—a modular, high-performance system designed for research-grade, interactive simulations across these domains.

### Target Users

**Researchers & Scientists**
- Computational research in physics, mathematics, and scientific ML
- Band structure calculations and mathematical modeling
- Monte Carlo simulations and ML-driven analysis
- Data export for publication and further simulation

**Educators & Students**
- Interactive demonstrations in physics, mathematics, and ML
- Self-paced learning across multiple scientific domains
- Visual exploration of complex scientific and mathematical concepts
- Research preparation with advanced interdisciplinary tools

### Key Capabilities

**Research-Grade Accuracy**
- Literature-verified implementations across physics, math, and ML
- Built-in validation with scientific law checking
- Publication-quality visualizations using WebGL
- Comprehensive theory integration with LaTeX equations

**Advanced Performance**
- WebGPU acceleration for intensive calculations
- Real-time simulations at 60fps with hardware acceleration
- Progressive Web App with offline capability
- Memory-optimized algorithms for large datasets

**Educational Excellence**
- Interactive tutorials from beginner to research level
- Step-by-step theory explanations with mathematical derivations
- Scientific validation tools for physics, mathematics, and ML
- Mobile-optimized responsive design

---

## Scientific Module Guide

### Learning Progression

**Beginner Level** (Conceptual Foundation)
1. **Crystal Visualizer** - Interactive 3D lattice structures
2. **Bloch Sphere Dynamics** - Qubit states and quantum gate operations  
3. **Microstates & Entropy** - Statistical foundations of thermodynamics

**Intermediate Level** (Interactive Simulations)
1. **Graphene Band Structure** - Electronic properties and Dirac physics
2. **2D Ising Model** - Magnetic phase transitions with Monte Carlo
3. **LLG Dynamics** - Magnetization switching and spin torque

**Advanced Level** (Research Applications)
1. **MoS₂ Valley Physics** - Berry curvature and valley Hall effects
2. **TDSE Solver** - Wave packet dynamics and quantum tunneling
3. **Canonical Ensemble** - Advanced thermodynamic calculations

**Research Level** (Cutting-Edge Methods)
1. **Physics-Informed Neural Networks** - PINN-based PDE solvers
2. **Quantum Field Theory** - Field quantization and vacuum fluctuations
3. **Scientific Machine Learning** - AI-driven discovery and neural simulation
4. **Advanced Mathematical Methods** - Computational mathematics for scientific applications

#### MoS₂ Valley Physics
**Difficulty**: Advanced | **Category**: Materials & Crystals

**Physics Overview**:
Valley-contrasting physics in transition metal dichalcogenides:
- Berry curvature calculation
- Valley Hall conductivity
- Optical selection rules
- Circular dichroism effects

**Key Equations**:
```math
Ω(k) = ∇ₖ × ⟨uₖ|i∇ₖ|uₖ⟩
σ_H = e²/ℏ ∫ f(Eₖ)Ω(k) d²k
```

**Interactive Parameters**:
- Valley index (±1)
- Spin-orbit coupling strength
- Mass terms (m₀, mz)
- Chemical potential

**Visualizations**:
1. **Valley Band Structure**: K and K' valley dispersions
2. **Berry Curvature Map**: Valley-contrasting Berry curvature
3. **Optical Response**: Circular dichroism spectrum
4. **Spin Texture**: In-plane spin orientation

### Quantum Dynamics

#### Time-Dependent Schrödinger Equation
**Difficulty**: Advanced | **Category**: Quantum Dynamics

**Physics Overview**:
Numerical solution of the TDSE in 1D and 2D:
- Split-operator method
- Arbitrary potential profiles
- Wave packet evolution
- Tunneling dynamics

**Key Equations**:
```math
iℏ∂ψ/∂t = Ĥψ
ψ(x,t+dt) = e^(-iĤdt/ℏ)ψ(x,t)
```

**Numerical Methods**:
- Split-operator technique
- FFT for kinetic energy
- Absorbing boundary conditions
- Adaptive time stepping

**Interactive Parameters**:
- Potential barrier height/width
- Initial wave packet parameters
- Time evolution speed
- Grid resolution

**Visualizations**:
1. **Wave Function**: |ψ(x,t)|² real-time evolution
2. **Probability Current**: j(x,t) = (ℏ/2mi)[ψ*∇ψ - ψ∇ψ*]
3. **Energy Expectation**: ⟨Ĥ⟩ vs time
4. **Transmission/Reflection**: Probability coefficients

#### Bloch Sphere Dynamics
**Difficulty**: Beginner | **Category**: Quantum Dynamics

**Physics Overview**:
Geometric representation of qubit states and evolution:
- Bloch vector representation
- Rabi oscillations
- Quantum gate operations
- Decoherence effects

**Key Equations**:
```math
|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
Ĥ = (ℏω₀/2)σz + (ℏΩ/2)(cosφ σx + sinφ σy)
```

**Interactive Parameters**:
- Rabi frequency (Ω)
- Detuning (Δ)
- Phase (φ)
- Pulse duration

**Visualizations**:
1. **3D Bloch Sphere**: Real-time state evolution
2. **Bloch Vector Components**: x(t), y(t), z(t)
3. **Population Dynamics**: P₀(t), P₁(t)
4. **Gate Operations**: Pauli gates visualization

### Statistical Physics

#### 2D Ising Model
**Difficulty**: Intermediate | **Category**: Statistical Physics

**Physics Overview**:
Monte Carlo simulation of magnetic phase transitions:
- Metropolis algorithm
- Critical temperature determination
- Order parameter calculation
- Finite-size scaling

**Key Equations**:
```math
H = -J∑⟨i,j⟩ SᵢSⱼ - h∑ᵢ Sᵢ
P(config) = e^(-βH)/Z
⟨m⟩ = (1/N)∑ᵢ ⟨Sᵢ⟩
```

**Interactive Parameters**:
- Temperature (T/Tc)
- External field (h/J)
- Lattice size (L×L)
- Coupling strength (J)

**Visualizations**:
1. **Spin Configuration**: Real-time lattice visualization
2. **Magnetization**: Order parameter vs temperature
3. **Energy**: Internal energy and heat capacity
4. **Correlation Function**: Spatial correlations

#### Boltzmann Distribution
**Difficulty**: Intermediate | **Category**: Statistical Physics

**Physics Overview**:
Statistical mechanics of energy level populations:
- Partition function calculation
- Maxwell-Boltzmann statistics
- Quantum vs classical statistics
- Temperature dependence

**Key Equations**:
```math
Pᵢ = gᵢe^(-Eᵢ/kᵦT)/Z
Z = ∑ᵢ gᵢe^(-Eᵢ/kᵦT)
⟨E⟩ = -∂lnZ/∂β
```

**Interactive Parameters**:
- Temperature (kBT)
- Energy levels (Ei)
- Degeneracies (gi)
- Statistics type (MB/FD/BE)

**Visualizations**:
1. **Population Distribution**: Bar chart of level occupations
2. **Temperature Dependence**: Populations vs T
3. **Average Energy**: ⟨E⟩(T) and heat capacity
4. **Entropy**: S(T) calculation

### Materials & Crystals

#### Crystal Structure Visualizer
**Difficulty**: Beginner | **Category**: Materials & Crystals

**Physics Overview**:
Interactive 3D visualization of crystal lattices:
- Bravais lattices
- Space group operations
- Miller indices
- Structure factors

**Key Concepts**:
- Lattice parameters (a, b, c, α, β, γ)
- Atomic positions and occupancies
- Symmetry operations
- Reciprocal lattice

**Interactive Features**:
- Crystal system selection
- Atomic species placement
- Symmetry operation visualization
- Miller plane display

**Visualizations**:
1. **3D Crystal**: Interactive lattice with atoms
2. **Unit Cell**: Highlighted primitive cell
3. **Reciprocal Lattice**: k-space representation
4. **Structure Factor**: |F(hkl)|² calculation

---

## Technical Architecture

### Frontend Stack

**Core Technologies**:
- React 18.3.1 with TypeScript
- Vite build system
- Tailwind CSS with shadcn/ui
- React Router for navigation

**Physics Libraries**:
- plotly.js for 2D/3D plotting
- KaTeX for mathematical typesetting
- mathjs for numerical calculations
- React Three Fiber for 3D graphics

**State Management**:
- Zustand for global state
- React Query for data fetching
- Local state for component-specific data

**Performance Optimizations**:
- Lazy loading with React.lazy()
- Code splitting by route
- Memoization with useMemo/useCallback
- Web Workers for heavy calculations

### Component Architecture

```
src/
├── components/
│   ├── ui/                 # shadcn/ui base components
│   ├── [module]/          # Module-specific components
│   │   ├── ControlPanel.tsx
│   │   ├── Visualization.tsx
│   │   ├── TheoryPanel.tsx
│   │   └── index.tsx
│   ├── shared/            # Shared components
│   │   ├── ScientificPlot.tsx
│   │   ├── ParameterSlider.tsx
│   │   └── TheoryDisplay.tsx
│   └── layout/            # Layout components
├── lib/                   # Business logic
│   ├── physics/          # Physics calculations
│   ├── utils/            # Utility functions
│   └── constants/        # Physical constants
├── pages/                # Route components
├── hooks/                # Custom React hooks
└── types/                # TypeScript definitions
```

### Physics Implementation

#### Calculation Pipeline

1. **Parameter Validation**: Input sanitization and range checking
2. **Physics Computation**: Numerical algorithms implementation
3. **Result Validation**: Physical consistency checks
4. **Visualization Preparation**: Data formatting for plots
5. **Theory Integration**: Relevant equations and explanations

#### Numerical Methods

**Linear Algebra**:
- Matrix diagonalization (LAPACK-style algorithms)
- Sparse matrix operations
- Eigenvalue/eigenvector calculations

**Differential Equations**:
- Runge-Kutta methods
- Split-operator techniques
- Finite difference schemes

**Monte Carlo**:
- Metropolis algorithm
- Importance sampling
- Statistical error estimation

**Optimization**:
- Gradient descent
- Simulated annealing
- Genetic algorithms

#### Error Handling

**Numerical Stability**:
- Condition number monitoring
- Overflow/underflow protection
- Convergence criteria

**Physical Validation**:
- Conservation law checks
- Symmetry verification
- Unitarity constraints

---

## API Reference

### Core Interfaces

#### PhysicsModule Interface
```typescript
interface PhysicsModule {
  id: string;
  title: string;
  description: string;
  category: ModuleCategory;
  difficulty: DifficultyLevel;
  tags: string[];
  equation?: string;
  isImplemented: boolean;
  route?: string;
  theory: TheoryContent;
}
```

#### Parameters Interface
```typescript
interface Parameters {
  [key: string]: number | string | boolean | number[];
}
```

#### Results Interface
```typescript
interface Results {
  data: any;
  metadata: {
    computationTime: number;
    convergence: boolean;
    accuracy: number;
  };
  validation: ValidationReport;
}
```

### Physics Libraries

#### Graphene Physics
```typescript
// lib/graphene-physics.ts
export function calculateGrapheneBands(
  kPoints: number[][],
  params: GrapheneParams
): BandResult[];

export function generateHighSymmetryPath(
  nPoints?: number
): { kPoints: number[][], labels: string[], distances: number[] };
```

#### Ising Model
```typescript
// lib/ising-simulator.ts
export class IsingSimulator {
  constructor(size: number, temperature: number);
  step(): void;
  getMagnetization(): number;
  getEnergy(): number;
}
```

### Component Props

#### ScientificPlot
```typescript
interface ScientificPlotProps {
  data: PlotData[];
  layout: Partial<Layout>;
  config?: Partial<Config>;
  onPlotUpdate?: (figure: Figure) => void;
}
```

#### ParameterControl
```typescript
interface ParameterControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (value: number) => void;
  tooltip?: string;
}
```

---

## Development Guide

### Setup Instructions

1. **Prerequisites**:
   ```bash
   node --version  # v18.0.0+
   npm --version   # v8.0.0+
   ```

2. **Installation**:
   ```bash
   git clone [repository-url]
   cd simcore-explorer
   npm install
   ```

3. **Development Server**:
   ```bash
   npm run dev
   # Opens at http://localhost:5173
   ```

### Collaboration & Sharing

SimCore includes built-in collaboration tooling to make research reproducible and sharable across physics, mathematics, and scientific ML modules:

- Shareable URLs: Encode full simulation state into a link for 1‑click loading
- Local Save/Load: Archive sessions with metadata in the browser
- Import/Export JSON: Portable state files for papers, reviews, or bug reports
- Sessions: Create public/private collaboration sessions (UI in Share dialog)

Access via the Share button in module UIs (Share → Save/Load → Import → Collaborate). Backed by src/lib/collaboration-manager.ts.

### Category–Domain Mapping Utility

To avoid duplication and ensure consistent theming and navigation grouping, use the centralized mapping utility:

```ts
import { categoryToThemeDomain, categoryToGroupDomain } from '@/lib/category-domain-map';

const themeDomain = categoryToThemeDomain('Band Structure'); // 'quantum'
const groupDomain = categoryToGroupDomain('Field Theory');   // 'Mathematics'
```

This utility powers DomainThemeProvider auto-detection and sidebar grouping.

### Creating New Modules

#### 1. Physics Implementation
Create the core physics calculations:

```typescript
// src/lib/my-module-physics.ts
export interface MyModuleParams {
  parameter1: number;
  parameter2: number;
}

export interface MyModuleResult {
  data: number[];
  energy: number;
}

export function calculateMyModule(params: MyModuleParams): MyModuleResult {
  // Implementation here
  return {
    data: computedData,
    energy: computedEnergy
  };
}
```

#### 2. Control Panel Component
```typescript
// src/components/my-module/ControlPanel.tsx
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';

interface ControlPanelProps {
  params: MyModuleParams;
  onParamsChange: (params: MyModuleParams) => void;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  params,
  onParamsChange
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label>Parameter 1</label>
          <Slider
            value={[params.parameter1]}
            onValueChange={([value]) => 
              onParamsChange({ ...params, parameter1: value })
            }
            min={0}
            max={10}
            step={0.1}
          />
        </div>
      </CardContent>
    </Card>
  );
};
```

#### 3. Visualization Component
```typescript
// src/components/my-module/Visualization.tsx
import { ScientificPlot } from '@/components/ScientificPlot';

interface VisualizationProps {
  result: MyModuleResult;
}

export const Visualization: React.FC<VisualizationProps> = ({ result }) => {
  const plotData = [{
    x: result.data.map((_, i) => i),
    y: result.data,
    type: 'scatter',
    mode: 'lines',
    name: 'My Module Data'
  }];

  return (
    <ScientificPlot
      data={plotData}
      layout={{
        title: 'My Module Visualization',
        xaxis: { title: 'X Axis' },
        yaxis: { title: 'Y Axis' }
      }}
    />
  );
};
```

#### 4. Main Module Component
```typescript
// src/pages/MyModule.tsx
import { useState, useMemo } from 'react';
import { ControlPanel } from '@/components/my-module/ControlPanel';
import { Visualization } from '@/components/my-module/Visualization';
import { calculateMyModule, MyModuleParams } from '@/lib/my-module-physics';

export const MyModule = () => {
  const [params, setParams] = useState<MyModuleParams>({
    parameter1: 1.0,
    parameter2: 2.0
  });

  const result = useMemo(() => calculateMyModule(params), [params]);

  return (
    <div className="container mx-auto p-4">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <ControlPanel params={params} onParamsChange={setParams} />
        </div>
        <div className="lg:col-span-2">
          <Visualization result={result} />
        </div>
      </div>
    </div>
  );
};
```

#### 5. Register Module
Add to module data and routing:

```typescript
// src/data/modules.ts
{
  id: 'my-module',
  title: 'My Module',
  description: 'Description of my module',
  category: 'Quantum Dynamics',
  difficulty: 'Intermediate',
  tags: ['Tag1', 'Tag2'],
  equation: 'H = ...',
  isImplemented: true,
  route: '/modules/my-module',
  theory: {
    overview: 'Theory overview...',
    mathematics: ['Equation 1', 'Equation 2'],
    references: ['Reference 1', 'Reference 2']
  }
}

// src/App.tsx
<Route path="/modules/my-module" element={<MyModule />} />
```

### Testing Guidelines

#### Unit Tests
```typescript
// src/lib/__tests__/my-module-physics.test.ts
import { calculateMyModule } from '../my-module-physics';

describe('MyModule Physics', () => {
  test('should calculate correctly', () => {
    const params = { parameter1: 1, parameter2: 2 };
    const result = calculateMyModule(params);
    
    expect(result.energy).toBeCloseTo(expectedValue, 6);
    expect(result.data).toHaveLength(expectedLength);
  });
});
```

#### Component Tests
```typescript
// src/components/__tests__/MyModule.test.tsx
import { render, screen } from '@testing-library/react';
import { MyModule } from '../MyModule';

test('renders module correctly', () => {
  render(<MyModule />);
  expect(screen.getByText('My Module')).toBeInTheDocument();
});
```

### Performance Optimization

#### Code Splitting
```typescript
// Lazy load heavy components
const MyModule = lazy(() => import('./pages/MyModule'));

// Use React.memo for expensive renders
export const Visualization = React.memo<VisualizationProps>(({ result }) => {
  // Component implementation
});
```

#### Calculation Optimization
```typescript
// Use Web Workers for heavy calculations
const worker = new Worker('/physics-worker.js');

// Memoize expensive calculations
const result = useMemo(() => 
  calculateExpensivePhysics(params), 
  [params]
);

// Debounce parameter changes
const debouncedParams = useDebounce(params, 300);
```

---

## Deployment

### Build Process

```bash
# Production build
npm run build

# Preview build locally
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint
```

### Environment Configuration

```bash
# .env.production
VITE_APP_TITLE="SimCore Explorer"
VITE_API_URL="https://api.simcore.example.com"
VITE_ANALYTICS_ID="your-analytics-id"
```

### Performance Monitoring

- **Bundle Size**: Monitor with `npm run analyze`
- **Core Web Vitals**: Lighthouse audits
- **Error Tracking**: Production error monitoring
- **User Analytics**: Usage pattern analysis

---

## Troubleshooting

### Common Issues

#### Build Errors

**TypeScript Errors**:
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Check TypeScript configuration
npx tsc --noEmit
```

**Memory Issues**:
```bash
# Increase Node.js memory limit
export NODE_OPTIONS="--max-old-space-size=4096"
npm run build
```

#### Runtime Issues

**Physics Calculation Errors**:
- Check parameter ranges and validation
- Verify numerical stability
- Monitor console for NaN/Infinity values

**Performance Issues**:
- Enable React DevTools Profiler
- Check for memory leaks in calculations
- Optimize expensive renders with React.memo

**Accessibility Issues**:
- Run axe-core accessibility tests
- Verify keyboard navigation
- Check screen reader compatibility

#### Browser Compatibility

**WebGL Issues**:
```javascript
// Check WebGL support
const canvas = document.createElement('canvas');
const gl = canvas.getContext('webgl2');
if (!gl) {
  console.warn('WebGL 2.0 not supported');
}
```

**ES Module Issues**:
- Ensure browser supports ES2020
- Check for polyfill requirements
- Verify module loading

### Debug Tools

#### Physics Debugging
```typescript
// Enable debug mode
const DEBUG_PHYSICS = process.env.NODE_ENV === 'development';

if (DEBUG_PHYSICS) {
  console.log('Physics parameters:', params);
  console.log('Calculation result:', result);
  console.log('Validation status:', validation);
}
```

#### Performance Profiling
```typescript
// Measure calculation time
const startTime = performance.now();
const result = calculatePhysics(params);
const endTime = performance.now();
console.log(`Calculation took ${endTime - startTime} ms`);
```

### Support Resources

- **Documentation**: This file and inline code comments
- **Issues**: GitHub issue tracker
- **Community**: Physics education forums
- **Contact**: Direct contact with development team

---

*This documentation is maintained alongside the codebase. For the most current information, refer to the source code and inline documentation.*