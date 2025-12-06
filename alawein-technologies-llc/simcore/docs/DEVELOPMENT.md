# SimCore Development Guide

*Comprehensive guide for contributing to the SimCore physics simulation platform*

## 🚀 Quick Setup

### Prerequisites
- **Node.js 18.0+** (recommended: 20.0+)
- **Modern browser** with WebGL 2.0 and WebGPU support
- **4GB+ RAM** for complex simulations

### Installation
```bash
git clone https://github.com/alaweimm90/SimCore.git
cd SimCore
npm install
npm run dev
```

### Development Commands
```bash
npm run dev          # Start development server
npm run build        # Production build
npm run build:dev    # Development build with source maps
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

## 🏗️ Architecture Overview

### Framework Stack
- **Frontend**: React 18.3 + TypeScript + Vite
- **UI Components**: shadcn/ui with Radix primitives
- **Styling**: Tailwind CSS with custom physics themes
- **State Management**: Zustand stores for complex physics simulations
- **Routing**: React Router with lazy-loaded components
- **Mathematics**: plotly.js, KaTeX, mathjs
- **3D Graphics**: React Three Fiber + Three.js

### Directory Structure
```
src/
├── components/              # React components
│   ├── ui/                 # shadcn/ui base components
│   ├── graphene/           # Graphene physics components
│   ├── ising/              # Ising model components
│   ├── llg/                # LLG dynamics components
│   └── mos2/               # TMD valley physics components
├── lib/                    # Physics calculation libraries
│   ├── physics-*.ts        # Domain-specific physics engines
│   ├── webgpu-*.ts         # GPU acceleration utilities
│   └── utils.ts            # General utilities
├── pages/                  # Route components
├── data/                   # Module registry and configurations
└── types/                  # TypeScript definitions
```

## 🧪 Adding New Physics Modules

### 1. Create Physics Engine
Implement calculations in `src/lib/physics-[module].ts`:

```typescript
// src/lib/physics-my-module.ts
export interface MyModuleParams {
  parameter1: number;
  parameter2: number;
}

export interface MyModuleResult {
  data: number[];
  energy: number;
}

export function calculateMyModule(params: MyModuleParams): MyModuleResult {
  // Physics implementation here
  return {
    data: computedData,
    energy: computedEnergy
  };
}
```

### 2. Build Components
Create UI components in `src/components/[module]/`:

```typescript
// src/components/my-module/ControlPanel.tsx
export const MyModuleControlPanel: React.FC<ControlPanelProps> = ({
  parameters,
  onParametersChange
}) => {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Parameters</CardTitle>
      </CardHeader>
      <CardContent>
        {/* Parameter controls */}
      </CardContent>
    </Card>
  );
};
```

### 3. Create Page Component
Add main module page in `src/pages/[ModuleName].tsx`:

```typescript
// src/pages/MyModule.tsx
export default function MyModule() {
  const [parameters, setParameters] = useState<MyModuleParams>(defaultParams);
  const result = useMemo(() => calculateMyModule(parameters), [parameters]);

  return (
    <div className="min-h-screen bg-gradient-cosmic">
      <PhysicsModuleHeader
        title="My Physics Module"
        description="Description of the physics simulation"
        category="Your Category"
        difficulty="Intermediate"
        equation="E = mc^2"
      />
      {/* Module content */}
    </div>
  );
}
```

### 4. Register Module
Add to `src/data/modules.ts`:

```typescript
{
  id: 'my-module',
  title: 'My Physics Module',
  description: 'Detailed description of the physics',
  category: 'Your Category',
  difficulty: 'Intermediate',
  tags: ['quantum', 'mechanics'],
  equation: 'E = mc^2',
  isImplemented: true,
  route: '/modules/my-module',
  theory: {
    overview: 'Physics overview...',
    keyEquations: [
      { label: 'Energy', equation: 'E = mc^2', description: 'Mass-energy equivalence' }
    ],
    // ... more theory content
  }
}
```

### 5. Add Route
Register in `src/App.tsx` and lazy load:

```typescript
// Add to LazyRoutes.tsx
const MyModule = lazy(() => import('@/pages/MyModule'));

// Add to App.tsx routes
<Route path="/modules/my-module" element={<MyModule />} />
```

## 🎨 Design System Guidelines

### Physics Domain Theming
SimCore uses automatic domain theming based on scientific categories. Use the centralized mapping utility to avoid duplication:

```typescript
import { categoryToThemeDomain, categoryToGroupDomain } from '@/lib/category-domain-map';

const domain = categoryToThemeDomain('Band Structure'); // 'quantum'
const group  = categoryToGroupDomain('Field Theory');   // 'Mathematics'
```


### Component Styling
Always use semantic design tokens:

```typescript
// ✅ Good - Uses design system
<Card variant="physics" domain="quantum">
<Button variant="primary">Launch Module</Button>

// ❌ Bad - Hardcoded styles
<Card className="bg-purple-500 text-white">
<Button style={{ backgroundColor: '#3b82f6' }}>Launch</Button>
```

### Color System
Use HSL values with CSS custom properties:

```css
/* Use semantic tokens */
background-color: hsl(var(--primary));
border-color: hsl(var(--border));

/* Avoid direct colors */
background-color: #3b82f6; /* ❌ Don't do this */
```

## 🔬 Physics Implementation Standards

### Mathematical Accuracy
- **Literature verification**: All implementations must match published equations
- **Unit consistency**: Use consistent unit systems (eV, Å, fs)
- **Validation checks**: Include physics consistency tests
- **Error handling**: Graceful handling of numerical instabilities

### Calculation Patterns
```typescript
// Standard physics calculation pattern
export function calculateBandStructure(
  kPoints: Vector2D[],
  params: BandStructureParams
): BandResult {
  // Input validation
  if (kPoints.length === 0) throw new Error('Empty k-point array');
  
  // Physics calculations
  const bands = kPoints.map(k => {
    const hamiltonian = buildHamiltonian(k, params);
    return diagonalize(hamiltonian);
  });
  
  // Output validation
  validateEnergyConservation(bands);
  
  return { energies: bands, kPoints };
}
```

### Performance Optimization
- **Memoization**: Use `useMemo` for expensive calculations
- **React.memo**: Wrap visualization components
- **Web Workers**: For heavy computational tasks
- **WebGPU**: For parallel calculations when available

```typescript
// Optimized component pattern
export const BandStructurePlot = React.memo<PlotProps>(({ data, parameters }) => {
  const bandData = useMemo(() => 
    calculateBandStructure(kPoints, parameters),
    [kPoints, parameters]
  );
  
  return <Plot data={bandData} />;
});
```

## 🧪 Testing Guidelines

### Physics Validation
Every module should include validation tests:

```typescript
// Example validation tests
export function validateGrapheneBands(params: GrapheneParams): ValidationResult {
  const testPoint = [Math.PI/3, Math.PI/3]; // K point
  const result = calculateGrapheneBands([testPoint], params);
  
  // Test Dirac point condition: f₁(K) = 0
  const f1AtK = calculateStructureFactor(testPoint);
  const isValidDiracPoint = Math.abs(f1AtK) < 1e-10;
  
  return {
    isValid: isValidDiracPoint,
    errors: isValidDiracPoint ? [] : ['Dirac point condition failed']
  };
}
```

### Component Testing
- **Unit tests**: Test individual physics functions
- **Integration tests**: Test component interactions
- **Visual tests**: Ensure plot outputs are correct

### Browser Testing
- **Chrome/Firefox/Safari**: Desktop compatibility
- **Mobile browsers**: Touch interaction testing
- **WebGL/WebGPU**: Hardware acceleration verification

## 📊 Performance Guidelines

### Bundle Optimization
- **Lazy loading**: All routes and heavy components
- **Tree shaking**: Import only used functions
- **Code splitting**: Physics modules loaded on demand

### Memory Management
```typescript
// Proper cleanup for physics stores
useEffect(() => {
  return () => {
    // Clean up physics simulations
    clearAnimationFrame(animationId);
    physicsStore.cleanup();
  };
}, []);
```

### Calculation Optimization
- **TypedArrays**: For large numerical data
- **Object pooling**: Reuse mathematical objects
- **GPU acceleration**: Use WebGPU when available

## 🎯 Code Quality Standards

### TypeScript
- **Strict mode**: Full type coverage required
- **Interfaces**: Define clear contracts for physics functions
- **Generics**: Reusable mathematical utilities

### Physics Documentation
```typescript
/**
 * Calculates the electronic band structure for graphene
 * using the tight-binding approximation.
 * 
 * @param kPoints - Array of k-points in reciprocal space [kx, ky]
 * @param params - Tight-binding parameters (t1, t2, etc.)
 * @returns Band energies and velocities at each k-point
 * 
 * @example
 * const kPath = generateHighSymmetryPath();
 * const bands = calculateGrapheneBands(kPath, DEFAULT_PARAMS);
 * 
 * @reference
 * Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009)
 */
```

### Error Handling
- **Physics validation**: Check conservation laws
- **Numerical stability**: Handle edge cases gracefully
- **User feedback**: Clear error messages for parameter issues

## 🤝 Contributing Workflow

### 1. Development Process
1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/new-physics-module`
3. **Implement** with comprehensive TypeScript types
4. **Validate** physics accuracy against literature
5. **Test** across browsers and devices
6. **Document** mathematical formulations
7. **Submit** pull request with detailed description

### 2. Code Review Checklist
- [ ] Physics equations match literature references
- [ ] TypeScript types are comprehensive
- [ ] Performance optimizations implemented
- [ ] Design system tokens used correctly
- [ ] Documentation includes theory and examples
- [ ] Tests cover edge cases and validation

### 3. Continuous Integration
- **ESLint**: Code quality enforcement
- **TypeScript**: Compile-time validation
- **Physics tests**: Automated accuracy verification
- **Bundle analysis**: Performance monitoring

## 🔧 Debugging Tools

### Development Tools
```bash
# Physics validation
npm run validate-physics

# Performance profiling
npm run analyze-bundle

# Type checking
npm run type-check
```

### Browser DevTools
- **React DevTools**: Component state inspection
- **Performance tab**: Animation and calculation profiling
- **WebGPU inspector**: GPU acceleration debugging

### Physics Debugging
```typescript
// Enable debug mode for detailed physics logging
const DEBUG_PHYSICS = process.env.NODE_ENV === 'development';

if (DEBUG_PHYSICS) {
  console.log('Band calculation:', { kPoints, parameters, result });
}
```

## 📚 Learning Resources

### Physics Background
- **Condensed Matter**: Ashcroft & Mermin
- **Quantum Mechanics**: Griffiths
- **Computational Physics**: Newman
- **Band Theory**: Marder

### Development Skills
- **React Performance**: React documentation
- **TypeScript**: Official handbook
- **WebGL/WebGPU**: MDN web docs
- **Scientific Visualization**: D3.js and plotly.js guides

---

*For questions or discussions, reach out to [meshal@berkeley.edu](mailto:meshal@berkeley.edu)*