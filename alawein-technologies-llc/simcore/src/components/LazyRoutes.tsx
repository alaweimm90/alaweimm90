import { lazy, Suspense } from 'react';
import { ProgressiveLoader } from '@/components/AdvancedLoadingStates';
import { ErrorBoundary } from '@/components/ErrorBoundary';

// Lazy load all route components for better performance
const GrapheneBandStructure = lazy(() => import('@/pages/GrapheneBandStructure'));
const MoS2ValleyPhysics = lazy(() => import('@/pages/MoS2ValleyPhysics'));
const TDSESolver = lazy(() => import('@/pages/TDSESolver'));
const CrystalVisualizer = lazy(() => import('@/pages/CrystalVisualizer'));
const LLGDynamics = lazy(() => import('@/pages/LLGDynamics'));
const BlochSphereDynamics = lazy(() => import('@/pages/BlochSphereDynamics'));
const QuantumTunneling = lazy(() => import('@/pages/QuantumTunneling'));
const BZFolding = lazy(() => import('@/pages/BZFolding'));
const PhononBandStructure = lazy(() => import('@/pages/PhononBandStructure'));
const QuantumFieldTheory = lazy(() => import('@/pages/QuantumFieldTheory'));
const LaplaceEigenmodes = lazy(() => import('@/pages/LaplaceEigenmodes'));
const PINNSchrodinger = lazy(() => import('@/pages/PINNSchrodinger'));
const MLShowcase = lazy(() => import('@/pages/MLShowcase'));
const AdvancedSimulation = lazy(() => import('@/pages/AdvancedSimulation'));

const WebWorkerDemo = lazy(() => import('@/pages/WebWorkerDemo'));
const ErrorReporting = lazy(() => import('@/pages/ErrorReporting'));
const AccessibilityFeatures = lazy(() => import('@/pages/AccessibilityFeatures'));
const PWAFeatures = lazy(() => import('@/pages/PWAFeatures'));
const IsingModel = lazy(() => import('@/pages/IsingModel'));
const BoltzmannDistribution = lazy(() => import('@/pages/BoltzmannDistribution'));
const MicrostatesEntropy = lazy(() => import('@/pages/MicrostatesEntropy'));
const CanonicalEnsemble = lazy(() => import('@/pages/CanonicalEnsemble'));
const BrownianMotion = lazy(() => import('@/pages/BrownianMotion'));
const Documentation = lazy(() => import('@/pages/Documentation'));
const NotFound = lazy(() => import('@/pages/NotFound'));
const SimulationDashboard = lazy(() => import('@/pages/SimulationDashboard'));
const ScientificComputing = lazy(() => import('@/pages/ScientificComputing'));
const AboutPlatform = lazy(() => import('@/pages/AboutPlatform'));
const InteractiveLearning = lazy(() => import('@/pages/InteractiveLearning'));
const EnhancedVisualization = lazy(() => import('@/pages/EnhancedVisualization'));
const DataExport = lazy(() => import('@/pages/DataExport'));
const TestingDashboard = lazy(() => import('@/pages/TestingDashboard'));

// Loading fallback component
const ModuleLoadingFallback = ({ moduleName }: { moduleName?: string }) => (
  <div className="min-h-screen relative overflow-hidden flex items-center justify-center">
    <div className="absolute inset-0 bg-gradient-cosmic" />
    <ProgressiveLoader 
      stage={1}
      progress={50}
      stages={['Loading', `Loading ${moduleName || 'module'}`, 'Ready']}
    />
  </div>
);

// HOC to wrap lazy components with error boundary and loading state
const withLazyWrapper = (
  Component: React.ComponentType<any>, 
  moduleName: string
) => {
  return (props: any) => (
    <ErrorBoundary>
      <Suspense fallback={<ModuleLoadingFallback moduleName={moduleName} />}>
        <Component {...props} />
      </Suspense>
    </ErrorBoundary>
  );
};

// Export wrapped components
export const LazyGrapheneBandStructure = withLazyWrapper(GrapheneBandStructure, 'Graphene Band Structure');
export const LazyMoS2ValleyPhysics = withLazyWrapper(MoS2ValleyPhysics, 'MoS₂ Valley Physics');
export const LazyTDSESolver = withLazyWrapper(TDSESolver, 'TDSE Solver');
export const LazyCrystalVisualizer = withLazyWrapper(CrystalVisualizer, 'Crystal Visualizer');
export const LazyLLGDynamics = withLazyWrapper(LLGDynamics, 'LLG Dynamics');
export const LazyBlochSphereDynamics = withLazyWrapper(BlochSphereDynamics, 'Bloch Sphere Dynamics');
export const LazyQuantumTunneling = withLazyWrapper(QuantumTunneling, 'Quantum Tunneling');
export const LazyBZFolding = withLazyWrapper(BZFolding, 'BZ Folding');
export const LazyPhononBandStructure = withLazyWrapper(PhononBandStructure, 'Phonon Band Structure');
export const LazyQuantumFieldTheory = withLazyWrapper(QuantumFieldTheory, 'Quantum Field Theory');
export const LazyLaplaceEigenmodes = withLazyWrapper(LaplaceEigenmodes, 'Laplace Eigenmodes');
export const LazyPINNSchrodinger = withLazyWrapper(PINNSchrodinger, 'PINN Schrödinger');
export const LazyMLShowcase = withLazyWrapper(MLShowcase, 'ML Showcase');
export const LazyWebWorkerDemo = withLazyWrapper(WebWorkerDemo, 'WebWorker Demo');
export const LazyIsingModel = withLazyWrapper(IsingModel, 'Ising Model');
export const LazyBoltzmannDistribution = withLazyWrapper(BoltzmannDistribution, 'Boltzmann Distribution');
export const LazyMicrostatesEntropy = withLazyWrapper(MicrostatesEntropy, 'Microstates & Entropy');
export const LazyCanonicalEnsemble = withLazyWrapper(CanonicalEnsemble, 'Canonical Ensemble');
export const LazyBrownianMotion = withLazyWrapper(BrownianMotion, 'Brownian Motion');
export const LazyAdvancedSimulation = withLazyWrapper(AdvancedSimulation, 'Advanced Simulation');

export const LazyDocumentation = withLazyWrapper(Documentation, 'Documentation');
export const LazyErrorReporting = withLazyWrapper(ErrorReporting, 'Error Reporting');
export const LazyAccessibilityFeatures = withLazyWrapper(AccessibilityFeatures, 'Accessibility Features');
export const LazyPWAFeatures = withLazyWrapper(PWAFeatures, 'PWA Features');
export const LazyNotFound = withLazyWrapper(NotFound, 'Page');
export const LazySimulationDashboard = withLazyWrapper(SimulationDashboard, 'Simulation Dashboard');
export const LazyScientificComputing = withLazyWrapper(ScientificComputing, 'Scientific Computing');
export const LazyAboutPlatform = withLazyWrapper(AboutPlatform, 'About Platform');
export const LazyInteractiveLearning = withLazyWrapper(InteractiveLearning, 'Interactive Learning');
export const LazyEnhancedVisualization = withLazyWrapper(EnhancedVisualization, 'Enhanced Visualization');
export const LazyDataExport = withLazyWrapper(DataExport, 'Data Export');
export const LazyTestingDashboard = withLazyWrapper(TestingDashboard, 'Testing Dashboard');