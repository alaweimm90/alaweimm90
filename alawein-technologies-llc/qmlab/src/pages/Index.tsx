import { Suspense, lazy, useEffect, useState } from 'react';
import { PageChrome } from '@/components/PageChrome';
import { HeroSectionProfessional } from '@/components/HeroSectionProfessional';
import { HeaderProfessional } from '@/components/HeaderProfessional';
import { QuantumFooter } from '@/components/QuantumFooter';
import { QuantumBackground } from '@/components/QuantumBackground';
import { ComingSoonCard } from '@/components/ComingSoonCard';
import { ComponentSkeleton } from '@/components/ComponentSkeleton';
import { FloatingCTA } from '@/components/FloatingCTA';
import { QuantumBreadcrumbs, useQuantumBreadcrumbs } from '@/components/QuantumBreadcrumbs';
import { QuantumConceptTooltip } from '@/components/QuantumConceptTooltip';
import { QuantumLearningTracker } from '@/components/QuantumLearningTracker';
import { PWAManifest } from '@/components/PWAManifest';
import { QLogo } from '@/components/QLogo';
import { CollapsibleSection } from '@/components/CollapsibleSection';
import { MathematicalDerivation } from '@/components/MathematicalDerivation';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Button } from '@/components/ui/button';
import {
  BarChart3,
  Library,
  Database,
  BookOpen,
  Lightbulb,
  Calculator,
  Atom,
  Brain,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { useWebVitals, useQuantumMetrics } from '@/hooks/useWebVitals';
import { useKeyboardNavigation, useScreenReaderOptimizations } from '@/hooks/useKeyboardNavigation';
import { preloadCriticalChunks } from '@/hooks/useChunkLoading';

// Enhanced lazy loading with error boundaries
const CircuitBuilder = lazy(() =>
  import('@/components/CircuitBuilder')
    .then((module) => ({ default: module.CircuitBuilder }))
    .catch(() => ({
      default: () => <ComponentSkeleton className="h-96" error="Failed to load Circuit Builder" />,
    })),
);

const BlochSphere = lazy(() =>
  import('@/components/BlochSphere')
    .then((module) => ({ default: module.BlochSphere }))
    .catch(() => ({
      default: () => <ComponentSkeleton className="h-96" error="Failed to load Bloch Sphere" />,
    })),
);

const TrainingDashboard = lazy(() =>
  import('@/components/TrainingDashboard')
    .then((module) => ({ default: module.TrainingDashboard }))
    .catch(() => ({
      default: () => (
        <ComponentSkeleton className="h-96" error="Failed to load Training Dashboard" />
      ),
    })),
);

export default function Index() {
  const webVitals = useWebVitals();
  const { measureQuantumOperation } = useQuantumMetrics();
  const keyboardNav = useKeyboardNavigation();
  const { announceToScreenReader } = useScreenReaderOptimizations();
  const { breadcrumbs } = useQuantumBreadcrumbs();

  // Collapsible section states
  const [labSectionOpen, setLabSectionOpen] = useState(true);
  const [learningResourcesOpen, setLearningResourcesOpen] = useState(false);
  const [advancedFeaturesOpen, setAdvancedFeaturesOpen] = useState(false);
  const [learningJourneyOpen, setLearningJourneyOpen] = useState(false);
  const [quickStatsOpen, setQuickStatsOpen] = useState(false);

  useEffect(() => {
    // Preload critical chunks during idle time
    preloadCriticalChunks();

    // Announce page load to screen readers
    announceToScreenReader('QMLab quantum machine learning playground loaded');
  }, [announceToScreenReader]);

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      {/* PWA Manifest */}
      <PWAManifest />

      {/* Quantum background animation */}
      {/* Removed flashy quantum background */}

      <PageChrome
        title="QMLab - Interactive Quantum Machine Learning Laboratory"
        header={<HeaderProfessional />}
        footer={<QuantumFooter />}
      >
        <main id="main-content" tabIndex={-1}>
          {/* Professional Hero Section */}
          <HeroSectionProfessional />

          <div className="relative z-10 py-16">
            <div className="container mx-auto px-6">
              {/* Quantum Breadcrumbs */}
              <div className="mb-8 pt-4 border-t border-slate-700/30">
                <QuantumBreadcrumbs
                  items={breadcrumbs}
                  className="sticky top-20 bg-slate-950/80 backdrop-blur-sm py-3 px-4 rounded-lg border border-slate-700/50 z-30"
                />
              </div>
              {/* Quantum Section Separator */}
              <div className="flex items-center justify-center mb-16">
                <div className="w-full max-w-4xl relative">
                  <div className="h-px bg-gradient-to-r from-transparent via-blue-400/30 to-transparent"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-slate-950 px-6">
                      <div className="w-3 h-3 rounded-full bg-gradient-to-r from-blue-400 to-purple-500 animate-pulse"></div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Lab Section */}
              <Collapsible open={labSectionOpen} onOpenChange={setLabSectionOpen}>
                <section id="lab-section" className="mb-20" aria-labelledby="lab-heading">
                  <div className="text-center mb-12 relative">
                    <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 w-24 h-px bg-gradient-to-r from-transparent via-blue-400/50 to-transparent"></div>
                    <h1 id="lab-heading" className="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 mb-6 tracking-tight">
                      Interactive Quantum Laboratory
                    </h1>
                    <CollapsibleTrigger asChild>
                      <Button
                        variant="ghost"
                        className="absolute top-0 right-0"
                        aria-label={labSectionOpen ? "Collapse laboratory section" : "Expand laboratory section"}
                        aria-expanded={labSectionOpen}
                      >
                        {labSectionOpen ? (
                          <ChevronUp className="w-8 h-8 text-blue-400 hover:text-blue-300 transition-colors" />
                        ) : (
                          <ChevronDown className="w-8 h-8 text-blue-400 hover:text-blue-300 transition-colors" />
                        )}
                      </Button>
                    </CollapsibleTrigger>
                    <div className="w-16 h-px bg-gradient-to-r from-blue-400 to-purple-500 mx-auto mb-4"></div>
                    <p className="text-muted-foreground max-w-3xl mx-auto text-lg leading-relaxed">
                      Hands-on learning environment for quantum machine learning. Build circuits,
                      explore quantum states, and train algorithms to understand the fundamental
                      principles of quantum computing.
                    </p>
                    {!labSectionOpen && (
                      <div className="mt-4 flex justify-center gap-2">
                        <div className="px-3 py-1 bg-blue-500/20 text-blue-300 text-sm rounded-full border border-blue-400/30">
                          <Calculator className="w-3 h-3 mr-1 inline" /> Circuit Builder
                        </div>
                        <div className="px-3 py-1 bg-purple-500/20 text-purple-300 text-sm rounded-full border border-purple-400/30">
                          <Atom className="w-3 h-3 mr-1 inline" /> Bloch Sphere
                        </div>
                        <div className="px-3 py-1 bg-indigo-500/20 text-indigo-300 text-sm rounded-full border border-indigo-400/30">
                          <Brain className="w-3 h-3 mr-1 inline" /> ML Training
                        </div>
                      </div>
                    )}
                  </div>

                  <CollapsibleContent>
                    {/* Interactive Quantum Modules */}
                    <div className="max-w-7xl mx-auto space-y-12">
                      {/* Circuit Builder Section */}
                      <div className="space-y-6">
                <div className="p-8 rounded-2xl bg-slate-900/80 border border-blue-400/40 hover:border-blue-400/60 transition-all duration-300 shadow-xl backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-2 rounded-lg bg-blue-400/20 border border-blue-400/30" aria-hidden="true">
                        <Calculator className="w-6 h-6 text-blue-400" />
                      </div>
                      <div className="flex items-center gap-2 flex-1">
                        <h3 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-300">Circuit Builder</h3>
                        <QuantumConceptTooltip concept="circuit" />
                      </div>
                    </div>
                  <p className="text-muted-foreground text-sm">
                    <strong className="text-slate-300">Learn:</strong> Quantum gates, circuit construction, and quantum logic.<br/>
                    <strong className="text-slate-300">How:</strong> Drag and drop quantum gates to build circuits.<br/>
                    <strong className="text-slate-300">Why:</strong> Understanding gate operations is fundamental to quantum programming.
                  </p>
                </div>
                <div id="circuit-builder">
                  <Suspense fallback={<ComponentSkeleton />}>
                    <CircuitBuilder />
                  </Suspense>
                </div>
              </div>

              {/* Bloch Sphere Section */}
              <div className="space-y-6">
                <div className="p-8 rounded-2xl bg-slate-900/80 border border-purple-400/40 hover:border-purple-400/60 transition-all duration-300 shadow-xl backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-2 rounded-lg bg-purple-400/20 border border-purple-400/30" aria-hidden="true">
                        <Atom className="w-6 h-6 text-purple-400" />
                      </div>
                      <div className="flex items-center gap-2 flex-1">
                        <h3 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-purple-300">Bloch Sphere</h3>
                        <QuantumConceptTooltip concept="bloch-sphere" />
                      </div>
                    </div>
                  <p className="text-muted-foreground text-sm">
                    <strong className="text-slate-300">Learn:</strong> Quantum state representation and superposition.<br/>
                    <strong className="text-slate-300">How:</strong> Visualize qubit states as points on a 3D sphere.<br/>
                    <strong className="text-slate-300">Why:</strong> The Bloch sphere provides intuitive understanding of quantum mechanics.
                  </p>
                </div>
                <div id="visualization">
                  <Suspense fallback={<ComponentSkeleton />}>
                    <BlochSphere />
                  </Suspense>
                </div>
              </div>

              {/* ML Training Section */}
              <div className="space-y-6">
                <div className="p-8 rounded-2xl bg-slate-900/80 border border-indigo-400/40 hover:border-indigo-400/60 transition-all duration-300 shadow-xl backdrop-blur-sm">
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-2 rounded-lg bg-indigo-400/20 border border-indigo-400/30" aria-hidden="true">
                        <Brain className="w-6 h-6 text-indigo-400" />
                      </div>
                      <div className="flex items-center gap-2 flex-1">
                        <h3 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-indigo-300">ML Training</h3>
                        <QuantumConceptTooltip concept="vqe" />
                      </div>
                    </div>
                          <p className="text-muted-foreground text-sm">
                            <strong className="text-slate-300">Learn:</strong> Quantum gates,
                            circuit construction, and quantum logic.
                            <br />
                            <strong className="text-slate-300">How:</strong> Drag and drop quantum
                            gates to build circuits.
                            <br />
                            <strong className="text-slate-300">Why:</strong> Understanding gate
                            operations is fundamental to quantum programming.
                          </p>
                        </div>
                        <div id="circuit-builder">
                          <Suspense fallback={<ComponentSkeleton />}>
                            <CircuitBuilder />
                          </Suspense>
                        </div>
                      </div>

                      {/* Bloch Sphere Section */}
                      <div className="space-y-6">
                        <div className="p-8 rounded-2xl bg-slate-900/80 border border-purple-400/40 hover:border-purple-400/60 transition-all duration-300 shadow-xl backdrop-blur-sm">
                          <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-purple-400/20 border border-purple-400/30">
                              <Atom className="w-6 h-6 text-purple-400" />
                            </div>
                            <div className="flex items-center gap-2 flex-1">
                              <h3 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-purple-300">
                                Bloch Sphere
                              </h3>
                              <QuantumConceptTooltip concept="bloch-sphere" />
                            </div>
                          </div>
                          <p className="text-muted-foreground text-sm">
                            <strong className="text-slate-300">Learn:</strong> Quantum state
                            representation and superposition.
                            <br />
                            <strong className="text-slate-300">How:</strong> Visualize qubit states
                            as points on a 3D sphere.
                            <br />
                            <strong className="text-slate-300">Why:</strong> The Bloch sphere
                            provides intuitive understanding of quantum mechanics.
                          </p>
                        </div>
                        <div id="visualization">
                          <Suspense fallback={<ComponentSkeleton />}>
                            <BlochSphere />
                          </Suspense>
                        </div>
                      </div>

                      {/* ML Training Section */}
                      <div className="space-y-6">
                        <div className="p-8 rounded-2xl bg-slate-900/80 border border-indigo-400/40 hover:border-indigo-400/60 transition-all duration-300 shadow-xl backdrop-blur-sm">
                          <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 rounded-lg bg-indigo-400/20 border border-indigo-400/30">
                              <Brain className="w-6 h-6 text-indigo-400" />
                            </div>
                            <div className="flex items-center gap-2 flex-1">
                              <h3 className="text-xl font-semibold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-indigo-300">
                                ML Training
                              </h3>
                              <QuantumConceptTooltip concept="vqe" />
                            </div>
                          </div>
                          <p className="text-muted-foreground text-sm">
                            <strong className="text-slate-300">Learn:</strong> Variational quantum
                            circuits and optimization.
                            <br />
                            <strong className="text-slate-300">How:</strong> Train quantum
                            classifiers on real datasets.
                            <br />
                            <strong className="text-slate-300">Why:</strong> Quantum ML may offer
                            advantages for certain problem types.
                          </p>
                        </div>
                        <div id="training">
                          <Suspense fallback={<ComponentSkeleton />}>
                            <TrainingDashboard />
                          </Suspense>
                        </div>
                      </div>
                    </div>
                  </CollapsibleContent>
                </section>
              </Collapsible>

              {/* Quantum Section Separator */}
              <div className="flex items-center justify-center mb-16">
                <div className="w-full max-w-4xl relative">
                  <div className="h-px bg-gradient-to-r from-transparent via-purple-400/30 to-transparent"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="bg-slate-950 px-6">
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-purple-400 animate-pulse"></div>
                        <div
                          className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-400 to-indigo-500 animate-pulse"
                          style={{ animationDelay: '0.5s' }}
                        ></div>
                        <div
                          className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse"
                          style={{ animationDelay: '1s' }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Learning Resources Section */}
              <CollapsibleSection
                id="learning-resources"
                title="Quantum Learning Resources"
                description="Educational materials and explanations to help you master quantum machine learning concepts."
                isOpen={learningResourcesOpen}
                onOpenChange={setLearningResourcesOpen}
                titleClassName="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-indigo-400 to-blue-400"
                chevronColor="text-purple-400"
                previewTags={[
                  {
                    label: 'Fundamentals',
                    icon: <BookOpen className="w-3 h-3" />,
                    color: 'bg-blue-500/20 text-blue-300 border-blue-400/30',
                  },
                  {
                    label: 'ML Concepts',
                    icon: <Lightbulb className="w-3 h-3" />,
                    color: 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30',
                  },
                ]}
              >
                <div className="grid lg:grid-cols-2 gap-8 max-w-6xl mx-auto">
                  <div className="p-8 rounded-2xl bg-slate-900/90 border border-blue-400/30 shadow-xl hover:border-blue-400/50 transition-all duration-300 backdrop-blur-sm">
                    <div className="flex items-center gap-4 mb-8">
                      <div className="p-3 rounded-xl bg-blue-400/20 border border-blue-400/30">
                        <BookOpen className="w-7 h-7 text-blue-400" />
                      </div>
                      <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-300">
                        Quantum Fundamentals
                      </h2>
                    </div>
                    <div className="space-y-4">
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="font-medium text-slate-300">What is Superposition?</h3>
                          <QuantumConceptTooltip concept="superposition" />
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Unlike classical bits that are either 0 or 1, quantum bits (qubits) can
                          exist in a superposition of both states simultaneously, enabling quantum
                          parallelism.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="font-medium text-slate-300">Understanding Entanglement</h3>
                          <QuantumConceptTooltip concept="entanglement" />
                        </div>
                        <p className="text-sm text-muted-foreground">
                          When qubits become entangled, measuring one instantly affects the other,
                          regardless of distance. This enables quantum algorithms to process
                          information in fundamentally new ways.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="font-medium text-slate-300">Quantum Interference</h3>
                          <QuantumConceptTooltip concept="phase" />
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Quantum states can interfere constructively or destructively, allowing
                          quantum algorithms to amplify correct answers and cancel out wrong ones.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div className="p-8 rounded-2xl bg-slate-900/90 border border-yellow-400/30 shadow-xl hover:border-yellow-400/50 transition-all duration-300 backdrop-blur-sm">
                    <div className="flex items-center gap-4 mb-8">
                      <div className="p-3 rounded-xl bg-yellow-400/20 border border-yellow-400/30">
                        <Lightbulb className="w-7 h-7 text-yellow-400" />
                      </div>
                      <h2 className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-yellow-400 to-yellow-300">
                        Quantum ML Concepts
                      </h2>
                    </div>
                    <div className="space-y-4">
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <h3 className="font-medium text-slate-300 mb-2">
                          Variational Quantum Eigensolver (VQE)
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          A hybrid quantum-classical algorithm that uses parameterized quantum
                          circuits to find the ground state energy of quantum systems.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <h3 className="font-medium text-slate-300 mb-2">Quantum Neural Networks</h3>
                        <p className="text-sm text-muted-foreground">
                          Quantum circuits that can learn patterns in data by adjusting gate
                          parameters, potentially offering advantages for certain machine learning
                          tasks.
                        </p>
                      </div>
                      <div className="p-4 rounded-lg bg-slate-800/50 border border-slate-600/30">
                        <h3 className="font-medium text-slate-300 mb-2">Quantum Feature Maps</h3>
                        <p className="text-sm text-muted-foreground">
                          Methods for encoding classical data into quantum states, enabling quantum
                          algorithms to process classical machine learning problems.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Mathematical Foundations Section */}
                <div className="max-w-6xl mx-auto mt-12">
                  <h3 className="text-xl font-bold text-slate-200 mb-4 flex items-center gap-2">
                    <Calculator className="w-5 h-5 text-blue-400" />
                    Mathematical Foundations
                  </h3>
                  <div className="grid gap-4">
                    <MathematicalDerivation
                      title="Quantum Gate Operations"
                      concept="Unitary Matrix Representation"
                      difficulty="intermediate"
                      introduction="Quantum gates are represented as unitary matrices that preserve probability when applied to quantum states."
                      steps={[
                        {
                          equation: 'U† U = I',
                          explanation:
                            'Unitary condition: the conjugate transpose times the matrix equals identity',
                        },
                        {
                          equation: 'X = |0⟩⟨1| + |1⟩⟨0| = [[0,1], [1,0]]',
                          explanation:
                            'Pauli-X gate matrix representation - swaps |0⟩ and |1⟩ states',
                        },
                        {
                          equation: 'H = (1/√2)[[1,1], [1,-1]]',
                          explanation:
                            'Hadamard gate creates equal superposition from basis states',
                          note: 'This is the most important gate for creating superposition',
                        },
                      ]}
                      conclusion="Unitary matrices ensure quantum operations are reversible and preserve probability normalization."
                      applications={[
                        'Quantum circuit design and optimization',
                        'Error correction through gate decomposition',
                        'Quantum compiler optimization',
                      ]}
                    />

                    <MathematicalDerivation
                      title="Measurement Probabilities"
                      concept="Born Rule and State Collapse"
                      difficulty="beginner"
                      introduction="The Born rule determines measurement probabilities and describes how quantum superposition collapses to classical outcomes."
                      steps={[
                        {
                          equation: '|ψ⟩ = α|0⟩ + β|1⟩',
                          explanation: 'General qubit state with complex amplitudes α and β',
                        },
                        {
                          equation: 'P(0) = |α|², P(1) = |β|²',
                          explanation: 'Measurement probabilities from amplitude magnitudes',
                        },
                        {
                          equation: '⟨ψ|M†M|ψ⟩',
                          explanation: 'General measurement expectation value for operator M',
                          note: 'This generalizes to any quantum observable',
                        },
                      ]}
                      conclusion="The Born rule connects the mathematical formalism of quantum mechanics to experimental observations."
                      applications={[
                        'Quantum state tomography',
                        'Quantum error correction',
                        'Quantum algorithm analysis',
                      ]}
                    />
                  </div>
                </div>
              </CollapsibleSection>

              {/* Advanced Features Coming Soon */}
              <CollapsibleSection
                id="advanced-features"
                title="Advanced Features in Development"
                description="Additional tools and capabilities being developed to enhance your quantum learning experience."
                isOpen={advancedFeaturesOpen}
                onOpenChange={setAdvancedFeaturesOpen}
                titleClassName="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400"
                chevronColor="text-indigo-400"
                previewTags={[
                  {
                    label: 'Data Visualization',
                    icon: <BarChart3 className="w-3 h-3" />,
                    color: 'bg-slate-500/20 text-slate-300 border-slate-400/30',
                  },
                  {
                    label: 'Circuit Library',
                    icon: <Library className="w-3 h-3" />,
                    color: 'bg-slate-500/20 text-slate-300 border-slate-400/30',
                  },
                  {
                    label: 'Datasets',
                    icon: <Database className="w-3 h-3" />,
                    color: 'bg-slate-500/20 text-slate-300 border-slate-400/30',
                  },
                ]}
              >
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-6xl mx-auto">
                  <ComingSoonCard
                    title="Quantum Data Visualization"
                    description="Advanced plotting and analysis tools for quantum measurement data with interactive charts, statistical analysis, and publication-ready exports for quantum research."
                    features={[
                      'Interactive quantum state tomography plots with 3D visualization',
                      'Process fidelity analysis with confidence intervals',
                      'Real-time measurement analytics with statistical distributions',
                      'Export to LaTeX, PDF, and CSV formats for research papers',
                    ]}
                    eta="Q2 2025"
                    priority="high"
                    preview={
                      <div className="flex items-center justify-center h-full">
                        <BarChart3 className="w-12 h-12 text-primary/60" />
                      </div>
                    }
                  />

                  <ComingSoonCard
                    title="Circuit Library"
                    description="Comprehensive collection of pre-built quantum algorithms, educational circuit templates, and collaborative sharing platform for quantum computing enthusiasts and researchers."
                    features={[
                      'Curated quantum algorithm gallery with educational explanations',
                      'Variational circuit templates for machine learning applications',
                      'Educational circuit collections for quantum computing courses',
                      'One-click import with parameter customization and documentation',
                    ]}
                    eta="Q3 2025"
                    priority="medium"
                    preview={
                      <div className="flex items-center justify-center h-full">
                        <Library className="w-12 h-12 text-accent/60" />
                      </div>
                    }
                  />

                  <ComingSoonCard
                    title="Quantum Datasets"
                    description="Carefully curated and preprocessed datasets specifically designed for quantum machine learning research, education, and benchmarking with comprehensive documentation and tutorials."
                    features={[
                      'Quantum chemistry molecular datasets with ground state energies',
                      'Combinatorial optimization problem sets (QAOA, MaxCut, TSP)',
                      'Benchmark quantum classification and regression datasets',
                      'Custom dataset upload with quantum feature mapping tools',
                    ]}
                    eta="Q4 2025"
                    priority="medium"
                    preview={
                      <div className="flex items-center justify-center h-full">
                        <Database className="w-12 h-12 text-state-pure/60" />
                      </div>
                    }
                  />
                </div>
              </CollapsibleSection>

              {/* Learning Progress Tracker */}
              <CollapsibleSection
                id="learning-journey"
                title="Your Learning Journey"
                description="Track your progress through quantum concepts and unlock achievements as you master quantum machine learning."
                isOpen={learningJourneyOpen}
                onOpenChange={setLearningJourneyOpen}
                titleClassName="text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-indigo-400 to-pink-400"
                chevronColor="text-purple-400"
                previewTags={[
                  {
                    label: 'Progress Tracking',
                    color: 'bg-green-500/20 text-green-300 border-green-400/30',
                  },
                  {
                    label: 'Achievements',
                    color: 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30',
                  },
                ]}
              >
                <div className="max-w-4xl mx-auto">
                  <QuantumLearningTracker />
                </div>
              </CollapsibleSection>

              {/* Quick Stats */}
              <CollapsibleSection
                title="Platform Stats"
                isOpen={quickStatsOpen}
                onOpenChange={setQuickStatsOpen}
                titleClassName="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-slate-400 to-slate-300"
                chevronColor="text-muted-foreground"
              >
                <div className="grid grid-cols-2 md:grid-cols-4 gap-8 max-w-5xl mx-auto">
                  <div className="text-center p-6 rounded-xl bg-slate-900/60 border border-blue-400/30 hover:border-blue-400/50 transition-all duration-300">
                    <div className="text-4xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-blue-300 mb-2">
                      15+
                    </div>
                    <div className="text-sm text-muted-foreground font-medium">Quantum Gates</div>
                    <div className="w-8 h-px bg-gradient-to-r from-blue-400/50 to-transparent mx-auto mt-2"></div>
                  </div>
                  <div className="text-center p-6 rounded-xl bg-slate-900/60 border border-purple-400/30 hover:border-purple-400/50 transition-all duration-300">
                    <div className="text-4xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-purple-300 mb-2">
                      ∞
                    </div>
                    <div className="text-sm text-muted-foreground font-medium">Circuit Depth</div>
                    <div className="w-8 h-px bg-gradient-to-r from-purple-400/50 to-transparent mx-auto mt-2"></div>
                  </div>
                  <div className="text-center p-6 rounded-xl bg-slate-900/60 border border-indigo-400/30 hover:border-indigo-400/50 transition-all duration-300">
                    <div className="text-4xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-indigo-300 mb-2">
                      Live
                    </div>
                    <div className="text-sm text-muted-foreground font-medium">Visualization</div>
                    <div className="w-8 h-px bg-gradient-to-r from-indigo-400/50 to-transparent mx-auto mt-2"></div>
                  </div>
                  <div className="text-center p-6 rounded-xl bg-slate-900/60 border border-teal-400/30 hover:border-teal-400/50 transition-all duration-300">
                    <div className="text-4xl font-mono font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-teal-300 mb-2">
                      Cloud
                    </div>
                    <div className="text-sm text-muted-foreground font-medium">Based</div>
                    <div className="w-8 h-px bg-gradient-to-r from-teal-400/50 to-transparent mx-auto mt-2"></div>
                  </div>
                </div>
              </CollapsibleSection>
            </div>
          </div>

          {/* Floating CTA for enhanced user engagement */}
          <FloatingCTA />
        </main>
      </PageChrome>
    </div>
  );
}
