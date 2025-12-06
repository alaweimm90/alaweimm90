import React, { useState } from 'react';
import { Search, Download, Share2, Settings } from 'lucide-react';
import { IconButton } from '@/components/ui/icon-button';
import { QuantumTooltip, MathInsight } from '@/components/QuantumTooltip';
import { AtomicBlochSphere } from '@/components/AtomicBlochSphere';
import { QuantumMLPipeline } from '@/components/QuantumMLPipeline';
import { QuantumStatusBar } from '@/components/QuantumStatusBar';
import { QuantumCommandPalette, useQuantumCommandPalette } from '@/components/QuantumCommandPalette';

// Quantum Circuit Builder Panel
const CircuitBuilderPanel: React.FC = () => {
  const [selectedGate, setSelectedGate] = useState<string | null>(null);

  const quantumGates = [
    { id: 'H', label: 'H', name: 'Hadamard', color: 'orbital-s' },
    { id: 'X', label: 'X', name: 'Pauli-X', color: 'orbital-s' },
    { id: 'Y', label: 'Y', name: 'Pauli-Y', color: 'orbital-s' },
    { id: 'Z', label: 'Z', name: 'Pauli-Z', color: 'orbital-s' },
    { id: 'RX', label: 'RX', name: 'Rotation-X', color: 'orbital-p' },
    { id: 'RY', label: 'RY', name: 'Rotation-Y', color: 'orbital-p' },
    { id: 'RZ', label: 'RZ', name: 'Rotation-Z', color: 'orbital-p' },
    { id: 'CNOT', label: '⊕', name: 'CNOT', color: 'orbital-d' },
  ];

  const mlAlgorithms = [
    { id: 'VQC', name: 'Variational Quantum Classifier', desc: 'Supervised learning' },
    { id: 'QAOA', name: 'Quantum Approximate Optimization', desc: 'Combinatorial optimization' },
    { id: 'VQE', name: 'Variational Quantum Eigensolver', desc: 'Ground state finding' },
    { id: 'QGAN', name: 'Quantum GAN', desc: 'Generative modeling' },
  ];

  return (
    <div className="glass-panel p-6">
      <div className="mb-6">
        <h2 className="heading-refined-3 mb-2">Circuit Studio</h2>
        <p className="text-subtitle">Build your quantum ML circuit</p>
      </div>

      {/* Gate Library */}
      <div className="mb-6">
        <h3 className="text-caption mb-3">Quantum Gates</h3>
        <div className="grid grid-cols-4 gap-2">
          {quantumGates.map((gate) => (
            <QuantumTooltip key={gate.id} concept={gate.id} position="right">
              <button
                className={`quantum-gate-enhanced quantum-gate-${gate.id.toLowerCase()} ${
                  selectedGate === gate.id ? 'selected' : ''
                } button-quantum-enhanced focus-ring-advanced quantum-energy-field hardware-accelerated`}
                onClick={() => setSelectedGate(gate.id)}
                aria-label={`Add ${gate.name} gate to circuit`}
                aria-pressed={selectedGate === gate.id}
              >
                {gate.label}
              </button>
            </QuantumTooltip>
          ))}
        </div>
      </div>

      {/* ML Algorithms */}
      <div className="mb-6">
        <h3 className="text-caption mb-3">ML Algorithms</h3>
        <div className="space-y-2">
          {mlAlgorithms.map((algo) => (
            <QuantumTooltip key={algo.id} concept={algo.id} position="right">
              <button 
                className="w-full quantum-algorithm-card text-left card-advanced-hover focus-ring-advanced hardware-accelerated"
                aria-label={`Start training ${algo.name}`}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <div className="text-label text-primary font-semibold">{algo.id}</div>
                    <div className="body-elegant-sm mt-1">{algo.desc}</div>
                  </div>
                  <div className="quantum-status-indicator quantum-status-pure quantum-micro-animation"></div>
                </div>
              </button>
            </QuantumTooltip>
          ))}
        </div>
      </div>

      {/* Circuit Canvas */}
      <div className="glass-minimal p-4">
        <div className="flex items-center justify-between mb-4">
          <span className="text-mono">|q₀⟩ ─</span>
          <span className="body-elegant-sm">2 qubits</span>
        </div>
        
        {/* Simple circuit visualization */}
        <div className="space-y-6">
          <div className="flex items-center gap-3">
            <span className="text-mono w-8 text-muted">q₀:</span>
            <div className="quantum-circuit-wire flex-1 relative">
              <QuantumTooltip concept="H" position="top">
                <div className="absolute left-12 top-[-24px] quantum-gate-enhanced quantum-gate-hadamard text-xs min-w-8 min-h-8">H</div>
              </QuantumTooltip>
              <QuantumTooltip concept="CNOT" position="top">
                <div className="absolute left-32 top-[-24px] quantum-gate-enhanced quantum-gate-cnot text-xs min-w-8 min-h-8">●</div>
              </QuantumTooltip>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-mono w-8 text-muted">q₁:</span>
            <div className="quantum-circuit-wire flex-1 relative">
              <div className="absolute left-32 top-[-24px] quantum-gate-enhanced quantum-gate-cnot text-xs min-w-8 min-h-8">⊕</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Quantum Visualization Panel
const VisualizationPanel: React.FC = () => {
  return (
    <div className="glass-panel p-6">
      <div className="mb-6">
        <h2 className="heading-refined-3 mb-2">Quantum State</h2>
        <p className="text-subtitle">Live visualization of your quantum system</p>
      </div>

      {/* Atomic Bloch Sphere */}
      <div className="bloch-sphere-container mb-6">
        <div className="relative w-full h-80">
          <AtomicBlochSphere 
            showOrbitals={true}
            animateTransition={true}
          />
        </div>
      </div>

      {/* ML Pipeline Integration */}
      <QuantumMLPipeline />
    </div>
  );
};

// Mathematical Context Panel
const MathContextPanel: React.FC<{ selectedGate?: string | null }> = ({ selectedGate }) => {
  const [activeTab, setActiveTab] = useState<'gate' | 'algorithm' | 'theory'>('gate');
  
  return (
    <div className="math-context-panel">
      <div className="relative z-10">
        <div className="mb-6">
          <h2 className="heading-refined-3 mb-2">Mathematical Context</h2>
          <p className="text-subtitle">Deep understanding at your fingertips</p>
        </div>

        {/* Context Tabs */}
        <div className="flex gap-2 mb-4">
          <button
            onClick={() => setActiveTab('gate')}
            className={`px-3 py-1.5 rounded-lg text-xs transition-all min-h-[44px] ${
              activeTab === 'gate' 
                ? 'glass-subtle border border-primary text-primary' 
                : 'glass-minimal text-muted hover:text-primary'
            }`}
            aria-label="View gate mathematics"
            aria-pressed={activeTab === 'gate'}
          >
            Gate Math
          </button>
          <button
            onClick={() => setActiveTab('algorithm')}
            className={`px-3 py-1.5 rounded-lg text-xs transition-all min-h-[44px] ${
              activeTab === 'algorithm' 
                ? 'glass-subtle border border-primary text-primary' 
                : 'glass-minimal text-muted hover:text-primary'
            }`}
            aria-label="View algorithm details"
            aria-pressed={activeTab === 'algorithm'}
          >
            Algorithms
          </button>
          <button
            onClick={() => setActiveTab('theory')}
            className={`px-3 py-1.5 rounded-lg text-xs transition-all min-h-[44px] ${
              activeTab === 'theory' 
                ? 'glass-subtle border border-primary text-primary' 
                : 'glass-minimal text-muted hover:text-primary'
            }`}
            aria-label="View quantum theory"
            aria-pressed={activeTab === 'theory'}
          >
            Theory
          </button>
        </div>

        {/* Dynamic Content based on selected gate/tab */}
        {activeTab === 'gate' && (
          <MathInsight concept={selectedGate || 'H'} />
        )}
        
        {activeTab === 'algorithm' && (
          <div className="space-y-3">
            <div className="glass-minimal p-3">
              <h3 className="text-label mb-2">VQC Optimization</h3>
              <div className="quantum-formula-display">
                L(θ) = Σᵢ |⟨ψᵢ|U(θ)|0⟩ - yᵢ|²
              </div>
              <div className="body-elegant-sm">
                Minimize loss via parameter gradient descent in Hilbert space
              </div>
            </div>
          </div>
        )}
        
        {activeTab === 'theory' && (
          <div className="space-y-3">
            <div className="glass-minimal p-3">
              <h3 className="text-label mb-2">Quantum Supremacy</h3>
              <div className="body-elegant mb-2">
                For n qubits, quantum systems explore 2ⁿ dimensional Hilbert space
                simultaneously, while classical computers require exponential time.
              </div>
              <div className="body-elegant-sm">
                Key insight: Entanglement + Superposition = Exponential speedup
              </div>
            </div>
          </div>
        )}

        {/* Quick Actions */}
        <div>
          <h3 className="text-caption mb-3">Export</h3>
          <div className="space-y-2">
            <button 
              className="w-full p-2 glass-subtle text-left glass-interactive min-h-[44px]"
              aria-label="Export circuit as Qiskit Python code"
            >
              <div className="flex items-center justify-between">
                <span>Export Qiskit Python</span>
                <Download className="w-4 h-4" aria-hidden="true" />
              </div>
            </button>
            <button 
              className="w-full p-2 glass-subtle text-left glass-interactive min-h-[44px]"
              aria-label="Export circuit as PennyLane code"
            >
              <div className="flex items-center justify-between">
                <span>Export PennyLane</span>
                <Download className="w-4 h-4" aria-hidden="true" />
              </div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Workspace Component
export const QuantumWorkspace: React.FC = () => {
  const [selectedGate, setSelectedGate] = useState<string | null>('H');
  const { isOpen, openPalette, closePalette } = useQuantumCommandPalette();
  
  return (
    <div className="min-h-screen quantum-background critical-above-fold">
      {/* Enhanced Skip Link */}
      <a href="#main-content" className="skip-link-enhanced sr-only-focusable">
        Skip to main content
      </a>
      
      {/* Header */}
      <header className="glass-panel-advanced border-b border-white/10 p-4 hardware-accelerated" style={{zIndex: 'var(--z-fixed)'}}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-quantum-hydrogen-blue to-quantum-superposition flex items-center justify-center">
                <span className="text-white font-bold text-sm">Q</span>
              </div>
              <span className="heading-refined-2 font-display">QMLab</span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button 
              onClick={openPalette}
              className="flex items-center gap-2 px-3 py-2 glass-subtle glass-interactive"
              aria-label="Open command palette"
            >
              <Search className="w-4 h-4 text-slate-400" aria-hidden="true" />
              <span className="text-label">Search</span>
              <kbd className="px-1.5 py-0.5 text-xs bg-surface-2 rounded text-mono">⌘K</kbd>
            </button>
            <IconButton 
              label="Share current workspace"
              variant="quantum"
            >
              <Share2 className="w-5 h-5" />
            </IconButton>
            <IconButton 
              label="Open workspace settings"
              variant="quantum"
            >
              <Settings className="w-5 h-5" />
            </IconButton>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main id="main-content" tabIndex={-1} className="quantum-workspace p-6 grid grid-cols-1 lg:grid-cols-3 gap-6 optimized-grid quantum-scrollbar" role="main">
        <div className="lazy-load-section">
          <CircuitBuilderPanel />
        </div>
        <div className="lazy-load-section">
          <VisualizationPanel />
        </div>
        <div className="lazy-load-section">
          <MathContextPanel selectedGate={selectedGate} />
        </div>
      </main>

      {/* Status Bar */}
      <QuantumStatusBar />

      {/* Command Palette */}
      <QuantumCommandPalette isOpen={isOpen} onClose={closePalette} />
    </div>
  );
};