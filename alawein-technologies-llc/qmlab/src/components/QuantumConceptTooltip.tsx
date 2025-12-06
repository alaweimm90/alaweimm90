import React from 'react';
import { HelpCircle, Calculator, BookOpen } from 'lucide-react';
import { 
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { MathematicalDerivation } from '@/components/MathematicalDerivation';

interface QuantumConceptTooltipProps {
  concept: 'superposition' | 'entanglement' | 'measurement' | 'bloch-sphere' | 'qubit' | 'gate' | 'circuit' | 'vqe' | 'phase';
  size?: 'sm' | 'md';
}

interface MathDerivation {
  title: string;
  concept: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  introduction: string;
  steps: Array<{
    equation: string;
    explanation: string;
    note?: string;
  }>;
  conclusion: string;
  applications: string[];
}

interface ConceptData {
  title: string;
  explanation: string;
  example: string;
  hasMath?: boolean;
  mathDerivation?: MathDerivation;
}

const conceptExplanations: Record<QuantumConceptTooltipProps['concept'], ConceptData> = {
  superposition: {
    title: 'Superposition',
    explanation: 'A quantum state where a qubit exists in multiple states simultaneously. Unlike classical bits that are either 0 or 1, qubits can be in a combination of both states until measured.',
    example: 'A coin spinning in the air is both heads and tails until it lands.',
    hasMath: true,
    mathDerivation: {
      title: 'Superposition State Mathematics',
      concept: 'Linear Combination of Basis States',
      difficulty: 'beginner',
      introduction: 'A quantum superposition is mathematically represented as a linear combination of basis states with complex coefficients called amplitudes.',
      steps: [
        {
          equation: '|ψ⟩ = α|0⟩ + β|1⟩',
          explanation: 'General form of a qubit state where α and β are complex probability amplitudes'
        },
        {
          equation: '|α|² + |β|² = 1',
          explanation: 'Normalization condition - the probabilities of measuring |0⟩ and |1⟩ must sum to 1',
          note: 'This ensures the total probability equals 100%'
        },
        {
          equation: 'P(0) = |α|², P(1) = |β|²',
          explanation: 'Born rule: measurement probabilities are the squared magnitudes of amplitudes'
        }
      ],
      conclusion: 'Superposition allows quantum systems to exist in multiple states simultaneously, enabling quantum parallelism and interference effects.',
      applications: [
        'Quantum algorithms like Grover\'s search',
        'Quantum Fourier Transform',
        'Creating equal superpositions with Hadamard gates'
      ]
    }
  },
  entanglement: {
    title: 'Quantum Entanglement',
    explanation: 'A phenomenon where two or more qubits become interconnected, so measuring one instantly affects the others regardless of distance. Einstein called this "spooky action at a distance."',
    example: 'Like having two magical coins that always land on opposite sides, no matter how far apart they are.',
    hasMath: false
  },
  measurement: {
    title: 'Quantum Measurement',
    explanation: 'The process of observing a quantum system, which collapses the superposition into a definite state. The outcome is probabilistic based on the quantum state amplitudes.',
    example: 'Like stopping a spinning coin - it randomly lands on heads or tails based on how it was spinning.',
    hasMath: false
  },
  'bloch-sphere': {
    title: 'Bloch Sphere',
    explanation: 'A geometric representation of qubit states as points on a 3D sphere. The north pole represents |0⟩, south pole |1⟩, and the equator represents superposition states.',
    example: 'Think of it as a globe where every point represents a different quantum state.',
    hasMath: true,
    mathDerivation: {
      title: 'Bloch Sphere Parameterization',
      concept: 'Spherical Coordinate Representation',
      difficulty: 'intermediate',
      introduction: 'Any qubit state can be represented as a point on the Bloch sphere using spherical coordinates θ (polar angle) and φ (azimuthal angle).',
      steps: [
        {
          equation: '|ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩',
          explanation: 'General qubit state in terms of Bloch sphere angles θ and φ'
        },
        {
          equation: 'x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)',
          explanation: 'Cartesian coordinates on the unit sphere surface'
        },
        {
          equation: '|0⟩: θ=0, |1⟩: θ=π, |+⟩: θ=π/2, φ=0',
          explanation: 'Common quantum states mapped to specific sphere points',
          note: 'The north pole (θ=0) represents |0⟩, south pole (θ=π) represents |1⟩'
        }
      ],
      conclusion: 'The Bloch sphere provides geometric intuition for quantum operations as rotations in 3D space.',
      applications: [
        'Visualizing quantum gate operations as rotations',
        'Understanding quantum state evolution',
        'Designing pulse sequences in NMR and quantum control'
      ]
    }
  },
  qubit: {
    title: 'Quantum Bit (Qubit)',
    explanation: 'The fundamental unit of quantum information. Unlike classical bits, qubits can exist in superposition, holding both 0 and 1 states simultaneously until measured.',
    example: 'Like a magical bit that can be 0, 1, or both at the same time.',
    hasMath: false
  },
  gate: {
    title: 'Quantum Gate',
    explanation: 'Operations that manipulate qubits by rotating their state on the Bloch sphere. Gates are the building blocks of quantum circuits, similar to logic gates in classical computing.',
    example: 'Like instructions that tell a qubit how to change its state, such as "flip" or "rotate".',
    hasMath: false
  },
  circuit: {
    title: 'Quantum Circuit',
    explanation: 'A sequence of quantum gates applied to qubits to perform quantum algorithms. Circuits are read from left to right, with each gate transforming the quantum state.',
    example: 'Like a recipe with step-by-step instructions for manipulating quantum information.',
    hasMath: false
  },
  vqe: {
    title: 'Variational Quantum Eigensolver',
    explanation: 'A hybrid quantum-classical algorithm that finds the minimum energy of quantum systems. It\'s used in quantum chemistry and optimization problems.',
    example: 'Like having a quantum computer and classical computer work together to solve chemistry problems.',
    hasMath: true,
    mathDerivation: {
      title: 'Variational Quantum Eigensolver Algorithm',
      concept: 'Variational Principle in Quantum Mechanics',
      difficulty: 'advanced',
      introduction: 'VQE uses the variational principle: any trial wavefunction gives an energy expectation value greater than or equal to the true ground state energy.',
      steps: [
        {
          equation: 'E₀ ≤ ⟨ψ(θ)|H|ψ(θ)⟩',
          explanation: 'Variational principle: trial state energy is upper bound to ground state energy E₀'
        },
        {
          equation: '|ψ(θ)⟩ = U(θ)|0⟩',
          explanation: 'Parameterized quantum circuit prepares trial state with parameters θ'
        },
        {
          equation: 'E(θ) = ⟨0|U†(θ)HU(θ)|0⟩',
          explanation: 'Expected energy as function of circuit parameters'
        },
        {
          equation: 'θ* = argmin_θ E(θ)',
          explanation: 'Classical optimizer finds parameters that minimize the energy',
          note: 'This optimization loop is the hybrid classical-quantum aspect of VQE'
        }
      ],
      conclusion: 'VQE combines quantum state preparation with classical optimization to find ground states of molecular Hamiltonians.',
      applications: [
        'Quantum chemistry calculations for drug discovery',
        'Materials science and catalyst design',
        'Optimization problems in finance and logistics'
      ]
    }
  },
  phase: {
    title: 'Quantum Phase',
    explanation: 'A property of quantum states that affects how they interfere with each other. Phase differences can cause constructive or destructive interference in quantum algorithms.',
    example: 'Like the timing of waves - when they align they amplify, when they oppose they cancel out.',
    hasMath: false
  }
};

export const QuantumConceptTooltip: React.FC<QuantumConceptTooltipProps> = ({ 
  concept, 
  size = 'sm' 
}) => {
  const conceptData = conceptExplanations[concept];
  
  if (!conceptData) {
    return null;
  }

  return (
    <TooltipProvider>
      <Tooltip delayDuration={300}>
        <TooltipTrigger asChild>
          <button
            className={`inline-flex items-center justify-center rounded-full bg-primary/20 hover:bg-primary/30 border border-primary/30 hover:border-primary/50 text-primary transition-all duration-200 ${
              size === 'sm' ? 'w-4 h-4' : 'w-5 h-5'
            }`}
            aria-label={`Learn about ${conceptData.title}`}
          >
            <HelpCircle className={size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'} />
          </button>
        </TooltipTrigger>
        <TooltipContent 
          className="max-w-sm p-4 bg-slate-900 border border-slate-700 shadow-2xl"
          side="top"
          sideOffset={5}
        >
          <div className="space-y-3">
            <div className="font-semibold text-primary text-sm">
              {conceptData.title}
            </div>
            
            <div className="text-xs text-slate-300 leading-relaxed">
              {conceptData.explanation}
            </div>
            
            <div className="pt-2 border-t border-slate-700">
              <div className="text-xs text-slate-400">
                <span className="font-medium text-slate-300">Simple analogy:</span>
                <div className="mt-1 italic">
                  {conceptData.example}
                </div>
              </div>
            </div>
            
            <div className="flex items-center justify-between pt-2">
              <div className="flex items-center gap-2">
                <div className="w-1 h-1 rounded-full bg-blue-400 animate-pulse"></div>
                <div className="text-xs text-blue-400">Quantum Concept</div>
              </div>
              
              {conceptData.hasMath && conceptData.mathDerivation && (
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-6 px-2 text-xs hover:bg-slate-800">
                      <Calculator className="w-3 h-3 mr-1" />
                      Math
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto bg-slate-900 border-slate-700">
                    <DialogHeader>
                      <DialogTitle className="flex items-center gap-2 text-slate-200">
                        <BookOpen className="w-5 h-5 text-blue-400" />
                        Mathematical Deep Dive: {conceptData.title}
                      </DialogTitle>
                    </DialogHeader>
                    <MathematicalDerivation {...conceptData.mathDerivation} />
                  </DialogContent>
                </Dialog>
              )}
            </div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};