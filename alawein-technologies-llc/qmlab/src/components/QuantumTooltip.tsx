import React, { useState } from 'react';
import { HelpCircle, Zap, Atom } from 'lucide-react';

interface MathConcept {
  operation: string;
  matrix?: string;
  equation?: string;
  explanation: string;
  quantumAdvantage?: string;
  mlRelevance?: string;
  complexity?: {
    classical: string;
    quantum: string;
  };
}

const quantumConcepts: Record<string, MathConcept> = {
  'H': {
    operation: 'Hadamard Gate',
    matrix: 'H = (1/√2) [1  1]\n           [1 -1]',
    equation: '|0⟩ → (|0⟩ + |1⟩)/√2',
    explanation: 'Creates equal superposition by rotating qubit state 90° around X+Z axis',
    quantumAdvantage: 'Enables quantum parallelism - fundamental for search algorithms',
    mlRelevance: 'Feature space expansion in quantum ML models',
    complexity: {
      classical: 'O(2ⁿ) for n-bit superposition',
      quantum: 'O(1) single gate operation'
    }
  },
  'CNOT': {
    operation: 'Controlled-NOT Gate',
    matrix: 'CNOT = [1 0 0 0]\n       [0 1 0 0]\n       [0 0 0 1]\n       [0 0 1 0]',
    equation: '|00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |11⟩, |11⟩ → |10⟩',
    explanation: 'Creates entanglement by flipping target qubit if control is |1⟩',
    quantumAdvantage: 'Non-local correlations impossible classically',
    mlRelevance: 'Feature correlation learning in quantum neural networks',
    complexity: {
      classical: 'O(n) for n coupled classical bits',
      quantum: 'O(1) regardless of entanglement degree'
    }
  },
  'RY': {
    operation: 'Y-Rotation Gate',
    matrix: 'RY(θ) = [cos(θ/2)  -sin(θ/2)]\n        [sin(θ/2)   cos(θ/2)]',
    equation: 'Rotates |ψ⟩ around Y-axis by angle θ',
    explanation: 'Parametric gate for amplitude modulation - key for variational algorithms',
    quantumAdvantage: 'Continuous parameter space for optimization',
    mlRelevance: 'Trainable weights in variational quantum circuits',
    complexity: {
      classical: 'O(2ⁿ) matrix multiplication',
      quantum: 'O(1) rotation operation'
    }
  },
  'VQC': {
    operation: 'Variational Quantum Classifier',
    equation: 'L(θ) = Σᵢ |⟨ψᵢ|U(θ)|0⟩ - yᵢ|²',
    explanation: 'Parametric quantum circuit optimized via classical feedback loop',
    quantumAdvantage: 'Exponential Hilbert space for feature mapping',
    mlRelevance: 'Quantum kernel methods + classical optimization',
    complexity: {
      classical: 'O(2ᵈ) for d-dimensional feature space',
      quantum: 'O(poly(d)) for quantum feature maps'
    }
  },
  'Bell': {
    operation: 'Bell State Creation',
    equation: '|Φ⁺⟩ = (|00⟩ + |11⟩)/√2',
    explanation: 'Maximally entangled two-qubit state with perfect correlation',
    quantumAdvantage: 'Non-local quantum correlations violate Bell inequality',
    mlRelevance: 'Quantum feature entanglement for enhanced pattern recognition'
  }
};

interface QuantumTooltipProps {
  concept: string;
  children: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

export const QuantumTooltip: React.FC<QuantumTooltipProps> = ({ 
  concept, 
  children, 
  position = 'top' 
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const mathConcept = quantumConcepts[concept];

  if (!mathConcept) return <>{children}</>;

  const positionClasses = {
    top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 transform -translate-y-1/2 ml-2'
  };

  return (
    <div className="relative inline-block">
      <div
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        className="cursor-pointer"
      >
        {children}
      </div>
      
      {isVisible && (
        <div className={`absolute z-50 ${positionClasses[position]} quantum-fade-in`}>
          <div className="quantum-tooltip w-80 max-w-md">
            {/* Header */}
            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-white/20">
              <Atom className="w-5 h-5 state-superposition" />
              <h3 className="font-medium text-white">{mathConcept.operation}</h3>
            </div>

            {/* Mathematical Representation */}
            {mathConcept.matrix && (
              <div className="mb-3">
                <div className="text-xs text-slate-400 mb-1">Matrix Form:</div>
                <div className="quantum-mono text-xs text-slate-200 bg-slate-900/50 p-2 rounded border">
                  <pre>{mathConcept.matrix}</pre>
                </div>
              </div>
            )}

            {mathConcept.equation && (
              <div className="mb-3">
                <div className="text-xs text-slate-400 mb-1">Operation:</div>
                <div className="quantum-math text-sm text-slate-200 bg-slate-900/50 p-2 rounded border">
                  {mathConcept.equation}
                </div>
              </div>
            )}

            {/* Explanation */}
            <div className="mb-3">
              <div className="text-xs text-slate-400 mb-1">How it works:</div>
              <div className="text-xs text-slate-200 leading-relaxed">
                {mathConcept.explanation}
              </div>
            </div>

            {/* Quantum Advantage */}
            {mathConcept.quantumAdvantage && (
              <div className="mb-3">
                <div className="flex items-center gap-1 mb-1">
                  <Zap className="w-3 h-3 state-coherent" />
                  <div className="text-xs text-slate-400">Quantum Advantage:</div>
                </div>
                <div className="text-xs state-coherent leading-relaxed">
                  {mathConcept.quantumAdvantage}
                </div>
              </div>
            )}

            {/* ML Relevance */}
            {mathConcept.mlRelevance && (
              <div className="mb-3">
                <div className="text-xs text-slate-400 mb-1">ML Application:</div>
                <div className="text-xs text-slate-200 leading-relaxed">
                  {mathConcept.mlRelevance}
                </div>
              </div>
            )}

            {/* Complexity Comparison */}
            {mathConcept.complexity && (
              <div>
                <div className="text-xs text-slate-400 mb-2">Computational Complexity:</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-slate-900/50 p-2 rounded border">
                    <div className="text-slate-400 mb-1">Classical:</div>
                    <div className="quantum-mono text-slate-300">{mathConcept.complexity.classical}</div>
                  </div>
                  <div className="bg-slate-900/50 p-2 rounded border">
                    <div className="text-slate-400 mb-1">Quantum:</div>
                    <div className="quantum-mono state-coherent">{mathConcept.complexity.quantum}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Tooltip Arrow */}
            <div className={`absolute w-2 h-2 bg-slate-800 border border-white/20 transform rotate-45 ${
              position === 'top' ? 'top-full left-1/2 -translate-x-1/2 -mt-1' :
              position === 'bottom' ? 'bottom-full left-1/2 -translate-x-1/2 -mb-1' :
              position === 'left' ? 'left-full top-1/2 -translate-y-1/2 -ml-1' :
              'right-full top-1/2 -translate-y-1/2 -mr-1'
            }`}></div>
          </div>
        </div>
      )}
    </div>
  );
};

// Quick Math Insight Component
interface MathInsightProps {
  concept: string;
  compact?: boolean;
}

export const MathInsight: React.FC<MathInsightProps> = ({ concept, compact = false }) => {
  const mathConcept = quantumConcepts[concept];
  
  if (!mathConcept) return null;

  if (compact) {
    return (
      <div className="quantum-glass p-3 rounded-lg">
        <div className="flex items-center gap-2 mb-2">
          <HelpCircle className="w-4 h-4 text-slate-400" />
          <span className="text-sm font-medium text-white">{mathConcept.operation}</span>
        </div>
        <div className="text-xs text-slate-300 leading-relaxed">
          {mathConcept.explanation}
        </div>
      </div>
    );
  }

  return (
    <div className="math-context-panel">
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <Atom className="w-5 h-5 state-superposition" />
          <h3 className="text-lg font-medium text-white">{mathConcept.operation}</h3>
        </div>
        
        {mathConcept.equation && (
          <div className="quantum-glass p-4 rounded-lg mb-4">
            <div className="quantum-math text-base text-slate-200 mb-2">
              {mathConcept.equation}
            </div>
          </div>
        )}

        <div className="space-y-3 text-sm">
          <div className="text-slate-200 leading-relaxed">
            {mathConcept.explanation}
          </div>
          
          {mathConcept.quantumAdvantage && (
            <div className="flex items-start gap-2">
              <Zap className="w-4 h-4 state-coherent mt-0.5 flex-shrink-0" />
              <div>
                <div className="font-medium state-coherent mb-1">Quantum Advantage:</div>
                <div className="text-slate-300 text-xs leading-relaxed">
                  {mathConcept.quantumAdvantage}
                </div>
              </div>
            </div>
          )}
          
          {mathConcept.mlRelevance && (
            <div className="text-xs text-slate-400 leading-relaxed">
              <strong className="text-slate-300">ML Impact:</strong> {mathConcept.mlRelevance}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};