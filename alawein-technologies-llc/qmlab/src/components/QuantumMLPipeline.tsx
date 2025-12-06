import React, { useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Download, TrendingUp, Cpu, Zap, Target } from 'lucide-react';
import { IconButton } from '@/components/ui/icon-button';
import { QuantumTooltip } from './QuantumTooltip';

interface TrainingMetrics {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  quantumAdvantage: number;
  convergenceRate: number;
  parameters: number;
}

interface MLAlgorithm {
  id: string;
  name: string;
  description: string;
  parameters: {
    layers: number;
    qubits: number;
    entanglementType: 'full' | 'linear' | 'circular';
    optimizer: 'COBYLA' | 'SPSA' | 'Adam';
    learningRate: number;
  };
}

export const QuantumMLPipeline: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('VQC');
  const [metrics, setMetrics] = useState<TrainingMetrics>({
    epoch: 0,
    totalEpochs: 100,
    loss: 1.0,
    accuracy: 0,
    quantumAdvantage: 1,
    convergenceRate: 0,
    parameters: 12
  });

  const algorithms: Record<string, MLAlgorithm> = {
    VQC: {
      id: 'VQC',
      name: 'Variational Quantum Classifier',
      description: 'Parametric quantum circuit for classification',
      parameters: {
        layers: 3,
        qubits: 4,
        entanglementType: 'full',
        optimizer: 'COBYLA',
        learningRate: 0.01
      }
    },
    QAOA: {
      id: 'QAOA',
      name: 'Quantum Approximate Optimization',
      description: 'Combinatorial optimization solver',
      parameters: {
        layers: 2,
        qubits: 6,
        entanglementType: 'circular',
        optimizer: 'SPSA',
        learningRate: 0.1
      }
    },
    VQE: {
      id: 'VQE',
      name: 'Variational Quantum Eigensolver',
      description: 'Ground state energy finder',
      parameters: {
        layers: 4,
        qubits: 4,
        entanglementType: 'linear',
        optimizer: 'Adam',
        learningRate: 0.001
      }
    }
  };

  // Simulate training progress
  useEffect(() => {
    if (!isTraining) return;

    const interval = setInterval(() => {
      setMetrics(prev => {
        if (prev.epoch >= prev.totalEpochs) {
          setIsTraining(false);
          return prev;
        }

        const newEpoch = prev.epoch + 1;
        const progress = newEpoch / prev.totalEpochs;
        
        // Simulate convergence
        const loss = Math.max(0.01, 1 - progress * 0.9 + Math.random() * 0.1);
        const accuracy = Math.min(0.98, progress * 0.85 + Math.random() * 0.1);
        const convergenceRate = 1 - loss;
        const quantumAdvantage = 1 + Math.log2(prev.parameters) * progress;

        return {
          ...prev,
          epoch: newEpoch,
          loss,
          accuracy,
          convergenceRate,
          quantumAdvantage
        };
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isTraining]);

  const handleStartTraining = () => {
    setIsTraining(true);
    setMetrics(prev => ({ ...prev, epoch: 0, loss: 1.0, accuracy: 0 }));
  };

  const handleStopTraining = () => {
    setIsTraining(false);
  };

  const handleReset = () => {
    setIsTraining(false);
    setMetrics({
      epoch: 0,
      totalEpochs: 100,
      loss: 1.0,
      accuracy: 0,
      quantumAdvantage: 1,
      convergenceRate: 0,
      parameters: 12
    });
  };

  const currentAlgo = algorithms[selectedAlgorithm];

  return (
    <div className="space-y-6">
      {/* Algorithm Selection */}
      <div className="quantum-glass-strong p-6 rounded-2xl">
        <h2 className="quantum-display text-xl font-light text-white mb-4">ML Pipeline Configuration</h2>
        
        <div className="grid grid-cols-3 gap-3 mb-4">
          {Object.values(algorithms).map(algo => (
            <QuantumTooltip key={algo.id} concept={algo.id} position="top">
              <button
                onClick={() => setSelectedAlgorithm(algo.id)}
                className={`p-3 rounded-lg transition-all min-h-[44px] ${
                  selectedAlgorithm === algo.id
                    ? 'quantum-glass border-2 border-quantum-photon-blue'
                    : 'quantum-glass border border-transparent hover:border-slate-600'
                }`}
                aria-label={`Select ${algo.name} algorithm`}
                aria-pressed={selectedAlgorithm === algo.id}
              >
                <div className="text-sm font-medium text-white mb-1">{algo.id}</div>
                <div className="text-xs text-slate-400">{algo.description}</div>
              </button>
            </QuantumTooltip>
          ))}
        </div>

        {/* Algorithm Parameters */}
        <div className="quantum-glass p-4 rounded-xl">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-xs text-slate-400 mb-1">Architecture</div>
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-slate-500" />
                <span className="quantum-mono text-sm text-white">
                  {currentAlgo.parameters.layers} layers × {currentAlgo.parameters.qubits} qubits
                </span>
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400 mb-1">Entanglement</div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-slate-500" />
                <span className="quantum-mono text-sm text-white capitalize">
                  {currentAlgo.parameters.entanglementType}
                </span>
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400 mb-1">Optimizer</div>
              <div className="quantum-mono text-sm text-white">
                {currentAlgo.parameters.optimizer}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-400 mb-1">Learning Rate</div>
              <div className="quantum-mono text-sm text-white">
                {currentAlgo.parameters.learningRate}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Controls */}
      <div className="quantum-glass-strong p-6 rounded-2xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-light text-white">Training Progress</h3>
          <div className="flex items-center gap-2">
            {!isTraining ? (
              <button
                onClick={handleStartTraining}
                className="quantum-btn px-4 py-2 rounded-lg flex items-center gap-2 min-h-[44px]"
                aria-label="Start training quantum machine learning model"
              >
                <Play className="w-4 h-4" aria-hidden="true" />
                <span className="text-sm">Start Training</span>
              </button>
            ) : (
              <button
                onClick={handleStopTraining}
                className="quantum-btn px-4 py-2 rounded-lg flex items-center gap-2 min-h-[44px]"
                aria-label="Pause training process"
              >
                <Pause className="w-4 h-4" aria-hidden="true" />
                <span className="text-sm">Pause Training</span>
              </button>
            )}
            <IconButton
              onClick={handleReset}
              label="Reset training — clears all progress"
              variant="quantum"
              size="sm"
            >
              <RotateCcw className="w-4 h-4" />
            </IconButton>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between text-xs text-slate-400 mb-2">
            <span>Epoch {metrics.epoch}/{metrics.totalEpochs}</span>
            <span>{((metrics.epoch / metrics.totalEpochs) * 100).toFixed(0)}%</span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-quantum-neon-green to-quantum-photon-blue rounded-full transition-all duration-300"
              style={{ width: `${(metrics.epoch / metrics.totalEpochs) * 100}%` }}
            />
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 gap-4">
          <div className="quantum-glass p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400">Loss</span>
              <TrendingUp className="w-3 h-3 text-slate-500" />
            </div>
            <div className="quantum-mono text-xl text-white">
              {metrics.loss.toFixed(4)}
            </div>
            <div className="mt-2 h-8">
              <svg className="w-full h-full">
                <polyline
                  points={Array.from({ length: 20 }, (_, i) => {
                    const x = (i / 19) * 100;
                    const y = 30 - (Math.random() * 20 * (i / 19));
                    return `${x},${y}`;
                  }).join(' ')}
                  fill="none"
                  stroke="rgb(34, 197, 94)"
                  strokeWidth="1"
                  opacity="0.5"
                />
              </svg>
            </div>
          </div>

          <div className="quantum-glass p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-slate-400">Accuracy</span>
              <Target className="w-3 h-3 text-slate-500" />
            </div>
            <div className="quantum-mono text-xl state-coherent">
              {(metrics.accuracy * 100).toFixed(1)}%
            </div>
            <div className="mt-2">
              <div className="text-xs text-slate-500">
                Converging at {(metrics.convergenceRate * 100).toFixed(0)}% rate
              </div>
            </div>
          </div>
        </div>

        {/* Quantum Advantage Indicator */}
        <div className="mt-4 quantum-glass p-4 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-xs text-slate-400 mb-1">Quantum Advantage</div>
              <div className="flex items-center gap-2">
                <div className="quantum-mono text-lg state-superposition">
                  {metrics.quantumAdvantage.toFixed(2)}×
                </div>
                <span className="text-xs text-slate-500">speedup vs classical</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                metrics.quantumAdvantage > 2 ? 'bg-quantum-coherent' : 
                metrics.quantumAdvantage > 1.5 ? 'bg-quantum-photon-blue' : 
                'bg-slate-500'
              } animate-pulse`} />
              <span className="text-xs text-slate-400">
                {metrics.quantumAdvantage > 2 ? 'Quantum supremacy' : 
                 metrics.quantumAdvantage > 1.5 ? 'Quantum advantage' : 
                 'Building advantage'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Export Options */}
      <div className="quantum-glass p-4 rounded-xl">
        <div className="flex items-center justify-between">
          <span className="text-sm text-slate-300">Export Trained Model</span>
          <div className="flex items-center gap-2">
            <button 
              className="px-3 py-1.5 quantum-glass rounded-lg text-xs text-white quantum-interactive flex items-center gap-2 min-h-[44px]"
              aria-label="Export trained model as Qiskit code (Python file)"
            >
              <Download className="w-3 h-3" aria-hidden="true" />
              Export Qiskit
            </button>
            <button 
              className="px-3 py-1.5 quantum-glass rounded-lg text-xs text-white quantum-interactive flex items-center gap-2 min-h-[44px]"
              aria-label="Export trained model as PennyLane code (Python file)"
            >
              <Download className="w-3 h-3" aria-hidden="true" />
              Export PennyLane
            </button>
            <button 
              className="px-3 py-1.5 quantum-glass rounded-lg text-xs text-white quantum-interactive flex items-center gap-2 min-h-[44px]"
              aria-label="Export trained model as Cirq code (Python file)"
            >
              <Download className="w-3 h-3" aria-hidden="true" />
              Export Cirq
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};