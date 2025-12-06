import React, { useState, useEffect } from 'react';
import { Cpu, Zap, Activity, Atom, TrendingUp } from 'lucide-react';

interface SystemStatus {
  qubits: number;
  entanglement: number;
  coherenceTime: number;
  fidelity: number;
  gateCount: number;
  circuitDepth: number;
  quantumVolume: number;
}

export const QuantumStatusBar: React.FC = () => {
  const [status, setStatus] = useState<SystemStatus>({
    qubits: 4,
    entanglement: 0.73,
    coherenceTime: 125.4, // microseconds
    fidelity: 0.9892,
    gateCount: 12,
    circuitDepth: 6,
    quantumVolume: 32
  });

  const [isConnected, setIsConnected] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(Date.now());

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setStatus(prev => ({
        ...prev,
        entanglement: Math.max(0, Math.min(1, prev.entanglement + (Math.random() - 0.5) * 0.05)),
        coherenceTime: Math.max(100, prev.coherenceTime + (Math.random() - 0.5) * 5),
        fidelity: Math.max(0.95, Math.min(0.999, prev.fidelity + (Math.random() - 0.5) * 0.002))
      }));
      setLastUpdate(Date.now());
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (value: number, threshold: number = 0.8) => {
    if (value >= threshold) return 'state-coherent';
    if (value >= threshold * 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(1)}μs`;
    return `${(ms / 1000).toFixed(2)}ms`;
  };

  return (
    <div className="quantum-glass border-t border-white/10 p-3">
      <div className="flex items-center justify-between">
        {/* System Status */}
        <div className="flex items-center gap-6">
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className="text-xs text-slate-400">
              {isConnected ? 'Quantum Backend' : 'Disconnected'}
            </span>
          </div>

          {/* Qubit Count */}
          <div className="flex items-center gap-2">
            <Atom className="w-4 h-4 text-slate-500" />
            <span className="quantum-mono text-xs text-white">{status.qubits}</span>
            <span className="text-xs text-slate-400">qubits</span>
          </div>

          {/* Entanglement */}
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-slate-500" />
            <span className={`quantum-mono text-xs ${getStatusColor(status.entanglement, 0.7)}`}>
              {(status.entanglement * 100).toFixed(0)}%
            </span>
            <span className="text-xs text-slate-400">entangled</span>
          </div>

          {/* Coherence */}
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-slate-500" />
            <span className={`quantum-mono text-xs ${getStatusColor(status.coherenceTime / 200)}`}>
              {formatTime(status.coherenceTime)}
            </span>
            <span className="text-xs text-slate-400">T₂</span>
          </div>

          {/* Fidelity */}
          <div className="flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-slate-500" />
            <span className={`quantum-mono text-xs ${getStatusColor(status.fidelity, 0.98)}`}>
              {(status.fidelity * 100).toFixed(2)}%
            </span>
            <span className="text-xs text-slate-400">fidelity</span>
          </div>
        </div>

        {/* Circuit Metrics */}
        <div className="flex items-center gap-6">
          {/* Gate Count */}
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-slate-500" />
            <span className="quantum-mono text-xs text-white">{status.gateCount}</span>
            <span className="text-xs text-slate-400">gates</span>
          </div>

          {/* Circuit Depth */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">depth:</span>
            <span className="quantum-mono text-xs text-white">{status.circuitDepth}</span>
          </div>

          {/* Quantum Volume */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400">QV:</span>
            <span className="quantum-mono text-xs state-superposition">{status.quantumVolume}</span>
          </div>

          {/* Last Update */}
          <div className="text-xs text-slate-500">
            Updated {((Date.now() - lastUpdate) / 1000).toFixed(0)}s ago
          </div>
        </div>
      </div>
    </div>
  );
};