import * as React from "react"
import { cn } from "@/lib/utils"

interface GateData {
  id: string;
  name: string;
  qubit: number;
}

interface QuantumCircuitWireProps {
  qubitIndex: number;
  gates: GateData[];
  availableGates: Array<{ name: string; symbol: string; label: string; description: string }>;
  selectedGate?: string;
  isRunning?: boolean;
  onGateAdd?: (gateName: string, qubit: number) => void;
  onGateRemove?: (gateId: string) => void;
  onDrop?: (qubit: number) => void;
  onDragOver?: (event: React.DragEvent) => void;
}

export const QuantumCircuitWire: React.FC<QuantumCircuitWireProps> = ({
  qubitIndex,
  gates,
  availableGates,
  selectedGate,
  isRunning = false,
  onGateAdd,
  onGateRemove,
  onDrop,
  onDragOver
}) => {
  const qubitGates = gates.filter(gate => gate.qubit === qubitIndex);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.key === 'Enter' || e.key === ' ') && selectedGate && !isRunning) {
      e.preventDefault();
      onGateAdd?.(selectedGate, qubitIndex);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    onDrop?.(qubitIndex);
  };

  return (
    <div className="flex items-center gap-4">
      {/* Qubit label */}
      <div className="text-sm font-mono text-slate-400 w-8" aria-label={`Qubit ${qubitIndex}`}>
        |{qubitIndex}⟩
      </div>
      
      {/* Quantum wire line */}
      <div className="flex-1 h-0.5 bg-slate-600/50 relative" aria-hidden="true">
        <div className="absolute inset-0 bg-gradient-to-r from-slate-600/50 via-slate-500/70 to-slate-600/50" />
      </div>
      
      {/* Drop zone and gate container */}
      <div
        className={cn(
          "flex-1 min-h-[3rem] border-2 border-dashed transition-all duration-200 rounded-lg p-2",
          "flex items-center gap-2 flex-wrap",
          "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
          isRunning 
            ? "border-slate-600/30 cursor-not-allowed" 
            : "border-slate-600/30 hover:border-blue-400/40"
        )}
        onDrop={handleDrop}
        onDragOver={onDragOver}
        role="region"
        aria-label={`Drop zone for qubit |${qubitIndex}⟩ - contains ${qubitGates.length} gates`}
        tabIndex={0}
        onKeyDown={handleKeyDown}
      >
        {/* Existing gates */}
        {qubitGates.map((gate) => {
          const gateInfo = availableGates.find(g => g.name === gate.name);
          return (
            <button
              key={gate.id}
              onClick={(e) => {
                e.stopPropagation();
                onGateRemove?.(gate.id);
              }}
              className={cn(
                "group relative bg-blue-500/20 border border-blue-400/40 rounded-md",
                "px-3 py-1.5 text-sm font-mono font-bold text-blue-300",
                "hover:bg-blue-500/30 hover:border-blue-400/60 transition-all duration-200",
                "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
                "min-h-[36px] min-w-[36px]" // Smaller but still accessible
              )}
              disabled={isRunning}
              aria-label={`${gateInfo?.label || gate.name} gate on qubit |${qubitIndex}⟩ - click to remove`}
              title={`${gateInfo?.label || gate.name}: ${gateInfo?.description || 'Quantum gate'} - Click to remove`}
            >
              {gateInfo?.symbol || gate.name}
              <span className="sr-only">Remove {gateInfo?.label || gate.name} gate</span>
              
              {/* Remove indicator on hover */}
              <div 
                className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center text-xs text-white font-bold" 
                aria-hidden="true"
              >
                ×
              </div>
            </button>
          );
        })}
        
        {/* Add gate button for keyboard users */}
        {!isRunning && selectedGate && (
          <button
            onClick={() => onGateAdd?.(selectedGate, qubitIndex)}
            className={cn(
              "bg-blue-500/10 border border-blue-400/30 border-dashed rounded-md",
              "px-3 py-1.5 text-sm text-blue-400",
              "hover:bg-blue-500/20 hover:border-blue-400/50 transition-all duration-200",
              "focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-2",
              "min-h-[36px]"
            )}
            aria-label={`Add ${availableGates.find(g => g.name === selectedGate)?.label || selectedGate} gate to qubit |${qubitIndex}⟩`}
          >
            + {availableGates.find(g => g.name === selectedGate)?.symbol || selectedGate}
          </button>
        )}
      </div>
      
      {/* Measurement indicator */}
      <div 
        className="text-xs text-slate-400 font-mono w-12 text-center border border-slate-600/30 rounded px-2 py-1" 
        aria-label="Measurement output"
        title="Quantum measurement result"
      >
        📊
      </div>
    </div>
  );
};
