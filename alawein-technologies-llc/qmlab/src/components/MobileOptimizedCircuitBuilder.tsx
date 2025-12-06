import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Zap, 
  RotateCw, 
  Split, 
  Trash2, 
  Play, 
  RotateCcw, 
  ChevronLeft, 
  ChevronRight,
  Info,
  TouchpadOff
} from 'lucide-react';

// Quantum gate definitions optimized for mobile
const QUANTUM_GATES = [
  { id: 'H', symbol: 'H', name: 'Hadamard', color: 'bg-blue-500', description: 'Creates superposition' },
  { id: 'X', symbol: 'X', name: 'Pauli-X', color: 'bg-red-500', description: 'Bit flip gate' },
  { id: 'Y', symbol: 'Y', name: 'Pauli-Y', color: 'bg-green-500', description: 'Bit and phase flip' },
  { id: 'Z', symbol: 'Z', name: 'Pauli-Z', color: 'bg-purple-500', description: 'Phase flip gate' },
  { id: 'RX', symbol: 'RX', name: 'Rotation-X', color: 'bg-orange-500', description: 'X-axis rotation' },
  { id: 'RY', symbol: 'RY', name: 'Rotation-Y', color: 'bg-teal-500', description: 'Y-axis rotation' },
  { id: 'RZ', symbol: 'RZ', name: 'Rotation-Z', color: 'bg-indigo-500', description: 'Z-axis rotation' },
  { id: 'CNOT', symbol: '⊕', name: 'CNOT', color: 'bg-pink-500', description: 'Controlled NOT' }
];

// Touch-optimized gate button
const TouchGateButton: React.FC<{
  gate: typeof QUANTUM_GATES[0];
  isSelected: boolean;
  onSelect: (gate: typeof QUANTUM_GATES[0]) => void;
}> = ({ gate, isSelected, onSelect }) => (
  <Button
    className={`
      min-h-[56px] min-w-[56px] touch-manipulation relative
      ${gate.color} hover:opacity-90 active:scale-95
      ${isSelected ? 'ring-2 ring-white ring-offset-2' : ''}
      transition-all duration-150 ease-in-out
    `}
    size="lg"
    onClick={() => onSelect(gate)}
    aria-label={`Select ${gate.name} gate - ${gate.description}`}
    aria-pressed={isSelected}
  >
    <span className="text-white font-bold text-lg">{gate.symbol}</span>
    {isSelected && (
      <div className="absolute -top-1 -right-1 w-3 h-3 bg-white rounded-full border-2 border-blue-500" />
    )}
  </Button>
);

// Mobile-optimized circuit wire
const CircuitWire: React.FC<{
  qubitIndex: number;
  gates: Array<{ id: string; type: string; qubit: number; angle?: number }>;
  onGateAdd: (qubitIndex: number, position: number) => void;
  onGateRemove: (gateIndex: number) => void;
  selectedGate: typeof QUANTUM_GATES[0] | null;
}> = ({ qubitIndex, gates, onGateAdd, onGateRemove, selectedGate }) => {
  const wireGates = gates.filter(gate => gate.qubit === qubitIndex);
  const maxGates = 8; // Maximum gates per wire for mobile

  return (
    <div className="relative">
      {/* Qubit label */}
      <div className="flex items-center mb-3">
        <Badge variant="outline" className="text-sm font-mono min-w-[60px] justify-center">
          |q{qubitIndex}⟩
        </Badge>
      </div>

      {/* Wire with gates */}
      <div className="relative">
        {/* Wire line */}
        <div className="absolute top-1/2 left-0 right-0 h-0.5 bg-gray-300 -translate-y-0.5 z-0" />
        
        {/* Gate positions */}
        <div className="flex items-center gap-2 relative z-10">
          {Array.from({ length: maxGates }, (_, position) => {
            const gateAtPosition = wireGates.find((_, index) => index === position);
            
            return (
              <div key={position} className="flex-shrink-0">
                {gateAtPosition ? (
                  <Button
                    className="min-h-[48px] min-w-[48px] touch-manipulation bg-white border-2 border-gray-300 hover:border-red-400 active:scale-95"
                    variant="outline"
                    size="sm"
                    onClick={() => onGateRemove(gates.indexOf(gateAtPosition))}
                    aria-label={`Remove ${gateAtPosition.type} gate from qubit ${qubitIndex}`}
                  >
                    <span className="text-gray-800 font-semibold">
                      {QUANTUM_GATES.find(g => g.id === gateAtPosition.type)?.symbol || gateAtPosition.type}
                    </span>
                  </Button>
                ) : (
                  <Button
                    className="min-h-[48px] min-w-[48px] touch-manipulation border-2 border-dashed border-gray-300 hover:border-blue-400 hover:bg-blue-50 active:scale-95"
                    variant="ghost"
                    size="sm"
                    onClick={() => selectedGate && onGateAdd(qubitIndex, position)}
                    disabled={!selectedGate}
                    aria-label={`Add ${selectedGate?.name || 'gate'} to qubit ${qubitIndex} at position ${position + 1}`}
                  >
                    <span className="text-gray-400 text-lg">+</span>
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// Mobile circuit builder component
export const MobileOptimizedCircuitBuilder: React.FC = () => {
  const [selectedGate, setSelectedGate] = useState<typeof QUANTUM_GATES[0] | null>(null);
  const [qubits] = useState(3); // Fixed for mobile simplicity
  const [gates, setGates] = useState<Array<{ id: string; type: string; qubit: number; angle?: number }>>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [showGatePalette, setShowGatePalette] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Handle gate addition
  const handleGateAdd = useCallback((qubitIndex: number, position: number) => {
    if (!selectedGate) return;

    const newGate = {
      id: `${selectedGate.id}-${Date.now()}`,
      type: selectedGate.id,
      qubit: qubitIndex,
      angle: selectedGate.id.startsWith('R') ? Math.PI / 4 : undefined
    };

    setGates(prev => {
      const newGates = [...prev];
      newGates.splice(position, 0, newGate);
      return newGates;
    });

    // Haptic feedback on supported devices
    if ('vibrate' in navigator) {
      navigator.vibrate(50);
    }
  }, [selectedGate]);

  // Handle gate removal
  const handleGateRemove = useCallback((gateIndex: number) => {
    setGates(prev => prev.filter((_, index) => index !== gateIndex));
    
    // Haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate(100);
    }
  }, []);

  // Run circuit simulation
  const handleRunCircuit = useCallback(async () => {
    setIsRunning(true);
    
    // Simulate circuit execution
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    setIsRunning(false);
    
    // Success haptic feedback
    if ('vibrate' in navigator) {
      navigator.vibrate([100, 50, 100]);
    }
  }, [gates]);

  // Reset circuit
  const handleReset = useCallback(() => {
    setGates([]);
    setSelectedGate(null);
  }, []);

  // Auto-scroll to show new gates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        left: scrollRef.current.scrollWidth,
        behavior: 'smooth'
      });
    }
  }, [gates]);

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg sm:text-xl flex items-center gap-2">
            <Zap className="w-5 h-5 text-blue-500" />
            Quantum Circuit Builder
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowGatePalette(!showGatePalette)}
              className="sm:hidden touch-manipulation"
              aria-label={showGatePalette ? "Hide gate palette" : "Show gate palette"}
            >
              <TouchpadOff className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {/* Mobile instruction */}
        <div className="text-sm text-gray-600 bg-blue-50 p-3 rounded-lg border">
          <div className="flex items-start gap-2">
            <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
            <div>
              <p className="font-medium text-blue-800">How to use:</p>
              <p>1. Select a gate from the palette below</p>
              <p>2. Tap the + buttons to add gates to qubits</p>
              <p>3. Tap existing gates to remove them</p>
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Gate Palette */}
        {showGatePalette && (
          <div className="space-y-3">
            <h3 className="text-sm font-semibold text-gray-700 flex items-center gap-2">
              Gate Palette
              {selectedGate && (
                <Badge variant="secondary" className="text-xs">
                  {selectedGate.name} selected
                </Badge>
              )}
            </h3>
            
            <ScrollArea className="w-full">
              <div className="flex gap-3 pb-2">
                {QUANTUM_GATES.map(gate => (
                  <TouchGateButton
                    key={gate.id}
                    gate={gate}
                    isSelected={selectedGate?.id === gate.id}
                    onSelect={setSelectedGate}
                  />
                ))}
              </div>
            </ScrollArea>
          </div>
        )}

        {/* Circuit Display */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Quantum Circuit</h3>
            <div className="text-xs text-gray-500">
              {gates.length} gate{gates.length !== 1 ? 's' : ''} • {qubits} qubit{qubits !== 1 ? 's' : ''}
            </div>
          </div>

          <div className="bg-gray-50 p-4 rounded-lg border-2 border-dashed border-gray-200">
            <ScrollArea className="w-full" ref={scrollRef}>
              <div className="space-y-6 min-w-max pr-4">
                {Array.from({ length: qubits }, (_, qubitIndex) => (
                  <CircuitWire
                    key={qubitIndex}
                    qubitIndex={qubitIndex}
                    gates={gates}
                    onGateAdd={handleGateAdd}
                    onGateRemove={handleGateRemove}
                    selectedGate={selectedGate}
                  />
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-3 pt-4">
          <Button
            className="flex-1 touch-manipulation min-h-[48px]"
            onClick={handleRunCircuit}
            disabled={gates.length === 0 || isRunning}
            aria-label={isRunning ? "Running circuit simulation" : "Run quantum circuit"}
          >
            <Play className="w-4 h-4 mr-2" />
            {isRunning ? 'Running...' : 'Run Circuit'}
          </Button>
          
          <Button
            variant="outline"
            onClick={handleReset}
            disabled={gates.length === 0}
            className="touch-manipulation min-h-[48px]"
            aria-label="Reset circuit (remove all gates)"
          >
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>

        {/* Results Display */}
        {isRunning && (
          <div className="bg-blue-50 p-4 rounded-lg border">
            <div className="flex items-center gap-2 mb-2">
              <RotateCw className="w-4 h-4 animate-spin text-blue-500" />
              <span className="text-sm font-medium text-blue-800">Simulating quantum circuit...</span>
            </div>
            <div className="w-full bg-blue-200 rounded-full h-2">
              <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }} />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default MobileOptimizedCircuitBuilder;