import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { StatusChip } from '@/components/ui/status-chip';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { useAccessibilityContext } from '@/components/AccessibilityProvider';
import { Play, Square, RotateCcw, Zap, Cpu, Activity, Plus, HelpCircle, GraduationCap, Settings, Undo, Redo, Trash2, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { trackQuantumEvents } from '@/lib/analytics';

type DifficultyLevel = 'beginner' | 'intermediate' | 'advanced';

interface CircuitOperation {
  type: 'add' | 'remove';
  gate: { id: string; name: string; qubit: number };
  index?: number;
}

interface FeedbackMessage {
  type: 'success' | 'error' | 'info';
  message: string;
  timestamp: number;
}

export const CircuitBuilder = () => {
  const [selectedGate, setSelectedGate] = useState<string>('');
  const [selectedQubit, setSelectedQubit] = useState<number>(0);
  const [isRunning, setIsRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [circuitGates, setCircuitGates] = useState<Array<{ id: string; name: string; qubit: number }>>([]);
  const [difficultyLevel, setDifficultyLevel] = useState<DifficultyLevel>('beginner');
  const [history, setHistory] = useState<Array<{ id: string; name: string; qubit: number }[]>>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [feedback, setFeedback] = useState<FeedbackMessage | null>(null);
  const [draggedGate, setDraggedGate] = useState<string | null>(null);
  const { announce } = useAccessibilityContext();

  // Progressive gate sets based on difficulty
  const allGates = {
    beginner: [
      { name: "H", symbol: "H", label: "Hadamard", description: "Creates superposition - puts qubit in equal probability of |0⟩ and |1⟩" },
      { name: "X", symbol: "X", label: "Pauli-X", description: "Bit flip - converts |0⟩ to |1⟩ and vice versa" },
    ],
    intermediate: [
      { name: "H", symbol: "H", label: "Hadamard", description: "Creates superposition" },
      { name: "X", symbol: "X", label: "Pauli-X", description: "Bit flip gate" },
      { name: "Y", symbol: "Y", label: "Pauli-Y", description: "Bit and phase flip" },
      { name: "Z", symbol: "Z", label: "Pauli-Z", description: "Phase flip gate" },
    ],
    advanced: [
      { name: "H", symbol: "H", label: "Hadamard", description: "Creates superposition" },
      { name: "X", symbol: "X", label: "Pauli-X", description: "Bit flip gate" },
      { name: "Y", symbol: "Y", label: "Pauli-Y", description: "Bit and phase flip" },
      { name: "Z", symbol: "Z", label: "Pauli-Z", description: "Phase flip gate" },
      { name: "CNOT", symbol: "⊕", label: "CNOT", description: "Controlled-NOT - flips target if control is |1⟩" },
      { name: "T", symbol: "T", label: "T Gate", description: "π/4 phase rotation" },
      { name: "S", symbol: "S", label: "S Gate", description: "π/2 phase rotation" },
    ]
  };

  const gates = allGates[difficultyLevel];
  const maxQubits = difficultyLevel === 'beginner' ? 1 : 2;

  const selectGate = (gateName: string) => {
    const gate = gates.find(g => g.name === gateName);
    setSelectedGate(gateName);
    announce(`Selected ${gate?.label || gateName} gate`);
    trackQuantumEvents.circuitChange(gateName, circuitGates.length);
  };

  // Add feedback message
  const showFeedback = (type: FeedbackMessage['type'], message: string) => {
    setFeedback({ type, message, timestamp: Date.now() });
    setTimeout(() => setFeedback(null), 3000);
  };

  // Save state to history
  const saveToHistory = (newGates: Array<{ id: string; name: string; qubit: number }>) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push([...newGates]);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const addGateToCircuit = (gateName: string, qubit?: number) => {
    if (!gateName) {
      showFeedback('error', 'Please select a gate first');
      return;
    }
    
    const targetQubit = qubit !== undefined ? qubit : selectedQubit;
    const newGate = {
      id: `${gateName}-${Date.now()}`,
      name: gateName,
      qubit: targetQubit
    };
    
    const updatedGates = [...circuitGates, newGate];
    saveToHistory(circuitGates); // Save current state before changing
    setCircuitGates(updatedGates);
    
    const gateInfo = gates.find(g => g.name === gateName);
    showFeedback('success', `Added ${gateInfo?.label || gateName} to qubit |${targetQubit}⟩`);
    announce(`Added ${gateName} gate to qubit |${targetQubit}⟩`);
    trackQuantumEvents.circuitChange(gateName, updatedGates.length, targetQubit);
  };

  const removeGateFromCircuit = (gateId: string) => {
    const gateToRemove = circuitGates.find(g => g.id === gateId);
    if (!gateToRemove) return;
    
    saveToHistory(circuitGates);
    const updatedGates = circuitGates.filter(g => g.id !== gateId);
    setCircuitGates(updatedGates);
    
    const gateInfo = gates.find(g => g.name === gateToRemove.name);
    showFeedback('info', `Removed ${gateInfo?.label || gateToRemove.name} from circuit`);
    announce(`Removed ${gateToRemove.name} gate from circuit`);
  };

  const undoLastAction = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setCircuitGates([...history[historyIndex - 1]]);
      showFeedback('info', 'Undid last action');
      announce('Undid last circuit change');
    } else {
      showFeedback('error', 'Nothing to undo');
    }
  };

  const redoLastAction = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setCircuitGates([...history[historyIndex + 1]]);
      showFeedback('info', 'Redid last action');
      announce('Redid last circuit change');
    } else {
      showFeedback('error', 'Nothing to redo');
    }
  };

  const clearCircuit = () => {
    if (circuitGates.length > 0) {
      saveToHistory(circuitGates);
      setCircuitGates([]);
      showFeedback('info', 'Circuit cleared');
      announce('Circuit cleared');
      trackQuantumEvents.circuitReset(0);
    } else {
      showFeedback('error', 'Circuit is already empty');
    }
  };

  const runCircuit = async () => {
    if (circuitGates.length === 0) {
      showFeedback('error', 'Add some gates to the circuit first!');
      return;
    }
    
    setIsRunning(true);
    setProgress(0);
    showFeedback('info', 'Starting quantum circuit simulation...');
    trackQuantumEvents.circuitRun(circuitGates.length, 1000);
    
    // Simulate circuit execution with progress
    for (let i = 0; i <= 100; i += 10) {
      setProgress(i);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    setTimeout(() => {
      setIsRunning(false);
      setProgress(0);
      showFeedback('success', `Circuit executed successfully! Used ${circuitGates.length} gates.`);
      announce('Circuit execution completed successfully');
    }, 500);
  };

  // Keyboard shortcuts
  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (isRunning) return;
    
    // Prevent default only for our shortcuts
    switch (event.key.toLowerCase()) {
      case 'z':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          if (event.shiftKey) {
            redoLastAction();
          } else {
            undoLastAction();
          }
        }
        break;
      case 'r':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          runCircuit();
        }
        break;
      case 'c':
        if (event.ctrlKey || event.metaKey) {
          event.preventDefault();
          clearCircuit();
        }
        break;
      case '1':
      case '2':
        if (!event.ctrlKey && !event.metaKey) {
          const qubitIndex = parseInt(event.key) - 1;
          if (qubitIndex <= maxQubits) {
            setSelectedQubit(qubitIndex);
            announce(`Selected qubit |${qubitIndex}⟩`);
          }
        }
        break;
    }
  };

  // Drag and drop handlers
  const handleDragStart = (gateName: string) => {
    setDraggedGate(gateName);
    setSelectedGate(gateName);
  };

  const handleDragEnd = () => {
    setDraggedGate(null);
  };

  const handleDrop = (qubit: number) => {
    if (draggedGate) {
      addGateToCircuit(draggedGate, qubit);
      setDraggedGate(null);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  return (
    <div
      onKeyDown={handleKeyDown}
      tabIndex={0}
      className="focus:outline-none"
      role="application"
      aria-label="Quantum Circuit Builder"
    >
      <Card className="relative rounded-2xl border border-blue-400/30 bg-slate-900/80 shadow-xl hover:border-blue-400/50 transition-all duration-300">
        {/* Primary accent bar */}
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-400 rounded-r-full"></div>
      
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500/20 border border-blue-400/30 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <CardTitle className="text-lg font-semibold text-slate-200">Circuit Builder</CardTitle>
                <Badge className={`text-xs ${
                  difficultyLevel === 'beginner' ? 'bg-green-500/20 text-green-300 border-green-400/30' :
                  difficultyLevel === 'intermediate' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30' :
                  'bg-red-500/20 text-red-300 border-red-400/30'
                }`}>
                  <GraduationCap className="w-3 h-3 mr-1" />
                  {difficultyLevel.charAt(0).toUpperCase() + difficultyLevel.slice(1)}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">Build quantum circuits with {gates.length} available gates</p>
            </div>
          </div>
          <div className="flex gap-2">
            <StatusChip 
              variant={isRunning ? "running" : "idle"} 
              icon={isRunning ? <Activity className="w-3 h-3" /> : <Cpu className="w-3 h-3" />}
            >
              {isRunning ? "Running" : "Ready"}
            </StatusChip>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Difficulty Level Selector */}
        <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-600/30">
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-slate-300 flex items-center gap-2">
              <Settings className="w-4 h-4" />
              Learning Mode
            </h4>
          </div>
          <div className="flex gap-2">
            {(['beginner', 'intermediate', 'advanced'] as DifficultyLevel[]).map((level) => (
              <Button
                key={level}
                onClick={() => {
                  setDifficultyLevel(level);
                  setSelectedGate('');
                  setCircuitGates([]);
                  announce(`Switched to ${level} mode`);
                }}
                variant={difficultyLevel === level ? "primary" : "outline"}
                size="sm"
                className={`flex-1 text-xs ${
                  level === 'beginner' ? 'hover:bg-green-500/20 hover:border-green-400/40' :
                  level === 'intermediate' ? 'hover:bg-yellow-500/20 hover:border-yellow-400/40' :
                  'hover:bg-red-500/20 hover:border-red-400/40'
                }`}
                disabled={isRunning}
              >
                {level === 'beginner' && '🌱'}
                {level === 'intermediate' && '🎯'}
                {level === 'advanced' && '🚀'}
                <span className="ml-1 capitalize">{level}</span>
              </Button>
            ))}
          </div>
          <div className="mt-3 p-3 rounded-lg bg-slate-900/50 border border-slate-700/30">
            <p className="text-xs text-muted-foreground">
              {difficultyLevel === 'beginner' && '🌱 Learn basic gates (H, X) on single qubits with detailed explanations'}
              {difficultyLevel === 'intermediate' && '🎯 Explore more gates (H, X, Y, Z) and two-qubit systems'}
              {difficultyLevel === 'advanced' && '🚀 Master all gates including CNOT, T, S for complex quantum algorithms'}
            </p>
          </div>
        </div>

        {/* Feedback Messages */}
        {feedback && (
          <div className={`p-3 rounded-lg border flex items-center gap-3 transition-all duration-200 ${
            feedback.type === 'success' ? 'bg-green-500/10 border-green-400/30' :
            feedback.type === 'error' ? 'bg-red-500/10 border-red-400/30' :
            'bg-blue-500/10 border-blue-400/30'
          }`}>
            {feedback.type === 'success' && <CheckCircle className="w-4 h-4 text-green-400" />}
            {feedback.type === 'error' && <AlertCircle className="w-4 h-4 text-red-400" />}
            {feedback.type === 'info' && <Info className="w-4 h-4 text-blue-400" />}
            <span className={`text-sm ${
              feedback.type === 'success' ? 'text-green-300' :
              feedback.type === 'error' ? 'text-red-300' :
              'text-blue-300'
            }`}>
              {feedback.message}
            </span>
          </div>
        )}

        {/* Control buttons */}
        <div className="space-y-3">
          <div className="flex gap-2">
            <Button
              onClick={runCircuit}
              disabled={isRunning || circuitGates.length === 0}
              variant={isRunning ? "secondary" : "primary"}
              size="sm"
              className="flex-1"
              aria-label="Run quantum circuit simulation"
            >
              {isRunning ? (
                <>
                  <Activity className="w-4 h-4 mr-2 animate-pulse" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Run Circuit
                </>
              )}
            </Button>
            
            <Button
              onClick={clearCircuit}
              variant="outline"
              size="sm"
              disabled={isRunning || circuitGates.length === 0}
              aria-label="Clear all gates from circuit"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Clear
            </Button>
          </div>
          
          {/* Undo/Redo buttons */}
          <div className="flex gap-2">
            <Button
              onClick={undoLastAction}
              variant="outline"
              size="sm"
              disabled={isRunning || historyIndex <= 0}
              className="flex-1"
              aria-label="Undo last action"
            >
              <Undo className="w-4 h-4 mr-2" />
              Undo
            </Button>
            
            <Button
              onClick={redoLastAction}
              variant="outline"
              size="sm"
              disabled={isRunning || historyIndex >= history.length - 1}
              className="flex-1"
              aria-label="Redo last action"
            >
              <Redo className="w-4 h-4 mr-2" />
              Redo
            </Button>
          </div>
        </div>

        {/* Progress indicator */}
        {isRunning && (
          <div className="space-y-2">
            <div className="flex justify-between text-small text-muted">
              <span>Executing circuit...</span>
              <span className="font-mono">{progress}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>
        )}

        {/* Qubit Selection */}
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3">Target Qubit</h4>
          <div className="flex gap-2 mb-4">
            {Array.from({ length: maxQubits + 1 }, (_, i) => i).map((qubit) => (
              <Button
                key={qubit}
                onClick={() => {
                  setSelectedQubit(qubit);
                  announce(`Selected qubit |${qubit}⟩`);
                }}
                variant={selectedQubit === qubit ? "primary" : "outline"}
                size="sm"
                className="flex-1 h-10"
                disabled={isRunning}
                aria-label={`Select qubit |${qubit}⟩`}
                aria-pressed={selectedQubit === qubit}
              >
                |{qubit}⟩
              </Button>
            ))}
          </div>
        </div>

        {/* Gate palette with educational descriptions */}
        <div>
          <h4 className="text-sm font-medium text-slate-300 mb-3">Available Gates</h4>
          <div className={`grid gap-3 ${
            difficultyLevel === 'beginner' ? 'grid-cols-1' :
            difficultyLevel === 'intermediate' ? 'grid-cols-2' :
            'grid-cols-3'
          }`}>
            {gates.map((gate) => (
              <div
                key={gate.name}
                className={`relative group p-3 rounded-lg border transition-all duration-200 ${
                  selectedGate === gate.name 
                    ? 'border-blue-400/50 bg-blue-500/10' 
                    : 'border-slate-600/50 hover:border-slate-500/60 bg-slate-800/30'
                }`}
              >
                <Button
                  onClick={() => selectGate(gate.name)}
                  onDragStart={(e) => {
                    e.dataTransfer.effectAllowed = 'copy';
                    handleDragStart(gate.name);
                  }}
                  onDragEnd={handleDragEnd}
                  draggable={!isRunning}
                  variant={selectedGate === gate.name ? "primary" : "ghost"}
                  size="sm"
                  className={`w-full h-auto p-2 flex-col gap-2 ${
                    draggedGate === gate.name ? 'opacity-50' : 'hover:scale-105'
                  } transition-all`}
                  disabled={isRunning}
                  aria-label={`Select or drag ${gate.label} gate`}
                  aria-pressed={selectedGate === gate.name}
                >
                  <div className="flex items-center gap-2 w-full">
                    <span className="font-mono font-bold text-lg">{gate.symbol}</span>
                    <div className="flex-1 text-left">
                      <div className="font-semibold text-sm">{gate.name}</div>
                      <div className="text-xs text-muted-foreground">{gate.label}</div>
                    </div>
                  </div>
                </Button>
                {difficultyLevel === 'beginner' && (
                  <div className="mt-2 text-xs text-muted-foreground leading-relaxed">
                    {gate.description}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Circuit visualization */}
        <div>
          <h4 className="text-small font-medium text-text mb-3">Circuit</h4>
          <div className="min-h-32 border border-border rounded-lg bg-surface-2/30 p-4">
            {circuitGates.length === 0 ? (
              <div className="flex items-center justify-center h-24 text-center">
                <div className="space-y-2">
                  <HelpCircle className="w-8 h-8 mx-auto text-muted" />
                  <p className="text-muted text-small">
                    Select a gate and target qubit, then click to add it to your circuit
                  </p>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => selectedGate && addGateToCircuit(selectedGate)}
                    disabled={!selectedGate}
                    className="mt-2"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add {selectedGate || 'Gate'} to |{selectedQubit}⟩
                  </Button>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Qubit lines */}
                {Array.from({ length: maxQubits + 1 }, (_, i) => i).map((qubit) => (
                  <div 
                    key={qubit} 
                    className={`flex items-center gap-2 p-2 rounded-lg border transition-all duration-200 ${
                      selectedQubit === qubit 
                        ? 'border-blue-400/50 bg-blue-500/10' 
                        : 'border-transparent hover:border-slate-500/60 hover:bg-slate-800/50'
                    }`}
                    onClick={() => setSelectedQubit(qubit)}
                    onDrop={(e) => {
                      e.preventDefault();
                      handleDrop(qubit);
                    }}
                    onDragOver={handleDragOver}
                    style={{ cursor: 'pointer' }}
                  >
                    <span className={`text-sm font-mono w-8 ${
                      selectedQubit === qubit ? 'text-blue-400 font-semibold' : 'text-muted-foreground'
                    }`}>|{qubit}⟩</span>
                    <div className="flex-1 h-0.5 bg-slate-600/50 relative">
                      {circuitGates
                        .filter(gate => gate.qubit === qubit)
                        .map((gate, index) => (
                          <div
                            key={gate.id}
                            className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 bg-blue-500 text-white rounded border border-blue-400 flex items-center justify-center text-xs font-mono font-semibold hover:scale-110 transition-all cursor-pointer group shadow-lg"
                            style={{ left: `${(index + 1) * 60}px` }}
                            title={`${gates.find(g => g.name === gate.name)?.label} on qubit |${qubit}⟩ - Click to remove`}
                            onClick={(e) => {
                              e.stopPropagation();
                              removeGateFromCircuit(gate.id);
                            }}
                          >
                            {gates.find(g => g.name === gate.name)?.symbol}
                            {/* Remove indicator on hover */}
                            <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 text-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                              <span className="text-xs">×</span>
                            </div>
                          </div>
                        ))}
                    </div>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="w-8 h-8 p-0 opacity-60 hover:opacity-100"
                      onClick={(e) => {
                        e.stopPropagation();
                        selectedGate && addGateToCircuit(selectedGate, qubit);
                      }}
                      disabled={!selectedGate}
                      title={`Add ${selectedGate || 'gate'} to |${qubit}⟩`}
                    >
                      <Plus className="w-4 h-4" />
                    </Button>
                  </div>
                ))}
                
                {/* Add gate button */}
                <div className="flex justify-center pt-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => selectedGate && addGateToCircuit(selectedGate)}
                    disabled={!selectedGate}
                    className="hover:bg-blue-500/10 hover:border-blue-400/50"
                  >
                    <Plus className="w-4 h-4 mr-2" />
                    Add {selectedGate ? gates.find(g => g.name === selectedGate)?.label : 'Gate'} to |{selectedQubit}⟩
                  </Button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Circuit info and shortcuts */}
        <div className="space-y-4 pt-4 border-t border-slate-600/30">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-lg font-mono font-semibold text-slate-200">{circuitGates.length}</div>
              <div className="text-sm text-muted-foreground">Gates</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-mono font-semibold text-slate-200">{maxQubits + 1}</div>
              <div className="text-sm text-muted-foreground">Qubits</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-mono font-semibold text-slate-200">{Math.max(1, circuitGates.length)}</div>
              <div className="text-sm text-muted-foreground">Depth</div>
            </div>
          </div>
          
          {/* Keyboard shortcuts help */}
          <div className="p-3 rounded-lg bg-slate-800/50 border border-slate-700/30">
            <h5 className="text-xs font-medium text-slate-300 mb-2">💡 Keyboard Shortcuts:</h5>
            <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
              <div><kbd className="px-1 py-0.5 bg-slate-600 rounded text-xs">Ctrl+Z</kbd> Undo</div>
              <div><kbd className="px-1 py-0.5 bg-slate-600 rounded text-xs">Ctrl+Shift+Z</kbd> Redo</div>
              <div><kbd className="px-1 py-0.5 bg-slate-600 rounded text-xs">Ctrl+R</kbd> Run</div>
              <div><kbd className="px-1 py-0.5 bg-slate-600 rounded text-xs">Ctrl+C</kbd> Clear</div>
              <div><kbd className="px-1 py-0.5 bg-slate-600 rounded text-xs">1/2</kbd> Select Qubit</div>
              <div><span className="text-xs">🖱️ Drag gates to qubits</span></div>
            </div>
          </div>
        </div>
      </CardContent>
      </Card>
    </div>
  );
};