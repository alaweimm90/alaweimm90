import React, { useState, useEffect, useRef } from 'react';
import { Search, Zap, Download, Play, RotateCcw, Settings, Atom, Calculator } from 'lucide-react';
import { logger } from '@/lib/logger';

interface Command {
  id: string;
  name: string;
  description: string;
  shortcut?: string;
  icon: React.ReactNode;
  category: 'gates' | 'algorithms' | 'actions' | 'export';
  action: () => void;
}

interface QuantumCommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export const QuantumCommandPalette: React.FC<QuantumCommandPaletteProps> = ({ isOpen, onClose }) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const commands: Command[] = [
    // Quantum Gates
    {
      id: 'add-hadamard',
      name: 'Add Hadamard Gate',
      description: 'Create superposition with H gate',
      shortcut: 'H',
      icon: <Atom className="w-4 h-4" />,
      category: 'gates',
      action: () => logger.debug('Add Hadamard gate action triggered')
    },
    {
      id: 'add-cnot',
      name: 'Add CNOT Gate',
      description: 'Create entanglement with controlled-X',
      shortcut: 'C',
      icon: <Zap className="w-4 h-4" />,
      category: 'gates',
      action: () => logger.debug('Add CNOT gate action triggered')
    },
    {
      id: 'add-rotation',
      name: 'Add Rotation Gate',
      description: 'Parametric rotation (RX, RY, RZ)',
      shortcut: 'R',
      icon: <RotateCcw className="w-4 h-4" />,
      category: 'gates',
      action: () => logger.debug('Add Rotation gate action triggered')
    },

    // ML Algorithms
    {
      id: 'start-vqc',
      name: 'Start VQC Training',
      description: 'Train Variational Quantum Classifier',
      shortcut: '⌘T',
      icon: <Play className="w-4 h-4" />,
      category: 'algorithms',
      action: () => logger.debug('Start VQC training action triggered')
    },
    {
      id: 'run-qaoa',
      name: 'Run QAOA',
      description: 'Quantum Approximate Optimization Algorithm',
      icon: <Calculator className="w-4 h-4" />,
      category: 'algorithms',
      action: () => logger.debug('Run QAOA action triggered')
    },

    // Actions
    {
      id: 'reset-circuit',
      name: 'Reset Circuit',
      description: 'Clear all gates and start fresh',
      shortcut: '⌘R',
      icon: <RotateCcw className="w-4 h-4" />,
      category: 'actions',
      action: () => logger.debug('Reset circuit action triggered')
    },
    {
      id: 'settings',
      name: 'Open Settings',
      description: 'Configure quantum parameters',
      shortcut: '⌘,',
      icon: <Settings className="w-4 h-4" />,
      category: 'actions',
      action: () => logger.debug('Open settings action triggered')
    },

    // Export
    {
      id: 'export-qiskit',
      name: 'Export to Qiskit',
      description: 'Generate Python code for Qiskit',
      shortcut: '⌘E',
      icon: <Download className="w-4 h-4" />,
      category: 'export',
      action: () => logger.debug('Export to Qiskit action triggered')
    },
    {
      id: 'export-pennylane',
      name: 'Export to PennyLane',
      description: 'Generate code for PennyLane',
      icon: <Download className="w-4 h-4" />,
      category: 'export',
      action: () => logger.debug('Export to PennyLane action triggered')
    }
  ];

  // Filter commands based on query
  const filteredCommands = commands.filter(command =>
    command.name.toLowerCase().includes(query.toLowerCase()) ||
    command.description.toLowerCase().includes(query.toLowerCase())
  );

  // Group commands by category
  const groupedCommands = filteredCommands.reduce((groups, command) => {
    if (!groups[command.category]) {
      groups[command.category] = [];
    }
    groups[command.category].push(command);
    return groups;
  }, {} as Record<string, Command[]>);

  const categoryNames = {
    gates: 'Quantum Gates',
    algorithms: 'ML Algorithms', 
    actions: 'Actions',
    export: 'Export'
  };

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            filteredCommands[selectedIndex].action();
            onClose();
          }
          break;
        case 'Escape':
          onClose();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, filteredCommands, onClose]);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
      setQuery('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  let commandIndex = 0;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="command-palette-backdrop" 
        onClick={onClose}
      />
      
      {/* Command Palette */}
      <div className="command-palette-content max-w-2xl">
        {/* Header */}
        <div className="p-4 border-b border-white/10">
          <div className="flex items-center gap-3">
            <Search className="w-5 h-5 text-muted" aria-hidden="true" />
            <input
              ref={inputRef}
              type="text"
              placeholder="Search quantum operations..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 bg-transparent text-primary placeholder-muted border-none outline-none text-lg focus:outline-none"
              aria-label="Search quantum operations"
            />
            <kbd className="px-2 py-1 text-xs text-muted bg-surface-2 rounded text-mono">ESC</kbd>
          </div>
        </div>

        {/* Commands */}
        <div className="max-h-96 overflow-y-auto p-2">
          {Object.entries(groupedCommands).map(([category, categoryCommands]) => (
            <div key={category} className="mb-4">
              <div className="px-3 py-2 text-caption">
                {categoryNames[category as keyof typeof categoryNames]}
              </div>
              
              {categoryCommands.map((command) => {
                const isSelected = commandIndex === selectedIndex;
                commandIndex++;
                
                return (
                  <button
                    key={command.id}
                    onClick={() => {
                      command.action();
                      onClose();
                    }}
                    className={`w-full flex items-center justify-between p-3 rounded-lg transition-all glass-interactive ${
                      isSelected 
                        ? 'glass-subtle border border-primary' 
                        : 'hover:glass-subtle'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`${isSelected ? 'text-primary' : 'text-muted'}`} aria-hidden="true">
                        {command.icon}
                      </div>
                      <div className="text-left">
                        <div className="text-primary font-medium">{command.name}</div>
                        <div className="body-elegant-sm">{command.description}</div>
                      </div>
                    </div>
                    
                    {command.shortcut && (
                      <kbd className="px-2 py-1 text-xs text-muted bg-surface-2 rounded text-mono">
                        {command.shortcut}
                      </kbd>
                    )}
                  </button>
                );
              })}
            </div>
          ))}

          {filteredCommands.length === 0 && (
            <div className="p-8 text-center">
              <Search className="w-8 h-8 text-muted mx-auto mb-3" aria-hidden="true" />
              <div className="text-secondary font-medium">No commands found</div>
              <div className="body-elegant-sm mt-1">
                Try searching for gates, algorithms, or actions
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-white/10 body-elegant-sm text-muted flex justify-between">
          <div>Use ↑↓ to navigate, ⏎ to select</div>
          <div className="text-mono">⌘K to open anytime</div>
        </div>
      </div>
    </>
  );
};

// Hook to manage command palette state
export const useQuantumCommandPalette = () => {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return {
    isOpen,
    openPalette: () => setIsOpen(true),
    closePalette: () => setIsOpen(false)
  };
};