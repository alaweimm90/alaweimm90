import { memo, useEffect, useState, createContext, useContext, ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Keyboard, Command, ArrowRight, CornerDownLeft, X } from 'lucide-react';

interface Shortcut {
  key: string;
  description: string;
  action: () => void;
  category: 'navigation' | 'ui' | 'actions' | 'dev';
  modifiers?: ('ctrl' | 'alt' | 'shift' | 'meta')[];
}

interface KeyboardContextType {
  addShortcut: (shortcut: Shortcut) => void;
  removeShortcut: (key: string) => void;
  toggleHelp: () => void;
}

const KeyboardContext = createContext<KeyboardContextType | null>(null);

export const useKeyboard = () => {
  const context = useContext(KeyboardContext);
  if (!context) {
    return { addShortcut: () => {}, removeShortcut: () => {}, toggleHelp: () => {} };
  }
  return context;
};

interface KeyboardProviderProps {
  children: ReactNode;
}

export const KeyboardProvider = ({ children }: KeyboardProviderProps) => {
  const [shortcuts, setShortcuts] = useState<Shortcut[]>([]);
  const [showHelp, setShowHelp] = useState(false);
  const [lastPressed, setLastPressed] = useState<string[]>([]);
  const navigate = useNavigate();

  const addShortcut = (shortcut: Shortcut) => {
    setShortcuts(prev => [...prev.filter(s => s.key !== shortcut.key), shortcut]);
  };

  const removeShortcut = (key: string) => {
    setShortcuts(prev => prev.filter(s => s.key !== key));
  };

  const toggleHelp = () => {
    setShowHelp(prev => !prev);
  };

  // Default shortcuts
  useEffect(() => {
    const defaultShortcuts: Shortcut[] = [
      {
        key: '?',
        description: 'Show keyboard shortcuts',
        action: toggleHelp,
        category: 'ui'
      },
      {
        key: 'Escape',
        description: 'Close dialogs/help',
        action: () => setShowHelp(false),
        category: 'ui'
      },
      {
        key: 'h',
        description: 'Go to home',
        action: () => navigate('/'),
        category: 'navigation'
      },
      {
        key: 'a',
        description: 'Go to arena',
        action: () => navigate('/arena'),
        category: 'navigation'
      },
      {
        key: 'b',
        description: 'Go to bench',
        action: () => navigate('/bench'),
        category: 'navigation'
      },
      {
        key: 'r',
        description: 'Refresh page',
        action: () => window.location.reload(),
        category: 'actions'
      }
    ];

    defaultShortcuts.forEach(addShortcut);
  }, [navigate]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) {
        return;
      }

      const pressedKeys = [
        e.ctrlKey && 'ctrl',
        e.altKey && 'alt',
        e.shiftKey && 'shift',
        e.metaKey && 'meta',
        e.key.toLowerCase()
      ].filter(Boolean) as string[];

      setLastPressed(pressedKeys);

      // Find matching shortcut
      const matchingShortcut = shortcuts.find(shortcut => {
        const shortcutKeys = [
          ...(shortcut.modifiers || []),
          shortcut.key.toLowerCase()
        ];
        
        return shortcutKeys.length === pressedKeys.length &&
               shortcutKeys.every(key => pressedKeys.includes(key));
      });

      if (matchingShortcut) {
        e.preventDefault();
        matchingShortcut.action();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);

  const groupedShortcuts = shortcuts.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, Shortcut[]>);

  const categoryIcons = {
    navigation: ArrowRight,
    ui: Command,
    actions: CornerDownLeft,
    dev: Keyboard
  };

  const categoryNames = {
    navigation: 'Navigation',
    ui: 'Interface', 
    actions: 'Actions',
    dev: 'Development'
  };

  return (
    <KeyboardContext.Provider value={{ addShortcut, removeShortcut, toggleHelp }}>
      {children}
      
      {/* Keyboard Shortcuts Help Modal */}
      {showHelp && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="glass-panel max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="p-6">
              {/* Header */}
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="glass-subtle p-2 rounded-xl">
                    <Keyboard className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <h2 className="heading-refined text-lg">Keyboard Shortcuts</h2>
                    <p className="text-xs text-muted-foreground">
                      Master the Strategic Command Center with hotkeys
                    </p>
                  </div>
                </div>
                <button
                  onClick={toggleHelp}
                  className="glass-minimal p-2 rounded-lg hover:bg-muted/20 transition-colors"
                >
                  <X className="h-4 w-4 text-muted-foreground" />
                </button>
              </div>

              {/* Shortcuts by Category */}
              <div className="space-y-6">
                {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => {
                  const IconComponent = categoryIcons[category as keyof typeof categoryIcons];
                  const categoryName = categoryNames[category as keyof typeof categoryNames];

                  return (
                    <div key={category}>
                      <div className="flex items-center gap-2 mb-3">
                        <IconComponent className="h-4 w-4 text-primary" />
                        <h3 className="heading-refined text-sm">{categoryName}</h3>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                        {categoryShortcuts.map(shortcut => (
                          <div 
                            key={shortcut.key}
                            className="glass-minimal p-3 rounded-lg flex items-center justify-between hover:bg-muted/10 transition-colors"
                          >
                            <span className="text-sm">{shortcut.description}</span>
                            <div className="flex gap-1">
                              {shortcut.modifiers?.map(modifier => (
                                <kbd 
                                  key={modifier}
                                  className="px-2 py-1 text-xs bg-muted/30 border border-muted/50 rounded font-mono"
                                >
                                  {modifier}
                                </kbd>
                              ))}
                              <kbd className="px-2 py-1 text-xs bg-primary/20 border border-primary/30 rounded font-mono text-primary">
                                {shortcut.key}
                              </kbd>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard Hint Toast */}
      <div className="fixed bottom-4 right-4 glass-minimal px-3 py-2 rounded-lg opacity-50 hover:opacity-100 transition-opacity z-40">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Keyboard className="h-3 w-3" />
          <span>Press</span>
          <kbd className="px-1 py-0.5 bg-muted/30 rounded text-xs">?</kbd>
          <span>for shortcuts</span>
        </div>
      </div>
    </KeyboardContext.Provider>
  );
};

export const KeyboardShortcuts = () => {
  // This component is now handled by KeyboardProvider
  return null;
};