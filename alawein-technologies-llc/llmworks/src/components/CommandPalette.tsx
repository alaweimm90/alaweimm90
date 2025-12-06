import React, { memo, useState, useEffect, useRef, useMemo } from 'react';
import { Dialog, DialogContent } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { 
  Search, 
  Zap, 
  BarChart3, 
  Settings, 
  Github, 
  Keyboard, 
  Home,
  Command,
  ArrowRight,
  Clock,
  Star
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { trackEvent } from '@/lib/analytics';

interface CommandItem {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  action: () => void;
  category: string;
  keywords: string[];
  badge?: string;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const CommandPaletteComponent: React.FC<CommandPaletteProps> = ({ 
  open, 
  onOpenChange 
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const commands: CommandItem[] = useMemo(() => [
    {
      id: 'home',
      title: 'Home',
      description: 'Return to the main platform',
      icon: Home,
      action: () => navigate('/'),
      category: 'Navigation',
      keywords: ['home', 'main', 'platform', 'overview'],
    },
    {
      id: 'arena',
      title: 'Enter Arena',
      description: 'Start AI battles and competitions',
      icon: Zap,
      action: () => navigate('/arena'),
      category: 'Combat',
      keywords: ['arena', 'battle', 'fight', 'combat', 'competition'],
      badge: 'Hot'
    },
    {
      id: 'bench',
      title: 'View Bench',
      description: 'Check leaderboards and rankings',
      icon: BarChart3,
      action: () => navigate('/bench'),
      category: 'Analytics',
      keywords: ['bench', 'leaderboard', 'rankings', 'stats', 'performance'],
    },
    {
      id: 'dashboard',
      title: 'Command Center',
      description: 'Access management dashboard',
      icon: Settings,
      action: () => navigate('/dashboard'),
      category: 'Management',
      keywords: ['dashboard', 'settings', 'config', 'management', 'control'],
    },
    {
      id: 'github',
      title: 'View Source',
      description: 'Open GitHub repository',
      icon: Github,
      action: () => window.open('https://github.com/alaweimm90/aegis-ai-evaluator', '_blank'),
      category: 'External',
      keywords: ['github', 'source', 'code', 'repository', 'open source'],
    },
    {
      id: 'shortcuts',
      title: 'Keyboard Shortcuts',
      description: 'View all available shortcuts',
      icon: Keyboard,
      action: () => {
        // This would trigger the keyboard shortcuts modal
        onOpenChange(false);
      },
      category: 'Help',
      keywords: ['shortcuts', 'keyboard', 'help', 'hotkeys', 'commands'],
    },
  ], [navigate, onOpenChange]);

  const filteredCommands = useMemo(() => {
    if (!query.trim()) return commands;
    
    const normalizedQuery = query.toLowerCase().trim();
    return commands.filter(command => 
      command.title.toLowerCase().includes(normalizedQuery) ||
      command.description.toLowerCase().includes(normalizedQuery) ||
      command.keywords.some(keyword => keyword.toLowerCase().includes(normalizedQuery))
    );
  }, [query, commands]);

  const categories = useMemo(() => {
    const categoryMap = new Map<string, CommandItem[]>();
    filteredCommands.forEach(command => {
      const category = command.category;
      if (!categoryMap.has(category)) {
        categoryMap.set(category, []);
      }
      categoryMap.get(category)!.push(command);
    });
    return Array.from(categoryMap.entries());
  }, [filteredCommands]);

  useEffect(() => {
    if (open && inputRef.current) {
      inputRef.current.focus();
    }
  }, [open]);

  useEffect(() => {
    setSelectedIndex(0);
  }, [query]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      const selectedCommand = filteredCommands[selectedIndex];
      if (selectedCommand) {
        trackEvent('command_palette_action', { 
          command: selectedCommand.id,
          query: query.trim() 
        });
        selectedCommand.action();
        onOpenChange(false);
        setQuery('');
      }
    } else if (e.key === 'Escape') {
      onOpenChange(false);
      setQuery('');
    }
  };

  const handleCommandClick = (command: CommandItem) => {
    trackEvent('command_palette_action', { 
      command: command.id,
      query: query.trim() 
    });
    command.action();
    onOpenChange(false);
    setQuery('');
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent 
        className="glass-panel max-w-2xl p-0 gap-0 overflow-hidden border-border/20"
        style={{
          background: 'hsl(var(--glass-background))',
          backdropFilter: 'var(--glass-backdrop-filter)'
        }}
      >
        {/* Sophisticated Header */}
        <div className="flex items-center gap-3 p-6 pb-4 border-b border-border/10">
          <div className="glass-panel p-2 rounded-lg">
            <Command className="h-5 w-5 text-primary" />
          </div>
          <div className="flex-1">
            <Input
              ref={inputRef}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Search commands..."
              className="border-0 bg-transparent text-lg placeholder:text-muted-foreground/50 focus-visible:ring-0 focus-visible:ring-offset-0 px-0"
              autoComplete="off"
            />
          </div>
          <Badge className="glass-panel border-0 text-xs px-3 py-1">
            <span className="font-mono">⌘K</span>
          </Badge>
        </div>

        {/* Sophisticated Results */}
        <div 
          ref={listRef}
          className="max-h-96 overflow-y-auto scrollbar-elegant p-2"
        >
          {categories.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="glass-panel p-4 rounded-xl mb-4 opacity-60">
                <Search className="h-8 w-8 text-muted-foreground" />
              </div>
              <p className="body-elegant text-muted-foreground mb-2">No commands found</p>
              <p className="text-sm text-muted-foreground/60">
                Try searching for "arena", "bench", or "dashboard"
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {categories.map(([category, categoryCommands]) => (
                <div key={category}>
                  <div className="flex items-center gap-2 px-3 py-2 mb-2">
                    <div className="w-1 h-4 bg-gradient-to-b from-primary to-secondary rounded-full"></div>
                    <h3 className="heading-refined text-xs text-muted-foreground uppercase tracking-wider">
                      {category}
                    </h3>
                  </div>
                  
                  <div className="space-y-1">
                    {categoryCommands.map((command, index) => {
                      const globalIndex = filteredCommands.indexOf(command);
                      const isSelected = globalIndex === selectedIndex;
                      const IconComponent = command.icon;
                      
                      return (
                        <button
                          key={command.id}
                          onClick={() => handleCommandClick(command)}
                          className={`
                            w-full flex items-center gap-3 p-3 rounded-xl text-left transition-all duration-200 group
                            ${isSelected ? 'glass-panel shadow-md' : 'hover:bg-muted/30'}
                          `}
                        >
                          <div className={`
                            glass-panel p-2 rounded-lg transition-all duration-200
                            ${isSelected ? 'shadow-md' : 'group-hover:shadow-sm'}
                          `}>
                            <IconComponent className="h-4 w-4 text-primary" />
                          </div>
                          
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="heading-refined text-sm text-foreground truncate">
                                {command.title}
                              </span>
                              {command.badge && (
                                <Badge className="glass-panel border-0 text-xs px-2 py-0.5">
                                  <Star className="h-3 w-3 mr-1" />
                                  {command.badge}
                                </Badge>
                              )}
                            </div>
                            <p className="body-elegant text-xs text-muted-foreground truncate">
                              {command.description}
                            </p>
                          </div>
                          
                          <div className={`
                            transition-all duration-200
                            ${isSelected ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-2 group-hover:opacity-60 group-hover:translate-x-0'}
                          `}>
                            <ArrowRight className="h-4 w-4 text-muted-foreground" />
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </div>
              ))}
              
              {/* Usage Tip */}
              <div className="px-3 py-4 border-t border-border/10 mt-4">
                <div className="flex items-center gap-2 text-xs text-muted-foreground/60">
                  <Clock className="h-3 w-3" />
                  <span>Use ↑ ↓ to navigate, ⏎ to select, ⎋ to close</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export const CommandPalette = memo(CommandPaletteComponent);