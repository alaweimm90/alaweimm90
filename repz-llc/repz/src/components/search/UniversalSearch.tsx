import React, { useState, useRef, useEffect } from 'react';
import { Search, Filter, Clock, Star, Zap, X, ArrowRight } from 'lucide-react';
import { Input } from '@/ui/atoms/Input';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Card, CardContent } from '@/ui/molecules/Card';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/ui/molecules/Dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { useNavigate } from 'react-router-dom';
import { useDebounce } from '@/hooks/useDebounce';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  category: 'workouts' | 'nutrition' | 'progress' | 'messages' | 'features' | 'help';
  url: string;
  icon: string;
  tags: string[];
  priority: number;
  lastAccessed?: string;
  isBookmarked?: boolean;
}

interface SearchFilter {
  category?: string;
  dateRange?: 'today' | 'week' | 'month' | 'all';
  priority?: 'high' | 'medium' | 'low';
  hasAccess?: boolean;
}

export const UniversalSearch: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [filters, setFilters] = useState<SearchFilter>({});
  const [selectedIndex, setSelectedIndex] = useState(0);
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const debouncedQuery = useDebounce(query, 300);

  // Mock search data
  const allResults: SearchResult[] = [
    // Workouts
    {
      id: '1',
      title: 'Upper Body Strength Training',
      description: 'Week 4 - Advanced upper body workout with progressive overload',
      category: 'workouts',
      url: '/workouts/upper-body-strength',
      icon: '💪',
      tags: ['strength', 'upper-body', 'progressive'],
      priority: 1,
      lastAccessed: '2 hours ago',
      isBookmarked: true
    },
    {
      id: '2',
      title: 'HIIT Cardio Blast',
      description: '20-minute high-intensity interval training session',
      category: 'workouts',
      url: '/workouts/hiit-cardio',
      icon: '🔥',
      tags: ['cardio', 'hiit', 'fat-loss'],
      priority: 2
    },
    
    // Nutrition
    {
      id: '3',
      title: 'Protein Intake Goals',
      description: 'Track your daily protein consumption and optimize intake',
      category: 'nutrition',
      url: '/nutrition/protein-tracking',
      icon: '🥩',
      tags: ['protein', 'tracking', 'goals'],
      priority: 1,
      lastAccessed: '1 day ago'
    },
    {
      id: '4',
      title: 'Meal Planning Assistant',
      description: 'AI-powered meal planning based on your goals and preferences',
      category: 'nutrition',
      url: '/nutrition/meal-planner',
      icon: '🍽️',
      tags: ['meals', 'planning', 'ai'],
      priority: 2
    },

    // Progress
    {
      id: '5',
      title: 'Weight Progress Chart',
      description: 'Visualize your weight loss/gain journey over time',
      category: 'progress',
      url: '/progress/weight-chart',
      icon: '📈',
      tags: ['weight', 'progress', 'charts'],
      priority: 1,
      isBookmarked: true
    },
    {
      id: '6',
      title: 'Strength Gains Analysis',
      description: 'Track your strength improvements across all exercises',
      category: 'progress',
      url: '/progress/strength-analysis',
      icon: '💪',
      tags: ['strength', 'analysis', 'gains'],
      priority: 2
    },

    // Messages
    {
      id: '7',
      title: 'Coach Messages',
      description: 'View and respond to messages from your personal coach',
      category: 'messages',
      url: '/messages/coach',
      icon: '💬',
      tags: ['coach', 'messages', 'communication'],
      priority: 1,
      lastAccessed: '30 minutes ago'
    },

    // Features
    {
      id: '8',
      title: 'Live Workout Session',
      description: 'Start a real-time workout with AI coaching and form analysis',
      category: 'features',
      url: '/live-workout',
      icon: '⚡',
      tags: ['live', 'ai', 'coaching'],
      priority: 1
    },
    {
      id: '9',
      title: 'Habit Tracking',
      description: 'Build and track healthy habits beyond workouts',
      category: 'features',
      url: '/habits',
      icon: '✅',
      tags: ['habits', 'tracking', 'lifestyle'],
      priority: 2
    },

    // Help
    {
      id: '10',
      title: 'How to Log Workouts',
      description: 'Step-by-step guide to logging your workout sessions',
      category: 'help',
      url: '/help/workout-logging',
      icon: '❓',
      tags: ['help', 'workouts', 'logging'],
      priority: 3
    }
  ];

  // Search functionality
  useEffect(() => {
    if (debouncedQuery.length === 0) {
      setResults([]);
      return;
    }

    const searchResults = allResults.filter(result => {
      const matchesQuery = result.title.toLowerCase().includes(debouncedQuery.toLowerCase()) ||
                          result.description.toLowerCase().includes(debouncedQuery.toLowerCase()) ||
                          result.tags.some(tag => tag.toLowerCase().includes(debouncedQuery.toLowerCase()));

      const matchesCategory = !filters.category || result.category === filters.category;
      
      return matchesQuery && matchesCategory;
    });

    // Sort by priority and relevance
    searchResults.sort((a, b) => {
      if (a.priority !== b.priority) return a.priority - b.priority;
      if (a.isBookmarked && !b.isBookmarked) return -1;
      if (!a.isBookmarked && b.isBookmarked) return 1;
      return 0;
    });

    setResults(searchResults.slice(0, 10)); // Limit to 10 results
    setSelectedIndex(0);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQuery, filters]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, results.length - 1));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (results[selectedIndex]) {
            handleResultClick(results[selectedIndex]);
          }
          break;
        case 'Escape':
          setIsOpen(false);
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, results, selectedIndex]);

  // Global search shortcut
  useEffect(() => {
    const handleGlobalShortcut = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsOpen(true);
      }
    };

    document.addEventListener('keydown', handleGlobalShortcut);
    return () => document.removeEventListener('keydown', handleGlobalShortcut);
  }, []);

  // Focus input when opened
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const handleResultClick = (result: SearchResult) => {
    navigate(result.url);
    setIsOpen(false);
    setQuery('');
    
    // Add to recent searches
    setRecentSearches(prev => {
      const updated = [result.title, ...prev.filter(s => s !== result.title)];
      return updated.slice(0, 5);
    });
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'workouts': return '🏋️';
      case 'nutrition': return '🥗';
      case 'progress': return '📊';
      case 'messages': return '💬';
      case 'features': return '⚡';
      case 'help': return '❓';
      default: return '📄';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'workouts': return 'bg-blue-100 text-blue-800';
      case 'nutrition': return 'bg-green-100 text-green-800';
      case 'progress': return 'bg-purple-100 text-purple-800';
      case 'messages': return 'bg-repz-orange/10 text-repz-orange';
      case 'features': return 'bg-yellow-100 text-yellow-800';
      case 'help': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const categories = [
    { id: 'all', label: 'All', count: allResults.length },
    { id: 'workouts', label: 'Workouts', count: allResults.filter(r => r.category === 'workouts').length },
    { id: 'nutrition', label: 'Nutrition', count: allResults.filter(r => r.category === 'nutrition').length },
    { id: 'progress', label: 'Progress', count: allResults.filter(r => r.category === 'progress').length },
    { id: 'features', label: 'Features', count: allResults.filter(r => r.category === 'features').length },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" className="relative w-full justify-start text-sm text-muted-foreground sm:pr-12 md:w-64">
          <Search className="mr-2 h-4 w-4" />
          <span>Search everything...</span>
          <kbd className="pointer-events-none absolute right-1.5 top-1.5 hidden h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium opacity-100 sm:flex">
            <span className="text-xs">⌘</span>K
          </kbd>
        </Button>
      </DialogTrigger>
      
      <DialogContent className="max-w-2xl p-0">
        <div className="flex flex-col h-[600px]">
          {/* Header */}
          <div className="flex items-center gap-3 px-4 py-3 border-b">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              ref={inputRef}
              placeholder="Search workouts, nutrition, progress, and more..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="border-0 focus-visible:ring-0 text-sm"
            />
            {query && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setQuery('')}
                className="h-6 w-6 p-0"
              >
                <X className="h-3 w-3" />
              </Button>
            )}
          </div>

          {/* Content */}
          <div className="flex-1 overflow-hidden">
            {query === '' ? (
              // Empty state with recent searches and shortcuts
              <div className="p-4 space-y-4">
                {recentSearches.length > 0 && (
                  <div>
                    <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
                      <Clock className="h-4 w-4" />
                      Recent
                    </h3>
                    <div className="space-y-1">
                      {recentSearches.map((search, index) => (
                        <Button
                          key={index}
                          variant="ghost"
                          className="w-full justify-start text-sm h-8"
                          onClick={() => setQuery(search)}
                        >
                          {search}
                        </Button>
                      ))}
                    </div>
                  </div>
                )}

                <div>
                  <h3 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Star className="h-4 w-4" />
                    Quick Access
                  </h3>
                  <div className="grid grid-cols-2 gap-2">
                    {allResults.filter(r => r.isBookmarked).slice(0, 4).map((result) => (
                      <Button
                        key={result.id}
                        variant="outline"
                        className="h-auto p-3 flex flex-col items-start gap-1"
                        onClick={() => handleResultClick(result)}
                      >
                        <div className="flex items-center gap-2 w-full">
                          <span className="text-lg">{result.icon}</span>
                          <span className="font-medium text-xs truncate">{result.title}</span>
                        </div>
                      </Button>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-medium mb-2">Categories</h3>
                  <div className="grid grid-cols-3 gap-2">
                    {categories.slice(1).map((category) => (
                      <Button
                        key={category.id}
                        variant="outline"
                        className="h-auto p-3 flex flex-col items-center gap-1"
                        onClick={() => setFilters({ category: category.id })}
                      >
                        <span className="text-lg">{getCategoryIcon(category.id)}</span>
                        <span className="text-xs font-medium">{category.label}</span>
                        <Badge variant="secondary" className="text-xs">
                          {category.count}
                        </Badge>
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              // Search results
              <div className="flex h-full">
                {/* Filters sidebar */}
                <div className="w-48 border-r bg-muted/30 p-3">
                  <div className="space-y-3">
                    <div>
                      <label className="text-xs font-medium text-muted-foreground">Categories</label>
                      <div className="space-y-1 mt-1">
                        {categories.map((category) => (
                          <Button
                            key={category.id}
                            variant={filters.category === category.id || (!filters.category && category.id === 'all') ? 'default' : 'ghost'}
                            size="sm"
                            className="w-full justify-between text-xs h-7"
                            onClick={() => setFilters({ 
                              ...filters, 
                              category: category.id === 'all' ? undefined : category.id 
                            })}
                          >
                            <span>{category.label}</span>
                            <span className="text-xs text-muted-foreground">{category.count}</span>
                          </Button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Results */}
                <div className="flex-1 overflow-y-auto">
                  {results.length === 0 ? (
                    <div className="p-8 text-center text-muted-foreground">
                      <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p>No results found for "{query}"</p>
                      <p className="text-xs mt-1">Try different keywords or check the filters</p>
                    </div>
                  ) : (
                    <div className="p-2">
                      {results.map((result, index) => (
                        <Button
                          key={result.id}
                          variant="ghost"
                          className={`w-full justify-start p-3 h-auto mb-1 ${
                            index === selectedIndex ? 'bg-accent' : ''
                          }`}
                          onClick={() => handleResultClick(result)}
                        >
                          <div className="flex items-start gap-3 w-full">
                            <span className="text-lg flex-shrink-0">{result.icon}</span>
                            <div className="flex-1 text-left min-w-0">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="font-medium text-sm truncate">{result.title}</span>
                                <Badge variant="outline" className={`text-xs ${getCategoryColor(result.category)}`}>
                                  {result.category}
                                </Badge>
                                {result.isBookmarked && <Star className="h-3 w-3 text-yellow-500 fill-current" />}
                              </div>
                              <p className="text-xs text-muted-foreground line-clamp-2">{result.description}</p>
                              {result.lastAccessed && (
                                <p className="text-xs text-muted-foreground mt-1">Last accessed: {result.lastAccessed}</p>
                              )}
                            </div>
                            <ArrowRight className="h-4 w-4 text-muted-foreground flex-shrink-0" />
                          </div>
                        </Button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="border-t p-2 bg-muted/30">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center gap-4">
                <kbd className="px-1.5 py-0.5 bg-background border rounded">↑↓</kbd>
                <span>Navigate</span>
                <kbd className="px-1.5 py-0.5 bg-background border rounded">⏎</kbd>
                <span>Select</span>
                <kbd className="px-1.5 py-0.5 bg-background border rounded">Esc</kbd>
                <span>Close</span>
              </div>
              {results.length > 0 && (
                <span>{results.length} results</span>
              )}
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};