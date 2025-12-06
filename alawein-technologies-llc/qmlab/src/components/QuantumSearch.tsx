import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { Search, X, BookOpen, Calculator, Atom, Brain, Command } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { useAccessibilityContext } from '@/components/AccessibilityProvider';
import { trackQuantumEvents } from '@/lib/analytics';

// Quantum concepts and searchable content
const quantumConcepts = [
  {
    id: 'qubit',
    title: 'Qubit',
    description: 'The fundamental unit of quantum information',
    category: 'basics',
    tags: ['quantum-bit', 'superposition', 'state'],
    section: 'circuit-builder'
  },
  {
    id: 'superposition',
    title: 'Superposition',
    description: 'Quantum states existing in multiple possibilities simultaneously',
    category: 'basics',
    tags: ['quantum-state', 'probability', 'measurement'],
    section: 'bloch-sphere'
  },
  {
    id: 'entanglement',
    title: 'Entanglement',
    description: 'Quantum correlation between particles',
    category: 'advanced',
    tags: ['correlation', 'spooky-action', 'bell-states'],
    section: 'training-dashboard'
  },
  {
    id: 'bloch-sphere',
    title: 'Bloch Sphere',
    description: 'Geometric representation of qubit states',
    category: 'visualization',
    tags: ['visualization', 'geometry', 'state-representation'],
    section: 'bloch-sphere'
  },
  {
    id: 'quantum-gates',
    title: 'Quantum Gates',
    description: 'Operations that manipulate quantum states',
    category: 'operations',
    tags: ['pauli-x', 'hadamard', 'cnot', 'operations'],
    section: 'circuit-builder'
  },
  {
    id: 'quantum-circuits',
    title: 'Quantum Circuits',
    description: 'Sequences of quantum gates applied to qubits',
    category: 'basics',
    tags: ['circuit', 'gates', 'algorithm'],
    section: 'circuit-builder'
  },
  {
    id: 'measurement',
    title: 'Quantum Measurement',
    description: 'Process of observing quantum states',
    category: 'basics',
    tags: ['collapse', 'probability', 'observation'],
    section: 'bloch-sphere'
  },
  {
    id: 'quantum-ml',
    title: 'Quantum Machine Learning',
    description: 'Machine learning algorithms using quantum computers',
    category: 'ml',
    tags: ['algorithm', 'optimization', 'neural-networks'],
    section: 'training-dashboard'
  }
];

const tutorials = [
  {
    id: 'build-first-circuit',
    title: 'Build Your First Quantum Circuit',
    description: 'Learn to create basic quantum circuits with gates',
    section: 'circuit-builder',
    difficulty: 'beginner'
  },
  {
    id: 'understanding-bloch',
    title: 'Understanding the Bloch Sphere',
    description: 'Visualize quantum states in 3D space',
    section: 'bloch-sphere',
    difficulty: 'beginner'
  },
  {
    id: 'quantum-ml-basics',
    title: 'Quantum Machine Learning Basics',
    description: 'Introduction to quantum-enhanced ML algorithms',
    section: 'training-dashboard',
    difficulty: 'intermediate'
  }
];

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: 'concept' | 'tutorial';
  category?: string;
  section: string;
  relevance: number;
}

interface QuantumSearchProps {
  isOpen: boolean;
  onClose: () => void;
  onNavigate?: (section: string) => void;
}

export const QuantumSearch: React.FC<QuantumSearchProps> = ({
  isOpen,
  onClose,
  onNavigate
}) => {
  const [query, setQuery] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const { announce } = useAccessibilityContext();

  // Search algorithm with relevance scoring
  const searchResults = useMemo(() => {
    if (!query.trim()) return [];

    const searchTerm = query.toLowerCase();
    const results: SearchResult[] = [];

    // Search concepts
    quantumConcepts.forEach(concept => {
      let relevance = 0;
      
      // Title match (highest relevance)
      if (concept.title.toLowerCase().includes(searchTerm)) {
        relevance += 100;
      }
      
      // Description match
      if (concept.description.toLowerCase().includes(searchTerm)) {
        relevance += 50;
      }
      
      // Tags match
      concept.tags.forEach(tag => {
        if (tag.includes(searchTerm)) {
          relevance += 30;
        }
      });

      // Category match
      if (concept.category.includes(searchTerm)) {
        relevance += 20;
      }

      if (relevance > 0) {
        results.push({
          id: concept.id,
          title: concept.title,
          description: concept.description,
          type: 'concept',
          category: concept.category,
          section: concept.section,
          relevance
        });
      }
    });

    // Search tutorials
    tutorials.forEach(tutorial => {
      let relevance = 0;
      
      if (tutorial.title.toLowerCase().includes(searchTerm)) {
        relevance += 100;
      }
      
      if (tutorial.description.toLowerCase().includes(searchTerm)) {
        relevance += 50;
      }

      if (tutorial.difficulty.includes(searchTerm)) {
        relevance += 20;
      }

      if (relevance > 0) {
        results.push({
          id: tutorial.id,
          title: tutorial.title,
          description: tutorial.description,
          type: 'tutorial',
          section: tutorial.section,
          relevance
        });
      }
    });

    // Sort by relevance
    return results.sort((a, b) => b.relevance - a.relevance).slice(0, 8);
  }, [query]);

  // Keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, searchResults.length - 1));
        break;
      case 'ArrowUp':
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
        break;
      case 'Enter':
        e.preventDefault();
        if (searchResults[selectedIndex]) {
          handleSelectResult(searchResults[selectedIndex]);
        }
        break;
      case 'Escape':
        e.preventDefault();
        onClose();
        break;
    }
  }, [selectedIndex, searchResults, onClose]);

  const handleSelectResult = (result: SearchResult) => {
    trackQuantumEvents.searchResultClick(result.type, result.title);
    
    // Navigate to section
    if (onNavigate) {
      onNavigate(result.section);
    } else {
      // Scroll to section if no navigation handler
      const element = document.getElementById(result.section);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
      }
    }
    
    announce(`Navigating to ${result.title}`);
    onClose();
  };

  // Reset selection when results change
  useEffect(() => {
    setSelectedIndex(0);
  }, [searchResults]);

  // Announce search results for screen readers
  useEffect(() => {
    if (query && searchResults.length > 0) {
      announce(`Found ${searchResults.length} results for "${query}"`);
    }
  }, [searchResults.length, query, announce]);

  if (!isOpen) return null;

  const getCategoryIcon = (type: string, category?: string) => {
    if (type === 'tutorial') return <BookOpen className="w-4 h-4" />;
    
    switch (category) {
      case 'basics': return <Calculator className="w-4 h-4" />;
      case 'visualization': return <Atom className="w-4 h-4" />;
      case 'ml': return <Brain className="w-4 h-4" />;
      default: return <Calculator className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (type: string, category?: string) => {
    if (type === 'tutorial') return 'bg-purple-500/20 text-purple-300 border-purple-400/30';
    
    switch (category) {
      case 'basics': return 'bg-blue-500/20 text-blue-300 border-blue-400/30';
      case 'visualization': return 'bg-green-500/20 text-green-300 border-green-400/30';
      case 'ml': return 'bg-indigo-500/20 text-indigo-300 border-indigo-400/30';
      default: return 'bg-slate-500/20 text-slate-300 border-slate-400/30';
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-start justify-center pt-20">
      <Card className="w-full max-w-2xl mx-4 bg-slate-900/95 border-primary/20 shadow-xl">
        <CardContent className="p-0">
          {/* Search Input */}
          <div className="relative border-b border-slate-700/50">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search quantum concepts, tutorials..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              className="pl-12 pr-12 py-4 text-lg bg-transparent border-0 focus:ring-0 focus:ring-offset-0"
              autoFocus
              aria-label="Search quantum concepts and tutorials"
            />
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="absolute right-2 top-1/2 transform -translate-y-1/2"
              aria-label="Close search"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>

          {/* Search Results */}
          <div className="max-h-96 overflow-y-auto">
            {query && searchResults.length === 0 && (
              <div className="p-6 text-center text-muted-foreground">
                <Search className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No results found for "{query}"</p>
                <p className="text-sm mt-1">Try searching for "qubit", "circuit", or "bloch"</p>
              </div>
            )}

            {searchResults.map((result, index) => (
              <div
                key={result.id}
                className={`p-4 border-b border-slate-700/30 cursor-pointer transition-colors ${
                  index === selectedIndex
                    ? 'bg-primary/10 border-primary/30'
                    : 'hover:bg-slate-800/50'
                }`}
                onClick={() => handleSelectResult(result)}
                role="option"
                aria-selected={index === selectedIndex}
              >
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-lg ${getCategoryColor(result.type, result.category)}`}>
                    {getCategoryIcon(result.type, result.category)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h3 className="font-medium text-slate-200 truncate">{result.title}</h3>
                      <Badge 
                        variant="outline" 
                        className={`text-xs ${getCategoryColor(result.type, result.category)}`}
                      >
                        {result.type === 'tutorial' ? 'Tutorial' : result.category}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground line-clamp-2">{result.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Search Tips */}
          {!query && (
            <div className="p-4 border-t border-slate-700/50">
              <div className="text-xs text-muted-foreground-light mb-2">Quick search tips:</div>
              <div className="flex flex-wrap gap-2">
                {['qubit', 'superposition', 'bloch sphere', 'quantum gates'].map((tip) => (
                  <button
                    key={tip}
                    onClick={() => setQuery(tip)}
                    className="px-2 py-1 bg-slate-800 rounded text-xs text-slate-300 hover:bg-slate-700 transition-colors"
                  >
                    {tip}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-3 text-xs text-muted-foreground-light">
                <Command className="w-3 h-3" />
                <span>Use ↑↓ to navigate, ↵ to select, ESC to close</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};