import React, { useState, useMemo, useCallback } from 'react';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { 
  Search, 
  Filter, 
  X, 
  SortAsc, 
  SortDesc,
  BookOpen,
  Zap,
  Target,
  Atom
} from 'lucide-react';
import { PhysicsModule, moduleCategories } from '@/data/modules';
import { useToast } from '@/hooks/use-toast';

interface SearchAndFilterProps {
  modules: PhysicsModule[];
  onFilteredModules: (modules: PhysicsModule[]) => void;
  className?: string;
}

type SortOption = 'title' | 'category' | 'difficulty' | 'default';
type SortDirection = 'asc' | 'desc';

export const SearchAndFilter: React.FC<SearchAndFilterProps> = ({
  modules,
  onFilteredModules,
  className = ''
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('');
  const [selectedTags, setSelectedTags] = useState<string[]>([]);
  const [sortBy, setSortBy] = useState<SortOption>('default');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [showImplementedOnly, setShowImplementedOnly] = useState(false);
  const { toast } = useToast();

  // Get unique tags from all modules
  const allTags = useMemo(() => {
    const tags = new Set<string>();
    modules.forEach(module => {
      module.tags.forEach(tag => tags.add(tag));
    });
    return Array.from(tags).sort();
  }, [modules]);

  const difficulties = ['Beginner', 'Intermediate', 'Advanced', 'Research'];

  // Advanced filtering logic
  const filteredAndSortedModules = useMemo(() => {
    const filtered = modules.filter(module => {
      // Search filter
      const searchMatch = !searchTerm || 
        module.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));

      // Category filter
      const categoryMatch = !selectedCategory || module.category === selectedCategory;

      // Difficulty filter
      const difficultyMatch = !selectedDifficulty || module.difficulty === selectedDifficulty;

      // Tags filter
      const tagsMatch = selectedTags.length === 0 || 
        selectedTags.every(tag => module.tags.includes(tag));

      // Implementation filter
      const implementationMatch = !showImplementedOnly || module.isImplemented;

      return searchMatch && categoryMatch && difficultyMatch && tagsMatch && implementationMatch;
    });

    // Sorting logic
    if (sortBy !== 'default') {
      filtered.sort((a, b) => {
        let comparison = 0;
        
        switch (sortBy) {
          case 'title':
            comparison = a.title.localeCompare(b.title);
            break;
          case 'category':
            comparison = a.category.localeCompare(b.category);
            break;
          case 'difficulty':
            const diffOrder = { 'Beginner': 1, 'Intermediate': 2, 'Advanced': 3, 'Research': 4 };
            comparison = (diffOrder[a.difficulty as keyof typeof diffOrder] || 0) - 
                        (diffOrder[b.difficulty as keyof typeof diffOrder] || 0);
            break;
        }
        
        return sortDirection === 'desc' ? -comparison : comparison;
      });
    }

    return filtered;
  }, [modules, searchTerm, selectedCategory, selectedDifficulty, selectedTags, sortBy, sortDirection, showImplementedOnly]);

  // Update parent component when filters change
  React.useEffect(() => {
    onFilteredModules(filteredAndSortedModules);
  }, [filteredAndSortedModules, onFilteredModules]);

  const handleTagToggle = useCallback((tag: string) => {
    setSelectedTags(prev => 
      prev.includes(tag) 
        ? prev.filter(t => t !== tag)
        : [...prev, tag]
    );
  }, []);

  const clearAllFilters = useCallback(() => {
    setSearchTerm('');
    setSelectedCategory('');
    setSelectedDifficulty('');
    setSelectedTags([]);
    setSortBy('default');
    setSortDirection('asc');
    setShowImplementedOnly(false);
    
    toast({
      title: "Filters Cleared",
      description: "All search and filter criteria have been reset.",
    });
  }, [toast]);

  const toggleSort = useCallback((option: SortOption) => {
    if (sortBy === option) {
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(option);
      setSortDirection('asc');
    }
  }, [sortBy]);

  const getDifficultyIcon = (difficulty: string) => {
    switch (difficulty) {
      case 'Beginner': return BookOpen;
      case 'Intermediate': return Target;
      case 'Advanced': return Zap;
      case 'Research': return Atom;
      default: return BookOpen;
    }
  };

  const activeFiltersCount = [
    searchTerm,
    selectedCategory,
    selectedDifficulty,
    ...selectedTags,
    showImplementedOnly ? 'implemented' : ''
  ].filter(Boolean).length;

  return (
    <Card className={`w-full ${className}`}>
      <CardContent className="p-spacing-lg space-y-spacing-lg">
        {/* Search Bar */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search modules, descriptions, or tags..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-10 min-h-touch"
          />
          {searchTerm && (
            <Button
              variant="ghost"
              size="sm"
              className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0 min-h-touch"
              onClick={() => setSearchTerm('')}
            >
              <X className="w-4 h-4" />
            </Button>
          )}
        </div>

        {/* Filter Controls */}
        <div className="flex flex-wrap gap-spacing-md items-center">
          {/* Category Filter */}
          <div className="flex flex-wrap gap-spacing-xs">
            <span className="text-sm font-medium text-muted-foreground">Category:</span>
            {moduleCategories.map(category => (
              <Badge
                key={category}
                variant={selectedCategory === category ? "default" : "outline"}
                className="cursor-pointer hover:bg-primary/10 transition-colors min-h-touch"
                onClick={() => setSelectedCategory(selectedCategory === category ? '' : category)}
              >
                {category}
              </Badge>
            ))}
          </div>

          {/* Difficulty Filter */}
          <div className="flex flex-wrap gap-2">
            <span className="text-sm font-medium text-muted-foreground">Difficulty:</span>
            {difficulties.map(difficulty => {
              const Icon = getDifficultyIcon(difficulty);
              return (
                <Badge
                  key={difficulty}
                  variant={selectedDifficulty === difficulty ? "default" : "outline"}
                  className="cursor-pointer hover:bg-primary/10 transition-colors flex items-center gap-1"
                  onClick={() => setSelectedDifficulty(selectedDifficulty === difficulty ? '' : difficulty)}
                >
                  <Icon className="w-3 h-3" />
                  {difficulty}
                </Badge>
              );
            })}
          </div>

          {/* Implementation Filter */}
          <Badge
            variant={showImplementedOnly ? "default" : "outline"}
            className="cursor-pointer hover:bg-primary/10 transition-colors"
            onClick={() => setShowImplementedOnly(!showImplementedOnly)}
          >
            <Zap className="w-3 h-3 mr-1" />
            Ready to Use
          </Badge>
        </div>

        {/* Tags Filter */}
        {allTags.length > 0 && (
          <div className="space-y-2">
            <span className="text-sm font-medium text-muted-foreground">Tags:</span>
            <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto">
              {allTags.map(tag => (
                <Badge
                  key={tag}
                  variant={selectedTags.includes(tag) ? "default" : "secondary"}
                  className="cursor-pointer hover:bg-primary/10 transition-colors text-xs"
                  onClick={() => handleTagToggle(tag)}
                >
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Sort Controls */}
        <div className="flex flex-wrap gap-2 items-center">
          <span className="text-sm font-medium text-muted-foreground">Sort by:</span>
          {([
            { key: 'title', label: 'Title' },
            { key: 'category', label: 'Category' },
            { key: 'difficulty', label: 'Difficulty' }
          ] as const).map(({ key, label }) => (
            <Button
              key={key}
              variant={sortBy === key ? "default" : "outline"}
              size="sm"
              className="flex items-center gap-1"
              onClick={() => toggleSort(key)}
            >
              {label}
              {sortBy === key && (
                sortDirection === 'asc' ? 
                  <SortAsc className="w-3 h-3" /> : 
                  <SortDesc className="w-3 h-3" />
              )}
            </Button>
          ))}
        </div>

        {/* Results Summary and Clear Filters */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            Showing {filteredAndSortedModules.length} of {modules.length} modules
            {activeFiltersCount > 0 && (
              <span className="ml-2">
                ({activeFiltersCount} filter{activeFiltersCount > 1 ? 's' : ''} active)
              </span>
            )}
          </div>
          
          {activeFiltersCount > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={clearAllFilters}
              className="flex items-center gap-1"
            >
              <Filter className="w-3 h-3" />
              Clear All
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
};