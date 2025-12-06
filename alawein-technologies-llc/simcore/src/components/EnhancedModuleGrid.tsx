/**
 * Enhanced Module Grid Component
 * 
 * Mobile-first responsive grid with semantic tokens, accessibility,
 * and performance optimizations for physics module cards.
 */

import { ReactNode, useMemo, useState, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ModuleCard } from '@/components/ModuleCard';
import { ResponsiveGrid } from '@/components/ResponsivePhysicsLayout';
import { AccessibleAlert } from '@/components/AccessibilityEnhancedComponents';
import { useAccessibility } from '@/hooks/use-accessibility';
import { useResponsive } from '@/hooks/use-responsive';
import { Search, Filter, Grid, List, SortAsc, SortDesc } from 'lucide-react';
import { cn } from '@/lib/utils';
import { physicsModules, moduleCategories } from '@/data/modules';

interface EnhancedModuleGridProps {
  modules?: typeof physicsModules;
  searchTerm?: string;
  selectedCategory?: string;
  onModuleSelect?: (route: string) => void;
  onSearchChange?: (term: string) => void;
  onCategoryChange?: (category: string) => void;
  className?: string;
  showFilters?: boolean;
  layout?: 'grid' | 'list';
}

export function EnhancedModuleGrid({
  modules = physicsModules,
  searchTerm: externalSearchTerm,
  selectedCategory: externalSelectedCategory,
  onModuleSelect,
  onSearchChange,
  onCategoryChange,
  className,
  showFilters = true,
  layout: externalLayout = 'grid'
}: EnhancedModuleGridProps) {
  const { isMobile, isTablet } = useResponsive();
  const { announce } = useAccessibility();
  
  // Internal state for standalone usage
  const [internalSearchTerm, setInternalSearchTerm] = useState('');
  const [internalSelectedCategory, setInternalSelectedCategory] = useState('All');
  const [internalLayout, setInternalLayout] = useState<'grid' | 'list'>('grid');
  const [sortBy, setSortBy] = useState<'name' | 'difficulty' | 'category'>('name');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');

  // Use external props if provided, otherwise use internal state
  const searchTerm = externalSearchTerm ?? internalSearchTerm;
  const selectedCategory = externalSelectedCategory ?? internalSelectedCategory;
  const layout = externalLayout ?? internalLayout;

  const handleSearchChange = useCallback((value: string) => {
    if (onSearchChange) {
      onSearchChange(value);
    } else {
      setInternalSearchTerm(value);
    }
    announce(`Search updated: ${value || 'cleared'}`);
  }, [onSearchChange, announce]);

  const handleCategoryChange = useCallback((value: string) => {
    if (onCategoryChange) {
      onCategoryChange(value);
    } else {
      setInternalSelectedCategory(value);
    }
    announce(`Category filter changed to ${value}`);
  }, [onCategoryChange, announce]);

  const handleLayoutChange = useCallback((value: 'grid' | 'list') => {
    setInternalLayout(value);
    announce(`Layout changed to ${value} view`);
  }, [announce]);

  // Filter and sort modules
  const filteredAndSortedModules = useMemo(() => {
    const filtered = modules.filter(module => {
      const matchesSearch = searchTerm === '' || 
        module.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.category.toLowerCase().includes(searchTerm.toLowerCase()) ||
        module.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      
      const matchesCategory = selectedCategory === 'All' || module.category === selectedCategory;
      
      return matchesSearch && matchesCategory;
    });

    // Sort modules
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'name':
          comparison = a.title.localeCompare(b.title);
          break;
        case 'difficulty':
          const difficultyOrder = ['Beginner', 'Intermediate', 'Advanced', 'Research'];
          comparison = difficultyOrder.indexOf(a.difficulty) - difficultyOrder.indexOf(b.difficulty);
          break;
        case 'category':
          comparison = a.category.localeCompare(b.category);
          break;
      }
      
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return filtered;
  }, [modules, searchTerm, selectedCategory, sortBy, sortOrder]);

  const implementedCount = filteredAndSortedModules.filter(m => m.isImplemented).length;
  const totalCount = filteredAndSortedModules.length;

  return (
    <div className={cn("space-y-6", className)}>
      {/* Search and Filter Section */}
      {showFilters && (
        <Card className="bg-surface/90 backdrop-blur-sm border-muted/30 shadow-elevation1">
          <CardHeader className="pb-4">
            <CardTitle className="text-textPrimary">Explore Physics Modules</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Search Bar */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-textMuted" />
              <Input
                type="search"
                placeholder="Search modules, theories, or concepts..."
                value={searchTerm}
                onChange={(e) => handleSearchChange(e.target.value)}
                className="pl-10 bg-surfaceElevated border-muted/30 text-textPrimary placeholder:text-textMuted"
                aria-label="Search physics modules"
              />
            </div>

            {/* Filters Row - Responsive */}
            <div className="flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
              <div className="flex flex-col sm:flex-row gap-3 flex-1">
                {/* Category Filter */}
                <Select value={selectedCategory} onValueChange={handleCategoryChange}>
                  <SelectTrigger className="w-full sm:w-48 bg-surfaceElevated border-muted/30">
                    <SelectValue placeholder="All Categories" />
                  </SelectTrigger>
                  <SelectContent className="bg-surface border-muted/30">
                    <SelectItem value="All">All Categories</SelectItem>
                    {moduleCategories.map(category => (
                      <SelectItem key={category} value={category}>
                        {category}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {/* Sort Controls */}
                <div className="flex gap-2">
                  <Select value={sortBy} onValueChange={(value: 'name' | 'difficulty' | 'category') => setSortBy(value)}>
                    <SelectTrigger className="w-32 bg-surfaceElevated border-muted/30">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent className="bg-surface border-muted/30">
                      <SelectItem value="name">Name</SelectItem>
                      <SelectItem value="difficulty">Difficulty</SelectItem>
                      <SelectItem value="category">Category</SelectItem>
                    </SelectContent>
                  </Select>

                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
                    className="bg-surfaceElevated border-muted/30"
                    aria-label={`Sort ${sortOrder === 'asc' ? 'descending' : 'ascending'}`}
                  >
                    {sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />}
                  </Button>
                </div>
              </div>

              {/* Layout Toggle - Desktop Only */}
              {!isMobile && (
                <div className="flex gap-1 bg-surfaceMuted rounded-lg p-1">
                  <Button
                    variant={layout === 'grid' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleLayoutChange('grid')}
                    className="px-3"
                    aria-label="Grid view"
                  >
                    <Grid className="h-4 w-4" />
                  </Button>
                  <Button
                    variant={layout === 'list' ? 'default' : 'ghost'}
                    size="sm"
                    onClick={() => handleLayoutChange('list')}
                    className="px-3"
                    aria-label="List view"
                  >
                    <List className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>

            {/* Results Summary */}
            <div className="flex flex-wrap items-center gap-2 text-sm text-textSecondary">
              <span>
                Showing {totalCount} module{totalCount !== 1 ? 's' : ''}
              </span>
              <span className="text-textMuted">•</span>
              <Badge variant="outline" className="bg-accentPhysics/10 text-accentPhysics border-accentPhysics/20">
                {implementedCount} Ready
              </Badge>
              <Badge variant="outline" className="bg-textMuted/10 text-textMuted border-textMuted/20">
                {totalCount - implementedCount} Coming Soon
              </Badge>
            </div>
          </CardContent>
        </Card>
      )}

      {/* No Results Message */}
      {filteredAndSortedModules.length === 0 && (
        <AccessibleAlert type="info" title="No modules found">
          Try adjusting your search terms or category filter to find more modules.
        </AccessibleAlert>
      )}

      {/* Module Grid/List */}
      {layout === 'grid' ? (
        <ResponsiveGrid
          columns={{
            mobile: 1,
            tablet: 2,
            desktop: isMobile ? 1 : isTablet ? 2 : 3
          }}
          gap={{
            mobile: '1rem',
            tablet: '1.5rem',
            desktop: '2rem'
          }}
          className="auto-rows-fr"
        >
          {filteredAndSortedModules.map((module, index) => (
            <ModuleCard
              key={module.id}
              title={module.title}
              description={module.description}
              category={module.category}
              difficulty={module.difficulty}
              tags={module.tags}
              equation={module.equation}
              isImplemented={module.isImplemented}
              onExplore={() => onModuleSelect?.(module.route)}
              onTheory={() => console.log('Theory for', module.title)}
            />
          ))}
        </ResponsiveGrid>
      ) : (
        <div className="space-y-4">
          {filteredAndSortedModules.map((module, index) => (
            <Card key={module.id} className="bg-surface/90 backdrop-blur-sm border-muted/30">
              <CardContent className="p-4">
                <div className="flex items-center justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-3 mb-2">
                      <h3 className="font-semibold text-textPrimary">{module.title}</h3>
                      <Badge variant="outline" className="text-xs">
                        {module.category}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {module.difficulty}
                      </Badge>
                    </div>
                    <p className="text-sm text-textSecondary line-clamp-2">
                      {module.description}
                    </p>
                  </div>
                  <Button
                    variant={module.isImplemented ? "default" : "outline"}
                    size="sm"
                    disabled={!module.isImplemented}
                    onClick={() => onModuleSelect?.(module.route)}
                  >
                    {module.isImplemented ? 'Launch' : 'Coming Soon'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}