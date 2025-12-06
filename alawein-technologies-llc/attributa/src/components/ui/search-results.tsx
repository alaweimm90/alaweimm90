import React, { useState } from 'react';
import { Search, X, Filter, ArrowUpDown } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  category: string;
  tags?: string[];
  href?: string;
}

interface SearchResultsProps {
  results: SearchResult[];
  query: string;
  onResultClick?: (result: SearchResult) => void;
  className?: string;
  loading?: boolean;
  emptyMessage?: string;
}

export function SearchResults({
  results,
  query,
  onResultClick,
  className,
  loading = false,
  emptyMessage = "No results found"
}: SearchResultsProps) {
  const [sortBy, setSortBy] = useState<'relevance' | 'category'>('relevance');
  const [filterBy, setFilterBy] = useState<string>('');

  const sortedResults = React.useMemo(() => {
    let filtered = results;
    
    if (filterBy) {
      filtered = results.filter(result => 
        result.category.toLowerCase() === filterBy.toLowerCase()
      );
    }

    if (sortBy === 'category') {
      return filtered.sort((a, b) => a.category.localeCompare(b.category));
    }
    
    return filtered; // Default relevance order
  }, [results, sortBy, filterBy]);

  const categories = React.useMemo(() => {
    return Array.from(new Set(results.map(r => r.category)));
  }, [results]);

  if (loading) {
    return (
      <div className={cn('space-y-4', className)}>
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="glass-card p-4 space-y-2 animate-pulse">
            <div className="h-4 bg-muted rounded w-3/4"></div>
            <div className="h-3 bg-muted rounded w-full"></div>
            <div className="h-3 bg-muted rounded w-2/3"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Search controls */}
      <div className="flex items-center justify-between gap-4 pb-2 border-b border-border/50">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span className="font-mono">
            {sortedResults.length} result{sortedResults.length !== 1 ? 's' : ''}
            {query && (
              <>
                {' '}for{' '}
                <span className="font-semibold text-foreground">"{query}"</span>
              </>
            )}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          {categories.length > 1 && (
            <select
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value)}
              className="bg-background border border-border rounded px-2 py-1 text-xs font-mono"
            >
              <option value="">All categories</option>
              {categories.map(category => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
          )}
          
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortBy(sortBy === 'relevance' ? 'category' : 'relevance')}
            className="text-xs font-mono gap-1"
          >
            <ArrowUpDown className="h-3 w-3" />
            {sortBy}
          </Button>
        </div>
      </div>

      {/* Results */}
      {sortedResults.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground">
          <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
          <p className="font-mono text-sm">{emptyMessage}</p>
        </div>
      ) : (
        <div className="space-y-3">
          {sortedResults.map((result) => (
            <div
              key={result.id}
              className={cn(
                'glass-card p-4 rounded-lg transition-all duration-200',
                'hover:shadow-md hover:-translate-y-0.5 cursor-pointer',
                'border border-border/50 hover:border-primary/30'
              )}
              onClick={() => onResultClick?.(result)}
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <h3 className="font-semibold font-mono text-sm text-foreground">
                      {result.title}
                    </h3>
                    <Badge variant="outline" className="text-xs font-mono px-2 py-0.5">
                      {result.category}
                    </Badge>
                  </div>
                  
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {result.description}
                  </p>
                  
                  {result.tags && result.tags.length > 0 && (
                    <div className="flex gap-1 flex-wrap">
                      {result.tags.map((tag) => (
                        <Badge 
                          key={tag} 
                          variant="secondary" 
                          className="text-xs px-2 py-0.5 font-mono"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}