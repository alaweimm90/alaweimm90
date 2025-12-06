import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import { Card, CardContent } from '@/ui/molecules/Card';
import { Input } from '@/ui/atoms/Input';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { useDebounce } from '@/hooks/useDebounce';

interface SearchResult {
  id: string;
  category: string;
  type: string;
  relevanceScore: number;
  name?: string;
  client_name?: string;
  description?: string;
  daily_notes?: string;
  primary_goal?: string;
  created_at?: string;
  muscle_groups?: string[] | string;
  matchDetails: Record<string, { matches?: string[] }>;
  [key: string]: unknown;
}

interface SearchResponse {
  query: string;
  totalResults: number;
  results: SearchResult[];
  groupedResults: Record<string, SearchResult[]>;
  categories: string[];
  suggestions: string[];
}

const CATEGORY_CONFIG = {
  workouts: {
    name: 'Workouts',
    color: 'text-blue-500',
    bgColor: 'bg-blue-50',
  },
  exercises: {
    name: 'Exercises',
    color: 'text-green-500',
    bgColor: 'bg-green-50',
  },
  nutrition: {
    name: 'Nutrition',
    color: 'text-orange-500',
    bgColor: 'bg-orange-50',
  },
  progress: {
    name: 'Progress',
    color: 'text-purple-500',
    bgColor: 'bg-purple-50',
  },
  clients: {
    name: 'Clients',
    color: 'text-indigo-500',
    bgColor: 'bg-indigo-50',
  },
};

export default function UniversalSearch() {
  const { toast } = useToast();
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState('all');
  const [highlightIndex, setHighlightIndex] = useState<number>(-1);
  const activeRequestId = useRef(0);

  const debouncedQuery = useDebounce(query, 300);

  // Perform search
  const performSearch = useCallback(
    async (searchQuery: string) => {
      if (!searchQuery.trim() || searchQuery.length < 2) {
        setSearchResults(null);
        return;
      }

      activeRequestId.current += 1;
      const requestId = activeRequestId.current;
      setIsSearching(true);

      try {
        const { data, error } = await supabase.functions.invoke('universal-search', {
          body: {
            query: searchQuery,
            categories: selectedCategories.length > 0 ? selectedCategories : undefined,
            limit: 50,
          },
        });

        if (error) throw error;

        if (requestId === activeRequestId.current) {
          setSearchResults(data);
        }
      } catch (error) {
        console.error('Search error:', error);
        if (requestId === activeRequestId.current) {
          toast({
            title: 'Search Error',
            description: 'Could not perform search. Please try again.',
            variant: 'destructive',
          });
        }
      } finally {
        if (requestId === activeRequestId.current) {
          setIsSearching(false);
        }
      }
    },
    [selectedCategories, toast]
  );

  // Search when debounced query changes
  useEffect(() => {
    performSearch(debouncedQuery);
  }, [debouncedQuery, performSearch]);

  // Toggle category filter
  const toggleCategory = useCallback((category: string) => {
    setSelectedCategories((prev) =>
      prev.includes(category) ? prev.filter((c) => c !== category) : [...prev, category]
    );
  }, []);

  // Get filtered results for current tab
  const filteredResults = useMemo(() => {
    if (!searchResults) return [];

    if (activeTab === 'all') {
      return searchResults.results;
    }

    return searchResults.groupedResults[activeTab] || [];
  }, [searchResults, activeTab]);

  const onKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!filteredResults.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setHighlightIndex((i) => (i + 1) % filteredResults.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIndex((i) => (i <= 0 ? filteredResults.length - 1 : i - 1));
    } else if (e.key === 'Enter' && highlightIndex >= 0) {
      const item = filteredResults[highlightIndex];
      setQuery(item.name || item.client_name || item.type || '');
    }
  }, [filteredResults, highlightIndex]);

  // Render search result item
  const renderResultItem = useCallback((result: SearchResult) => {
    const config = CATEGORY_CONFIG[result.category as keyof typeof CATEGORY_CONFIG];

    return (
      <Card key={result.id} className="hover:shadow-md transition-shadow cursor-pointer">
        <CardContent className="p-4">
          <div className="flex items-start gap-3">
            <div className={`p-2 rounded-lg ${config?.bgColor || 'bg-gray-50'}`}>
              <span
                className={`inline-block h-5 w-5 rounded ${config?.color || 'text-gray-500'}`}
              ></span>
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-medium truncate">
                  {result.name || result.client_name || `${result.type} #${result.id}`}
                </h3>
                <Badge variant="outline" className="text-xs">
                  {config?.name || result.category}
                </Badge>
                <div className="text-xs text-muted-foreground">Score: {result.relevanceScore}</div>
              </div>

              {/* Description or relevant content */}
              <p className="text-sm text-muted-foreground line-clamp-2">
                {result.description ||
                  result.daily_notes ||
                  result.primary_goal ||
                  'No description available'}
              </p>

              {/* Match highlights */}
              {result.matchDetails && Object.keys(result.matchDetails).length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {Object.entries(result.matchDetails).map(([field, details]) => {
                    const matchDetails = details as { matches?: string[] };
                    return (
                      <Badge key={field} variant="secondary" className="text-xs">
                        {field}: {matchDetails.matches?.join(', ') || 'match'}
                      </Badge>
                    );
                  })}
                </div>
              )}

              {/* Additional metadata */}
              <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                {result.created_at && (
                  <div className="flex items-center gap-1">
                    {new Date(result.created_at).toLocaleDateString()}
                  </div>
                )}
                {result.muscle_groups && (
                  <div className="flex items-center gap-1">
                    {Array.isArray(result.muscle_groups)
                      ? result.muscle_groups.slice(0, 2).join(', ')
                      : result.muscle_groups}
                  </div>
                )}
              </div>
            </div>

            <span className="h-4 w-4 text-muted-foreground">›</span>
          </div>
        </CardContent>
      </Card>
    );
  }, []);

  return (
    <div className="space-y-6">
      {/* Search Header */}
      <Card>
        <CardContent className="p-6">
          <div className="space-y-4">
            {/* Search Input */}
            <div className="relative">
              <span aria-hidden="true" className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground">
                🔎
              </span>
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder="Search workouts, exercises, nutrition plans, progress..."
                className="pl-10 pr-10"
                aria-label="Universal search"
                role="combobox"
                aria-expanded={!!searchResults}
                aria-activedescendant={highlightIndex >= 0 ? `result-${filteredResults[highlightIndex]?.id}` : undefined}
              />
              {isSearching && (
                <span role="status" aria-live="polite" aria-label="Loading" className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground">
                  ⏳
                </span>
              )}
            </div>

            {/* Category Filters */}
            <div className="flex flex-wrap gap-2">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span aria-hidden="true" className="h-4 w-4">⚙️</span>
                Filters:
              </div>
              {Object.entries(CATEGORY_CONFIG).map(([key, config]) => (
                <Button
                  key={key}
                  variant={selectedCategories.includes(key) ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => toggleCategory(key)}
                  className="h-7"
                  aria-pressed={selectedCategories.includes(key)}
                >
                  {config.name}
                </Button>
              ))}
            </div>

            {/* Search Suggestions */}
            {searchResults?.suggestions && searchResults.suggestions.length > 0 && (
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground">Suggestions:</p>
                <div className="flex flex-wrap gap-2">
                  {searchResults.suggestions.map((suggestion, index) => (
                    <Button
                      key={index}
                      variant="ghost"
                      size="sm"
                      onClick={() => setQuery(suggestion)}
                      className="h-6 text-xs"
                      aria-label={`Use suggestion ${suggestion}`}
                    >
                      {suggestion}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Search Results */}
      {searchResults && (
        <Card>
          <CardContent className="p-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <div className="flex items-center justify-between mb-4">
                <TabsList>
                  <TabsTrigger value="all">All ({searchResults.totalResults})</TabsTrigger>
                  {searchResults.categories.map((category) => {
                    const config = CATEGORY_CONFIG[category as keyof typeof CATEGORY_CONFIG];
                    const count = searchResults.groupedResults[category]?.length || 0;
                    return (
                      <TabsTrigger key={category} value={category}>
                        {config?.name || category} ({count})
                      </TabsTrigger>
                    );
                  })}
                </TabsList>

                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <div aria-live="polite">{searchResults.totalResults} results</div>
                </div>
              </div>

              <TabsContent value={activeTab} className="mt-0">
                {filteredResults.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    {query ? 'No results found for your search.' : 'Start typing to search...'}
                  </div>
                ) : (
                  <div className="space-y-3" role="listbox">
                    {filteredResults.map((r, idx) => (
                      <div key={r.id} id={`result-${r.id}`} role="option" aria-selected={idx === highlightIndex}>
                        {renderResultItem(r)}
                      </div>
                    ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
