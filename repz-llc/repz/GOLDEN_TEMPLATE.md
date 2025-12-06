# 🏆 Enterprise React/TypeScript Golden Template

> **Abstracted from REPZ Platform - Enterprise-Grade Architecture**
>
> **Status**: Production Ready | **Version**: 2.0 | **Last Updated**: 2025-11-15

## 🎯 Core Philosophy

**"Living It Iconic" through Code Excellence**

- Functions < 20 lines, single responsibility
- Self-documenting names, comments only for "why"
- TypeScript strict mode, zero `any` types
- Enterprise patterns with scalability built-in
- Security-first, performance-optimized, accessibility-compliant

---

## 📁 Repository Structure (Golden Standard)

```
project-root/
├── src/
│   ├── components/           # Feature-first organization
│   │   ├── feature-name/
│   │   │   ├── FeatureComponent.tsx
│   │   │   ├── FeatureHooks.ts
│   │   │   ├── FeatureUtils.ts
│   │   │   └── FeatureTypes.ts
│   │   ├── atoms/           # Atomic design - smallest components
│   │   ├── molecules/       # Combined atoms
│   │   ├── organisms/       # Complex components
│   │   └── shared/          # Cross-feature components
│   ├── hooks/              # Reusable React hooks
│   ├── lib/                # Business logic and utilities
│   │   ├── feature-systems/ # Complex feature implementations
│   │   ├── utils/          # Pure utility functions
│   │   └── integrations/   # Third-party integrations
│   ├── types/              # Global TypeScript definitions
│   ├── styles/             # Global styles and themes
│   └── assets/             # Static assets
├── packages/               # Monorepo packages
│   ├── ui-components/      # Shared UI library
│   ├── eslint-config/      # Shared linting rules
│   └── typescript-config/  # Shared TS configuration
├── scripts/                # Build and maintenance scripts
├── docs/                   # Comprehensive documentation
├── .github/               # CI/CD workflows
└── _graveyard/            # Deprecated code (never delete)
```

---

## 🧩 Component Architecture (Golden Pattern)

### 1. Component File Structure

```typescript
/**
 * @fileoverview Universal search system with real-time filtering
 * @author [Your Team]
 * @since 2025-01-01
 */

// 1. Type imports first
import type { FC, PropsWithChildren } from 'react';
import type { SearchResult, SearchResponse } from '@/types/search';

// 2. Library imports
import React, { useState, useCallback, useEffect, useMemo, useRef } from 'react';

// 3. Internal imports (absolute paths)
import { Card, CardContent } from '@/ui/molecules/Card';
import { Input } from '@/ui/atoms/Input';
import { useToast } from '@/hooks/useToast';
import { useDebounce } from '@/hooks/useDebounce';
import { supabase } from '@/integrations/supabase/client';

// 4. Constants and configuration
const CATEGORY_CONFIG = {
  workouts: { name: 'Workouts', color: 'text-blue-500', bgColor: 'bg-blue-50' },
  exercises: { name: 'Exercises', color: 'text-green-500', bgColor: 'bg-green-50' },
  nutrition: { name: 'Nutrition', color: 'text-orange-500', bgColor: 'bg-orange-50' },
} as const;

// 5. Type definitions
interface ComponentProps {
  /** Primary configuration object */
  config: SearchConfig;
  /** Optional CSS classes */
  className?: string;
  /** Callback for search completion */
  onSearchComplete?: (results: SearchResponse) => void;
}

interface ComponentState {
  query: string;
  results: SearchResponse | null;
  isSearching: boolean;
}

// 6. Main component (default export)
export default function UniversalSearch({ config, className, onSearchComplete }: ComponentProps) {
  // Hooks and state
  const { toast } = useToast();
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const activeRequestId = useRef(0);

  const debouncedQuery = useDebounce(query, 300);

  // Component logic (functions < 20 lines)
  const performSearch = useCallback(
    async (searchQuery: string) => {
      if (!searchQuery.trim() || searchQuery.length < 2) {
        setResults(null);
        return;
      }

      activeRequestId.current += 1;
      const requestId = activeRequestId.current;
      setIsSearching(true);

      try {
        const { data, error } = await supabase.functions.invoke('universal-search', {
          body: { query: searchQuery, limit: 50 },
        });

        if (error) throw error;
        if (requestId === activeRequestId.current) {
          setResults(data);
          onSearchComplete?.(data);
        }
      } catch (error) {
        console.error('Search error:', error);
        toast({
          title: 'Search Error',
          description: 'Could not perform search. Please try again.',
          variant: 'destructive',
        });
      } finally {
        if (requestId === activeRequestId.current) {
          setIsSearching(false);
        }
      }
    },
    [onSearchComplete, toast]
  );

  // Effects
  useEffect(() => {
    performSearch(debouncedQuery);
  }, [debouncedQuery, performSearch]);

  // Render
  return <div className={cn('space-y-6', className)}>{/* Search implementation */}</div>;
}
```

### 2. Component Patterns

#### ✅ **Do This (Golden Standard)**

```typescript
// Single responsibility, pure functions
const formatSearchResult = (result: SearchResult): FormattedResult => ({
  id: result.id,
  title: result.name || result.client_name || 'Untitled',
  description: result.description || result.daily_notes || 'No description',
  category: CATEGORY_CONFIG[result.category]?.name || result.category,
  score: Math.round(result.relevanceScore * 100),
});

// Memoized expensive operations
const filteredResults = useMemo(() => {
  if (!searchResults) return [];
  return activeTab === 'all'
    ? searchResults.results
    : searchResults.groupedResults[activeTab] || [];
}, [searchResults, activeTab]);

// Proper error boundaries and loading states
if (error) return <ErrorState message={error.message} onRetry={handleRetry} />;
if (loading) return <LoadingState />;
```

#### ❌ **Don't Do This**

```typescript
// Avoid complex inline logic
const result = data.find(
  (item) =>
    item.category === 'workouts' &&
    item.relevanceScore > 0.5 &&
    (item.name || item.client_name)?.includes(query)
) || { id: 'default', name: 'Not Found' };

// Avoid nested ternaries
const status = loading ? 'loading' : error ? 'error' : data ? 'success' : 'empty';

// Avoid magic numbers and strings
if (score > 0.75 && category === 'workouts' && type === 'strength') {
  // Business logic buried in component
}
```

---

## 🎯 Hook Architecture (Golden Pattern)

### Custom Hook Structure

```typescript
/**
 * Custom hook for debounced search functionality
 * @param delay - Debounce delay in milliseconds
 * @returns Debounced value and setter function
 */
export function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => clearTimeout(handler);
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Complex business logic hook with error handling
 */
interface UseSearchOptions {
  enabled?: boolean;
  onSuccess?: (data: SearchResponse) => void;
  onError?: (error: Error) => void;
}

export function useSearch(query: string, options: UseSearchOptions = {}) {
  const { enabled = true, onSuccess, onError } = options;
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const search = useCallback(
    async (searchQuery: string) => {
      if (!enabled || !searchQuery.trim()) return;

      setLoading(true);
      setError(null);

      try {
        const response = await performSearch(searchQuery);
        setData(response);
        onSuccess?.(response);
      } catch (err) {
        const error = err instanceof Error ? err : new Error('Search failed');
        setError(error);
        onError?.(error);
      } finally {
        setLoading(false);
      }
    },
    [enabled, onSuccess, onError]
  );

  return { data, loading, error, search };
}
```

---

## 🏗️ State Management (Golden Pattern)

### Zustand Store Structure

```typescript
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';

interface SearchStore {
  // State
  query: string;
  results: SearchResponse | null;
  history: string[];
  filters: SearchFilters;

  // Actions
  setQuery: (query: string) => void;
  setResults: (results: SearchResponse) => void;
  addToHistory: (query: string) => void;
  setFilters: (filters: SearchFilters) => void;
  clearSearch: () => void;
}

export const useSearchStore = create<SearchStore>()(
  devtools(
    persist(
      (set, get) => ({
        query: '',
        results: null,
        history: [],
        filters: {},

        setQuery: (query) => set({ query }),
        setResults: (results) => set({ results }),
        addToHistory: (query) => {
          const { history } = get();
          const newHistory = [query, ...history.filter((item) => item !== query)].slice(0, 10);
          set({ history: newHistory });
        },
        setFilters: (filters) => set({ filters }),
        clearSearch: () => set({ query: '', results: null }),
      }),
      {
        name: 'search-store',
        partialize: (state) => ({ history: state.history }),
      }
    )
  )
);
```

---

## 🔧 Utility Functions (Golden Pattern)

### Pure Function Standards

```typescript
/**
 * Validates and formats search query
 * @param query - Raw search query
 * @returns Validated query or null if invalid
 */
export const validateSearchQuery = (query: string): string | null => {
  if (!query || typeof query !== 'string') return null;

  const trimmed = query.trim();
  if (trimmed.length < 2) return null;
  if (trimmed.length > 200) return null;

  // Remove potentially harmful characters
  return trimmed.replace(/[<>]/g, '');
};

/**
 * Calculates relevance score with proper weighting
 * @param result - Search result item
 * @param query - Search query
 * @returns Normalized relevance score (0-1)
 */
export const calculateRelevanceScore = (result: SearchResult, query: string): number => {
  const queryLower = query.toLowerCase();
  let score = 0;

  // Title match (40% weight)
  const title = (result.name || result.client_name || '').toLowerCase();
  if (title.includes(queryLower)) score += 0.4;

  // Description match (30% weight)
  const description = (result.description || result.daily_notes || '').toLowerCase();
  if (description.includes(queryLower)) score += 0.3;

  // Category match (20% weight)
  if (result.category.toLowerCase().includes(queryLower)) score += 0.2;

  // Recency bonus (10% weight)
  if (result.created_at) {
    const daysOld = (Date.now() - new Date(result.created_at).getTime()) / (1000 * 60 * 60 * 24);
    if (daysOld < 7) score += 0.1;
  }

  return Math.min(score, 1);
};
```

---

## 🎨 Styling Standards (Golden Pattern)

### Tailwind CSS Patterns

```typescript
// Component variants using cva (class variance authority)
import { cva, type VariantProps } from 'class-variance-authority';

const searchResultVariants = cva('rounded-lg border p-4 transition-all duration-200', {
  variants: {
    variant: {
      default: 'bg-white border-gray-200 hover:shadow-md',
      highlighted: 'bg-blue-50 border-blue-200 hover:shadow-lg',
      selected: 'bg-blue-100 border-blue-300 shadow-md',
    },
    size: {
      sm: 'p-3 text-sm',
      default: 'p-4',
      lg: 'p-6 text-lg',
    },
  },
  defaultVariants: {
    variant: 'default',
    size: 'default',
  },
});

interface SearchResultProps extends VariantProps<typeof searchResultVariants> {
  result: SearchResult;
  onSelect?: (result: SearchResult) => void;
}

export const SearchResult: FC<SearchResultProps> = ({ result, variant, size, onSelect }) => {
  return (
    <div className={searchResultVariants({ variant, size })} onClick={() => onSelect?.(result)}>
      {/* Result content */}
    </div>
  );
};
```

---

## 🔒 Security Standards (Golden Pattern)

### Input Validation & Sanitization

```typescript
import { z } from 'zod';

// Comprehensive validation schemas
const SearchQuerySchema = z.object({
  query: z
    .string()
    .min(2, 'Search query must be at least 2 characters')
    .max(200, 'Search query too long')
    .regex(/^[a-zA-Z0-9\s\-_'.]+$/, 'Invalid characters in search query')
    .transform((val) => val.trim()),

  categories: z
    .array(z.enum(['workouts', 'exercises', 'nutrition', 'progress', 'clients']))
    .optional(),

  limit: z.number().int().min(1).max(100).optional().default(50),
});

// Secure API handler
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const validatedData = SearchQuerySchema.parse(body);

    // Sanitize query to prevent XSS
    const sanitizedQuery = validatedData.query
      .replace(/[<>]/g, '')
      .replace(/javascript:/gi, '')
      .replace(/on\w+\s*=/gi, '');

    const results = await performSearch(sanitizedQuery, validatedData);

    return Response.json({
      success: true,
      data: results,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return Response.json(
        {
          success: false,
          error: 'Validation failed',
          details: error.errors.map((err) => ({
            field: err.path.join('.'),
            message: err.message,
          })),
        },
        { status: 400 }
      );
    }

    // Log security events
    logger.error('Search validation failed', {
      error: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString(),
      ip: request.headers.get('x-forwarded-for') || 'unknown',
    });

    return Response.json({ success: false, error: 'Internal server error' }, { status: 500 });
  }
}
```

---

## 🧪 Testing Standards (Golden Pattern)

### Component Testing

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { UniversalSearch } from './UniversalSearch';

// Mock external dependencies
vi.mock('@/integrations/supabase/client', () => ({
  supabase: {
    functions: {
      invoke: vi.fn(),
    },
  },
}));

vi.mock('@/hooks/useToast', () => ({
  useToast: () => ({
    toast: vi.fn(),
  }),
}));

describe('UniversalSearch', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders search input and filters', () => {
    render(<UniversalSearch config={{ categories: ['workouts', 'exercises'] }} />);

    expect(screen.getByLabelText('Universal search')).toBeInTheDocument();
    expect(screen.getByText('Filters:')).toBeInTheDocument();
  });

  it('performs search when query is entered', async () => {
    const mockResults = {
      totalResults: 2,
      results: [
        { id: '1', category: 'workouts', type: 'strength', relevanceScore: 0.9 },
        { id: '2', category: 'exercises', type: 'cardio', relevanceScore: 0.8 },
      ],
      categories: ['workouts', 'exercises'],
      suggestions: [],
    };

    vi.mocked(supabase.functions.invoke).mockResolvedValueOnce({
      data: mockResults,
      error: null,
    });

    render(<UniversalSearch config={{ categories: ['workouts', 'exercises'] }} />);

    const input = screen.getByLabelText('Universal search');
    fireEvent.change(input, { target: { value: 'test query' } });

    await waitFor(() => {
      expect(supabase.functions.invoke).toHaveBeenCalledWith('universal-search', {
        body: { query: 'test query', limit: 50 },
      });
    });
  });

  it('handles search errors gracefully', async () => {
    const mockToast = vi.fn();
    vi.mocked(useToast).mockReturnValue({ toast: mockToast });

    vi.mocked(supabase.functions.invoke).mockRejectedValueOnce(
      new Error('Search service unavailable')
    );

    render(<UniversalSearch config={{ categories: ['workouts'] }} />);

    const input = screen.getByLabelText('Universal search');
    fireEvent.change(input, { target: { value: 'error query' } });

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith({
        title: 'Search Error',
        description: 'Could not perform search. Please try again.',
        variant: 'destructive',
      });
    });
  });
});
```

---

## 📊 Performance Standards (Golden Pattern)

### Optimization Techniques

```typescript
// Memoization for expensive calculations
const processedResults = useMemo(() => {
  return (
    searchResults?.results.map((result) => ({
      ...result,
      formattedScore: Math.round(result.relevanceScore * 100),
      displayTitle: result.name || result.client_name || 'Untitled',
      categoryColor: CATEGORY_CONFIG[result.category]?.color || 'text-gray-500',
    })) || []
  );
}, [searchResults]);

// Debounced search to prevent excessive API calls
const debouncedSearch = useDebounce(searchQuery, 300);

// Virtual scrolling for large lists
import { FixedSizeList } from 'react-window';

const VirtualizedSearchResults: FC<{ results: SearchResult[] }> = ({ results }) => {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style}>
      <SearchResultItem result={results[index]} />
    </div>
  );

  return (
    <FixedSizeList height={600} itemCount={results.length} itemSize={120} width="100%">
      {Row}
    </FixedSizeList>
  );
};

// Code splitting for heavy components
const HeavyAnalyticsComponent = lazy(() =>
  import('./AnalyticsComponent').then((module) => ({
    default: module.AnalyticsComponent,
  }))
);
```

---

## 🚀 Deployment & Infrastructure (Golden Pattern)

### Environment Configuration

```typescript
// Environment validation
const envSchema = z.object({
  VITE_SUPABASE_URL: z.string().url(),
  VITE_SUPABASE_ANON_KEY: z.string().min(10),
  VITE_STRIPE_PUBLIC_KEY: z.string().startsWith('pk_'),
  VITE_ENVIRONMENT: z.enum(['development', 'staging', 'production']),
  VITE_ENABLE_DEBUG_MODE: z.string().transform((val) => val === 'true'),
});

export const config = envSchema.parse(import.meta.env);

// Feature flags with proper typing
interface FeatureFlags {
  enableSearch: boolean;
  enableAdvancedFiltering: boolean;
  enableRealTimeUpdates: boolean;
  maxSearchResults: number;
}

export const featureFlags: Record<string, FeatureFlags> = {
  development: {
    enableSearch: true,
    enableAdvancedFiltering: true,
    enableRealTimeUpdates: true,
    maxSearchResults: 100,
  },
  staging: {
    enableSearch: true,
    enableAdvancedFiltering: true,
    enableRealTimeUpdates: false,
    maxSearchResults: 50,
  },
  production: {
    enableSearch: true,
    enableAdvancedFiltering: false,
    enableRealTimeUpdates: false,
    maxSearchResults: 50,
  },
};
```

---

## 📋 Quality Checklist (Golden Standard)

### Before Committing Code

- [ ] Functions are < 20 lines with single responsibility
- [ ] All functions and variables have self-documenting names
- [ ] TypeScript strict mode compliance (zero `any` types)
- [ ] Comprehensive JSDoc for public APIs
- [ ] Error handling for all async operations
- [ ] Security validation for all user inputs
- [ ] Performance optimizations applied (memo, debounce, etc.)
- [ ] Accessibility standards met (ARIA labels, keyboard navigation)
- [ ] Unit tests with > 80% coverage
- [ ] Integration tests for API endpoints
- [ ] No console.logs or debugging code
- [ ] No secrets or sensitive data in code

### Code Review Standards

- [ ] Follows established patterns and conventions
- [ ] Business logic is separated from UI components
- [ ] State management is properly implemented
- [ ] External dependencies are properly mocked in tests
- [ ] Documentation is updated if needed
- [ ] Performance implications are considered
- [ ] Security vulnerabilities are addressed
- [ ] Mobile responsiveness is verified

---

## 🎯 Success Metrics (Golden Standard)

### Performance Targets

- **Bundle Size**: < 200KB for initial load
- **Time to Interactive**: < 3 seconds
- **API Response Time**: < 200ms for search operations
- **Error Rate**: < 0.1% for user-facing operations
- **Test Coverage**: > 80% for critical paths
- **Accessibility Score**: 100% on Lighthouse

### Code Quality Metrics

- **Cyclomatic Complexity**: < 10 per function
- **Duplication**: < 3% across codebase
- **Technical Debt**: < 5% of total development time
- **Code Review Time**: < 24 hours average
- **Deployment Frequency**: Daily with zero downtime

---

## 🏆 Living It Iconic Implementation

This golden template embodies the "Living It Iconic" philosophy through:

1. **Freedom to Scale**: Modular architecture supports business growth
2. **Innovation Ready**: AI integration points and extensible patterns
3. **Quality Excellence**: Enterprise-grade reliability and performance
4. **Mindset Transformation**: Clean, maintainable code that inspires
5. **Forward-Thinking**: Future-proof architecture with modern patterns

**Remember**: Every line of code should make the next developer's life easier. Build systems that
scale, perform, and inspire. This is how we Live It Iconic through technology excellence.

---

_This template is abstracted from the REPZ Platform - a production system handling millions of
requests with 99.9% uptime and enterprise-grade security._
