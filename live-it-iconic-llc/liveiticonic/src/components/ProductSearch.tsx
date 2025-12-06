import React, { useState, useEffect, useCallback } from 'react';
import { Search, X, Clock } from 'lucide-react';
import { Product } from '@/types/product';

interface ProductSearchProps {
  products: Product[];
  onSearch: (results: Product[]) => void;
  placeholder?: string;
}

/**
 * ProductSearch component provides real-time search with recent search history
 *
 * Features:
 * - Real-time fuzzy search across product names and descriptions
 * - Recent searches stored in localStorage
 * - Clear recent searches functionality
 * - Search suggestions as you type
 *
 * @component
 * @param {ProductSearchProps} props - Component props
 * @param {Product[]} props.products - Array of products to search
 * @param {Function} props.onSearch - Callback fired with search results
 * @param {string} [props.placeholder] - Input placeholder text
 *
 * @example
 * <ProductSearch
 *   products={productList}
 *   onSearch={(results) => setSearchResults(results)}
 *   placeholder="Search tees, hoodies..."
 * />
 */
const ProductSearch: React.FC<ProductSearchProps> = ({
  products,
  onSearch,
  placeholder = 'Search products...',
}) => {
  const [query, setQuery] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [suggestions, setSuggestions] = useState<Product[]>([]);

  const RECENT_SEARCHES_KEY = 'lii-recent-searches';
  const MAX_RECENT_SEARCHES = 5;

  // Load recent searches on mount
  useEffect(() => {
    const saved = localStorage.getItem(RECENT_SEARCHES_KEY);
    if (saved) {
      try {
        setRecentSearches(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load recent searches', e);
      }
    }
  }, []);

  // Fuzzy search implementation
  const fuzzySearch = useCallback(
    (searchQuery: string) => {
      if (!searchQuery.trim()) {
        setSuggestions([]);
        onSearch(products);
        return;
      }

      const query = searchQuery.toLowerCase();
      const results = products.filter(product => {
        const name = product.name.toLowerCase();
        const description = product.description.toLowerCase();
        const tagline = product.tagline.toLowerCase();

        return name.includes(query) || description.includes(query) || tagline.includes(query);
      });

      setSuggestions(results.slice(0, 8));
      onSearch(results);
    },
    [products, onSearch]
  );

  const handleSearch = useCallback(
    (value: string) => {
      setQuery(value);
      fuzzySearch(value);
    },
    [fuzzySearch]
  );

  const handleSelectSuggestion = (product: Product) => {
    setQuery(product.name);
    addToRecentSearches(product.name);
    setSuggestions([]);
    setIsOpen(false);
    onSearch([product]);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      addToRecentSearches(query);
      setSuggestions([]);
      setIsOpen(false);
    }
  };

  const addToRecentSearches = (search: string) => {
    setRecentSearches(prev => {
      const updated = [search, ...prev.filter(s => s !== search)].slice(0, MAX_RECENT_SEARCHES);
      localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(updated));
      return updated;
    });
  };

  const clearRecentSearches = () => {
    setRecentSearches([]);
    localStorage.removeItem(RECENT_SEARCHES_KEY);
  };

  const clearSearch = () => {
    setQuery('');
    setSuggestions([]);
    onSearch(products);
  };

  return (
    <div className="relative w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-lii-ash pointer-events-none" />
          <input
            type="text"
            value={query}
            onChange={e => handleSearch(e.target.value)}
            onFocus={() => setIsOpen(true)}
            onBlur={() => setTimeout(() => setIsOpen(false), 200)}
            placeholder={placeholder}
            className="w-full pl-10 pr-10 py-2 bg-lii-charcoal/20 border border-lii-gold/20 rounded-lg text-lii-cloud placeholder:text-lii-ash/50 focus:outline-none focus:border-lii-gold/50 transition-colors"
          />
          {query && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-lii-ash hover:text-lii-gold transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      </form>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-2 bg-lii-ink border border-lii-gold/20 rounded-lg shadow-lg z-50 max-h-[500px] overflow-y-auto">
          {/* Suggestions */}
          {query && suggestions.length > 0 && (
            <div className="border-b border-lii-gold/10">
              <div className="px-4 py-2 text-xs uppercase text-lii-ash font-ui font-medium">
                Suggestions
              </div>
              <div className="space-y-0">
                {suggestions.map(product => (
                  <button
                    key={product.id}
                    type="button"
                    onClick={() => handleSelectSuggestion(product)}
                    className="w-full text-left px-4 py-2 hover:bg-lii-charcoal/20 transition-colors"
                  >
                    <p className="text-lii-cloud font-ui font-medium">{product.name}</p>
                    <p className="text-lii-ash font-ui text-sm line-clamp-1">{product.tagline}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Recent Searches */}
          {!query && recentSearches.length > 0 && (
            <div className="px-4 py-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-lii-ash" />
                  <p className="text-xs uppercase text-lii-ash font-ui font-medium">Recent Searches</p>
                </div>
                <button
                  type="button"
                  onClick={clearRecentSearches}
                  className="text-xs text-lii-ash hover:text-lii-gold transition-colors"
                >
                  Clear
                </button>
              </div>
              <div className="space-y-1">
                {recentSearches.map((search, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => {
                      setQuery(search);
                      handleSearch(search);
                    }}
                    className="w-full text-left px-2 py-1 text-lii-ash font-ui text-sm hover:text-lii-gold hover:bg-lii-charcoal/20 rounded transition-colors"
                  >
                    {search}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* No Results */}
          {query && suggestions.length === 0 && (
            <div className="px-4 py-6 text-center">
              <p className="text-lii-ash font-ui">No products found matching "{query}"</p>
            </div>
          )}

          {/* Empty State */}
          {!query && recentSearches.length === 0 && (
            <div className="px-4 py-6 text-center">
              <p className="text-lii-ash font-ui text-sm">Start typing to search</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ProductSearch;
