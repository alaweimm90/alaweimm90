import React, { memo, useCallback } from 'react';
import { ProductFilter } from '@/types/product';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Separator } from '@/components/ui/separator';

interface FilterPanelProps {
  filters: ProductFilter;
  onChange: (filters: Partial<ProductFilter>) => void;
  categories?: string[];
}

/**
 * FilterPanel component provides product filtering controls (price, category, search)
 *
 * Renders a card with price range slider, category select dropdown, and product search input.
 * Memoized for performance. Fires onChange callback with updated filters when controls change.
 *
 * @component
 * @param {FilterPanelProps} props - Component props
 * @param {ProductFilter} props.filters - Current filter state
 * @param {Function} props.onChange - Callback fired with updated filters
 * @param {string[]} [props.categories] - Available product categories for dropdown
 *
 * @example
 * <FilterPanel
 *   filters={currentFilters}
 *   onChange={(newFilters) => setFilters(newFilters)}
 *   categories={['Performance', 'Lifestyle']}
 * />
 *
 * @remarks
 * - Memoized to prevent unnecessary re-renders
 * - Price slider controls priceRange: [min, max]
 * - Category select updates category field
 * - Text input controls product search
 */
const FilterPanelComponent: React.FC<FilterPanelProps> = ({ filters, onChange, categories = [] }) => {
  const handlePriceRangeChange = useCallback(
    (values: number[]) => {
      onChange({ priceRange: [values[0], values[1]] });
    },
    [onChange]
  );

  return (
    <Card className="bg-lii-ink border-lii-gold/10">
      <CardHeader>
        <CardTitle className="text-lii-cloud font-display">Filters</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Search */}
        <div className="space-y-2">
          <Label htmlFor="search" className="text-lii-ash">
            Search
          </Label>
          <Input
            id="search"
            type="text"
            placeholder="Search products..."
            value={filters.search || ''}
            onChange={e => onChange({ search: e.target.value, page: 1 })}
            className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud placeholder:text-lii-ash/50"
          />
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Category Filter */}
        {categories.length > 0 && (
          <>
            <div className="space-y-2">
              <Label className="text-lii-ash">Category</Label>
              <Select
                value={filters.category || 'all'}
                onValueChange={value =>
                  onChange({ category: value === 'all' ? undefined : value, page: 1 })
                }
              >
                <SelectTrigger className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud">
                  <SelectValue placeholder="All Categories" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  {categories.map(category => (
                    <SelectItem key={category} value={category}>
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Separator className="bg-lii-gold/10" />
          </>
        )}

        {/* Price Range */}
        <div className="space-y-2">
          <Label className="text-lii-ash">Price Range</Label>
          <div className="px-2">
            <Slider
              value={filters.priceRange || [0, 1000]}
              onValueChange={handlePriceRangeChange}
              max={1000}
              min={0}
              step={10}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-lii-ash mt-2">
              <span>${filters.priceRange?.[0] || 0}</span>
              <span>${filters.priceRange?.[1] || 1000}</span>
            </div>
          </div>
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Sort By */}
        <div className="space-y-2">
          <Label className="text-lii-ash">Sort By</Label>
          <Select
            value={filters.sortBy || 'newest'}
            onValueChange={value => onChange({ sortBy: value as ProductFilter['sortBy'], page: 1 })}
          >
            <SelectTrigger className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="newest">Newest First</SelectItem>
              <SelectItem value="price-low">Price: Low to High</SelectItem>
              <SelectItem value="price-high">Price: High to Low</SelectItem>
              <SelectItem value="popular">Most Popular</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  );
};

const FilterPanel = memo(FilterPanelComponent);
FilterPanel.displayName = 'FilterPanel';

export default FilterPanel;
export { FilterPanelComponent };
