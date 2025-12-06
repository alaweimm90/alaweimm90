import React, { memo, useCallback } from 'react';
import { ProductFilter, Product } from '@/types/product';
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
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';

interface FilterSidebarProps {
  filters: ProductFilter;
  onChange: (filters: Partial<ProductFilter>) => void;
  categories?: string[];
  products?: Product[];
  onReset?: () => void;
}

/**
 * FilterSidebar component provides comprehensive product filtering
 *
 * Includes filters for:
 * - Search (text input)
 * - Category (dropdown)
 * - Collection (Heritage, Performance, Urban)
 * - Color (from available products)
 * - Size (XS-XXL)
 * - Price range (slider)
 * - Stock status (checkboxes)
 * - Sort options
 *
 * @component
 * @param {FilterSidebarProps} props - Component props
 * @param {ProductFilter} props.filters - Current filter state
 * @param {Function} props.onChange - Callback fired with updated filters
 * @param {string[]} [props.categories] - Available categories
 * @param {Product[]} [props.products] - Products for color extraction
 * @param {Function} [props.onReset] - Reset filters callback
 */
const FilterSidebarComponent: React.FC<FilterSidebarProps> = ({
  filters,
  onChange,
  categories = [],
  products = [],
  onReset,
}) => {
  // Extract unique colors from products
  const uniqueColors = React.useMemo(() => {
    const colors = new Set<string>();
    products.forEach(p => {
      p.colors.forEach(c => {
        colors.add(c.name);
      });
    });
    return Array.from(colors).sort();
  }, [products]);

  // Extract unique sizes from products
  const uniqueSizes = React.useMemo(() => {
    const sizes = new Set<string>();
    products.forEach(p => {
      p.sizes.forEach(s => {
        sizes.add(s.label);
      });
    });
    return Array.from(sizes).sort((a, b) => {
      const order = ['XS', 'S', 'M', 'L', 'XL', 'XXL', 'ONE'];
      return order.indexOf(a) - order.indexOf(b);
    });
  }, [products]);

  const handlePriceRangeChange = useCallback(
    (values: number[]) => {
      onChange({ priceRange: [values[0], values[1]] });
    },
    [onChange]
  );

  const handleReset = useCallback(() => {
    onReset?.();
  }, [onReset]);

  return (
    <Card className="bg-lii-ink border-lii-gold/10 sticky top-32">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lii-cloud font-display">Filters</CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="text-lii-ash hover:text-lii-gold text-xs h-auto p-0"
          >
            Reset
          </Button>
        </div>
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

        {/* Collection Filter */}
        <div className="space-y-2">
          <Label className="text-lii-ash">Collection</Label>
          <Select
            value={filters.collection || 'all'}
            onValueChange={value =>
              onChange({ collection: value === 'all' ? undefined : (value as any), page: 1 })
            }
          >
            <SelectTrigger className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud">
              <SelectValue placeholder="All Collections" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Collections</SelectItem>
              <SelectItem value="heritage">Heritage</SelectItem>
              <SelectItem value="performance">Performance</SelectItem>
              <SelectItem value="urban">Urban</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Color Filter */}
        {uniqueColors.length > 0 && (
          <>
            <div className="space-y-3">
              <Label className="text-lii-ash">Color</Label>
              <div className="space-y-2">
                {uniqueColors.map(color => (
                  <div key={color} className="flex items-center gap-2">
                    <Checkbox
                      id={`color-${color}`}
                      checked={filters.color === color}
                      onCheckedChange={checked =>
                        onChange({ color: checked ? color : undefined, page: 1 })
                      }
                      className="border-lii-gold/40"
                    />
                    <Label
                      htmlFor={`color-${color}`}
                      className="text-lii-ash font-ui text-sm cursor-pointer"
                    >
                      {color}
                    </Label>
                  </div>
                ))}
              </div>
            </div>

            <Separator className="bg-lii-gold/10" />
          </>
        )}

        {/* Size Filter */}
        {uniqueSizes.length > 0 && (
          <>
            <div className="space-y-3">
              <Label className="text-lii-ash">Size</Label>
              <div className="grid grid-cols-3 gap-2">
                {uniqueSizes.map(size => (
                  <button
                    key={size}
                    onClick={() => onChange({ size: filters.size === size ? undefined : size, page: 1 })}
                    className={`py-2 px-2 rounded text-sm font-ui transition-all border ${
                      filters.size === size
                        ? 'border-lii-gold bg-lii-gold/10 text-lii-gold'
                        : 'border-lii-gold/20 text-lii-ash hover:border-lii-gold/40'
                    }`}
                  >
                    {size}
                  </button>
                ))}
              </div>
            </div>

            <Separator className="bg-lii-gold/10" />
          </>
        )}

        {/* Price Range */}
        <div className="space-y-2">
          <Label className="text-lii-ash">Price Range</Label>
          <div className="px-2">
            <Slider
              value={filters.priceRange || [0, 500]}
              onValueChange={handlePriceRangeChange}
              max={500}
              min={0}
              step={10}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-lii-ash mt-2">
              <span>${filters.priceRange?.[0] || 0}</span>
              <span>${filters.priceRange?.[1] || 500}</span>
            </div>
          </div>
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Stock Status */}
        <div className="space-y-3">
          <Label className="text-lii-ash">Availability</Label>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Checkbox
                id="in-stock"
                checked={filters.inStockOnly || false}
                onCheckedChange={checked =>
                  onChange({ inStockOnly: checked ? true : false, page: 1 })
                }
                className="border-lii-gold/40"
              />
              <Label htmlFor="in-stock" className="text-lii-ash font-ui text-sm cursor-pointer">
                In Stock Only
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="new-arrivals"
                checked={filters.newArrivalsOnly || false}
                onCheckedChange={checked =>
                  onChange({ newArrivalsOnly: checked ? true : false, page: 1 })
                }
                className="border-lii-gold/40"
              />
              <Label htmlFor="new-arrivals" className="text-lii-ash font-ui text-sm cursor-pointer">
                New Arrivals
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <Checkbox
                id="featured"
                checked={filters.featuredOnly || false}
                onCheckedChange={checked =>
                  onChange({ featuredOnly: checked ? true : false, page: 1 })
                }
                className="border-lii-gold/40"
              />
              <Label htmlFor="featured" className="text-lii-ash font-ui text-sm cursor-pointer">
                Featured Items
              </Label>
            </div>
          </div>
        </div>

        <Separator className="bg-lii-gold/10" />

        {/* Sort By */}
        <div className="space-y-2">
          <Label className="text-lii-ash">Sort By</Label>
          <Select
            value={filters.sortBy || 'newest'}
            onValueChange={value => onChange({ sortBy: value as any, page: 1 })}
          >
            <SelectTrigger className="bg-lii-charcoal/20 border-lii-gold/20 text-lii-cloud">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="newest">Newest First</SelectItem>
              <SelectItem value="featured">Featured</SelectItem>
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

const FilterSidebar = memo(FilterSidebarComponent);
FilterSidebar.displayName = 'FilterSidebar';

export default FilterSidebar;
export { FilterSidebarComponent };
