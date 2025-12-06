/**
 * Product Filters Component
 * Filter controls for product management
 */
import { Search, Filter } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';

interface ProductFiltersProps {
  onSearchChange?: (query: string) => void;
  onCategoryChange?: (category: string) => void;
  onStatusChange?: (status: string) => void;
}

export function ProductFilters({
  onSearchChange,
  onCategoryChange,
  onStatusChange,
}: ProductFiltersProps) {
  return (
    <div className="bg-white rounded-lg border border-lii-cloud p-4 space-y-4 mb-6">
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex-1 min-w-64">
          <div className="relative">
            <Search className="absolute left-3 top-3 text-lii-ash" size={18} />
            <Input
              placeholder="Search products..."
              onChange={(e) => onSearchChange?.(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        <select
          onChange={(e) => onCategoryChange?.(e.target.value)}
          className="px-4 py-2 border border-lii-cloud rounded-lg focus:outline-none focus:ring-2 focus:ring-lii-gold text-sm"
        >
          <option value="">All Categories</option>
          <option value="tees">Tees</option>
          <option value="hoodies">Hoodies</option>
          <option value="caps">Caps</option>
          <option value="outerwear">Outerwear</option>
          <option value="accessories">Accessories</option>
        </select>

        <select
          onChange={(e) => onStatusChange?.(e.target.value)}
          className="px-4 py-2 border border-lii-cloud rounded-lg focus:outline-none focus:ring-2 focus:ring-lii-gold text-sm"
        >
          <option value="">All Status</option>
          <option value="in-stock">In Stock</option>
          <option value="low-stock">Low Stock</option>
          <option value="out-of-stock">Out of Stock</option>
        </select>

        <Button
          variant="ghost"
          size="sm"
          className="flex items-center gap-2"
        >
          <Filter size={18} />
          More Filters
        </Button>
      </div>
    </div>
  );
}
