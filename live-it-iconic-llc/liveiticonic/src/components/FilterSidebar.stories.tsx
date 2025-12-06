import type { Meta, StoryObj } from '@storybook/react';
import { FilterSidebar } from './FilterSidebar';
import { useState } from 'react';
import { ProductFilter } from '@/types/product';

const meta: Meta<typeof FilterSidebar> = {
  title: 'Components/FilterSidebar',
  component: FilterSidebar,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-sm">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof FilterSidebar>;

const categories = ['All', 'Performance', 'Recovery', 'Training', 'Lifestyle'];

export const Default: Story = {
  render: () => {
    const [filters, setFilters] = useState<ProductFilter>({
      search: '',
      category: 'All',
      collection: undefined,
      colors: [],
      sizes: [],
      priceRange: [0, 300],
      inStockOnly: false,
      sortBy: 'featured',
    });

    return (
      <FilterSidebar
        filters={filters}
        onChange={(newFilters) => setFilters({ ...filters, ...newFilters })}
        categories={categories}
        onReset={() =>
          setFilters({
            search: '',
            category: 'All',
            collection: undefined,
            colors: [],
            sizes: [],
            priceRange: [0, 300],
            inStockOnly: false,
            sortBy: 'featured',
          })
        }
      />
    );
  },
};

export const WithFiltersApplied: Story = {
  render: () => {
    const [filters, setFilters] = useState<ProductFilter>({
      search: '',
      category: 'Performance',
      collection: 'Heritage',
      colors: ['Black', 'Gold'],
      sizes: ['M', 'L'],
      priceRange: [50, 150],
      inStockOnly: true,
      sortBy: 'price-low',
    });

    return (
      <FilterSidebar
        filters={filters}
        onChange={(newFilters) => setFilters({ ...filters, ...newFilters })}
        categories={categories}
        onReset={() =>
          setFilters({
            search: '',
            category: 'All',
            collection: undefined,
            colors: [],
            sizes: [],
            priceRange: [0, 300],
            inStockOnly: false,
            sortBy: 'featured',
          })
        }
      />
    );
  },
};
