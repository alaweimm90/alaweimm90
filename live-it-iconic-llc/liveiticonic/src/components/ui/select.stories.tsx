import type { Meta, StoryObj } from '@storybook/react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './select';

const meta: Meta<typeof Select> = {
  title: 'UI/Select',
  component: Select,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Select>;

export const Default: Story = {
  render: () => (
    <Select>
      <SelectTrigger className="w-full max-w-xs">
        <SelectValue placeholder="Select an option" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="option1">Option 1</SelectItem>
        <SelectItem value="option2">Option 2</SelectItem>
        <SelectItem value="option3">Option 3</SelectItem>
      </SelectContent>
    </Select>
  ),
};

export const Categories: Story = {
  render: () => (
    <Select>
      <SelectTrigger className="w-full max-w-xs">
        <SelectValue placeholder="Select a category" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="performance">Performance</SelectItem>
        <SelectItem value="heritage">Heritage</SelectItem>
        <SelectItem value="urban">Urban</SelectItem>
        <SelectItem value="lifestyle">Lifestyle</SelectItem>
      </SelectContent>
    </Select>
  ),
};

export const SortBy: Story = {
  render: () => (
    <Select>
      <SelectTrigger className="w-full max-w-xs">
        <SelectValue placeholder="Sort by" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="featured">Featured</SelectItem>
        <SelectItem value="newest">Newest</SelectItem>
        <SelectItem value="price-low">Price: Low to High</SelectItem>
        <SelectItem value="price-high">Price: High to Low</SelectItem>
        <SelectItem value="popular">Most Popular</SelectItem>
      </SelectContent>
    </Select>
  ),
};

export const WithDefault: Story = {
  render: () => (
    <Select defaultValue="xs">
      <SelectTrigger className="w-full max-w-xs">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="xs">Extra Small</SelectItem>
        <SelectItem value="sm">Small</SelectItem>
        <SelectItem value="md">Medium</SelectItem>
        <SelectItem value="lg">Large</SelectItem>
        <SelectItem value="xl">Extra Large</SelectItem>
      </SelectContent>
    </Select>
  ),
};
