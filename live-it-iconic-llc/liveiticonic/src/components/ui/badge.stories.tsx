import type { Meta, StoryObj } from '@storybook/react';
import { Badge } from './badge';

const meta: Meta<typeof Badge> = {
  title: 'UI/Badge',
  component: Badge,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  argTypes: {
    variant: {
      control: 'select',
      options: ['default', 'secondary', 'destructive', 'outline'],
    },
  },
};

export default meta;
type Story = StoryObj<typeof Badge>;

export const Default: Story = {
  args: {
    children: 'Premium',
  },
};

export const Secondary: Story = {
  args: {
    variant: 'secondary',
    children: 'Limited Edition',
  },
};

export const Destructive: Story = {
  args: {
    variant: 'destructive',
    children: 'Out of Stock',
  },
};

export const Outline: Story = {
  args: {
    variant: 'outline',
    children: 'Featured',
  },
};

export const AllVariants: Story = {
  render: () => (
    <div className="flex gap-3 flex-wrap">
      <Badge>Default</Badge>
      <Badge variant="secondary">Secondary</Badge>
      <Badge variant="destructive">Destructive</Badge>
      <Badge variant="outline">Outline</Badge>
    </div>
  ),
};

export const WithStatus: Story = {
  render: () => (
    <div className="flex gap-3 flex-wrap">
      <Badge className="bg-green-600 text-white">In Stock</Badge>
      <Badge variant="destructive">Low Stock</Badge>
      <Badge className="bg-yellow-600 text-white">Pre-Order</Badge>
    </div>
  ),
};
