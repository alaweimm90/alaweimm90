import type { Meta, StoryObj } from '@storybook/react';
import { Input } from './input';

const meta: Meta<typeof Input> = {
  title: 'UI/Input',
  component: Input,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  args: {
    placeholder: 'Enter text...',
    disabled: false,
  },
};

export default meta;
type Story = StoryObj<typeof Input>;

export const Default: Story = {
  args: {
    placeholder: 'Search products...',
  },
};

export const WithValue: Story = {
  args: {
    placeholder: 'Search products...',
    defaultValue: 'Heritage Polo',
  },
};

export const Disabled: Story = {
  args: {
    placeholder: 'Disabled input',
    disabled: true,
  },
};

export const Email: Story = {
  args: {
    type: 'email',
    placeholder: 'Enter your email',
  },
};

export const Search: Story = {
  args: {
    type: 'text',
    placeholder: 'Search our collection...',
  },
};

export const Multiple: Story = {
  render: () => (
    <div className="space-y-4 max-w-sm">
      <Input type="text" placeholder="Full Name" />
      <Input type="email" placeholder="Email Address" />
      <Input type="tel" placeholder="Phone Number" />
      <Input type="password" placeholder="Password" />
    </div>
  ),
};
