import type { Meta, StoryObj } from '@storybook/react';
import { Label } from './label';

const meta: Meta<typeof Label> = {
  title: 'UI/Label',
  component: Label,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Label>;

export const Default: Story = {
  args: {
    htmlFor: 'email',
    children: 'Email Address',
  },
};

export const Required: Story = {
  args: {
    htmlFor: 'name',
    children: 'Full Name *',
  },
};

export const WithInput: Story = {
  render: () => (
    <div className="space-y-2 max-w-sm">
      <Label htmlFor="email">Email Address</Label>
      <input
        id="email"
        type="email"
        placeholder="you@example.com"
        className="w-full px-3 py-2 border border-lii-gold/20 rounded-lg bg-lii-charcoal text-lii-cloud"
      />
    </div>
  ),
};

export const FormSection: Story = {
  render: () => (
    <div className="space-y-6 max-w-sm">
      <div className="space-y-2">
        <Label htmlFor="firstName">First Name</Label>
        <input
          id="firstName"
          type="text"
          placeholder="John"
          className="w-full px-3 py-2 border border-lii-gold/20 rounded-lg bg-lii-charcoal text-lii-cloud"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="lastName">Last Name</Label>
        <input
          id="lastName"
          type="text"
          placeholder="Doe"
          className="w-full px-3 py-2 border border-lii-gold/20 rounded-lg bg-lii-charcoal text-lii-cloud"
        />
      </div>
      <div className="space-y-2">
        <Label htmlFor="email">Email</Label>
        <input
          id="email"
          type="email"
          placeholder="john@example.com"
          className="w-full px-3 py-2 border border-lii-gold/20 rounded-lg bg-lii-charcoal text-lii-cloud"
        />
      </div>
    </div>
  ),
};
