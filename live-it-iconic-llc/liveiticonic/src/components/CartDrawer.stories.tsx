import type { Meta, StoryObj } from '@storybook/react';
import { CartDrawer } from './CartDrawer';
import { useState } from 'react';

const meta: Meta<typeof CartDrawer> = {
  title: 'Components/CartDrawer',
  component: CartDrawer,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof CartDrawer>;

export const Open: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(true);
    return <CartDrawer isOpen={isOpen} onClose={() => setIsOpen(false)} />;
  },
};

export const Closed: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    return <CartDrawer isOpen={isOpen} onClose={() => setIsOpen(false)} />;
  },
};

export const Interactive: Story = {
  render: () => {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <div className="space-y-4">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="px-4 py-2 bg-lii-gold text-lii-bg rounded hover:opacity-90"
        >
          {isOpen ? 'Close Cart' : 'Open Cart'}
        </button>
        <CartDrawer isOpen={isOpen} onClose={() => setIsOpen(false)} />
      </div>
    );
  },
};
