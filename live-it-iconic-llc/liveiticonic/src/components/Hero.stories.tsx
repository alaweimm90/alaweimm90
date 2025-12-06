import type { Meta, StoryObj } from '@storybook/react';
import { Hero } from './Hero';

const meta: Meta<typeof Hero> = {
  title: 'Components/Hero',
  component: Hero,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Hero>;

export const Default: Story = {
  render: () => (
    <div className="w-full">
      <Hero />
    </div>
  ),
};

export const FullPage: Story = {
  render: () => (
    <div className="w-full h-screen bg-lii-bg">
      <Hero />
    </div>
  ),
  parameters: {
    layout: 'fullscreen',
  },
};
