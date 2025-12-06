import type { Meta, StoryObj } from '@storybook/react';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from './tooltip';

const meta: Meta<typeof Tooltip> = {
  title: 'UI/Tooltip',
  component: Tooltip,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof Tooltip>;

export const Default: Story = {
  render: () => (
    <Tooltip>
      <TooltipTrigger asChild>
        <button className="px-4 py-2 bg-lii-gold text-lii-bg rounded hover:opacity-90">
          Hover Me
        </button>
      </TooltipTrigger>
      <TooltipContent>
        <p>This is a helpful tooltip</p>
      </TooltipContent>
    </Tooltip>
  ),
};

export const OnIcon: Story = {
  render: () => (
    <Tooltip>
      <TooltipTrigger asChild>
        <button className="text-lii-gold hover:text-lii-gold/80">
          <span className="text-xl">?</span>
        </button>
      </TooltipTrigger>
      <TooltipContent>
        <p>Information about this feature</p>
      </TooltipContent>
    </Tooltip>
  ),
};

export const Multiple: Story = {
  render: () => (
    <div className="flex gap-4">
      <Tooltip>
        <TooltipTrigger asChild>
          <button className="px-4 py-2 bg-lii-gold text-lii-bg rounded">
            Save
          </button>
        </TooltipTrigger>
        <TooltipContent>Save your changes</TooltipContent>
      </Tooltip>

      <Tooltip>
        <TooltipTrigger asChild>
          <button className="px-4 py-2 border border-lii-gold text-lii-gold rounded">
            Cancel
          </button>
        </TooltipTrigger>
        <TooltipContent>Discard changes</TooltipContent>
      </Tooltip>
    </div>
  ),
};
