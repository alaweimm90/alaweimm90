import type { Meta, StoryObj } from '@storybook/react';
import { Alert, AlertDescription, AlertTitle } from './alert';

const meta: Meta<typeof Alert> = {
  title: 'UI/Alert',
  component: Alert,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Alert>;

export const Default: Story = {
  render: () => (
    <Alert>
      <AlertTitle>Heads up!</AlertTitle>
      <AlertDescription>
        You can add components to your project to enhance functionality.
      </AlertDescription>
    </Alert>
  ),
};

export const Success: Story = {
  render: () => (
    <Alert className="border-green-500 bg-green-500/10">
      <AlertTitle className="text-green-600">Success</AlertTitle>
      <AlertDescription className="text-green-600/80">
        Your order has been placed successfully. You will receive a confirmation email shortly.
      </AlertDescription>
    </Alert>
  ),
};

export const Warning: Story = {
  render: () => (
    <Alert className="border-yellow-500 bg-yellow-500/10">
      <AlertTitle className="text-yellow-600">Warning</AlertTitle>
      <AlertDescription className="text-yellow-600/80">
        Only 2 items left in stock. Order soon to avoid missing out on this collection.
      </AlertDescription>
    </Alert>
  ),
};

export const Error: Story = {
  render: () => (
    <Alert className="border-red-500 bg-red-500/10">
      <AlertTitle className="text-red-600">Error</AlertTitle>
      <AlertDescription className="text-red-600/80">
        Something went wrong. Please try again or contact support.
      </AlertDescription>
    </Alert>
  ),
};

export const Info: Story = {
  render: () => (
    <Alert className="border-blue-500 bg-blue-500/10">
      <AlertTitle className="text-blue-600">Information</AlertTitle>
      <AlertDescription className="text-blue-600/80">
        New collection now available. Shop the latest Heritage and Performance lines.
      </AlertDescription>
    </Alert>
  ),
};
