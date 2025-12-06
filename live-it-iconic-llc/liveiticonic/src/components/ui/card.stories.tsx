import type { Meta, StoryObj } from '@storybook/react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './card';
import { Button } from './button';

const meta: Meta<typeof Card> = {
  title: 'UI/Card',
  component: Card,
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
type Story = StoryObj<typeof Card>;

export const Default: Story = {
  render: () => (
    <Card>
      <CardHeader>
        <CardTitle>Premium Apparel</CardTitle>
        <CardDescription>Luxury lifestyle merchandise</CardDescription>
      </CardHeader>
      <CardContent>
        <p>Experience the pinnacle of luxury lifestyle products crafted for automotive enthusiasts.</p>
      </CardContent>
      <CardFooter>
        <Button variant="primary">Shop Now</Button>
      </CardFooter>
    </Card>
  ),
};

export const Featured: Story = {
  render: () => (
    <Card className="border-lii-gold/30">
      <CardHeader>
        <CardTitle className="text-lii-gold">Featured Collection</CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-lii-cloud/80">Limited edition items curated for discerning collectors.</p>
      </CardContent>
      <CardFooter className="flex gap-2">
        <Button variant="primary" size="sm">View</Button>
        <Button variant="secondary" size="sm">Details</Button>
      </CardFooter>
    </Card>
  ),
};

export const Multiple: Story = {
  render: () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {[1, 2, 3, 4].map((i) => (
        <Card key={i}>
          <CardHeader>
            <CardTitle>Product {i}</CardTitle>
          </CardHeader>
          <CardContent>Premium item</CardContent>
          <CardFooter>
            <Button variant="ghost" size="sm">Learn More</Button>
          </CardFooter>
        </Card>
      ))}
    </div>
  ),
};
