import type { Meta, StoryObj } from '@storybook/react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './tabs';

const meta: Meta<typeof Tabs> = {
  title: 'UI/Tabs',
  component: Tabs,
  tags: ['autodocs'],
  parameters: {
    backgrounds: {
      default: 'dark',
    },
  },
};

export default meta;
type Story = StoryObj<typeof Tabs>;

export const Default: Story = {
  render: () => (
    <Tabs defaultValue="tab1" className="w-full max-w-md">
      <TabsList className="grid w-full grid-cols-3">
        <TabsTrigger value="tab1">Collections</TabsTrigger>
        <TabsTrigger value="tab2">Details</TabsTrigger>
        <TabsTrigger value="tab3">Reviews</TabsTrigger>
      </TabsList>
      <TabsContent value="tab1" className="p-4">
        <h3 className="font-semibold mb-2">Featured Collections</h3>
        <p className="text-sm text-lii-ash">Browse our curated product collections.</p>
      </TabsContent>
      <TabsContent value="tab2" className="p-4">
        <h3 className="font-semibold mb-2">Product Details</h3>
        <p className="text-sm text-lii-ash">Premium craftsmanship and specifications.</p>
      </TabsContent>
      <TabsContent value="tab3" className="p-4">
        <h3 className="font-semibold mb-2">Customer Reviews</h3>
        <p className="text-sm text-lii-ash">What our customers are saying.</p>
      </TabsContent>
    </Tabs>
  ),
};

export const WithContent: Story = {
  render: () => (
    <Tabs defaultValue="products" className="w-full">
      <TabsList className="grid w-full grid-cols-2">
        <TabsTrigger value="products">Products</TabsTrigger>
        <TabsTrigger value="about">About</TabsTrigger>
      </TabsList>
      <TabsContent value="products" className="p-6">
        <div className="space-y-2">
          <h3 className="text-lg font-semibold text-lii-gold">Heritage Collection</h3>
          <p className="text-lii-cloud/80">Timeless pieces crafted for discerning collectors.</p>
        </div>
      </TabsContent>
      <TabsContent value="about" className="p-6">
        <div className="space-y-2">
          <h3 className="text-lg font-semibold text-lii-gold">Our Story</h3>
          <p className="text-lii-cloud/80">Live It Iconic represents the pinnacle of luxury lifestyle.</p>
        </div>
      </TabsContent>
    </Tabs>
  ),
};
