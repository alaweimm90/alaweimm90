/**
 * Product Form Component
 * Form for creating and editing products
 */
import { useState } from 'react';
import { Product } from '@/types/product';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';

interface ProductFormProps {
  product?: Product | null;
  onSuccess?: () => void;
}

export function ProductForm({ product, onSuccess }: ProductFormProps) {
  const [formData, setFormData] = useState({
    name: product?.name || '',
    price: product?.price || 0,
    stock: product?.availableQuantity || 0,
    category: product?.category || 'tees',
    description: product?.description || '',
  });

  const handleChange = (
    e: React.ChangeEvent<
      HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
    >
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'price' || name === 'stock' ? Number(value) : value,
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // In production, this would make an API call
    console.log('Form submitted:', formData);
    onSuccess?.();
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-4">
        <div>
          <Label htmlFor="name">Product Name</Label>
          <Input
            id="name"
            name="name"
            value={formData.name}
            onChange={handleChange}
            placeholder="Enter product name"
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <Label htmlFor="price">Price ($)</Label>
            <Input
              id="price"
              name="price"
              type="number"
              step="0.01"
              value={formData.price}
              onChange={handleChange}
              placeholder="0.00"
              required
            />
          </div>
          <div>
            <Label htmlFor="stock">Stock Quantity</Label>
            <Input
              id="stock"
              name="stock"
              type="number"
              value={formData.stock}
              onChange={handleChange}
              placeholder="0"
              required
            />
          </div>
        </div>

        <div>
          <Label htmlFor="category">Category</Label>
          <select
            id="category"
            name="category"
            value={formData.category}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-lii-cloud rounded-lg focus:outline-none focus:ring-2 focus:ring-lii-gold"
          >
            <option value="tees">Tees</option>
            <option value="hoodies">Hoodies</option>
            <option value="caps">Caps</option>
            <option value="outerwear">Outerwear</option>
            <option value="accessories">Accessories</option>
          </select>
        </div>

        <div>
          <Label htmlFor="description">Description</Label>
          <Textarea
            id="description"
            name="description"
            value={formData.description}
            onChange={handleChange}
            placeholder="Enter product description"
            rows={4}
          />
        </div>
      </div>

      <div className="flex gap-3 justify-end">
        <Button type="button" variant="outline" onClick={onSuccess}>
          Cancel
        </Button>
        <Button type="submit">
          {product ? 'Update Product' : 'Create Product'}
        </Button>
      </div>
    </form>
  );
}
