/**
 * Admin Products Page
 * Product management and CRUD operations
 */
import React, { useState } from 'react';
import { Plus } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { ProductTable } from '@/components/admin/Products/ProductTable';
import { ProductForm } from '@/components/admin/Products/ProductForm';
import { ProductFilters } from '@/components/admin/Products/ProductFilters';

const AdminProducts: React.FC = () => {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingProduct, setEditingProduct] = useState<unknown>(null);

  const handleEdit = (product: unknown) => {
    setEditingProduct(product);
    setIsDialogOpen(true);
  };

  const handleDelete = (productId: string) => {
    if (confirm('Are you sure you want to delete this product?')) {
      console.log('Deleting product:', productId);
      // In production, this would make an API call
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-display text-lii-bg">Products</h1>
          <p className="text-lii-ash">Manage your product catalog</p>
        </div>
        <Button onClick={() => {
          setEditingProduct(null);
          setIsDialogOpen(true);
        }} className="flex items-center gap-2">
          <Plus size={20} />
          Add Product
        </Button>
      </div>

      {/* Filters */}
      <ProductFilters />

      {/* Product Table */}
      <ProductTable onEdit={handleEdit} onDelete={handleDelete} />

      {/* Product Form Dialog */}
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {editingProduct ? 'Edit Product' : 'Add New Product'}
            </DialogTitle>
            <DialogDescription>Manage details, pricing, and inventory for your catalog.</DialogDescription>
          </DialogHeader>
          <ProductForm
            product={editingProduct}
            onSuccess={() => {
              setIsDialogOpen(false);
              setEditingProduct(null);
            }}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default AdminProducts;
