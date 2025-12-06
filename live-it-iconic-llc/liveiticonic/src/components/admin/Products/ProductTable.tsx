/**
 * Product Table Component
 * Displays products in a table with edit/delete/view actions
 */
import { Edit, Trash2, Eye } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { products } from '@/data/products';

interface ProductTableProps {
  onEdit?: (product: unknown) => void;
  onDelete?: (productId: string) => void;
}

export function ProductTable({ onEdit, onDelete }: ProductTableProps) {
  return (
    <div className="bg-white rounded-lg border border-lii-cloud overflow-hidden">
      <table className="w-full">
        <thead className="bg-lii-cloud/5 border-b border-lii-cloud/20">
          <tr>
            <th className="px-6 py-4 text-left text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Product
            </th>
            <th className="px-6 py-4 text-left text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Category
            </th>
            <th className="px-6 py-4 text-left text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Price
            </th>
            <th className="px-6 py-4 text-left text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Stock
            </th>
            <th className="px-6 py-4 text-left text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Status
            </th>
            <th className="px-6 py-4 text-right text-xs font-semibold text-lii-bg uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-lii-cloud/10">
          {products.map((product) => (
            <tr
              key={product.id}
              className="hover:bg-lii-cloud/5 transition-colors"
            >
              <td className="px-6 py-4">
                <div className="flex items-center gap-3">
                  <img
                    src={product.images[0]?.url}
                    alt={product.name}
                    className="w-10 h-10 rounded object-cover"
                  />
                  <div>
                    <p className="font-medium text-lii-bg">{product.name}</p>
                    <p className="text-xs text-lii-ash">{product.sku}</p>
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 text-sm text-lii-bg capitalize">
                {product.category}
              </td>
              <td className="px-6 py-4 font-semibold text-lii-gold">
                ${product.price}
              </td>
              <td className="px-6 py-4">
                <div>
                  <p className="font-medium text-lii-bg">
                    {product.availableQuantity}
                  </p>
                  <p className="text-xs text-lii-ash">Available</p>
                </div>
              </td>
              <td className="px-6 py-4">
                <span
                  className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                    product.inStock
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {product.inStock ? 'In Stock' : 'Out of Stock'}
                </span>
              </td>
              <td className="px-6 py-4 text-right">
                <div className="flex items-center justify-end gap-2">
                  <Button variant="ghost" size="sm" title="View">
                    <Eye size={16} />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onEdit?.(product)}
                    title="Edit"
                  >
                    <Edit size={16} />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => onDelete?.(product.id)}
                    title="Delete"
                  >
                    <Trash2 size={16} className="text-red-600" />
                  </Button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
