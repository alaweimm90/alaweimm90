/**
 * Top Products Component
 * Displays best performing products
 */
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { products } from '@/data/products';

export function TopProducts() {
  // Mock sales data - in production this would come from the API
  const topProducts = products.slice(0, 5).map((product, index) => ({
    ...product,
    sales: [450, 380, 320, 290, 240][index],
    revenue: (product.price * [450, 380, 320, 290, 240][index]).toLocaleString('en-US'),
  }));

  return (
    <Card className="bg-white border-lii-cloud">
      <CardHeader>
        <CardTitle className="text-lii-bg">Top Products</CardTitle>
        <p className="text-sm text-lii-ash mt-1">Best selling items this month</p>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {topProducts.map((product, index) => (
            <div
              key={product.id}
              className="flex items-center justify-between p-3 bg-lii-cloud/5 rounded-lg hover:bg-lii-cloud/10 transition-colors"
            >
              <div className="flex items-center gap-3 flex-1">
                <div className="flex-shrink-0 font-bold text-lii-gold text-lg w-6">
                  {index + 1}
                </div>
                <div>
                  <p className="font-medium text-lii-bg text-sm">{product.name}</p>
                  <p className="text-xs text-lii-ash">{product.sales} units sold</p>
                </div>
              </div>
              <div className="text-right">
                <p className="font-semibold text-lii-gold">${product.revenue}</p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
