import React from 'react';
import { Link } from 'react-router-dom';
import { Order } from '@/types/order';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Package, ArrowRight } from 'lucide-react';

interface OrderHistoryProps {
  orders: Order[];
}

export const OrderHistory: React.FC<OrderHistoryProps> = ({ orders }) => {
  const getStatusColor = (status: Order['status']) => {
    switch (status) {
      case 'delivered':
        return 'bg-green-500/10 text-green-400 border-green-500/20';
      case 'shipped':
        return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case 'processing':
        return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20';
      case 'pending':
      case 'confirmed':
        return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
      case 'cancelled':
        return 'bg-red-500/10 text-red-400 border-red-500/20';
      default:
        return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  if (orders.length === 0) {
    return (
      <div className="text-center py-12">
        <Package className="w-12 h-12 text-lii-ash/50 mx-auto mb-4" />
        <p className="text-lii-ash font-ui">No orders yet</p>
        <Button asChild variant="primary" className="mt-4">
          <Link to="/shop">Start Shopping</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {orders.map(order => (
        <Card key={order.id} className="bg-lii-ink border-lii-gold/10">
          <CardContent className="p-6">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-4 mb-2">
                  <h3 className="text-lii-cloud font-display font-semibold">
                    Order {order.orderNumber}
                  </h3>
                  <Badge className={getStatusColor(order.status)} variant="outline">
                    {order.status.charAt(0).toUpperCase() + order.status.slice(1)}
                  </Badge>
                </div>
                <p className="text-lii-ash font-ui text-sm mb-2">
                  {new Date(order.createdAt).toLocaleDateString()}
                </p>
                <p className="text-lii-cloud font-ui text-sm">
                  {order.items.length} item{order.items.length !== 1 ? 's' : ''} • $
                  {order.total.toFixed(2)}
                </p>
              </div>
              <Button asChild variant="outline" size="sm">
                <Link to={`/orders/${order.id}`}>
                  View Details
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

export default OrderHistory;
