import React from 'react';
import { Order } from '@/types/order';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Link } from 'react-router-dom';
import { orderService } from '@/services/orderService';
import { useMutation, useQueryClient } from '@tanstack/react-query';

interface OrderTableProps {
  orders: Order[];
}

export const OrderTable: React.FC<OrderTableProps> = ({ orders }) => {
  const queryClient = useQueryClient();

  const updateStatusMutation = useMutation({
    mutationFn: ({ orderId, status }: { orderId: string; status: string }) =>
      orderService.updateOrderStatus(orderId, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin/orders'] });
    },
  });

  const handleStatusChange = (orderId: string, newStatus: string) => {
    updateStatusMutation.mutate({ orderId, status: newStatus });
  };

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
        <p className="text-lii-ash font-ui">No orders found</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-lii-gold/10 overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="bg-lii-ink border-lii-gold/10">
            <TableHead className="text-lii-cloud">Order Number</TableHead>
            <TableHead className="text-lii-cloud">Customer</TableHead>
            <TableHead className="text-lii-cloud">Total</TableHead>
            <TableHead className="text-lii-cloud">Status</TableHead>
            <TableHead className="text-lii-cloud">Date</TableHead>
            <TableHead className="text-lii-cloud">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {orders.map(order => (
            <TableRow key={order.id} className="border-lii-gold/10">
              <TableCell className="font-mono text-lii-cloud">{order.orderNumber}</TableCell>
              <TableCell className="text-lii-ash">
                {order.shippingAddress?.email || 'N/A'}
              </TableCell>
              <TableCell className="text-lii-gold font-semibold">
                ${order.total.toFixed(2)}
              </TableCell>
              <TableCell>
                <Select
                  value={order.status}
                  onValueChange={value => handleStatusChange(order.id, value)}
                >
                  <SelectTrigger className="w-32 bg-lii-charcoal/20 border-lii-gold/20">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="confirmed">Confirmed</SelectItem>
                    <SelectItem value="processing">Processing</SelectItem>
                    <SelectItem value="shipped">Shipped</SelectItem>
                    <SelectItem value="delivered">Delivered</SelectItem>
                    <SelectItem value="cancelled">Cancelled</SelectItem>
                  </SelectContent>
                </Select>
              </TableCell>
              <TableCell className="text-lii-ash text-sm">
                {new Date(order.createdAt).toLocaleDateString()}
              </TableCell>
              <TableCell>
                <Button asChild variant="ghost" size="sm">
                  <Link to={`/orders/${order.id}`}>View</Link>
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
};

export default OrderTable;
