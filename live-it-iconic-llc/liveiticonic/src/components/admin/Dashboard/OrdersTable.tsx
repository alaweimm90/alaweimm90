/**
 * Orders Table Component
 * Displays recent orders in admin dashboard
 */
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Eye, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

// Mock orders data - in production this would come from the API
const recentOrders = [
  {
    id: 'ORD-001',
    customer: 'John Doe',
    email: 'john@example.com',
    total: 248.50,
    status: 'Delivered',
    date: '2024-11-12',
  },
  {
    id: 'ORD-002',
    customer: 'Jane Smith',
    email: 'jane@example.com',
    total: 156.75,
    status: 'Processing',
    date: '2024-11-12',
  },
  {
    id: 'ORD-003',
    customer: 'Mike Johnson',
    email: 'mike@example.com',
    total: 412.00,
    status: 'Shipped',
    date: '2024-11-11',
  },
  {
    id: 'ORD-004',
    customer: 'Sarah Williams',
    email: 'sarah@example.com',
    total: 198.25,
    status: 'Pending',
    date: '2024-11-11',
  },
];

const statusColors = {
  Pending: 'bg-yellow-100 text-yellow-800',
  Processing: 'bg-blue-100 text-blue-800',
  Shipped: 'bg-purple-100 text-purple-800',
  Delivered: 'bg-green-100 text-green-800',
  Cancelled: 'bg-red-100 text-red-800',
};

export function OrdersTable() {
  return (
    <Card className="bg-white border-lii-cloud">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-lii-bg">Recent Orders</CardTitle>
            <p className="text-sm text-lii-ash mt-1">Latest customer orders</p>
          </div>
          <Button asChild variant="ghost" size="sm">
            <Link to="/admin/orders" className="flex items-center gap-2">
              View All
              <ArrowRight size={16} />
            </Link>
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-lii-cloud/20">
                <th className="text-left py-3 px-4 font-semibold text-lii-bg">
                  Order ID
                </th>
                <th className="text-left py-3 px-4 font-semibold text-lii-bg">
                  Customer
                </th>
                <th className="text-left py-3 px-4 font-semibold text-lii-bg">
                  Total
                </th>
                <th className="text-left py-3 px-4 font-semibold text-lii-bg">
                  Status
                </th>
                <th className="text-left py-3 px-4 font-semibold text-lii-bg">
                  Date
                </th>
                <th className="text-center py-3 px-4 font-semibold text-lii-bg">
                  Action
                </th>
              </tr>
            </thead>
            <tbody>
              {recentOrders.map((order) => (
                <tr
                  key={order.id}
                  className="border-b border-lii-cloud/10 hover:bg-lii-cloud/5 transition-colors"
                >
                  <td className="py-3 px-4 font-semibold text-lii-bg">{order.id}</td>
                  <td className="py-3 px-4">
                    <div>
                      <p className="font-medium text-lii-bg">{order.customer}</p>
                      <p className="text-xs text-lii-ash">{order.email}</p>
                    </div>
                  </td>
                  <td className="py-3 px-4 font-semibold text-lii-gold">
                    ${order.total.toFixed(2)}
                  </td>
                  <td className="py-3 px-4">
                    <span
                      className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
                        statusColors[order.status as keyof typeof statusColors] ||
                        'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {order.status}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-lii-ash">{order.date}</td>
                  <td className="py-3 px-4 text-center">
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/orders/${order.id}`}>
                        <Eye size={16} />
                      </Link>
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}
