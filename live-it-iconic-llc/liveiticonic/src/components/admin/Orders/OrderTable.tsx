/**
 * Order Table Component
 * Displays all orders with filtering and pagination
 */
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Eye, Edit, Trash2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { OrderStatusBadge } from './OrderStatusBadge';

// Mock orders data - in production this would come from the API
const mockOrders = [
  {
    id: 'ORD-001',
    orderNumber: 'LII-001001',
    customer: 'John Doe',
    email: 'john@example.com',
    total: 248.50,
    status: 'delivered',
    date: '2024-11-12',
    items: 3,
  },
  {
    id: 'ORD-002',
    orderNumber: 'LII-001002',
    customer: 'Jane Smith',
    email: 'jane@example.com',
    total: 156.75,
    status: 'processing',
    date: '2024-11-12',
    items: 2,
  },
  {
    id: 'ORD-003',
    orderNumber: 'LII-001003',
    customer: 'Mike Johnson',
    email: 'mike@example.com',
    total: 412.00,
    status: 'shipped',
    date: '2024-11-11',
    items: 4,
  },
  {
    id: 'ORD-004',
    orderNumber: 'LII-001004',
    customer: 'Sarah Williams',
    email: 'sarah@example.com',
    total: 198.25,
    status: 'pending',
    date: '2024-11-11',
    items: 1,
  },
  {
    id: 'ORD-005',
    orderNumber: 'LII-001005',
    customer: 'Tom Brown',
    email: 'tom@example.com',
    total: 325.00,
    status: 'processing',
    date: '2024-11-10',
    items: 2,
  },
];

export function OrderTable() {
  const [orders] = useState(mockOrders);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const totalPages = Math.ceil(orders.length / itemsPerPage);
  const startIdx = (currentPage - 1) * itemsPerPage;
  const paginatedOrders = orders.slice(startIdx, startIdx + itemsPerPage);

  return (
    <Card className="bg-white border-lii-cloud">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lii-bg">All Orders</CardTitle>
          <p className="text-sm text-lii-ash">Total: {orders.length} orders</p>
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
                  Items
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
                  Actions
                </th>
              </tr>
            </thead>
            <tbody>
              {paginatedOrders.map((order) => (
                <tr
                  key={order.id}
                  className="border-b border-lii-cloud/10 hover:bg-lii-cloud/5 transition-colors"
                >
                  <td className="py-3 px-4 font-semibold text-lii-bg">
                    {order.orderNumber}
                  </td>
                  <td className="py-3 px-4">
                    <div>
                      <p className="font-medium text-lii-bg">{order.customer}</p>
                      <p className="text-xs text-lii-ash">{order.email}</p>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-lii-bg">{order.items}</td>
                  <td className="py-3 px-4 font-semibold text-lii-gold">
                    ${order.total.toFixed(2)}
                  </td>
                  <td className="py-3 px-4">
                    <OrderStatusBadge status={order.status} />
                  </td>
                  <td className="py-3 px-4 text-lii-ash">{order.date}</td>
                  <td className="py-3 px-4">
                    <div className="flex items-center justify-center gap-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        asChild
                        title="View Details"
                      >
                        <Link to={`/orders/${order.id}`}>
                          <Eye size={16} />
                        </Link>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Edit Order"
                      >
                        <Edit size={16} />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        title="Delete Order"
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

        {/* Pagination */}
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-lii-cloud/10">
          <p className="text-sm text-lii-ash">
            Showing {startIdx + 1} to{' '}
            {Math.min(startIdx + itemsPerPage, orders.length)} of {orders.length}{' '}
            orders
          </p>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
            >
              Previous
            </Button>
            <div className="flex items-center gap-1">
              {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                (page) => (
                  <Button
                    key={page}
                    variant={currentPage === page ? 'default' : 'outline'}
                    size="sm"
                    onClick={() => setCurrentPage(page)}
                  >
                    {page}
                  </Button>
                )
              )}
            </div>
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage === totalPages}
              onClick={() =>
                setCurrentPage((p) => Math.min(totalPages, p + 1))
              }
            >
              Next
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
