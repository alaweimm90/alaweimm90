/**
 * Admin Orders Page
 * View and manage all customer orders
 */
import React, { useState } from 'react';
import { Search, Filter } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { OrderTable } from '@/components/admin/Orders/OrderTable';

const AdminOrders: React.FC = () => {
  const [filters, setFilters] = useState({
    status: '',
    search: '',
  });

  const handleSearchChange = (query: string) => {
    setFilters((prev) => ({ ...prev, search: query }));
  };

  const handleStatusChange = (status: string) => {
    setFilters((prev) => ({ ...prev, status }));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display text-lii-bg">Orders</h1>
        <p className="text-lii-ash">View and manage customer orders</p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg border border-lii-cloud p-4 space-y-4">
        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex-1 min-w-64">
            <div className="relative">
              <Search className="absolute left-3 top-3 text-lii-ash" size={18} />
              <Input
                placeholder="Search by order ID or customer..."
                onChange={(e) => handleSearchChange(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>

          <select
            onChange={(e) => handleStatusChange(e.target.value)}
            className="px-4 py-2 border border-lii-cloud rounded-lg focus:outline-none focus:ring-2 focus:ring-lii-gold text-sm"
          >
            <option value="">All Status</option>
            <option value="pending">Pending</option>
            <option value="processing">Processing</option>
            <option value="shipped">Shipped</option>
            <option value="delivered">Delivered</option>
            <option value="cancelled">Cancelled</option>
          </select>

          <Button variant="ghost" size="sm" className="flex items-center gap-2">
            <Filter size={18} />
            More Filters
          </Button>
        </div>
      </div>

      {/* Orders Table */}
      <OrderTable />
    </div>
  );
};

export default AdminOrders;
