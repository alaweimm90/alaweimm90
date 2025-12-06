/**
 * Admin Dashboard Page
 * Main dashboard with key metrics, charts, and recent orders
 */
import React from 'react';
import { DollarSign, ShoppingCart, Users, Package } from 'lucide-react';
import { StatsCard } from '@/components/admin/Dashboard/StatsCard';
import { RevenueChart } from '@/components/admin/Dashboard/RevenueChart';
import { OrdersTable } from '@/components/admin/Dashboard/OrdersTable';
import { TopProducts } from '@/components/admin/Dashboard/TopProducts';

const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display text-lii-bg">Dashboard</h1>
        <p className="text-lii-ash">Welcome back! Here's your business overview.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          icon={DollarSign}
          label="Total Revenue"
          value="$48,574"
          change={12.5}
          changeLabel="vs last month"
        />
        <StatsCard
          icon={ShoppingCart}
          label="Orders"
          value="284"
          change={8.2}
          changeLabel="vs last month"
        />
        <StatsCard
          icon={Users}
          label="Customers"
          value="1,429"
          change={-3.1}
          changeLabel="vs last month"
        />
        <StatsCard
          icon={Package}
          label="Products"
          value="8"
          change={0}
          changeLabel="Active SKUs"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RevenueChart />
        <TopProducts />
      </div>

      {/* Recent Orders */}
      <OrdersTable />
    </div>
  );
};

export default Dashboard;
