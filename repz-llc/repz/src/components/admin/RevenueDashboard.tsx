import React, { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import {
  DollarSign,
  TrendingUp,
  TrendingDown,
  Users,
  CreditCard,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  Download,
  Calendar,
  AlertCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import { cn } from '@/lib/utils';
import { TIER_CONFIGS } from '@/constants/tiers';

interface RevenueMetrics {
  mrr: number;
  arr: number;
  churnRate: number;
  cac: number;
  ltv: number;
  growth: number;
}

interface SubscriptionBreakdown {
  tier: string;
  count: number;
  revenue: number;
  percentage: number;
}

interface Transaction {
  id: string;
  customer: string;
  tier: string;
  amount: number;
  date: string;
  status: 'success' | 'pending' | 'failed';
}

const RevenueDashboard: React.FC = () => {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d' | '1y'>('30d');
  const [isLoading, setIsLoading] = useState(false);

  // Mock revenue metrics
  const [metrics, setMetrics] = useState<RevenueMetrics>({
    mrr: 45670,
    arr: 548040,
    churnRate: 3.2,
    cac: 125,
    ltv: 3840,
    growth: 12.5
  });

  // Mock subscription breakdown
  const subscriptionBreakdown: SubscriptionBreakdown[] = [
    { tier: 'Core Program', count: 142, revenue: 12638, percentage: 27.7 },
    { tier: 'Adaptive Engine', count: 98, revenue: 14602, percentage: 32.0 },
    { tier: 'Performance Suite', count: 54, revenue: 12366, percentage: 27.1 },
    { tier: 'Longevity Concierge', count: 12, revenue: 4188, percentage: 9.2 }
  ];

  // Mock revenue trend data
  const revenueTrend = useMemo(() => {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    return months.map((month, i) => ({
      month,
      revenue: 35000 + (i * 2500) + Math.random() * 1000,
      newCustomers: 15 + Math.floor(Math.random() * 10),
      churn: 2 + Math.floor(Math.random() * 3)
    }));
  }, []);

  // Mock recent transactions
  const [transactions] = useState<Transaction[]>([
    {
      id: 'TXN-001',
      customer: 'John Smith',
      tier: 'Performance Suite',
      amount: 229,
      date: '2025-11-12',
      status: 'success'
    },
    {
      id: 'TXN-002',
      customer: 'Sarah Johnson',
      tier: 'Adaptive Engine',
      amount: 149,
      date: '2025-11-12',
      status: 'success'
    },
    {
      id: 'TXN-003',
      customer: 'Mike Davis',
      tier: 'Core Program',
      amount: 89,
      date: '2025-11-11',
      status: 'success'
    },
    {
      id: 'TXN-004',
      customer: 'Emily Brown',
      tier: 'Longevity Concierge',
      amount: 349,
      date: '2025-11-11',
      status: 'pending'
    },
    {
      id: 'TXN-005',
      customer: 'Chris Wilson',
      tier: 'Adaptive Engine',
      amount: 149,
      date: '2025-11-10',
      status: 'failed'
    }
  ]);

  // Tier colors from configuration
  const tierColors = {
    'Core Program': '#3b82f6',
    'Adaptive Engine': '#f15b23',
    'Performance Suite': '#a78bfa',
    'Longevity Concierge': '#fbbf24'
  };

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => {
      setMetrics(prev => ({
        ...prev,
        mrr: prev.mrr + Math.random() * 1000 - 500,
        growth: Math.random() * 20 - 5
      }));
      setIsLoading(false);
    }, 1000);
  };

  const handleExport = () => {
    console.log('Exporting revenue data...');
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`;
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Revenue Dashboard</h1>
          <p className="text-gray-600 mt-1">Track revenue metrics and subscription performance</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-white border rounded-lg p-1">
            {(['7d', '30d', '90d', '1y'] as const).map((range) => (
              <Button
                key={range}
                size="sm"
                variant={timeRange === range ? 'default' : 'ghost'}
                onClick={() => setTimeRange(range)}
                className="px-3 py-1"
              >
                {range}
              </Button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={isLoading}>
            <RefreshCw className={cn('h-4 w-4 mr-2', isLoading && 'animate-spin')} />
            Refresh
          </Button>
          <Button variant="outline" size="sm" onClick={handleExport}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">MRR</p>
                <p className="text-2xl font-bold text-gray-900">{formatCurrency(metrics.mrr)}</p>
              </div>
              <div className="h-10 w-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <DollarSign className="h-5 w-5 text-blue-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              {metrics.growth >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-600" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-600" />
              )}
              <span className={cn(
                'text-sm font-medium',
                metrics.growth >= 0 ? 'text-green-600' : 'text-red-600'
              )}>
                {formatPercentage(metrics.growth)}
              </span>
              <span className="text-sm text-gray-500">vs last month</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">ARR</p>
                <p className="text-2xl font-bold text-gray-900">{formatCurrency(metrics.arr)}</p>
              </div>
              <div className="h-10 w-10 bg-green-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-green-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <span className="text-sm text-gray-500">Annual run rate</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Churn Rate</p>
                <p className="text-2xl font-bold text-gray-900">{metrics.churnRate}%</p>
              </div>
              <div className="h-10 w-10 bg-yellow-100 rounded-lg flex items-center justify-center">
                <Users className="h-5 w-5 text-yellow-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <span className="text-sm text-gray-500">Monthly average</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">CAC</p>
                <p className="text-2xl font-bold text-gray-900">{formatCurrency(metrics.cac)}</p>
              </div>
              <div className="h-10 w-10 bg-purple-100 rounded-lg flex items-center justify-center">
                <Users className="h-5 w-5 text-purple-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <span className="text-sm text-gray-500">Customer acquisition</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">LTV</p>
                <p className="text-2xl font-bold text-gray-900">{formatCurrency(metrics.ltv)}</p>
              </div>
              <div className="h-10 w-10 bg-orange-100 rounded-lg flex items-center justify-center">
                <TrendingUp className="h-5 w-5 text-orange-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <span className="text-sm text-gray-500">Lifetime value</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">LTV:CAC</p>
                <p className="text-2xl font-bold text-gray-900">{(metrics.ltv / metrics.cac).toFixed(1)}x</p>
              </div>
              <div className="h-10 w-10 bg-indigo-100 rounded-lg flex items-center justify-center">
                <CreditCard className="h-5 w-5 text-indigo-600" />
              </div>
            </div>
            <div className="flex items-center gap-1 mt-2">
              <span className="text-sm text-green-600">Healthy ratio</span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Revenue Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={revenueTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" stroke="#666" />
                <YAxis stroke="#666" />
                <Tooltip
                  contentStyle={{ background: '#fff', border: '1px solid #e5e7eb', borderRadius: '8px' }}
                  formatter={(value: number) => formatCurrency(value)}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="revenue"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: '#3b82f6', r: 4 }}
                  name="Revenue"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Subscription Breakdown */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Subscription Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={subscriptionBreakdown}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ tier, percentage }) => `${tier.split(' ')[0]} ${percentage.toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="revenue"
                >
                  {subscriptionBreakdown.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={tierColors[entry.tier as keyof typeof tierColors]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value: number) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
            <div className="mt-4 grid grid-cols-2 gap-3">
              {subscriptionBreakdown.map((item) => (
                <div key={item.tier} className="flex items-center gap-2">
                  <div
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: tierColors[item.tier as keyof typeof tierColors] }}
                  />
                  <div className="flex-1">
                    <p className="text-sm font-medium text-gray-900">{item.tier}</p>
                    <p className="text-xs text-gray-600">{item.count} subscribers</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Transactions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Transaction ID</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Customer</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Tier</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Amount</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Date</th>
                  <th className="text-left py-3 px-4 text-sm font-medium text-gray-700">Status</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((transaction) => (
                  <tr key={transaction.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-900 font-mono">{transaction.id}</td>
                    <td className="py-3 px-4 text-sm text-gray-900">{transaction.customer}</td>
                    <td className="py-3 px-4 text-sm text-gray-700">{transaction.tier}</td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-900">
                      {formatCurrency(transaction.amount)}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {new Date(transaction.date).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4">
                      <Badge
                        className={cn(
                          'text-xs',
                          transaction.status === 'success' && 'bg-green-100 text-green-800',
                          transaction.status === 'pending' && 'bg-yellow-100 text-yellow-800',
                          transaction.status === 'failed' && 'bg-red-100 text-red-800'
                        )}
                      >
                        {transaction.status.charAt(0).toUpperCase() + transaction.status.slice(1)}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default RevenueDashboard;
