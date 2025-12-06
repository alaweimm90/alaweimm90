import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ChartBarIcon,
  UserGroupIcon,
  CursorArrowRaysIcon,
  CurrencyDollarIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const StatCard = ({ title, value, change, icon: Icon }: any) => (
  <div className="bg-white rounded-lg shadow p-6">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-sm font-medium text-gray-600">{title}</p>
        <p className="text-3xl font-bold text-gray-900 mt-2">{value}</p>
        <p className={`text-sm mt-2 ${change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {change >= 0 ? '+' : ''}{change}% from last month
        </p>
      </div>
      <div className="bg-blue-50 p-3 rounded-full">
        <Icon className="w-8 h-8 text-blue-600" />
      </div>
    </div>
  </div>
);

export const Dashboard: React.FC = () => {
  const { data: overview } = useQuery({
    queryKey: ['dashboard-overview'],
    queryFn: async () => {
      // API call to fetch overview data
      return {
        totalReach: 125000,
        totalEngagement: 15000,
        totalConversions: 850,
        revenue: 45000,
        reachChange: 12.5,
        engagementChange: 8.3,
        conversionsChange: -2.1,
        revenueChange: 15.7
      };
    }
  });

  const performanceData = [
    { month: 'Jan', reach: 85000, engagement: 9500 },
    { month: 'Feb', reach: 92000, engagement: 11200 },
    { month: 'Mar', reach: 108000, engagement: 13400 },
    { month: 'Apr', reach: 125000, engagement: 15000 },
  ];

  const platformData = [
    { name: 'Instagram', value: 35 },
    { name: 'Facebook', value: 28 },
    { name: 'LinkedIn', value: 20 },
    { name: 'Twitter', value: 17 },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-2">Welcome back! Here's your marketing performance overview.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Reach"
          value={overview?.totalReach.toLocaleString()}
          change={overview?.reachChange}
          icon={UserGroupIcon}
        />
        <StatCard
          title="Engagement"
          value={overview?.totalEngagement.toLocaleString()}
          change={overview?.engagementChange}
          icon={CursorArrowRaysIcon}
        />
        <StatCard
          title="Conversions"
          value={overview?.totalConversions.toLocaleString()}
          change={overview?.conversionsChange}
          icon={ChartBarIcon}
        />
        <StatCard
          title="Revenue"
          value={`$${overview?.revenue.toLocaleString()}`}
          change={overview?.revenueChange}
          icon={CurrencyDollarIcon}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Trend */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="reach" stroke="#3B82F6" strokeWidth={2} />
              <Line type="monotone" dataKey="engagement" stroke="#10B981" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Platform Performance */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Platform Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={platformData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#3B82F6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow">
        <div className="p-6 border-b border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900">Recent Activity</h3>
        </div>
        <div className="p-6">
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((item) => (
              <div key={item} className="flex items-start gap-4 pb-4 border-b border-gray-100 last:border-0">
                <div className="bg-blue-100 p-2 rounded-full">
                  <ChartBarIcon className="w-5 h-5 text-blue-600" />
                </div>
                <div className="flex-1">
                  <p className="font-medium text-gray-900">New post published on Instagram</p>
                  <p className="text-sm text-gray-600 mt-1">2 hours ago • 1,234 impressions</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
