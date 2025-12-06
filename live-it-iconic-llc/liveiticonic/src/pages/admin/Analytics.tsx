/**
 * Admin Analytics Page
 * Detailed business analytics and insights
 */
import React from 'react';
import { RevenueChart } from '@/components/admin/Dashboard/RevenueChart';
import { TopProducts } from '@/components/admin/Dashboard/TopProducts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

// Mock data for various charts
const conversionData = [
  { day: 'Mon', traffic: 2400, conversions: 240, rate: 10 },
  { day: 'Tue', traffic: 1398, conversions: 221, rate: 16 },
  { day: 'Wed', traffic: 9800, conversions: 229, rate: 2 },
  { day: 'Thu', traffic: 3908, conversions: 200, rate: 5 },
  { day: 'Fri', traffic: 4800, conversions: 221, rate: 4.6 },
  { day: 'Sat', traffic: 3800, conversions: 250, rate: 6.6 },
  { day: 'Sun', traffic: 4300, conversions: 210, rate: 4.9 },
];

const categoryData = [
  { name: 'Tees', value: 35, fill: '#D4A574' },
  { name: 'Hoodies', value: 28, fill: '#2C3E50' },
  { name: 'Outerwear', value: 22, fill: '#34495E' },
  { name: 'Accessories', value: 15, fill: '#7F8C8D' },
];

export function AdminAnalytics() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display text-lii-bg">Analytics</h1>
        <p className="text-lii-ash">Detailed business insights and performance metrics</p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Traffic</p>
            <p className="text-3xl font-bold text-lii-bg">31.6K</p>
            <p className="text-xs text-green-600 mt-2">+12% vs last week</p>
          </CardContent>
        </Card>
        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Conversion Rate</p>
            <p className="text-3xl font-bold text-lii-gold">8.1%</p>
            <p className="text-xs text-green-600 mt-2">+2.4% vs last week</p>
          </CardContent>
        </Card>
        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Avg Order Value</p>
            <p className="text-3xl font-bold text-lii-bg">$171.21</p>
            <p className="text-xs text-green-600 mt-2">+5.8% vs last week</p>
          </CardContent>
        </Card>
        <Card className="bg-white border-lii-cloud">
          <CardContent className="p-6">
            <p className="text-sm text-lii-ash mb-2">Customer LTV</p>
            <p className="text-3xl font-bold text-lii-bg">$485</p>
            <p className="text-xs text-green-600 mt-2">+8.3% vs last month</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Revenue Trend */}
        <RevenueChart />

        {/* Conversion Funnel */}
        <Card className="bg-white border-lii-cloud">
          <CardHeader>
            <CardTitle className="text-lii-bg">Traffic & Conversions</CardTitle>
            <p className="text-sm text-lii-ash mt-1">Daily metrics this week</p>
          </CardHeader>
          <CardContent>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={conversionData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="day" stroke="#666" />
                  <YAxis stroke="#666" yAxisId="left" />
                  <YAxis stroke="#666" yAxisId="right" orientation="right" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e0e0e0',
                      borderRadius: '8px',
                    }}
                  />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="traffic"
                    stroke="#D4A574"
                    strokeWidth={2}
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="conversions"
                    stroke="#2C3E50"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Top Products */}
        <TopProducts />

        {/* Sales by Category */}
        <Card className="bg-white border-lii-cloud">
          <CardHeader>
            <CardTitle className="text-lii-bg">Sales by Category</CardTitle>
            <p className="text-sm text-lii-ash mt-1">Distribution of sales</p>
          </CardHeader>
          <CardContent>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={categoryData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name} ${value}%`}
                    outerRadius={100}
                    fill="#D4A574"
                    dataKey="value"
                  >
                    {categoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default AdminAnalytics;
