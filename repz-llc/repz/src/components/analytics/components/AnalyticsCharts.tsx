import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { AreaChart, Area, PieChart as RechartsPieChart, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface SubscriptionAnalytics {
  tierDistribution: { tier: string; count: number; revenue: number; percentage: number }[];
  monthlyTrends: { month: string; new: number; churned: number; net: number; revenue: number }[];
  cohortAnalysis: { cohort: string; month0: number; month1: number; month3: number; month6: number; month12: number }[];
  churnPrediction: { clientId: string; riskScore: number; factors: string[] }[];
}

interface AnalyticsChartsProps {
  subscriptionAnalytics: SubscriptionAnalytics;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({ subscriptionAnalytics }) => {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Monthly Revenue Trends</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={subscriptionAnalytics.monthlyTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip formatter={(value) => formatCurrency(value as number)} />
              <Area 
                type="monotone" 
                dataKey="revenue" 
                stroke="#8884d8" 
                fill="#8884d8" 
                fillOpacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Tier Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <RechartsPieChart>
              <Tooltip formatter={(value) => formatCurrency(value as number)} />
              <RechartsPieChart data={subscriptionAnalytics.tierDistribution}>
                {subscriptionAnalytics.tierDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </RechartsPieChart>
              <Legend />
            </RechartsPieChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
};