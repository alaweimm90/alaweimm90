import React from 'react';
import { Card, CardContent } from '@/ui/molecules/Card';
import { DollarSign, TrendingUp, Users, AlertTriangle } from 'lucide-react';

interface RevenueMetrics {
  totalRevenue: number;
  monthlyRecurringRevenue: number;
  averageRevenuePerUser: number;
  churnRate: number;
  lifetimeValue: number;
  growthRate: number;
  newSubscriptions: number;
  cancellations: number;
  revenueGrowth: number;
}

interface RevenueMetricsCardsProps {
  metrics: RevenueMetrics;
}

export const RevenueMetricsCards: React.FC<RevenueMetricsCardsProps> = ({ metrics }) => {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  const getMetricColor = (value: number, isPositive: boolean = true) => {
    if (isPositive) {
      return value > 0 ? 'text-green-600' : 'text-red-600';
    } else {
      return value < 0 ? 'text-green-600' : 'text-red-600';
    }
  };

  const getTrendIcon = (value: number, isPositive: boolean = true) => {
    const isGood = isPositive ? value > 0 : value < 0;
    return isGood ? (
      <TrendingUp className="h-4 w-4 text-green-600" />
    ) : (
      <TrendingUp className="h-4 w-4 text-red-600 rotate-180" />
    );
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Revenue</p>
              <p className="text-2xl font-bold">{formatCurrency(metrics.totalRevenue)}</p>
              <div className="flex items-center gap-1 mt-1">
                {getTrendIcon(metrics.revenueGrowth)}
                <span className={`text-xs ${getMetricColor(metrics.revenueGrowth)}`}>
                  {formatPercentage(metrics.revenueGrowth)}
                </span>
              </div>
            </div>
            <DollarSign className="h-8 w-8 text-green-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Monthly Recurring Revenue</p>
              <p className="text-2xl font-bold">{formatCurrency(metrics.monthlyRecurringRevenue)}</p>
              <div className="flex items-center gap-1 mt-1">
                {getTrendIcon(metrics.growthRate)}
                <span className={`text-xs ${getMetricColor(metrics.growthRate)}`}>
                  {formatPercentage(metrics.growthRate)}
                </span>
              </div>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Average Revenue Per User</p>
              <p className="text-2xl font-bold">{formatCurrency(metrics.averageRevenuePerUser)}</p>
              <div className="flex items-center gap-1 mt-1">
                <span className="text-xs text-green-600">
                  LTV: {formatCurrency(metrics.lifetimeValue)}
                </span>
              </div>
            </div>
            <Users className="h-8 w-8 text-purple-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Churn Rate</p>
              <p className="text-2xl font-bold">{formatPercentage(metrics.churnRate)}</p>
              <div className="flex items-center gap-1 mt-1">
                <span className="text-xs text-muted-foreground">
                  Net: +{metrics.newSubscriptions - metrics.cancellations}
                </span>
              </div>
            </div>
            <AlertTriangle className="h-8 w-8 text-repz-orange" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};