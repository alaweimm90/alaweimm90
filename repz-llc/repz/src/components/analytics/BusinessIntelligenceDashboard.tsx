import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { AlertTriangle, Download, BarChart3, TrendingUp, TrendingDown } from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from '@/contexts/AuthContext';
import { RevenueMetricsCards } from './components/RevenueMetricsCards';
import { AnalyticsCharts } from './components/AnalyticsCharts';
import { ClientSuccessMetricsCards } from './components/ClientSuccessMetrics';

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

interface SubscriptionAnalytics {
  tierDistribution: { tier: string; count: number; revenue: number; percentage: number }[];
  monthlyTrends: { month: string; new: number; churned: number; net: number; revenue: number }[];
  cohortAnalysis: { cohort: string; month0: number; month1: number; month3: number; month6: number; month12: number }[];
  churnPrediction: { clientId: string; riskScore: number; factors: string[] }[];
}

interface ClientSuccessMetrics {
  engagementScore: number;
  retentionRate: number;
  goalAchievementRate: number;
  satisfactionScore: number;
  progressMetrics: { metric: string; value: number; change: number }[];
  riskFactors: { factor: string; impact: number; clients: number }[];
}

interface CoachPerformanceData {
  coachId: string;
  coachName: string;
  clientCount: number;
  retentionRate: number;
  clientSatisfaction: number;
  revenueGenerated: number;
  responseTime: number;
  engagementScore: number;
  goals: { achieved: number; total: number };
}

const BusinessIntelligenceDashboard: React.FC = () => {
  const { user } = useAuth();
  const [dateRange, setDateRange] = useState('30d');
  const [isLoading, setIsLoading] = useState(false);
  const [revenueMetrics, setRevenueMetrics] = useState<RevenueMetrics | null>(null);
  const [subscriptionAnalytics, setSubscriptionAnalytics] = useState<SubscriptionAnalytics | null>(null);
  const [clientMetrics, setClientMetrics] = useState<ClientSuccessMetrics | null>(null);
  const [coachPerformance, setCoachPerformance] = useState<CoachPerformanceData[]>([]);

  useEffect(() => {
    if (user && ['admin', 'coach'].includes(user.role || '')) {
      fetchAnalyticsData();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, dateRange]);

  const fetchAnalyticsData = async () => {
    setIsLoading(true);
    try {
      // Fetch revenue and subscription data
      await Promise.all([
        fetchRevenueMetrics(),
        fetchSubscriptionAnalytics(), 
        fetchClientSuccessMetrics(),
        fetchCoachPerformanceData()
      ]);
    } catch (error) {
      console.error('Error fetching analytics:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchRevenueMetrics = async () => {
    // Mock data - in production, this would query your subscription/payment tables
    const mockData: RevenueMetrics = {
      totalRevenue: 47850,
      monthlyRecurringRevenue: 12400,
      averageRevenuePerUser: 156,
      churnRate: 4.2,
      lifetimeValue: 1847,
      growthRate: 18.5,
      newSubscriptions: 23,
      cancellations: 3,
      revenueGrowth: 15.2
    };
    setRevenueMetrics(mockData);
  };

  const fetchSubscriptionAnalytics = async () => {
    const mockData: SubscriptionAnalytics = {
      tierDistribution: [
        { tier: 'Core', count: 45, revenue: 4950, percentage: 32 },
        { tier: 'Adaptive', count: 38, revenue: 7600, percentage: 29 },
        { tier: 'Performance', count: 32, revenue: 9600, percentage: 25 },
        { tier: 'Longevity', count: 18, revenue: 8100, percentage: 14 }
      ],
      monthlyTrends: [
        { month: 'Jan', new: 12, churned: 2, net: 10, revenue: 8400 },
        { month: 'Feb', new: 18, churned: 3, net: 15, revenue: 9200 },
        { month: 'Mar', new: 23, churned: 3, net: 20, revenue: 11800 },
        { month: 'Apr', new: 28, churned: 4, net: 24, revenue: 12400 }
      ],
      cohortAnalysis: [
        { cohort: 'Jan 2024', month0: 100, month1: 85, month3: 72, month6: 65, month12: 58 },
        { cohort: 'Feb 2024', month0: 100, month1: 88, month3: 75, month6: 68, month12: 0 },
        { cohort: 'Mar 2024', month0: 100, month1: 92, month3: 78, month6: 0, month12: 0 }
      ],
      churnPrediction: [
        { clientId: 'client-1', riskScore: 85, factors: ['Low engagement', 'Missed check-ins'] },
        { clientId: 'client-2', riskScore: 72, factors: ['Payment issues', 'Declining progress'] }
      ]
    };
    setSubscriptionAnalytics(mockData);
  };

  const fetchClientSuccessMetrics = async () => {
    const mockData: ClientSuccessMetrics = {
      engagementScore: 78,
      retentionRate: 87,
      goalAchievementRate: 73,
      satisfactionScore: 4.6,
      progressMetrics: [
        { metric: 'Weekly Check-ins', value: 85, change: 12 },
        { metric: 'Workout Completion', value: 92, change: 8 },
        { metric: 'Goal Progress', value: 73, change: -3 },
        { metric: 'Coach Interaction', value: 68, change: 15 }
      ],
      riskFactors: [
        { factor: 'Inconsistent Check-ins', impact: 45, clients: 12 },
        { factor: 'Low Workout Frequency', impact: 38, clients: 8 },
        { factor: 'Poor Goal Progress', impact: 32, clients: 15 }
      ]
    };
    setClientMetrics(mockData);
  };

  const fetchCoachPerformanceData = async () => {
    const mockData: CoachPerformanceData[] = [
      {
        coachId: 'coach-1',
        coachName: 'Sarah Johnson',
        clientCount: 24,
        retentionRate: 92,
        clientSatisfaction: 4.8,
        revenueGenerated: 8400,
        responseTime: 2.3,
        engagementScore: 89,
        goals: { achieved: 18, total: 24 }
      },
      {
        coachId: 'coach-2', 
        coachName: 'Mike Chen',
        clientCount: 19,
        retentionRate: 87,
        clientSatisfaction: 4.6,
        revenueGenerated: 6650,
        responseTime: 3.1,
        engagementScore: 82,
        goals: { achieved: 14, total: 19 }
      },
      {
        coachId: 'coach-3',
        coachName: 'Emma Davis',
        clientCount: 32,
        retentionRate: 89,
        clientSatisfaction: 4.7,
        revenueGenerated: 11200,
        responseTime: 2.8,
        engagementScore: 85,
        goals: { achieved: 24, total: 32 }
      }
    ];
    setCoachPerformance(mockData);
  };

  const exportData = (dataType: string) => {
    // Implement data export functionality
    console.log(`Exporting ${dataType} data...`);
  };

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
      <TrendingDown className="h-4 w-4 text-red-600" />
    );
  };

  if (!user || !['admin', 'coach'].includes(user.role || '')) {
    return (
      <div className="p-8">
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Access restricted. Administrator or coach privileges required to view analytics.
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
              <BarChart3 className="h-8 w-8 text-blue-600" />
              Business Intelligence Dashboard
            </h1>
            <p className="text-muted-foreground">
              Advanced analytics and performance insights
            </p>
          </div>
          <div className="flex items-center gap-3">
            <select 
              value={dateRange} 
              onChange={(e) => setDateRange(e.target.value)}
              className="border rounded-lg px-3 py-2"
            >
              <option value="7d">Last 7 days</option>
              <option value="30d">Last 30 days</option>
              <option value="90d">Last 90 days</option>
              <option value="365d">Last year</option>
            </select>
            <Button variant="outline" onClick={() => exportData('all')}>
              <Download className="h-4 w-4 mr-2" />
              Export Data
            </Button>
          </div>
        </div>

        <Tabs defaultValue="revenue" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="revenue">Revenue Analytics</TabsTrigger>
            <TabsTrigger value="clients">Client Success</TabsTrigger>
            <TabsTrigger value="coaches">Coach Performance</TabsTrigger>
            <TabsTrigger value="predictive">Predictive Analytics</TabsTrigger>
          </TabsList>

          {/* Revenue Analytics Tab */}
          <TabsContent value="revenue" className="space-y-6">
            {revenueMetrics && (
              <>
                <RevenueMetricsCards metrics={revenueMetrics} />
                {subscriptionAnalytics && (
                  <AnalyticsCharts subscriptionAnalytics={subscriptionAnalytics} />
                )}
              </>
            )}
          </TabsContent>

          {/* Client Success Tab */}
          <TabsContent value="clients" className="space-y-6">
            {clientMetrics && (
              <ClientSuccessMetricsCards metrics={clientMetrics} />
            )}
          </TabsContent>

          {/* Coach Performance Tab */}
          <TabsContent value="coaches" className="space-y-6">
            <div className="grid gap-4">
              {coachPerformance.map((coach) => (
                <Card key={coach.coachId}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>{coach.coachName}</CardTitle>
                      <Badge className="bg-blue-100 text-blue-700">
                        {coach.clientCount} clients
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-600">{formatPercentage(coach.retentionRate)}</p>
                        <p className="text-xs text-muted-foreground">Retention Rate</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-yellow-600">{coach.clientSatisfaction}/5.0</p>
                        <p className="text-xs text-muted-foreground">Satisfaction</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-600">{formatCurrency(coach.revenueGenerated)}</p>
                        <p className="text-xs text-muted-foreground">Revenue</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-purple-600">{coach.responseTime}h</p>
                        <p className="text-xs text-muted-foreground">Avg Response</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-indigo-600">{coach.goals.achieved}/{coach.goals.total}</p>
                        <p className="text-xs text-muted-foreground">Goals Achieved</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Predictive Analytics Tab */}
          <TabsContent value="predictive" className="space-y-6">
            {subscriptionAnalytics && (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle>Cohort Retention Analysis</CardTitle>
                    <CardDescription>
                      Track how well different user cohorts are retained over time
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b">
                            <th className="text-left p-2">Cohort</th>
                            <th className="text-center p-2">Month 0</th>
                            <th className="text-center p-2">Month 1</th>
                            <th className="text-center p-2">Month 3</th>
                            <th className="text-center p-2">Month 6</th>
                            <th className="text-center p-2">Month 12</th>
                          </tr>
                        </thead>
                        <tbody>
                          {subscriptionAnalytics.cohortAnalysis.map((cohort, index) => (
                            <tr key={index} className="border-b">
                              <td className="p-2 font-medium">{cohort.cohort}</td>
                              <td className="text-center p-2">{cohort.month0}%</td>
                              <td className="text-center p-2">{cohort.month1}%</td>
                              <td className="text-center p-2">{cohort.month3}%</td>
                              <td className="text-center p-2">{cohort.month6 || '-'}%</td>
                              <td className="text-center p-2">{cohort.month12 || '-'}%</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Churn Risk Prediction</CardTitle>
                    <CardDescription>
                      Clients with high probability of churning based on behavior patterns
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {subscriptionAnalytics.churnPrediction.map((prediction, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                          <div>
                            <p className="font-medium">Client {prediction.clientId}</p>
                            <p className="text-sm text-muted-foreground">
                              Risk factors: {prediction.factors.join(', ')}
                            </p>
                          </div>
                          <Badge variant={prediction.riskScore > 80 ? 'destructive' : 'default'}>
                            {prediction.riskScore}% risk
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default BusinessIntelligenceDashboard;