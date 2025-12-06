import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { TrendingUp, Target, Brain, AlertTriangle, CheckCircle, ArrowUp, ArrowDown } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface PredictiveData {
  userGrowth: {
    current: number;
    predicted: number;
    confidence: number;
    trend: 'up' | 'down' | 'stable';
  };
  conversionForecast: {
    nextMonth: number;
    confidence: number;
    factors: string[];
  };
  churnRisk: {
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    totalUsers: number;
  };
  revenueProjection: {
    monthly: number[];
    quarterly: number;
    annual: number;
    confidence: number;
  };
  insights: Array<{
    type: 'opportunity' | 'warning' | 'success';
    title: string;
    description: string;
    impact: 'high' | 'medium' | 'low';
    action: string;
  }>;
}

export const PredictiveInsights: React.FC = () => {
  const [data, setData] = useState<PredictiveData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const generatePredictiveData = () => {
      // Mock predictive analytics data
      const mockData: PredictiveData = {
        userGrowth: {
          current: 1247,
          predicted: 1580,
          confidence: 87,
          trend: 'up'
        },
        conversionForecast: {
          nextMonth: 12.4,
          confidence: 82,
          factors: ['Pricing optimization', 'Enhanced onboarding', 'Feature updates']
        },
        churnRisk: {
          highRisk: 23,
          mediumRisk: 67,
          lowRisk: 156,
          totalUsers: 246
        },
        revenueProjection: {
          monthly: [28400, 31200, 34600, 38900, 42300, 45800],
          quarterly: 128400,
          annual: 487600,
          confidence: 79
        },
        insights: [
          {
            type: 'opportunity',
            title: 'Conversion Rate Optimization',
            description: 'Performance tier has 23% higher conversion potential based on user behavior patterns',
            impact: 'high',
            action: 'Focus marketing efforts on performance tier benefits'
          },
          {
            type: 'warning',
            title: 'Churn Risk Alert',
            description: '23 users show high churn probability in the next 30 days',
            impact: 'medium',
            action: 'Implement retention campaign for at-risk users'
          },
          {
            type: 'success',
            title: 'Engagement Trend',
            description: 'In-person training bookings increased 34% this month',
            impact: 'high',
            action: 'Scale in-person training availability'
          },
          {
            type: 'opportunity',
            title: 'Upsell Potential',
            description: 'Core tier users with 3+ checkins show 67% upgrade likelihood',
            impact: 'medium',
            action: 'Trigger automated upgrade prompts for engaged core users'
          }
        ]
      };

      setData(mockData);
      setIsLoading(false);
    };

    const timer = setTimeout(generatePredictiveData, 1500);
    return () => clearTimeout(timer);
  }, []);

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'opportunity': return <TrendingUp className="h-4 w-4" />;
      case 'warning': return <AlertTriangle className="h-4 w-4" />;
      case 'success': return <CheckCircle className="h-4 w-4" />;
      default: return <Brain className="h-4 w-4" />;
    }
  };

  const getInsightColor = (type: string) => {
    switch (type) {
      case 'opportunity': return 'text-adaptive';
      case 'warning': return 'text-destructive';
      case 'success': return 'text-success';
      default: return 'text-muted-foreground';
    }
  };

  if (isLoading) {
    return (
      <div className="grid gap-6 md:grid-cols-2">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="glass-card">
            <CardHeader>
              <div className="h-6 bg-muted/20 rounded animate-pulse" />
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="h-4 bg-muted/20 rounded animate-pulse" />
                <div className="h-4 bg-muted/20 rounded animate-pulse w-3/4" />
                <div className="h-8 bg-muted/20 rounded animate-pulse" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (!data) return null;

  const revenueChartData = data.revenueProjection.monthly.map((value, index) => ({
    month: `Month ${index + 1}`,
    revenue: value,
    projected: index >= 2
  }));

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">User Growth</p>
                <div className="flex items-center gap-2">
                  <p className="text-2xl font-bold">{data.userGrowth.predicted.toLocaleString()}</p>
                  <Badge variant="outline" className="text-xs">
                    {data.userGrowth.trend === 'up' && <ArrowUp className="h-3 w-3 mr-1" />}
                    {((data.userGrowth.predicted - data.userGrowth.current) / data.userGrowth.current * 100).toFixed(1)}%
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">{data.userGrowth.confidence}% confidence</p>
              </div>
              <TrendingUp className="h-8 w-8 text-adaptive" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Conversion Rate</p>
                <p className="text-2xl font-bold">{data.conversionForecast.nextMonth}%</p>
                <p className="text-xs text-muted-foreground">{data.conversionForecast.confidence}% confidence</p>
              </div>
              <Target className="h-8 w-8 text-performance" />
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Quarterly Revenue</p>
                <p className="text-2xl font-bold">${(data.revenueProjection.quarterly / 1000).toFixed(0)}K</p>
                <p className="text-xs text-muted-foreground">{data.revenueProjection.confidence}% confidence</p>
              </div>
              <div className="h-8 w-8 rounded-full bg-success/10 flex items-center justify-center">
                <ArrowUp className="h-4 w-4 text-success" />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Churn Risk</p>
                <p className="text-2xl font-bold">{data.churnRisk.highRisk}</p>
                <p className="text-xs text-muted-foreground">High risk users</p>
              </div>
              <AlertTriangle className="h-8 w-8 text-destructive" />
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Revenue Projection Chart */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Revenue Projection
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={revenueChartData}>
                <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
                <XAxis dataKey="month" className="text-xs" />
                <YAxis className="text-xs" />
                <Tooltip 
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      return (
                        <div className="glass-card p-2 border">
                          <p className="text-sm font-medium">{label}</p>
                          <p className="text-sm">
                            Revenue: ${payload[0].value?.toLocaleString()}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area 
                  type="monotone" 
                  dataKey="revenue" 
                  stroke="hsl(var(--adaptive))" 
                  fill="hsl(var(--adaptive) / 0.2)" 
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Churn Risk Analysis */}
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Churn Risk Analysis
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>High Risk ({data.churnRisk.highRisk} users)</span>
                <span>{((data.churnRisk.highRisk / data.churnRisk.totalUsers) * 100).toFixed(1)}%</span>
              </div>
              <Progress value={(data.churnRisk.highRisk / data.churnRisk.totalUsers) * 100} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Medium Risk ({data.churnRisk.mediumRisk} users)</span>
                <span>{((data.churnRisk.mediumRisk / data.churnRisk.totalUsers) * 100).toFixed(1)}%</span>
              </div>
              <Progress value={(data.churnRisk.mediumRisk / data.churnRisk.totalUsers) * 100} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-2">
                <span>Low Risk ({data.churnRisk.lowRisk} users)</span>
                <span>{((data.churnRisk.lowRisk / data.churnRisk.totalUsers) * 100).toFixed(1)}%</span>
              </div>
              <Progress value={(data.churnRisk.lowRisk / data.churnRisk.totalUsers) * 100} className="h-2" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Insights */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            AI-Powered Insights
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2">
            {data.insights.map((insight, index) => (
              <div key={index} className="p-4 rounded-lg border bg-background/50">
                <div className="flex items-start gap-3">
                  <div className={`mt-0.5 ${getInsightColor(insight.type)}`}>
                    {getInsightIcon(insight.type)}
                  </div>
                  <div className="flex-1 space-y-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium text-sm">{insight.title}</h4>
                      <Badge variant="outline" className="text-xs">
                        {insight.impact} impact
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{insight.description}</p>
                    <p className="text-xs font-medium text-primary">{insight.action}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};