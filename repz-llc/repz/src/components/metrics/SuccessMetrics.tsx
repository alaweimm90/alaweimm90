import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Badge } from '@/ui/atoms/Badge';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  TrendingUp, Target, Users, DollarSign, 
  BarChart3, Zap, Award, Clock, Activity 
} from 'lucide-react';

interface KPI {
  name: string;
  value: number;
  target: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  change: number;
}

interface ConversionMetric {
  stage: string;
  rate: number;
  target: number;
  improvement: number;
}

export const SuccessMetrics: React.FC = () => {
  const [kpis, setKpis] = useState<KPI[]>([
    {
      name: 'Monthly Recurring Revenue',
      value: 89200,
      target: 100000,
      unit: '$',
      trend: 'up',
      change: 12.5
    },
    {
      name: 'Customer Acquisition Cost',
      value: 87,
      target: 75,
      unit: '$',
      trend: 'down',
      change: -8.2
    },
    {
      name: 'Customer Lifetime Value',
      value: 1847,
      target: 2000,
      unit: '$',
      trend: 'up',
      change: 15.7
    },
    {
      name: 'Monthly Active Users',
      value: 2847,
      target: 3000,
      unit: '',
      trend: 'up',
      change: 18.3
    },
    {
      name: 'Churn Rate',
      value: 4.2,
      target: 3.5,
      unit: '%',
      trend: 'down',
      change: -12.1
    },
    {
      name: 'User Satisfaction',
      value: 4.7,
      target: 4.5,
      unit: '/5',
      trend: 'up',
      change: 6.8
    }
  ]);

  const [conversionMetrics, setConversionMetrics] = useState<ConversionMetric[]>([
    {
      stage: 'Visitor to Sign-up',
      rate: 12.4,
      target: 15.0,
      improvement: 2.3
    },
    {
      stage: 'Sign-up to Trial',
      rate: 78.9,
      target: 80.0,
      improvement: 5.7
    },
    {
      stage: 'Trial to Paid',
      rate: 34.7,
      target: 40.0,
      improvement: 8.9
    },
    {
      stage: 'Paid to Retained',
      rate: 87.3,
      target: 90.0,
      improvement: 12.1
    }
  ]);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
      default: return <Target className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getTrendColor = (trend: string, isGoodDirection: boolean = true) => {
    const isPositive = (trend === 'up' && isGoodDirection) || (trend === 'down' && !isGoodDirection);
    return isPositive ? 'text-green-500' : 'text-red-500';
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          <h1 className="text-3xl font-bold">Success Metrics & KPIs</h1>
          <Badge variant="default">Real-time</Badge>
        </div>
        <Button variant="outline" className="gap-2" onClick={() => console.log("SuccessMetrics button clicked")}>
          <Target className="h-4 w-4" />
          Set Targets
        </Button>
      </div>

      <Tabs defaultValue="kpis" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="kpis">Key Metrics</TabsTrigger>
          <TabsTrigger value="conversion">Conversion</TabsTrigger>
          <TabsTrigger value="optimization">Optimization</TabsTrigger>
        </TabsList>

        {/* KPIs Tab */}
        <TabsContent value="kpis" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {kpis.map((kpi) => {
              const progress = (kpi.value / kpi.target) * 100;
              const isGoodDirection = !['Customer Acquisition Cost', 'Churn Rate'].includes(kpi.name);
              
              return (
                <Card key={kpi.name}>
                  <CardHeader className="pb-4">
                    <CardTitle className="text-sm font-medium text-muted-foreground">
                      {kpi.name}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-2xl font-bold">
                          {kpi.unit === '$' ? '$' : ''}{kpi.value.toLocaleString()}{kpi.unit === '$' ? '' : kpi.unit}
                        </span>
                        <div className={`flex items-center gap-1 text-sm ${getTrendColor(kpi.trend, isGoodDirection)}`}>
                          {getTrendIcon(kpi.trend)}
                          <span>{Math.abs(kpi.change)}%</span>
                        </div>
                      </div>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span>Target: {kpi.unit === '$' ? '$' : ''}{kpi.target.toLocaleString()}{kpi.unit === '$' ? '' : kpi.unit}</span>
                          <span>{Math.round(progress)}%</span>
                        </div>
                        <Progress value={Math.min(progress, 100)} className="h-2" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4 text-center">
                <Users className="h-6 w-6 text-blue-500 mx-auto mb-2" />
                <div className="text-2xl font-bold">2,847</div>
                <p className="text-sm text-muted-foreground">Total Users</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <DollarSign className="h-6 w-6 text-green-500 mx-auto mb-2" />
                <div className="text-2xl font-bold">$1.2M</div>
                <p className="text-sm text-muted-foreground">Annual Revenue</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <Award className="h-6 w-6 text-purple-500 mx-auto mb-2" />
                <div className="text-2xl font-bold">4.7/5</div>
                <p className="text-sm text-muted-foreground">User Rating</p>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4 text-center">
                <Zap className="h-6 w-6 text-orange-500 mx-auto mb-2" />
                <div className="text-2xl font-bold">98.9%</div>
                <p className="text-sm text-muted-foreground">Uptime</p>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Conversion Tab */}
        <TabsContent value="conversion" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {conversionMetrics.map((metric) => {
              const progress = (metric.rate / metric.target) * 100;
              
              return (
                <Card key={metric.stage}>
                  <CardHeader>
                    <CardTitle className="text-lg">{metric.stage}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-3xl font-bold">{metric.rate}%</span>
                        <Badge variant={progress >= 100 ? "default" : "secondary"}>
                          +{metric.improvement}% this month
                        </Badge>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span>Target: {metric.target}%</span>
                          <span>{Math.round(progress)}% of target</span>
                        </div>
                        <Progress value={Math.min(progress, 100)} className="h-3" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        {/* Optimization Tab */}
        <TabsContent value="optimization" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  Performance Optimization
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { name: 'Page Load Time', current: '1.2s', target: '< 2s', status: 'excellent' },
                    { name: 'Bundle Size', current: '642KB', target: '< 1MB', status: 'good' },
                    { name: 'API Response', current: '247ms', target: '< 500ms', status: 'excellent' },
                    { name: 'Memory Usage', current: '24MB', target: '< 50MB', status: 'excellent' }
                  ].map((item) => (
                    <div key={item.name} className="flex items-center justify-between">
                      <div>
                        <div className="font-medium">{item.name}</div>
                        <div className="text-sm text-muted-foreground">Target: {item.target}</div>
                      </div>
                      <div className="text-right">
                        <div className="font-bold">{item.current}</div>
                        <Badge variant={item.status === 'excellent' ? 'default' : 'secondary'} className="text-xs">
                          {item.status}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Clock className="h-5 w-5" />
                  Optimization Actions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <Button variant="outline" className="w-full justify-start gap-2" onClick={() => console.log("SuccessMetrics button clicked")}>
                    <TrendingUp className="h-4 w-4" />
                    Run A/B Test
                  </Button>
                  <Button variant="outline" className="w-full justify-start gap-2" onClick={() => console.log("SuccessMetrics button clicked")}>
                    <Target className="h-4 w-4" />
                    Optimize Conversion
                  </Button>
                  <Button variant="outline" className="w-full justify-start gap-2" onClick={() => console.log("SuccessMetrics button clicked")}>
                    <Users className="h-4 w-4" />
                    Analyze User Behavior
                  </Button>
                  <Button variant="outline" className="w-full justify-start gap-2" onClick={() => console.log("SuccessMetrics button clicked")}>
                    <BarChart3 className="h-4 w-4" />
                    Export Analytics
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};