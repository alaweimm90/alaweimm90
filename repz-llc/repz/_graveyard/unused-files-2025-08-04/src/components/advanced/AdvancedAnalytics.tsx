import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { TrendingUp, Target, Zap, Award, Calendar, Clock } from 'lucide-react';
import { useTierAccess } from '@/hooks/useTierAccess';
import { TierGate } from '@/components/shared/TierGate';

interface AnalyticsMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  change: number;
  requiredTier: 'adaptive' | 'performance' | 'longevity';
}

interface PredictiveInsight {
  id: string;
  title: string;
  prediction: string;
  confidence: number;
  actionable: boolean;
  requiredTier: 'performance' | 'longevity';
}

export const AdvancedAnalytics: React.FC = () => {
  const { hasMinimumTier, userTier } = useTierAccess();
  const [metrics, setMetrics] = useState<AnalyticsMetric[]>([]);
  const [insights, setInsights] = useState<PredictiveInsight[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate loading advanced analytics data
    const loadAnalytics = async () => {
      setLoading(true);
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setMetrics([
        {
          id: '1',
          name: 'Strength Progression Rate',
          value: 12.5,
          unit: '%/month',
          trend: 'up',
          change: 2.3,
          requiredTier: 'adaptive'
        },
        {
          id: '2', 
          name: 'Recovery Efficiency',
          value: 87,
          unit: '%',
          trend: 'up',
          change: 5.2,
          requiredTier: 'adaptive'
        },
        {
          id: '3',
          name: 'Form Consistency Score',
          value: 94,
          unit: '/100',
          trend: 'stable',
          change: 0.8,
          requiredTier: 'performance'
        },
        {
          id: '4',
          name: 'Injury Risk Index',
          value: 15,
          unit: '%',
          trend: 'down',
          change: -8.1,
          requiredTier: 'performance'
        },
        {
          id: '5',
          name: 'Metabolic Age',
          value: 28,
          unit: 'years',
          trend: 'down',
          change: -2.5,
          requiredTier: 'longevity'
        },
        {
          id: '6',
          name: 'Longevity Score',
          value: 92,
          unit: '/100',
          trend: 'up',
          change: 4.7,
          requiredTier: 'longevity'
        }
      ]);

      setInsights([
        {
          id: '1',
          title: 'Strength Plateau Prevention',
          prediction: 'Based on your progression pattern, consider deload week in 3 weeks',
          confidence: 89,
          actionable: true,
          requiredTier: 'performance'
        },
        {
          id: '2',
          title: 'Optimal Training Window',
          prediction: 'Your peak performance occurs between 2-4 PM based on HRV data',
          confidence: 76,
          actionable: true,
          requiredTier: 'performance'
        },
        {
          id: '3',
          title: 'Longevity Optimization',
          prediction: 'Adding 15min meditation will likely improve recovery by 12%',
          confidence: 82,
          actionable: true,
          requiredTier: 'longevity'
        }
      ]);
      
      setLoading(false);
    };

    loadAnalytics();
  }, []);

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'down': return <TrendingUp className="h-4 w-4 text-red-500 rotate-180" />;
      default: return <Target className="h-4 w-4 text-yellow-500" />;
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'up': return 'text-green-500';
      case 'down': return 'text-red-500';
      default: return 'text-yellow-500';
    }
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card key={i} className="animate-pulse">
              <CardHeader className="pb-2">
                <div className="h-4 bg-muted rounded w-3/4"></div>
              </CardHeader>
              <CardContent>
                <div className="h-8 bg-muted rounded w-1/2 mb-2"></div>
                <div className="h-3 bg-muted rounded w-1/4"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Advanced Metrics Grid */}
      <div>
        <div className="flex items-center gap-2 mb-6">
          <Zap className="h-5 w-5 text-primary" />
          <h2 className="text-2xl font-bold">Advanced Analytics</h2>
          <Badge variant="secondary" className="ml-2">
            {userTier || 'core'}
          </Badge>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {metrics.map((metric) => (
            <TierGate key={metric.id} requiredTier={metric.requiredTier} feature="enhanced_analytics">
              <Card className="hover:shadow-md transition-shadow">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    {metric.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-2xl font-bold">
                        {metric.value}
                        <span className="text-sm font-normal text-muted-foreground ml-1">
                          {metric.unit}
                        </span>
                      </div>
                      <div className={`flex items-center gap-1 text-sm ${getTrendColor(metric.trend)}`}>
                        {getTrendIcon(metric.trend)}
                        <span>{Math.abs(metric.change)}%</span>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant="outline" className="text-xs">
                        {metric.requiredTier}+
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TierGate>
          ))}
        </div>
      </div>

      {/* Predictive Insights */}
      <div>
        <div className="flex items-center gap-2 mb-6">
          <Target className="h-5 w-5 text-primary" />
          <h2 className="text-2xl font-bold">AI Predictive Insights</h2>
          <Badge variant="outline" className="ml-2">Performance+</Badge>
        </div>

        <div className="space-y-4">
          {insights.map((insight) => (
            <TierGate key={insight.id} requiredTier={insight.requiredTier} feature="ai_coaching">
              <Card className="hover:shadow-md transition-shadow">
                <CardContent className="pt-6">
                  <div className="flex items-start justify-between">
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <Award className="h-4 w-4 text-primary" />
                        <h3 className="font-semibold">{insight.title}</h3>
                        <Badge 
                          variant={insight.confidence > 80 ? "default" : "secondary"}
                          className="text-xs"
                        >
                          {insight.confidence}% confidence
                        </Badge>
                      </div>
                      <p className="text-muted-foreground">{insight.prediction}</p>
                    </div>
                    {insight.actionable && (
                      <Button size="sm" variant="outline">
                        Apply
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TierGate>
          ))}
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex flex-wrap gap-3">
        <TierGate requiredTier="adaptive" feature="enhanced_analytics">
          <Button variant="outline" className="gap-2">
            <Calendar className="h-4 w-4" />
            Generate Report
          </Button>
        </TierGate>
        <TierGate requiredTier="performance" feature="enhanced_analytics">
          <Button variant="outline" className="gap-2">
            <Clock className="h-4 w-4" />
            Schedule Analysis
          </Button>
        </TierGate>
        <TierGate requiredTier="longevity" feature="ai_coaching">
          <Button variant="outline" className="gap-2">
            <Target className="h-4 w-4" />
            Optimize Program
          </Button>
        </TierGate>
      </div>
    </div>
  );
};