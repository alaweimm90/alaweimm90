import React from 'react';
import { Card, CardContent } from '@/ui/molecules/Card';
import { Activity, Users, Target, Star } from 'lucide-react';

interface ClientSuccessMetrics {
  engagementScore: number;
  retentionRate: number;
  goalAchievementRate: number;
  satisfactionScore: number;
  progressMetrics: { metric: string; value: number; change: number }[];
  riskFactors: { factor: string; impact: number; clients: number }[];
}

interface ClientSuccessMetricsProps {
  metrics: ClientSuccessMetrics;
}

export const ClientSuccessMetricsCards: React.FC<ClientSuccessMetricsProps> = ({ metrics }) => {
  const formatPercentage = (value: number) => {
    return `${value.toFixed(1)}%`;
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardContent className="p-4 text-center">
            <Activity className="h-8 w-8 mx-auto mb-2 text-blue-500" />
            <p className="text-2xl font-bold">{metrics.engagementScore}</p>
            <p className="text-sm text-muted-foreground">Engagement Score</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Users className="h-8 w-8 mx-auto mb-2 text-green-500" />
            <p className="text-2xl font-bold">{formatPercentage(metrics.retentionRate)}</p>
            <p className="text-sm text-muted-foreground">Retention Rate</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Target className="h-8 w-8 mx-auto mb-2 text-purple-500" />
            <p className="text-2xl font-bold">{formatPercentage(metrics.goalAchievementRate)}</p>
            <p className="text-sm text-muted-foreground">Goal Achievement</p>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4 text-center">
            <Star className="h-8 w-8 mx-auto mb-2 text-yellow-500" />
            <p className="text-2xl font-bold">{metrics.satisfactionScore.toFixed(1)}</p>
            <p className="text-sm text-muted-foreground">Satisfaction Score</p>
          </CardContent>
        </Card>
      </div>

      {/* Progress Metrics */}
      <Card>
        <CardContent className="p-6">
          <h4 className="font-medium mb-4">Progress Metrics</h4>
          <div className="space-y-4">
            {metrics.progressMetrics.map((metric, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-sm font-medium">{metric.metric}</span>
                <div className="flex items-center gap-2">
                  <span className="text-sm">{metric.value}%</span>
                  <span className={`text-xs ${metric.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {metric.change > 0 ? '+' : ''}{metric.change}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Risk Factors */}
      <Card>
        <CardContent className="p-6">
          <h4 className="font-medium mb-4">Risk Factors</h4>
          <div className="space-y-3">
            {metrics.riskFactors.map((factor, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-red-50 border border-red-200 rounded-lg">
                <div>
                  <p className="font-medium text-sm">{factor.factor}</p>
                  <p className="text-xs text-muted-foreground">{factor.clients} clients affected</p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-bold text-red-600">{factor.impact}%</p>
                  <p className="text-xs text-muted-foreground">Impact</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};