import React from 'react';
import { Card } from './card';
import { Badge } from './badge';
import { Progress } from './progress';
import { useAppStore } from '@/store';
import { BarChart3, Clock, CheckCircle, AlertTriangle } from 'lucide-react';

export function StatsDashboard() {
  const { reports = [], analysisOptions } = useAppStore();

  // Calculate stats from recent analyses
  const stats = React.useMemo(() => {
    const total = reports.length;
    if (total === 0) {
      return {
        total: 0,
        humanLikely: 0,
        aiLikely: 0,
        uncertain: 0,
        avgProcessingTime: 0,
        successRate: 100
      };
    }
    
    const humanLikely = reports.filter(r => r.summary?.aiScore && r.summary.aiScore < 30).length;
    const aiLikely = reports.filter(r => r.summary?.aiScore && r.summary.aiScore > 70).length;
    const uncertain = total - humanLikely - aiLikely;
    
    const avgProcessingTime = reports.reduce((sum, r) => sum + (r.processingTimeMs || 0), 0) / total / 1000 || 0;
    const successRate = reports.filter(r => r.status === 'completed').length / total * 100 || 100;
    
    return {
      total,
      humanLikely,
      aiLikely,
      uncertain,
      avgProcessingTime: Math.round(avgProcessingTime),
      successRate: Math.round(successRate)
    };
  }, [reports]);

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <BarChart3 className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">Total Analyses</span>
        </div>
        <div className="text-2xl font-bold">{stats.total}</div>
        <div className="text-xs text-muted-foreground">
          {analysisOptions?.useLocalOnly ? 'Local only' : 'Mixed mode'}
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <CheckCircle className="h-4 w-4 text-success" />
          <span className="text-sm font-medium">Success Rate</span>
        </div>
        <div className="text-2xl font-bold">{stats.successRate}%</div>
        <Progress value={stats.successRate} className="h-1 mt-1" />
      </Card>

      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <Clock className="h-4 w-4 text-info" />
          <span className="text-sm font-medium">Avg Time</span>
        </div>
        <div className="text-2xl font-bold">{stats.avgProcessingTime}s</div>
        <div className="text-xs text-muted-foreground">
          Per analysis
        </div>
      </Card>

      <Card className="p-4">
        <div className="flex items-center gap-2 mb-2">
          <AlertTriangle className="h-4 w-4 text-warning" />
          <span className="text-sm font-medium">Detection Rate</span>
        </div>
        <div className="flex gap-1 mt-2">
          <Badge variant="outline" className="text-xs">
            Human: {stats.humanLikely}
          </Badge>
          <Badge variant="outline" className="text-xs">
            AI: {stats.aiLikely}
          </Badge>
        </div>
      </Card>
    </div>
  );
}