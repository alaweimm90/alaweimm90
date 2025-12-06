import { useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { BarChart3, TrendingUp, Target, Clock, AlertTriangle, CheckCircle } from 'lucide-react';

interface Attribution {
  confidence_level: string;
  confidence_score: number;
  attribution_type: string;
  signals?: {
    combinedAnalysis?: {
      components: {
        gltr: number;
        detectgpt: number;
        semantic: number;
      };
    };
  };
}

interface PerformanceMetricsProps {
  attributions: Attribution[];
  analysisTime?: number;
  projectStats?: {
    totalArtifacts: number;
    processedArtifacts: number;
    averageProcessingTime: number;
  };
}

export default function PerformanceMetrics({ 
  attributions, 
  analysisTime,
  projectStats 
}: PerformanceMetricsProps) {
  
  const metrics = useMemo(() => {
    if (!attributions.length) return null;

    // Confidence distribution
    const confidenceDistribution = attributions.reduce((acc, attr) => {
      const level = attr.confidence_level.toLowerCase();
      acc[level] = (acc[level] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Attribution type distribution
    const typeDistribution = attributions.reduce((acc, attr) => {
      acc[attr.attribution_type] = (acc[attr.attribution_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    // Average confidence score
    const avgConfidence = attributions.reduce((sum, attr) => 
      sum + (attr.confidence_score * 100), 0) / attributions.length;

    // Signal quality metrics
    const signalMetrics = attributions.reduce((acc, attr) => {
      if (attr.signals?.combinedAnalysis) {
        const analysis = attr.signals.combinedAnalysis;
        acc.gltr.push(analysis.components.gltr * 100);
        acc.detectgpt.push(analysis.components.detectgpt * 100);
        acc.semantic.push(analysis.components.semantic * 100);
      }
      return acc;
    }, { gltr: [] as number[], detectgpt: [] as number[], semantic: [] as number[] });

    const avgSignals = {
      gltr: signalMetrics.gltr.length ? signalMetrics.gltr.reduce((a, b) => a + b, 0) / signalMetrics.gltr.length : 0,
      detectgpt: signalMetrics.detectgpt.length ? signalMetrics.detectgpt.reduce((a, b) => a + b, 0) / signalMetrics.detectgpt.length : 0,
      semantic: signalMetrics.semantic.length ? signalMetrics.semantic.reduce((a, b) => a + b, 0) / signalMetrics.semantic.length : 0
    };

    // Risk assessment
    const highRisk = attributions.filter(attr => attr.confidence_level === 'High').length;
    const mediumRisk = attributions.filter(attr => attr.confidence_level === 'Medium').length;
    const lowRisk = attributions.filter(attr => attr.confidence_level === 'Low').length;

    return {
      confidenceDistribution,
      typeDistribution,
      avgConfidence,
      avgSignals,
      riskBreakdown: { high: highRisk, medium: mediumRisk, low: lowRisk }
    };
  }, [attributions]);

  if (!metrics) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-8">
          <div className="text-center space-y-2">
            <BarChart3 className="h-8 w-8 text-muted-foreground mx-auto" />
            <p className="text-sm text-muted-foreground">No data available for metrics</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'high': return 'text-destructive';
      case 'medium': return 'text-warning';
      case 'low': return 'text-success';
      default: return 'text-muted-foreground';
    }
  };

  const getPerformanceStatus = () => {
    if (metrics.avgConfidence >= 80) return { status: 'excellent', color: 'text-success', icon: CheckCircle };
    if (metrics.avgConfidence >= 60) return { status: 'good', color: 'text-info', icon: TrendingUp };
    if (metrics.avgConfidence >= 40) return { status: 'moderate', color: 'text-warning', icon: AlertTriangle };
    return { status: 'needs attention', color: 'text-destructive', icon: AlertTriangle };
  };

  const performanceStatus = getPerformanceStatus();
  const PerformanceIcon = performanceStatus.icon;

  return (
    <div className="space-y-6">
      {/* Overall Performance */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <PerformanceIcon className={`h-5 w-5 ${performanceStatus.color}`} />
            Overall Performance
          </CardTitle>
          <CardDescription>
            Analysis quality and confidence metrics
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center space-y-1">
              <div className="text-2xl font-bold">{metrics.avgConfidence.toFixed(1)}%</div>
              <div className="text-xs text-muted-foreground">Avg Confidence</div>
            </div>
            <div className="text-center space-y-1">
              <div className="text-2xl font-bold">{attributions.length}</div>
              <div className="text-xs text-muted-foreground">Total Attributions</div>
            </div>
            {typeof analysisTime === 'number' && (
              <div className="text-center space-y-1">
                <div className="text-2xl font-bold">{(analysisTime / 1000).toFixed(1)}s</div>
                <div className="text-xs text-muted-foreground">Analysis Time</div>
              </div>
            )}
            <div className="text-center space-y-1">
              <div className={`text-2xl font-bold ${performanceStatus.color}`}>
                {performanceStatus.status.toUpperCase()}
              </div>
              <div className="text-xs text-muted-foreground">Status</div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Risk Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="h-4 w-4" />
              Risk Distribution
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {Object.entries(metrics.riskBreakdown).map(([level, count]) => (
              <div key={level} className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="capitalize font-medium">{level} Risk</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-muted-foreground">{count}</span>
                    <Badge 
                      variant="outline" 
                      className={getRiskColor(level)}
                    >
                      {((count / attributions.length) * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </div>
                <Progress 
                  value={(count / attributions.length) * 100} 
                  className="h-2"
                />
              </div>
            ))}
          </CardContent>
        </Card>

        {/* Signal Quality */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Signal Quality
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <TooltipProvider>
              {Object.entries(metrics.avgSignals).map(([signal, score]) => (
                <div key={signal} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="capitalize font-medium cursor-help">
                          {signal === 'detectgpt' ? 'DetectGPT' : signal.toUpperCase()}
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>{getSignalDescription(signal)}</p>
                      </TooltipContent>
                    </Tooltip>
                    <Badge variant="outline">
                      {score.toFixed(1)}%
                    </Badge>
                  </div>
                  <Progress 
                    value={score} 
                    className="h-2"
                  />
                </div>
              ))}
            </TooltipProvider>
          </CardContent>
        </Card>

        {/* Attribution Types */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Attribution Types
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {Object.entries(metrics.typeDistribution)
              .sort(([,a], [,b]) => (b as number) - (a as number))
              .map(([type, count]) => (
                <div key={type} className="flex justify-between items-center">
                  <span className="capitalize font-medium">{type}</span>
                  <div className="flex items-center gap-2">
                    <Progress 
                      value={((count as number) / attributions.length) * 100} 
                      className="w-20 h-2"
                    />
                    <Badge variant="secondary" className="min-w-12 text-center">
                      {count as number}
                    </Badge>
                  </div>
                </div>
              ))}
          </CardContent>
        </Card>

        {/* Processing Statistics */}
        {projectStats && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Processing Stats
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span>Artifacts Processed</span>
                  <Badge variant="outline">
                    {projectStats.processedArtifacts}/{projectStats.totalArtifacts}
                  </Badge>
                </div>
                <Progress 
                  value={(projectStats.processedArtifacts / projectStats.totalArtifacts) * 100}
                  className="h-2"
                />
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>Avg Processing Time</span>
                  <span>{(projectStats.averageProcessingTime / 1000).toFixed(1)}s</span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function getSignalDescription(signal: string): string {
  switch (signal) {
    case 'gltr':
      return 'Token probability distribution analysis for detecting generated text patterns';
    case 'detectgpt':
      return 'Curvature analysis measuring how text sits in probability space';
    case 'semantic':
      return 'Semantic similarity comparison with source materials';
    default:
      return 'Signal analysis metric';
  }
}