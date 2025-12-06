import React, { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';
import {
  Play,
  Pause,
  Square,
  Eye,
  Download,
  Clock,
  Users,
  BarChart3,
  Swords,
  Lightbulb,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Activity,
  Timer,
  TrendingUp,
  Zap,
  Target,
  Filter,
  Calendar,
} from 'lucide-react';
import { ChartContainer, ChartTooltip, ChartTooltipContent } from '@/components/ui/chart';
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from 'recharts';
import { trackEvent } from '@/lib/analytics';

interface Evaluation {
  id: string;
  name: string;
  type: 'arena' | 'benchmark' | 'custom';
  status: 'running' | 'completed' | 'failed' | 'paused' | 'queued';
  progress: number;
  models: string[];
  startTime: string;
  endTime?: string;
  duration?: string;
  results?: {
    winner?: string;
    scores: Record<string, number>;
    accuracy?: number;
  };
  metrics: {
    totalQuestions: number;
    completedQuestions: number;
    successRate: number;
    avgResponseTime: number;
    cost: number;
  };
  queue: {
    position?: number;
    estimatedStart?: string;
  };
}

const mockEvaluations: Evaluation[] = [
  {
    id: 'eval_001',
    name: 'GPT-4 vs Claude Reasoning Test',
    type: 'arena',
    status: 'running',
    progress: 67,
    models: ['GPT-4', 'Claude 3.5 Sonnet'],
    startTime: '2024-01-15T10:30:00Z',
    metrics: {
      totalQuestions: 50,
      completedQuestions: 34,
      successRate: 96.2,
      avgResponseTime: 2.3,
      cost: 4.56,
    },
    queue: {},
  },
  {
    id: 'eval_002',
    name: 'MMLU Mathematics Benchmark',
    type: 'benchmark',
    status: 'running',
    progress: 23,
    models: ['Gemini 1.5 Pro', 'Llama 3.1 70B'],
    startTime: '2024-01-15T09:45:00Z',
    metrics: {
      totalQuestions: 200,
      completedQuestions: 46,
      successRate: 89.1,
      avgResponseTime: 1.8,
      cost: 8.23,
    },
    queue: {},
  },
  {
    id: 'eval_003',
    name: 'Creative Writing Challenge',
    type: 'arena',
    status: 'queued',
    progress: 0,
    models: ['GPT-4', 'Claude 3.5 Sonnet', 'Gemini 1.5 Pro'],
    startTime: '2024-01-15T11:15:00Z',
    metrics: {
      totalQuestions: 25,
      completedQuestions: 0,
      successRate: 0,
      avgResponseTime: 0,
      cost: 0,
    },
    queue: {
      position: 1,
      estimatedStart: '2024-01-15T12:30:00Z',
    },
  },
  {
    id: 'eval_004',
    name: 'Code Generation Test',
    type: 'custom',
    status: 'completed',
    progress: 100,
    models: ['GPT-4', 'Claude 3.5 Sonnet'],
    startTime: '2024-01-15T08:00:00Z',
    endTime: '2024-01-15T09:30:00Z',
    duration: '1h 30m',
    results: {
      winner: 'GPT-4',
      scores: {
        'GPT-4': 87.5,
        'Claude 3.5 Sonnet': 84.2,
      },
      accuracy: 85.9,
    },
    metrics: {
      totalQuestions: 100,
      completedQuestions: 100,
      successRate: 94.7,
      avgResponseTime: 3.1,
      cost: 12.45,
    },
    queue: {},
  },
  {
    id: 'eval_005',
    name: 'TruthfulQA Evaluation',
    type: 'benchmark',
    status: 'failed',
    progress: 15,
    models: ['GPT-4'],
    startTime: '2024-01-15T07:30:00Z',
    endTime: '2024-01-15T07:45:00Z',
    duration: '15m',
    metrics: {
      totalQuestions: 100,
      completedQuestions: 15,
      successRate: 0,
      avgResponseTime: 0,
      cost: 1.23,
    },
    queue: {},
  },
  {
    id: 'eval_006',
    name: 'Medical Question Analysis',
    type: 'custom',
    status: 'paused',
    progress: 45,
    models: ['GPT-4', 'Claude 3.5 Sonnet'],
    startTime: '2024-01-15T06:00:00Z',
    metrics: {
      totalQuestions: 150,
      completedQuestions: 67,
      successRate: 91.8,
      avgResponseTime: 2.8,
      cost: 8.97,
    },
    queue: {},
  },
];

const mockRealTimeMetrics = [
  { time: '10:00', activeEvaluations: 2, queuedEvaluations: 3, throughput: 45 },
  { time: '10:05', activeEvaluations: 3, queuedEvaluations: 2, throughput: 52 },
  { time: '10:10', activeEvaluations: 2, queuedEvaluations: 4, throughput: 38 },
  { time: '10:15', activeEvaluations: 4, queuedEvaluations: 1, throughput: 67 },
  { time: '10:20', activeEvaluations: 3, queuedEvaluations: 3, throughput: 59 },
  { time: '10:25', activeEvaluations: 2, queuedEvaluations: 2, throughput: 43 },
];

const chartConfig = {
  activeEvaluations: { label: 'Active', color: 'hsl(var(--primary))' },
  queuedEvaluations: { label: 'Queued', color: 'hsl(var(--accent))' },
  throughput: { label: 'Throughput', color: 'hsl(var(--secondary))' },
};

export const EvaluationMonitoringDashboard = () => {
  const [evaluations, setEvaluations] = useState<Evaluation[]>(mockEvaluations);
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [realTimeData, setRealTimeData] = useState(mockRealTimeMetrics);
  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setEvaluations(prev => 
        prev.map(evaluation => {
          if (evaluation.status === 'running' && evaluation.progress < 100) {
            return {
              ...evaluation,
              progress: Math.min(100, evaluation.progress + Math.random() * 5),
              metrics: {
                ...evaluation.metrics,
                completedQuestions: Math.min(
                  evaluation.metrics.totalQuestions,
                  evaluation.metrics.completedQuestions + Math.floor(Math.random() * 3)
                ),
                cost: evaluation.metrics.cost + Math.random() * 0.1,
              },
            };
          }
          return evaluation;
        })
      );

      // Update real-time metrics
      const newTime = new Date();
      const timeString = newTime.toLocaleTimeString('en-US', { 
        hour12: false, 
        hour: '2-digit', 
        minute: '2-digit' 
      });
      
      setRealTimeData(prev => [
        ...prev.slice(-5),
        {
          time: timeString,
          activeEvaluations: Math.floor(Math.random() * 5) + 1,
          queuedEvaluations: Math.floor(Math.random() * 4) + 1,
          throughput: Math.floor(Math.random() * 30) + 40,
        },
      ]);

      setLastUpdated(new Date());
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const filteredEvaluations = evaluations.filter(evaluation => {
    const matchesStatus = statusFilter === 'all' || evaluation.status === statusFilter;
    const matchesType = typeFilter === 'all' || evaluation.type === typeFilter;
    return matchesStatus && matchesType;
  });

  const getStatusIcon = (status: Evaluation['status']) => {
    switch (status) {
      case 'running':
        return <Play className="h-4 w-4 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'paused':
        return <Pause className="h-4 w-4 text-yellow-500" />;
      case 'queued':
        return <Clock className="h-4 w-4 text-gray-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: Evaluation['status']) => {
    switch (status) {
      case 'running':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'queued':
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400';
      default:
        return '';
    }
  };

  const getTypeIcon = (type: Evaluation['type']) => {
    switch (type) {
      case 'arena':
        return <Swords className="h-4 w-4" />;
      case 'benchmark':
        return <BarChart3 className="h-4 w-4" />;
      case 'custom':
        return <Lightbulb className="h-4 w-4" />;
    }
  };

  const handleEvaluationAction = (action: string, evaluationId: string) => {
    trackEvent('evaluation_action', { action, evaluationId });
    setEvaluations(prev =>
      prev.map(evaluation => {
        if (evaluation.id === evaluationId) {
          switch (action) {
            case 'pause':
              return { ...evaluation, status: 'paused' as const };
            case 'resume':
              return { ...evaluation, status: 'running' as const };
            case 'stop':
              return { ...evaluation, status: 'failed' as const, progress: evaluation.progress };
            default:
              return evaluation;
          }
        }
        return evaluation;
      })
    );
  };

  const runningEvaluations = evaluations.filter(e => e.status === 'running').length;
  const queuedEvaluations = evaluations.filter(e => e.status === 'queued').length;
  const completedToday = evaluations.filter(e => e.status === 'completed').length;
  const totalCost = evaluations.reduce((sum, e) => sum + e.metrics.cost, 0);

  return (
    <div className="space-y-6">
      {/* Real-time Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Running</p>
              <p className="text-2xl font-bold text-blue-600">{runningEvaluations}</p>
              <div className="flex items-center gap-1 mt-1">
                <Activity className="h-3 w-3 text-blue-500" />
                <span className="text-xs text-muted-foreground">Active now</span>
              </div>
            </div>
            <Play className="h-8 w-8 text-blue-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Queued</p>
              <p className="text-2xl font-bold text-yellow-600">{queuedEvaluations}</p>
              <div className="flex items-center gap-1 mt-1">
                <Timer className="h-3 w-3 text-yellow-500" />
                <span className="text-xs text-muted-foreground">Waiting</span>
              </div>
            </div>
            <Clock className="h-8 w-8 text-yellow-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Completed Today</p>
              <p className="text-2xl font-bold text-green-600">{completedToday}</p>
              <div className="flex items-center gap-1 mt-1">
                <TrendingUp className="h-3 w-3 text-green-500" />
                <span className="text-xs text-green-600">+3 vs yesterday</span>
              </div>
            </div>
            <CheckCircle className="h-8 w-8 text-green-500" />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Total Cost</p>
              <p className="text-2xl font-bold text-foreground">${totalCost.toFixed(2)}</p>
              <div className="flex items-center gap-1 mt-1">
                <Target className="h-3 w-3 text-primary" />
                <span className="text-xs text-muted-foreground">This session</span>
              </div>
            </div>
            <Zap className="h-8 w-8 text-primary" />
          </div>
        </Card>
      </div>

      {/* Real-time Chart */}
      <Card className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-foreground">Real-time Activity</h3>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <RefreshCw className="h-4 w-4" />
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>
        <ChartContainer config={chartConfig} className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={realTimeData}>
              <XAxis dataKey="time" />
              <YAxis />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Line 
                type="monotone" 
                dataKey="activeEvaluations" 
                stroke="var(--color-activeEvaluations)" 
                strokeWidth={2}
                dot={{ fill: 'var(--color-activeEvaluations)' }}
              />
              <Line 
                type="monotone" 
                dataKey="throughput" 
                stroke="var(--color-throughput)" 
                strokeWidth={2}
                dot={{ fill: 'var(--color-throughput)' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartContainer>
      </Card>

      {/* Filters */}
      <Card className="p-4">
        <div className="flex flex-col sm:flex-row gap-4 items-start sm:items-center">
          <div className="flex gap-4">
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-40">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue placeholder="All Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Status</SelectItem>
                <SelectItem value="running">Running</SelectItem>
                <SelectItem value="queued">Queued</SelectItem>
                <SelectItem value="completed">Completed</SelectItem>
                <SelectItem value="paused">Paused</SelectItem>
                <SelectItem value="failed">Failed</SelectItem>
              </SelectContent>
            </Select>

            <Select value={typeFilter} onValueChange={setTypeFilter}>
              <SelectTrigger className="w-40">
                <SelectValue placeholder="All Types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="arena">Arena</SelectItem>
                <SelectItem value="benchmark">Benchmark</SelectItem>
                <SelectItem value="custom">Custom</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="ml-auto text-sm text-muted-foreground">
            Showing {filteredEvaluations.length} of {evaluations.length} evaluations
          </div>
        </div>
      </Card>

      {/* Evaluations List */}
      <div className="space-y-4">
        {filteredEvaluations.map((evaluation) => (
          <Card key={evaluation.id} className="p-6">
            <div className="space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  {getTypeIcon(evaluation.type)}
                  <div>
                    <h3 className="font-semibold text-foreground">{evaluation.name}</h3>
                    <p className="text-sm text-muted-foreground">
                      {evaluation.models.join(' vs ')}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge className={getStatusColor(evaluation.status)}>
                    {getStatusIcon(evaluation.status)}
                    {evaluation.status}
                  </Badge>
                  {evaluation.queue.position && (
                    <Badge variant="outline">
                      Queue #{evaluation.queue.position}
                    </Badge>
                  )}
                </div>
              </div>

              {/* Progress */}
              {evaluation.status === 'running' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">
                      Progress: {evaluation.metrics.completedQuestions} / {evaluation.metrics.totalQuestions}
                    </span>
                    <span className="font-medium">{evaluation.progress.toFixed(1)}%</span>
                  </div>
                  <Progress value={evaluation.progress} className="h-2" />
                </div>
              )}

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Success Rate</p>
                  <p className="font-medium">{evaluation.metrics.successRate.toFixed(1)}%</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Avg Response</p>
                  <p className="font-medium">{evaluation.metrics.avgResponseTime.toFixed(1)}s</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Cost</p>
                  <p className="font-medium">${evaluation.metrics.cost.toFixed(2)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Duration</p>
                  <p className="font-medium">
                    {evaluation.duration || 
                     `${Math.floor((Date.now() - new Date(evaluation.startTime).getTime()) / 60000)}m`}
                  </p>
                </div>
              </div>

              {/* Results */}
              {evaluation.results && (
                <div className="bg-muted/20 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-muted-foreground">Results</p>
                      {evaluation.results.winner && (
                        <p className="font-medium">Winner: {evaluation.results.winner}</p>
                      )}
                      {evaluation.results.accuracy && (
                        <p className="text-sm">Accuracy: {evaluation.results.accuracy}%</p>
                      )}
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-muted-foreground">Scores</p>
                      {Object.entries(evaluation.results.scores).map(([model, score]) => (
                        <p key={model} className="text-sm font-medium">
                          {model}: {score}%
                        </p>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center justify-between pt-2 border-t border-border">
                <div className="text-sm text-muted-foreground">
                  Started: {new Date(evaluation.startTime).toLocaleString()}
                  {evaluation.queue.estimatedStart && (
                    <span className="ml-2">
                      • Est. start: {new Date(evaluation.queue.estimatedStart).toLocaleString()}
                    </span>
                  )}
                </div>
                <div className="flex gap-2">
                  {evaluation.status === 'running' && (
                    <>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => handleEvaluationAction('pause', evaluation.id)}
                      >
                        <Pause className="h-4 w-4 mr-1" />
                        Pause
                      </Button>
                      <AlertDialog>
                        <AlertDialogTrigger asChild>
                          <Button size="sm" variant="outline">
                            <Square className="h-4 w-4 mr-1" />
                            Stop
                          </Button>
                        </AlertDialogTrigger>
                        <AlertDialogContent>
                          <AlertDialogHeader>
                            <AlertDialogTitle>Stop Evaluation?</AlertDialogTitle>
                            <AlertDialogDescription>
                              This will permanently stop the evaluation and you'll lose current progress.
                            </AlertDialogDescription>
                          </AlertDialogHeader>
                          <AlertDialogFooter>
                            <AlertDialogCancel>Cancel</AlertDialogCancel>
                            <AlertDialogAction
                              onClick={() => handleEvaluationAction('stop', evaluation.id)}
                            >
                              Stop Evaluation
                            </AlertDialogAction>
                          </AlertDialogFooter>
                        </AlertDialogContent>
                      </AlertDialog>
                    </>
                  )}
                  {evaluation.status === 'paused' && (
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleEvaluationAction('resume', evaluation.id)}
                    >
                      <Play className="h-4 w-4 mr-1" />
                      Resume
                    </Button>
                  )}
                  {(evaluation.status === 'completed' || evaluation.status === 'failed') && (
                    <>
                      <Button size="sm" variant="outline">
                        <Eye className="h-4 w-4 mr-1" />
                        View Details
                      </Button>
                      <Button size="sm" variant="outline">
                        <Download className="h-4 w-4 mr-1" />
                        Export
                      </Button>
                    </>
                  )}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {filteredEvaluations.length === 0 && (
        <div className="text-center py-12">
          <Activity className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-medium text-foreground mb-2">No evaluations found</h3>
          <p className="text-muted-foreground">
            {statusFilter !== 'all' || typeFilter !== 'all'
              ? 'Try adjusting your filters.'
              : 'Start your first evaluation to see activity here.'}
          </p>
        </div>
      )}
    </div>
  );
};