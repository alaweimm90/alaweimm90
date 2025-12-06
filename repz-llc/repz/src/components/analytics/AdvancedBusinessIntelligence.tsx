import React, { useState, useEffect, Suspense, lazy } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  TrendingUp, 
  DollarSign, 
  Users, 
  Target, 
  BarChart3,
  PieChart,
  LineChart,
  ArrowUpRight,
  ArrowDownRight,
  Minus,
  AlertTriangle,
  CheckCircle,
  RefreshCw
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';
import { useToast } from '@/ui/molecules/useToast';
import { Analytics } from '@/lib/analytics';
import { buildAnalyticsDAG } from '@/lib/orchestration/dag';
import { runAnalyticsPlan } from '@/lib/orchestration/executor';
const AnalyticsCharts = lazy(() => import('./components/AnalyticsCharts').then(m => ({ default: m.AnalyticsCharts })));

interface MetricData {
  current: number;
  previous: number;
  change: number;
  trend: 'up' | 'down' | 'stable';
}

interface Recommendation {
  action: string;
  impact: string;
  effort: string;
  category: string;
  timeline: string;
}

interface Alert {
  type: 'warning' | 'info' | 'success';
  message: string;
}

interface Insights {
  keyFindings: string[];
  recommendations: Recommendation[];
  alerts: Alert[];
}

interface BusinessMetrics {
  revenue: MetricData;
  users: MetricData;
  conversions: MetricData;
  churn: MetricData;
  growth: MetricData;
  insights: Insights;
}

interface AnalyticsSnapshotRow {
  payload: BusinessMetrics;
  created_at?: string;
}

const AdvancedBusinessIntelligence: React.FC = () => {
  const { toast } = useToast();
  const [searchParams, setSearchParams] = useSearchParams();
  const [metrics, setMetrics] = useState<BusinessMetrics | null>(null);
  const [isSnapshot, setIsSnapshot] = useState(false);
  const [snapshotCreatedAt, setSnapshotCreatedAt] = useState<string | null>(null);
  const [workflowStatus, setWorkflowStatus] = useState<string | null>(null);
  const [workflowSteps, setWorkflowSteps] = useState<{ step: string; status: string; error_code?: string | null; error_message?: string | null }[]>([]);
  const [cohortData, setCohortData] = useState<Record<string, unknown> | null>(null);
  const [abTests, setAbTests] = useState<Record<string, unknown> | null>(null);
  const [timeframe, setTimeframe] = useState<'7d' | '30d' | '90d' | '1y'>('30d');
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const hasTraceId = !!searchParams.get('trace_id');
  const [showAllSteps, setShowAllSteps] = useState(false);
  const [snapshotList, setSnapshotList] = useState<{ id: string; created_at: string; timeframe: string }[]>([]);
  const [pollAttempts, setPollAttempts] = useState(0);
  const [viewingHistorical, setViewingHistorical] = useState(false);
  const [showCharts, setShowCharts] = useState(false);

  useEffect(() => {
    const tfParam = searchParams.get('timeframe') as '7d' | '30d' | '90d' | '1y' | null;
    if (tfParam && ['7d','30d','90d','1y'].includes(tfParam)) {
      setTimeframe(tfParam as '7d' | '30d' | '90d' | '1y');
    }
    const traceId = searchParams.get('trace_id');
    loadAllAnalytics(traceId ?? undefined);
    loadSnapshotHistory();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeframe, searchParams]);

  useEffect(() => {
    const traceId = searchParams.get('trace_id') ?? undefined;
    if (workflowStatus === 'succeeded' && traceId) {
      refreshSnapshot();
    }
  }, [workflowStatus, searchParams]);

  useEffect(() => {
    const traceId = searchParams.get('trace_id') ?? undefined;
    if (workflowStatus !== 'running' || !traceId) return;
    const delay = Math.min(5000 * Math.max(1, Math.pow(2, pollAttempts)), 60000);
    const jitter = Math.floor(Math.random() * 750);
    const t = setTimeout(() => {
      loadWorkflowStatus(traceId);
      setPollAttempts((p) => Math.min(p + 1, 10));
    }, delay + jitter);
    return () => clearTimeout(t);
  }, [workflowStatus, searchParams]);

  useEffect(() => {
    if (workflowStatus && workflowStatus !== 'running') {
      setPollAttempts(0);
    }
  }, [workflowStatus]);

  useEffect(() => {
    const traceId = searchParams.get('trace_id') ?? undefined;
    if (!traceId) return;
    if (!isSnapshot || !snapshotCreatedAt) return;
    if (workflowStatus === 'running') return;
    const ageMs = Date.now() - new Date(snapshotCreatedAt).getTime();
    const thresholdMs = 12 * 60 * 60 * 1000;
    if (ageMs > thresholdMs && !isRefreshing) {
      refreshSnapshot();
    }
  }, [isSnapshot, snapshotCreatedAt, workflowStatus, searchParams]);

  useEffect(() => {
    const traceId = searchParams.get('trace_id') ?? undefined;
    if (!traceId) { setSnapshotList([]); return; }
    loadSnapshotHistory();
  }, [searchParams, timeframe]);

  useEffect(() => {
    if (workflowStatus !== 'failed') return;
    toast({ title: 'Workflow failed', description: 'One or more steps failed', variant: 'destructive' });
  }, [workflowStatus]);

  const loadBusinessMetrics = async (traceId?: string) => {
    setIsLoading(true);
    try {
      if (traceId) {
        const snapshot = await supabase.functions.invoke<{ data: AnalyticsSnapshotRow | null }>('analytics-snapshots', {
          body: { trace_id: traceId, timeframe }
        });
        if (snapshot.error) throw snapshot.error;
        const payload = snapshot.data?.data?.payload;
        if (payload) {
          setMetrics(payload);
          setIsSnapshot(true);
          setSnapshotCreatedAt(snapshot.data?.data?.created_at ?? null);
          return;
        }
      }

      const bi = await supabase.functions.invoke<BusinessMetrics>('business-intelligence', {
        body: {
          timeframe,
          metrics: ['revenue', 'users', 'conversions', 'churn', 'growth'],
          segmentation: {}
        }
      });
      if (bi.error) throw bi.error;
      setMetrics(bi.data);
      setIsSnapshot(false);
      setSnapshotCreatedAt(null);
    } catch (error) {
      console.error('Error loading business metrics:', error);
      toast({
        title: "Error",
        description: "Failed to load business metrics",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const loadWorkflowStatus = async (traceId?: string) => {
    try {
      if (!traceId) { setWorkflowStatus(null); return; }
      const res = await supabase.functions.invoke<{ data: { workflow: { status: string }, steps: { step: string; status: string; error_code?: string | null; error_message?: string | null }[] } | null }>('workflow-status', { body: { trace_id: traceId } });
      if (res.error) throw res.error;
      const status = res.data?.data?.workflow?.status ?? null;
      const steps = res.data?.data?.steps ?? [];
      setWorkflowStatus(status);
      setWorkflowSteps(steps);
    } catch (error) {
      setWorkflowStatus(null);
      setWorkflowSteps([]);
    }
  };

  const refreshSnapshot = async () => {
    if (isRefreshing) return;
    setIsRefreshing(true);
    try {
      const traceId = searchParams.get('trace_id') ?? undefined;
      if (!traceId) {
        setIsRefreshing(false);
        return;
      }
      
      const dag = buildAnalyticsDAG({
        traceId,
        timeframe,
        includeSnapshots: true,
        includeBI: true,
        includeWorkflow: true,
        includeCohort: true,
        includeABTests: true,
      });
      
      const result = await runAnalyticsPlan(dag, supabase);
      
      if (result.metrics && !viewingHistorical) {
        setMetrics(result.metrics);
        setIsSnapshot(result.isSnapshot);
        setSnapshotCreatedAt(result.snapshotCreatedAt ?? null);
      }
      
      if (result.workflowStatus) {
        setWorkflowStatus(result.workflowStatus);
      }
      
      if (result.workflowSteps) {
        setWorkflowSteps(result.workflowSteps);
      }
      
      if (result.cohortData) {
        setCohortData(result.cohortData);
      }
      
      if (result.abTestData) {
        setAbTests(result.abTestData);
      }
      
      await loadSnapshotHistory();
      Analytics.trackCustom('analytics', 'snapshot_refresh', { timeframe, trace_id: traceId });
    } catch (error) {
      console.error('Failed to refresh snapshot via DAG:', error);
      toast({ title: 'Error', description: 'Failed to refresh snapshot', variant: 'destructive' });
    } finally {
      setIsRefreshing(false);
    }
  };

  const loadSnapshotHistory = async () => {
    try {
      const traceId = searchParams.get('trace_id') ?? undefined;
      if (!traceId) return;
      const res = await supabase.functions.invoke<{ data: { id: string; created_at: string; timeframe: string }[] }>('analytics-snapshots', {
        body: { action: 'list', trace_id: traceId, timeframe }
      });
      if (res.error) throw res.error;
      setSnapshotList(res.data?.data ?? []);
    } catch (error) {
      setSnapshotList([]);
    }
  };

  const loadSnapshotById = async (id: string) => {
    try {
      const traceId = searchParams.get('trace_id') ?? undefined;
      if (!traceId) return;
      const { data, error } = await supabase
        .from('analytics_snapshots')
        .select('payload, created_at')
        .eq('id', id)
        .maybeSingle();
      if (error) throw error;
      if (data?.payload) {
        setViewingHistorical(true);
        setMetrics(data.payload);
        setIsSnapshot(true);
        setSnapshotCreatedAt(data.created_at ?? null);
        Analytics.trackCustom('analytics', 'snapshot_select', { snapshot_id: id, timeframe });
      }
    } catch (error) {
      toast({ title: 'Error', description: 'Failed to load snapshot', variant: 'destructive' });
    }
  };

  const loadAllAnalytics = async (traceId?: string) => {
    try {
      const dag = buildAnalyticsDAG({
        traceId,
        timeframe,
        includeSnapshots: true,
        includeBI: true,
        includeWorkflow: true,
        includeCohort: true,
        includeABTests: true,
      });
      
      const result = await runAnalyticsPlan(dag, supabase);
      
      if (result.metrics) {
        setMetrics(result.metrics);
        setIsSnapshot(result.isSnapshot);
        setSnapshotCreatedAt(result.snapshotCreatedAt ?? null);
      }
      
      if (result.workflowStatus) {
        setWorkflowStatus(result.workflowStatus);
      }
      
      if (result.workflowSteps) {
        setWorkflowSteps(result.workflowSteps);
      }
      
      if (result.cohortData) {
        setCohortData(result.cohortData);
      }
      
      if (result.abTestData) {
        setAbTests(result.abTestData);
      }
    } catch (error) {
      console.error('Failed to load analytics via DAG:', error);
      toast({
        title: 'Analytics Error',
        description: 'Failed to load analytics data',
        variant: 'destructive',
      });
    }
  };

  const exportMetrics = () => {
    if (!metrics) return;
    const traceId = searchParams.get('trace_id') ?? 'no-trace';
    const blob = new Blob([JSON.stringify(metrics)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics_${traceId}_${timeframe}.json`;
    a.click();
    if (typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(url);
    }
  };

  const exportMetricsCsv = () => {
    if (!metrics) return;
    const rows: string[] = [
      'metric,value',
      `total_revenue,${metrics.revenue?.current ?? metrics.revenue?.totalRevenue ?? ''}`,
      `revenue_mom,${metrics.revenue?.change ?? metrics.revenue?.revenueGrowth?.monthOverMonth ?? ''}`,
      `active_users,${metrics.users?.current ?? metrics.users?.activeUsers ?? ''}`,
      `new_users,${metrics.users?.previous ?? metrics.users?.newUsers ?? ''}`,
    ];
    const csv = rows.join('\n');
    const traceId = searchParams.get('trace_id') ?? 'no-trace';
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analytics_${traceId}_${timeframe}.csv`;
    a.click();
    if (typeof URL.revokeObjectURL === 'function') {
      URL.revokeObjectURL(url);
    }
  };

  const shareLink = async () => {
    const traceId = searchParams.get('trace_id');
    const url = new URL(window.location.href);
    if (traceId) url.searchParams.set('trace_id', traceId);
    url.searchParams.set('timeframe', timeframe);
    url.searchParams.set('tab', 'analytics');
    await navigator.clipboard.writeText(url.toString());
    toast({ title: 'Link copied', description: 'Dashboard link with timeframe copied' });
  };

  const loadCohortAnalysis = async () => {
    try {
      const { data, error } = await supabase.functions.invoke('cohort-analysis', {
        body: {
          cohortType: 'monthly',
          metrics: ['retention', 'revenue'],
          timeRange: {
            start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
            end: new Date().toISOString()
          }
        }
      });

      if (error) throw error;
      setCohortData(data);
    } catch (error) {
      console.error('Error loading cohort analysis:', error);
    }
  };

  const loadABTests = async () => {
    try {
      const { data, error } = await supabase.functions.invoke('ab-testing-engine', {
        body: { action: 'list_tests' }
      });

      if (error) throw error;
      setAbTests(data);
    } catch (error) {
      console.error('Error loading A/B tests:', error);
    }
  };

  const createNewABTest = async () => {
    try {
      const { data, error } = await supabase.functions.invoke('ab-testing-engine', {
        body: {
          action: 'create',
          testConfig: {
            name: 'Dashboard Layout Test',
            hypothesis: 'New dashboard layout will improve user engagement by 15%',
            variants: [
              { name: 'control', description: 'Current layout', weight: 0.5, config: { layout: 'current' } },
              { name: 'new_layout', description: 'Improved layout', weight: 0.5, config: { layout: 'enhanced' } }
            ],
            metrics: ['engagement_rate', 'session_duration', 'feature_adoption'],
            duration: 30,
            targetSampleSize: 1000
          }
        }
      });

      if (error) throw error;
      
      toast({
        title: "A/B Test Created",
        description: `Test "${data.test.name}" has been created successfully.`,
      });
      
      loadABTests(); // Refresh tests list
    } catch (error) {
      console.error('Error creating A/B test:', error);
      toast({
        title: "Error",
        description: "Failed to create A/B test",
        variant: "destructive",
      });
    }
  };

  const getTrendIcon = (value: number) => {
    if (value > 0) return <ArrowUpRight className="h-4 w-4 text-success" />;
    if (value < 0) return <ArrowDownRight className="h-4 w-4 text-error" />;
    return <Minus className="h-4 w-4 text-gray-500" />;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', { 
      style: 'currency', 
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const isStaleSnapshot = (() => {
    if (!isSnapshot || !snapshotCreatedAt) return false;
    const ageMs = Date.now() - new Date(snapshotCreatedAt).getTime();
    return ageMs > 12 * 60 * 60 * 1000;
  })();

  if (isLoading || !metrics) {
    return (
      <div className="space-y-6">
        <div className="grid md:grid-cols-4 gap-4">
          {Array(4).fill(0).map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="animate-pulse space-y-2">
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-8 bg-muted rounded w-1/2"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Advanced Business Intelligence</h2>
          <p className="text-muted-foreground">
            AI-powered analytics and insights for data-driven decisions
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {isSnapshot && <Badge variant="secondary">Snapshot</Badge>}
          {isSnapshot && snapshotCreatedAt && <Badge variant="outline">{new Date(snapshotCreatedAt).toLocaleString()}</Badge>}
          {isSnapshot && isStaleSnapshot && <Badge variant="destructive">Stale</Badge>}
          {workflowStatus && <Badge variant="outline">Workflow: {workflowStatus}</Badge>}
          <Button onClick={refreshSnapshot} size="sm" disabled={isRefreshing || !hasTraceId}>
            <RefreshCw className={`h-4 w-4 mr-1 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={exportMetrics} size="sm" variant="outline" disabled={!metrics}>
            Export
          </Button>
          <Button onClick={exportMetricsCsv} size="sm" variant="outline" disabled={!metrics}>
            Export CSV
          </Button>
          <Button onClick={shareLink} size="sm" variant="outline">
            Share
          </Button>
          <Button onClick={loadSnapshotHistory} size="sm" variant="outline" disabled={!hasTraceId}>
            History
          </Button>
          <Button onClick={() => setShowCharts(!showCharts)} size="sm" variant="outline" disabled={!metrics}>
            {showCharts ? 'Hide Charts' : 'View Charts'}
          </Button>
          {snapshotList.length > 0 && (
            <div className="flex gap-2">
              {snapshotList.slice(0, 3).map((s, i) => (
                <Button key={s.id} size="sm" variant="outline" onClick={() => loadSnapshotById(s.id)}>
                  {`Snapshot ${i + 1}`}
                </Button>
              ))}
            </div>
          )}
          {(['7d', '30d', '90d', '1y'] as const).map((period) => (
            <Button
              key={period}
              variant={timeframe === period ? 'default' : 'outline'}
              onClick={() => {
                setTimeframe(period);
                const next = new URLSearchParams(searchParams);
                next.set('timeframe', period);
                setSearchParams(next);
              }}
              size="sm"
            >
              {period}
            </Button>
          ))}
        </div>
      </div>

      {showCharts && metrics && (
        <Suspense fallback={<div className="p-6">Loading charts…</div>}>
          <AnalyticsCharts subscriptionAnalytics={{
            tierDistribution: [],
            monthlyTrends: [{ month: timeframe, new: Number(metrics.users?.previous ?? 0), churned: Number(metrics.churn?.current ?? 0), net: Number(metrics.users?.current ?? 0), revenue: Number(metrics.revenue?.current ?? 0) }],
            cohortAnalysis: [],
            churnPrediction: []
          }} />
        </Suspense>
      )}

      {/* Key Metrics */}
      <div className="grid md:grid-cols-4 gap-4">
        {workflowStatus && (
          <Card>
            <CardHeader>
              <CardTitle>Workflow Progress</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm">
                {(showAllSteps ? workflowSteps : workflowSteps.slice(0, 5)).map((s, i) => (
                  <div key={`${s.step}-${i}`} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="capitalize">{s.step.replace('_', ' ')}</span>
                      <Badge variant={s.status === 'succeeded' ? 'secondary' : s.status === 'failed' ? 'destructive' : 'outline'}>{s.status}</Badge>
                    </div>
                    {s.error_message && (
                      <div className="text-xs text-destructive">{s.error_message}</div>
                    )}
                  </div>
                ))}
                {workflowSteps.length > 5 && (
                  <div className="flex justify-end">
                    <Button size="sm" variant="outline" onClick={() => setShowAllSteps(!showAllSteps)}>
                      {showAllSteps ? 'Hide Steps' : 'View Steps'}
                    </Button>
                  </div>
                )}
              </div>
              
            </CardContent>
          </Card>
        )}
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Total Revenue</p>
                <p className="text-2xl font-bold">{formatCurrency(metrics.revenue.totalRevenue)}</p>
                <div className="flex items-center gap-1 text-sm">
                  {getTrendIcon(metrics.revenue.revenueGrowth.monthOverMonth)}
                  <span className={metrics.revenue.revenueGrowth.monthOverMonth > 0 ? 'text-success' : 'text-error'}>
                    {metrics.revenue.revenueGrowth.monthOverMonth}%
                  </span>
                  <span className="text-muted-foreground">vs last month</span>
                </div>
              </div>
              <DollarSign className="h-8 w-8 text-success" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Active Users</p>
                <p className="text-2xl font-bold">{metrics.users.activeUsers.toLocaleString()}</p>
                <div className="flex items-center gap-1 text-sm">
                  {getTrendIcon(15.2)}
                  <span className="text-success">+{metrics.users.newUsers}</span>
                  <span className="text-muted-foreground">new this month</span>
                </div>
              </div>
              <Users className="h-8 w-8 text-info" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Conversion Rate</p>
                <p className="text-2xl font-bold">{formatPercentage(metrics.conversions.trialToSubscription)}</p>
                <div className="flex items-center gap-1 text-sm">
                  {getTrendIcon(2.3)}
                  <span className="text-success">+2.3%</span>
                  <span className="text-muted-foreground">improvement</span>
                </div>
              </div>
              <Target className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Churn Rate</p>
                <p className="text-2xl font-bold">{formatPercentage(metrics.churn.overallChurnRate)}</p>
                <div className="flex items-center gap-1 text-sm">
                  {getTrendIcon(-1.2)}
                  <span className="text-success">-1.2%</span>
                  <span className="text-muted-foreground">vs last month</span>
                </div>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analytics */}
      <Tabs defaultValue="revenue" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="revenue">Revenue</TabsTrigger>
          <TabsTrigger value="cohorts">Cohorts</TabsTrigger>
          <TabsTrigger value="experiments">A/B Tests</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        <TabsContent value="revenue">
          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Revenue by Tier</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(metrics.revenue.revenueByTier).map(([tier, revenue]) => (
                    <div key={tier} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className={`w-3 h-3 rounded-full ${
                          tier === 'core' ? 'bg-blue-500' :
                          tier === 'adaptive' ? 'bg-green-500' :
                          tier === 'performance' ? 'bg-purple-500' : 'bg-orange-500'
                        }`} />
                        <span className="capitalize">{tier}</span>
                      </div>
                      <span className="font-medium">{formatCurrency(revenue as number)}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Growth Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>CAC</span>
                    <span className="font-medium">{formatCurrency(metrics.growth.customerAcquisitionCost)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>LTV</span>
                    <span className="font-medium">{formatCurrency(metrics.growth.lifetimeValue)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>LTV:CAC Ratio</span>
                    <span className="font-medium">{metrics.growth.marketingEfficiency.ltv_cac_ratio}:1</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Payback Period</span>
                    <span className="font-medium">{metrics.growth.paybackPeriod} months</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="cohorts">
          {cohortData && (
            <div className="grid md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Retention Analysis</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(cohortData.analysis.retentionMetrics.overallRetention).map(([period, rate]) => (
                      <div key={period} className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{period.replace('day', 'Day ')}</span>
                          <span>{formatPercentage(rate as number)}</span>
                        </div>
                        <Progress value={(rate as number) * 100} className="h-2" />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Cohort Insights</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {cohortData.insights.keyFindings.map((finding: string, index: number) => (
                      <div key={index} className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{finding}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="experiments">
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold">A/B Testing Framework</h3>
              <Button onClick={createNewABTest}>
                Create New Test
              </Button>
            </div>
            
            {abTests && (
              <>
                <div className="grid md:grid-cols-2 gap-4">
                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Active Tests</p>
                          <p className="text-2xl font-bold">{abTests.activeTests?.length || 0}</p>
                        </div>
                        <BarChart3 className="h-8 w-8 text-info" />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-muted-foreground">Completed Tests</p>
                          <p className="text-2xl font-bold">{abTests.completedTests?.length || 0}</p>
                        </div>
                        <CheckCircle className="h-8 w-8 text-success" />
                      </div>
                    </CardContent>
                  </Card>
                </div>

                <div className="space-y-4">
                  {abTests.activeTests?.map((test) => (
                    <Card key={test.id}>
                      <CardContent className="p-6">
                        <div className="flex items-center justify-between mb-4">
                          <div>
                            <h3 className="font-semibold">{test.name}</h3>
                            <p className="text-sm text-muted-foreground">{test.hypothesis}</p>
                          </div>
                          <Badge variant="outline">Active</Badge>
                        </div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span>Progress</span>
                            <span>{Math.round(test.progress * 100)}%</span>
                          </div>
                          <Progress value={test.progress * 100} />
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </>
            )}
          </div>
        </TabsContent>

        <TabsContent value="insights">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Key Business Insights</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {metrics.insights.keyFindings.map((finding: string, index: number) => (
                    <div key={index} className="flex items-start gap-2">
                      <CheckCircle className="h-4 w-4 text-success mt-0.5 flex-shrink-0" />
                      <span className="text-sm">{finding}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Actionable Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {metrics.insights.recommendations.map((rec, index) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium">{rec.action}</h4>
                        <div className="flex gap-2">
                          <Badge variant="outline">{rec.impact} Impact</Badge>
                          <Badge variant="secondary">{rec.effort} Effort</Badge>
                        </div>
                      </div>
                      <p className="text-sm text-muted-foreground mb-2">Category: {rec.category}</p>
                      <p className="text-sm">Timeline: {rec.timeline}</p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>System Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {metrics.insights.alerts.map((alert, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <AlertTriangle className={`h-4 w-4 mt-0.5 ${
                        alert.type === 'warning' ? 'text-warning' : 'text-success'
                      }`} />
                      <span className="text-sm">{alert.message}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AdvancedBusinessIntelligence;
