/**
 * Admin Monitoring Dashboard
 * Comprehensive system monitoring and observability dashboard
 */

import { useEffect, useState, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Database,
  Gauge,
  AlertTriangle,
  TrendingUp,
  Server,
  Zap,
  GitBranch,
  Heart,
  Wifi,
} from 'lucide-react';
// Recharts will be loaded lazily when charts enter the viewport

import { performHealthCheck, type HealthCheckResult } from '@/lib/healthCheck';
import { perfMonitor } from '@/lib/performance-monitor';
import { alertManager, type Alert } from '@/lib/alerts';
import { uptimeMonitor } from '@/lib/uptime';

// Monitoring Dashboard Component
export function Monitoring() {
  const [alerts, setAlerts] = useState<Alert[]>(() => alertManager.getAlerts());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const chartContainerRef = useRef<HTMLDivElement | null>(null);
  const [chartsInView, setChartsInView] = useState(false);
  type RechartsSubset = Pick<
    typeof import('recharts'),
    | 'ResponsiveContainer'
    | 'LineChart'
    | 'Line'
    | 'BarChart'
    | 'Bar'
    | 'XAxis'
    | 'YAxis'
    | 'CartesianGrid'
    | 'Tooltip'
    | 'Legend'
    | 'PieChart'
    | 'Pie'
    | 'Cell'
  >;
  const [recharts, setRecharts] = useState<RechartsSubset | null>(null);

  // Health check query
  const { data: health, isLoading: isLoadingHealth } = useQuery({
    queryKey: ['health'],
    queryFn: performHealthCheck,
    refetchInterval: autoRefresh ? 30000 : false, // Every 30 seconds
    staleTime: 15000,
  });

  // Alert subscription
  useEffect(() => {
    const unsubscribe = alertManager.subscribe((newAlerts) => {
      setAlerts(newAlerts);
    });
    return unsubscribe;
  }, []);

  // Lazy load Recharts when charts container enters viewport
  useEffect(() => {
    const el = chartContainerRef.current;
    if (!el) return;
    const io = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setChartsInView(true);
          io.disconnect();
        }
      },
      { rootMargin: '200px' }
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    if (!chartsInView || recharts) return;
    import('recharts').then(mod => {
      setRecharts({
        ResponsiveContainer: mod.ResponsiveContainer,
        LineChart: mod.LineChart,
        Line: mod.Line,
        BarChart: mod.BarChart,
        Bar: mod.Bar,
        XAxis: mod.XAxis,
        YAxis: mod.YAxis,
        CartesianGrid: mod.CartesianGrid,
        Tooltip: mod.Tooltip,
        Legend: mod.Legend,
        PieChart: mod.PieChart,
        Pie: mod.Pie,
        Cell: mod.Cell,
      });
    });
  }, [chartsInView, recharts]);

  const perfMetrics = perfMonitor.getMetrics();
  const perfSummary = perfMonitor.getSummary();
  const webVitals = perfMonitor.getWebVitals();
  const uptimeReport = uptimeMonitor.getReport();
  const alertStats = alertManager.getStatistics();

  return (
    <div className="space-y-8 pb-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-display text-lii-bg">System Monitoring</h1>
          <p className="text-lii-ash mt-2">Real-time system health, performance, and alerts</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              autoRefresh
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            <Zap className="inline mr-2" size={16} />
            {autoRefresh ? 'Live' : 'Paused'}
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="System Status"
          value={health?.status.toUpperCase() || 'CHECKING'}
          icon={Activity}
          color={
            health?.status === 'healthy'
              ? 'green'
              : health?.status === 'degraded'
                ? 'yellow'
                : 'red'
          }
          subtext={health ? `${health.metrics.responseTime.toFixed(0)}ms response` : 'Loading...'}
        />

        <StatCard
          title="Uptime"
          value={uptimeReport.uptime.toFixed(2) + '%'}
          icon={Heart}
          color="green"
          subtext={`${uptimeReport.days}d ${uptimeReport.hours}h running`}
        />

        <StatCard
          title="Active Alerts"
          value={alertStats.activeAlerts.toString()}
          icon={AlertCircle}
          color={alertStats.activeAlerts > 0 ? (alertStats.criticalAlerts > 0 ? 'red' : 'yellow') : 'green'}
          subtext={`${alertStats.criticalAlerts} critical`}
        />

        <StatCard
          title="Performance Score"
          value={perfSummary.vitalsScore.toString()}
          icon={Gauge}
          color={perfSummary.vitalsScore >= 75 ? 'green' : perfSummary.vitalsScore >= 50 ? 'yellow' : 'red'}
          subtext={`${perfSummary.goodRatings} metrics healthy`}
        />
      </div>

      {/* Services Health Grid */}
      {health && (
        <div className="bg-white rounded-lg border border-lii-cloud p-6">
          <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
            <Server size={24} />
            Services Status
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(health.services).map(([name, status]) => (
              <ServiceStatus key={name} name={name} healthy={status} />
            ))}
          </div>
        </div>
      )}

      {/* Core Web Vitals */}
      <div className="bg-white rounded-lg border border-lii-cloud p-6">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
          <TrendingUp size={24} />
          Core Web Vitals
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          <WebVitalCard vital="LCP" value={webVitals.LCP} unit="ms" />
          <WebVitalCard vital="FID" value={webVitals.FID} unit="ms" />
          <WebVitalCard vital="CLS" value={webVitals.CLS} unit="" />
          <WebVitalCard vital="FCP" value={webVitals.FCP} unit="ms" />
          <WebVitalCard vital="TTFB" value={webVitals.TTFB} unit="ms" />
        </div>
      </div>

      {/* Charts */}
      <div ref={chartContainerRef} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Performance Metrics Chart */}
        <div className="bg-white rounded-lg border border-lii-cloud p-6">
          <h3 className="text-xl font-semibold mb-4">Performance Metrics</h3>
          {!recharts ? (
            <div className="h-[300px] w-full bg-lii-cloud/20 animate-pulse rounded-lg" />
          ) : (
            <recharts.ResponsiveContainer width="100%" height={300}>
              <recharts.BarChart
                data={perfMetrics.map(m => ({
                  name: m.name,
                  value: m.value,
                }))}
              >
                <recharts.CartesianGrid strokeDasharray="3 3" />
                <recharts.XAxis dataKey="name" />
                <recharts.YAxis />
                <recharts.Tooltip />
                <recharts.Bar
                  dataKey="value"
                  fill="#8884d8"
                  radius={[8, 8, 0, 0]}
                />
              </recharts.BarChart>
            </recharts.ResponsiveContainer>
          )}
        </div>

        {/* Alert Distribution */}
        <div className="bg-white rounded-lg border border-lii-cloud p-6">
          <h3 className="text-xl font-semibold mb-4">Alert Distribution</h3>
          {!recharts ? (
            <div className="h-[300px] w-full bg-lii-cloud/20 animate-pulse rounded-lg" />
          ) : (
            <recharts.ResponsiveContainer width="100%" height={300}>
              <recharts.PieChart>
                <recharts.Pie
                  data={[
                    { name: 'Info', value: alertStats.bySeverity.info || 0 },
                    { name: 'Warning', value: alertStats.bySeverity.warning || 0 },
                    { name: 'Error', value: alertStats.bySeverity.error || 0 },
                    { name: 'Critical', value: alertStats.bySeverity.critical || 0 },
                  ]}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  <recharts.Cell fill="#3b82f6" />
                  <recharts.Cell fill="#f59e0b" />
                  <recharts.Cell fill="#ef4444" />
                  <recharts.Cell fill="#991b1b" />
                </recharts.Pie>
                <recharts.Tooltip />
              </recharts.PieChart>
            </recharts.ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Recent Alerts */}
      <div className="bg-white rounded-lg border border-lii-cloud p-6">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
          <AlertTriangle size={24} />
          Recent Alerts ({alerts.length})
        </h2>

        {alerts.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <CheckCircle className="mx-auto mb-3 text-green-500" size={32} />
            <p>No alerts at this time</p>
          </div>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {alerts.slice(0, 20).map((alert) => (
              <AlertItem key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </div>

      {/* Uptime Summary */}
      <div className="bg-white rounded-lg border border-lii-cloud p-6">
        <h2 className="text-2xl font-semibold mb-6 flex items-center gap-2">
          <Wifi size={24} />
          Uptime Report
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <p className="text-sm text-gray-600">Uptime Percentage</p>
            <p className="text-3xl font-bold text-green-600">{uptimeReport.uptime.toFixed(2)}%</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Total Downtime</p>
            <p className="text-3xl font-bold text-red-600">
              {Math.floor(uptimeReport.downtime / (1000 * 60))}m
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Running Duration</p>
            <p className="text-3xl font-bold text-blue-600">
              {uptimeReport.days}d {uptimeReport.hours}h
            </p>
          </div>
        </div>

        {uptimeMonitor.getRecentDowntimeEvents(5).length > 0 && (
          <div className="mt-6 pt-6 border-t">
            <h3 className="font-semibold mb-4">Recent Downtime Events</h3>
            <div className="space-y-2">
              {uptimeMonitor.getRecentDowntimeEvents(5).map((event, idx) => (
                <div key={idx} className="flex justify-between text-sm py-2 border-b last:border-b-0">
                  <div>
                    <p className="font-medium">{event.reason || 'Downtime'}</p>
                    <p className="text-xs text-gray-500">{new Date(event.start).toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <p className={`font-medium ${event.severity === 'critical' ? 'text-red-600' : event.severity === 'major' ? 'text-orange-600' : 'text-yellow-600'}`}>
                      {event.severity}
                    </p>
                    <p className="text-xs text-gray-500">{Math.round(event.duration / 1000)}s</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Component Exports
// ============================================================================

interface StatCardProps {
  title: string;
  value: string;
  icon: React.ComponentType<{ size?: number; color?: string }>;
  color: 'green' | 'yellow' | 'red';
  subtext?: string;
}

function StatCard({ title, value, icon: Icon, color, subtext }: StatCardProps) {
  const bgColor = color === 'green' ? 'bg-green-50' : color === 'yellow' ? 'bg-yellow-50' : 'bg-red-50';
  const textColor = color === 'green' ? 'text-green-600' : color === 'yellow' ? 'text-yellow-600' : 'text-red-600';
  const iconColor = color === 'green' ? '#16a34a' : color === 'yellow' ? '#ca8a04' : '#dc2626';

  return (
    <div className={`${bgColor} rounded-lg p-6 border border-lii-cloud`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className={`text-3xl font-bold ${textColor}`}>{value}</p>
          {subtext && <p className="text-xs text-gray-500 mt-2">{subtext}</p>}
        </div>
        <Icon size={32} color={iconColor} />
      </div>
    </div>
  );
}

interface ServiceStatusProps {
  name: string;
  healthy?: boolean;
}

function ServiceStatus({ name, healthy }: ServiceStatusProps) {
  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <div className="flex items-center gap-3">
        {healthy ? (
          <CheckCircle className="text-green-500" size={24} />
        ) : (
          <AlertCircle className="text-red-500" size={24} />
        )}
        <div>
          <p className="font-medium">{name}</p>
          <p className={`text-xs ${healthy ? 'text-green-600' : 'text-red-600'}`}>
            {healthy ? 'Operational' : 'Down'}
          </p>
        </div>
      </div>
    </div>
  );
}

interface WebVitalCardProps {
  vital: string;
  value: { value: number; rating: 'good' | 'needs-improvement' | 'poor' } | null;
  unit: string;
}

function WebVitalCard({ vital, value, unit }: WebVitalCardProps) {
  if (!value) {
    return (
      <div className="p-4 border border-gray-200 rounded-lg">
        <p className="text-sm font-medium text-gray-600">{vital}</p>
        <p className="text-lg font-bold text-gray-400">-</p>
        <p className="text-xs text-gray-500 mt-1">No data</p>
      </div>
    );
  }

  const getRatingColor = (rating: string) => {
    return rating === 'good' ? 'text-green-600' : rating === 'needs-improvement' ? 'text-yellow-600' : 'text-red-600';
  };

  return (
    <div className="p-4 border border-gray-200 rounded-lg">
      <p className="text-sm font-medium text-gray-600">{vital}</p>
      <p className={`text-2xl font-bold ${getRatingColor(value.rating)}`}>
        {Math.round(value.value)}{unit}
      </p>
      <p className={`text-xs capitalize mt-1 ${getRatingColor(value.rating)}`}>
        {value.rating}
      </p>
    </div>
  );
}

interface AlertItemProps {
  alert: Alert;
}

function AlertItem({ alert }: AlertItemProps) {
  const severityColors = {
    info: 'border-blue-500 bg-blue-50',
    warning: 'border-yellow-500 bg-yellow-50',
    error: 'border-red-500 bg-red-50',
    critical: 'border-red-700 bg-red-50',
  };

  const severityIcons = {
    info: <AlertCircle size={18} className="text-blue-600" />,
    warning: <AlertTriangle size={18} className="text-yellow-600" />,
    error: <AlertCircle size={18} className="text-red-600" />,
    critical: <AlertCircle size={18} className="text-red-700" />,
  };

  return (
    <div className={`border-l-4 ${severityColors[alert.severity]} p-4 rounded`}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-3 flex-1">
          {severityIcons[alert.severity]}
          <div className="flex-1">
            <p className="font-semibold">{alert.title}</p>
            <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
            <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
              <span>{alert.source}</span>
              <span>{new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
          </div>
        </div>
        {alert.resolved ? (
          <span className="text-xs bg-green-200 text-green-800 px-2 py-1 rounded">Resolved</span>
        ) : (
          <span className="text-xs bg-yellow-200 text-yellow-800 px-2 py-1 rounded">Active</span>
        )}
      </div>
    </div>
  );
}

export default Monitoring;
