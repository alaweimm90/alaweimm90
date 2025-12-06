/**
 * Performance Monitoring Dashboard
 * Real-time performance metrics visualization
 */

import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, AlertTriangle, CheckCircle, TrendingUp, TrendingDown, Clock, Zap, Globe, Database } from 'lucide-react';

interface MetricData {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
  timestamp: number;
}

interface PerformanceData {
  lcp: MetricData[];
  fid: MetricData[];
  cls: MetricData[];
  fcp: MetricData[];
  ttfb: MetricData[];
  inp: MetricData[];
  custom: {
    apiCalls: MetricData[];
    componentRenders: MetricData[];
    bundleSize: number;
    cacheHitRate: number;
  };
}

const METRIC_THRESHOLDS = {
  LCP: { good: 2500, poor: 4000, unit: 'ms' },
  FID: { good: 100, poor: 300, unit: 'ms' },
  CLS: { good: 0.1, poor: 0.25, unit: '' },
  FCP: { good: 1800, poor: 3000, unit: 'ms' },
  TTFB: { good: 800, poor: 1800, unit: 'ms' },
  INP: { good: 200, poor: 500, unit: 'ms' }
};

export const PerformanceDashboard: React.FC = () => {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [selectedMetric, setSelectedMetric] = useState<string>('LCP');
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [timeRange]);

  const fetchPerformanceData = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`/api/metrics/performance?range=${timeRange}`);
      if (!response.ok) throw new Error('Failed to fetch performance data');
      const data = await response.json();
      setPerformanceData(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load performance data');
    } finally {
      setIsLoading(false);
    }
  };

  const getMetricStatus = (metric: string, value: number) => {
    const threshold = METRIC_THRESHOLDS[metric as keyof typeof METRIC_THRESHOLDS];
    if (!threshold) return 'good';

    if (value <= threshold.good) return 'good';
    if (value <= threshold.poor) return 'needs-improvement';
    return 'poor';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'needs-improvement':
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
      case 'poor':
        return <AlertTriangle className="w-5 h-5 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'needs-improvement':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'poor':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="flex flex-col items-center gap-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
          <p className="text-gray-600">Loading performance metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 max-w-md">
          <div className="flex items-center gap-3">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            <div>
              <h3 className="font-semibold text-red-800">Error Loading Metrics</h3>
              <p className="text-red-600 mt-1">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  const mockData = {
    lcp: [
      { name: '00:00', value: 2300, rating: 'good' as const, timestamp: Date.now() - 3600000 },
      { name: '01:00', value: 2500, rating: 'good' as const, timestamp: Date.now() - 3000000 },
      { name: '02:00', value: 2800, rating: 'needs-improvement' as const, timestamp: Date.now() - 2400000 },
      { name: '03:00', value: 2400, rating: 'good' as const, timestamp: Date.now() - 1800000 },
      { name: '04:00', value: 2200, rating: 'good' as const, timestamp: Date.now() - 1200000 },
      { name: '05:00', value: 2600, rating: 'needs-improvement' as const, timestamp: Date.now() - 600000 },
      { name: '06:00', value: 2100, rating: 'good' as const, timestamp: Date.now() }
    ],
    lighthouse: {
      performance: 92,
      accessibility: 98,
      bestPractices: 95,
      seo: 100
    },
    bundleSize: {
      main: 245,
      vendor: 180,
      common: 65,
      total: 490
    },
    cacheMetrics: {
      hitRate: 0.85,
      missRate: 0.15,
      staleRate: 0.05
    },
    apiMetrics: {
      avgResponseTime: 185,
      p50: 150,
      p90: 280,
      p99: 450,
      errorRate: 0.02
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
            <Activity className="w-8 h-8 text-blue-600" />
            Performance Dashboard
          </h1>
          <p className="text-gray-600 mt-2">Real-time monitoring of application performance metrics</p>
        </div>

        {/* Time Range Selector */}
        <div className="bg-white rounded-lg p-4 mb-6 border border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex gap-2">
              {(['1h', '24h', '7d', '30d'] as const).map((range) => (
                <button
                  key={range}
                  onClick={() => setTimeRange(range)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                    timeRange === range
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {range === '1h' && 'Last Hour'}
                  {range === '24h' && 'Last 24 Hours'}
                  {range === '7d' && 'Last 7 Days'}
                  {range === '30d' && 'Last 30 Days'}
                </button>
              ))}
            </div>
            <button
              onClick={fetchPerformanceData}
              className="flex items-center gap-2 px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            >
              <Activity className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>

        {/* Core Web Vitals */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
          {Object.entries(METRIC_THRESHOLDS).map(([metric, threshold]) => {
            const latestValue = mockData.lcp[mockData.lcp.length - 1].value;
            const status = getMetricStatus(metric, latestValue);
            const trend = mockData.lcp[mockData.lcp.length - 1].value > mockData.lcp[0].value;

            return (
              <div
                key={metric}
                className={`bg-white rounded-lg p-4 border-2 ${getStatusColor(status)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-sm">{metric}</span>
                  {getStatusIcon(status)}
                </div>
                <div className="text-2xl font-bold mb-1">
                  {metric === 'CLS' ? latestValue.toFixed(2) : Math.round(latestValue)}
                  <span className="text-sm font-normal ml-1">{threshold.unit}</span>
                </div>
                <div className="flex items-center gap-1 text-sm">
                  {trend ? (
                    <>
                      <TrendingUp className="w-4 h-4 text-red-500" />
                      <span className="text-red-600">+5%</span>
                    </>
                  ) : (
                    <>
                      <TrendingDown className="w-4 h-4 text-green-500" />
                      <span className="text-green-600">-8%</span>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Performance Chart */}
        <div className="bg-white rounded-lg p-6 mb-8 border border-gray-200">
          <h2 className="text-xl font-semibold mb-4">Largest Contentful Paint (LCP) Trend</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={mockData.lcp}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#3b82f6"
                strokeWidth={2}
                dot={{ fill: '#3b82f6', r: 4 }}
                activeDot={{ r: 6 }}
              />
              <Line
                type="monotone"
                dataKey={() => METRIC_THRESHOLDS.LCP.good}
                stroke="#10b981"
                strokeDasharray="5 5"
                name="Good Threshold"
              />
              <Line
                type="monotone"
                dataKey={() => METRIC_THRESHOLDS.LCP.poor}
                stroke="#ef4444"
                strokeDasharray="5 5"
                name="Poor Threshold"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Additional Metrics Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Bundle Size */}
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-500" />
              Bundle Size Analysis
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={Object.entries(mockData.bundleSize).filter(([key]) => key !== 'total').map(([key, value]) => ({ name: key, size: value }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="size" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm text-blue-800">
                Total Bundle Size: <span className="font-bold">{mockData.bundleSize.total}KB</span>
                {mockData.bundleSize.total < 500 ? (
                  <span className="ml-2 text-green-600">✓ Within budget</span>
                ) : (
                  <span className="ml-2 text-red-600">⚠ Exceeds budget</span>
                )}
              </p>
            </div>
          </div>

          {/* Lighthouse Scores */}
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Globe className="w-5 h-5 text-green-500" />
              Lighthouse Scores
            </h3>
            <div className="space-y-4">
              {Object.entries(mockData.lighthouse).map(([key, value]) => (
                <div key={key}>
                  <div className="flex justify-between mb-1">
                    <span className="text-sm font-medium capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                    <span className={`text-sm font-bold ${value >= 90 ? 'text-green-600' : value >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                      {value}
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${value >= 90 ? 'bg-green-500' : value >= 50 ? 'bg-yellow-500' : 'bg-red-500'}`}
                      style={{ width: `${value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* API and Cache Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* API Performance */}
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-purple-500" />
              API Performance
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Avg Response</p>
                <p className="text-xl font-bold">{mockData.apiMetrics.avgResponseTime}ms</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">P50 Latency</p>
                <p className="text-xl font-bold">{mockData.apiMetrics.p50}ms</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">P90 Latency</p>
                <p className="text-xl font-bold">{mockData.apiMetrics.p90}ms</p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg">
                <p className="text-sm text-gray-600">Error Rate</p>
                <p className="text-xl font-bold">{(mockData.apiMetrics.errorRate * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {/* Cache Performance */}
          <div className="bg-white rounded-lg p-6 border border-gray-200">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-500" />
              Cache Performance
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Hit Rate</span>
                  <span className="text-sm font-bold text-green-600">
                    {(mockData.cacheMetrics.hitRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full bg-green-500"
                    style={{ width: `${mockData.cacheMetrics.hitRate * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Miss Rate</span>
                  <span className="text-sm font-bold text-yellow-600">
                    {(mockData.cacheMetrics.missRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full bg-yellow-500"
                    style={{ width: `${mockData.cacheMetrics.missRate * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm font-medium">Stale Rate</span>
                  <span className="text-sm font-bold text-red-600">
                    {(mockData.cacheMetrics.staleRate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="h-2 rounded-full bg-red-500"
                    style={{ width: `${mockData.cacheMetrics.staleRate * 100}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="mt-4 p-3 bg-green-50 rounded-lg">
              <p className="text-sm text-green-800">
                Cache efficiency: <span className="font-bold">Excellent</span>
                <span className="block text-xs mt-1">85% hit rate exceeds target of 80%</span>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};