/**
 * Error Reporting Dashboard
 * Provides insights into application errors and performance
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  Filter,
  Download,
  RefreshCw,
  AlertCircle,
  Info,
  XCircle
} from 'lucide-react';
import { errorReporter, type ErrorReport } from '@/lib/error-reporting';
import Plot from 'react-plotly.js';

interface ErrorReportingDashboardProps {
  className?: string;
}

export function ErrorReportingDashboard({ className }: ErrorReportingDashboardProps) {
  const [reports, setReports] = useState<ErrorReport[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [selectedFilter, setSelectedFilter] = useState<string>('all');
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    const allReports = errorReporter.getReports();
    const errorStats = errorReporter.getErrorStats();
    setReports(allReports);
    setStats(errorStats);
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setTimeout(() => setRefreshing(false), 500);
  };

  const handleExport = () => {
    const dataStr = JSON.stringify(reports, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `error-reports-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="h-4 w-4 text-red-600" />;
      case 'high': return <AlertTriangle className="h-4 w-4 text-orange-600" />;
      case 'medium': return <AlertCircle className="h-4 w-4 text-accent" />;
      case 'low': return <Info className="h-4 w-4 text-primary" />;
      default: return <Info className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': return 'outline';
      default: return 'outline';
    }
  };

  if (!stats) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle>Error Reporting Dashboard</CardTitle>
          <CardDescription>Loading error reports...</CardDescription>
        </CardHeader>
        <CardContent>
          <Progress value={50} className="w-full" />
        </CardContent>
      </Card>
    );
  }

  // Prepare chart data
  const errorTrendData = reports.slice(-20).map((report, index) => ({
    x: index,
    y: 1,
    text: report.message,
    type: report.errorType
  }));

  const severityData = Object.entries(stats.bySeverity).map(([severity, count]) => ({
    labels: [severity],
    values: [count],
    type: 'pie'
  }));

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <AlertTriangle className="h-4 w-4" />
              Total Errors
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Clock className="h-4 w-4" />
              Recent (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">{stats.recentErrors}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <TrendingUp className="h-4 w-4" />
              Error Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {stats.total > 0 ? ((stats.recentErrors / stats.total) * 100).toFixed(1) : 0}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Most Affected</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm font-medium">
              {Object.entries(stats.byModule).sort(([,a], [,b]) => (b as number) - (a as number))[0]?.[0] || 'N/A'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2">
        <Button
          onClick={handleRefresh}
          disabled={refreshing}
          size="sm"
          variant="outline"
          className="flex items-center gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>

        <Button
          onClick={handleExport}
          size="sm"
          variant="outline"
          className="flex items-center gap-2"
        >
          <Download className="h-4 w-4" />
          Export
        </Button>

        <Button
          onClick={() => errorReporter.clearReports()}
          size="sm"
          variant="outline"
          className="text-red-600"
        >
          Clear All
        </Button>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="reports">Reports</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Errors by Type</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(stats.byType).map(([type, count]) => (
                    <div key={type} className="flex items-center justify-between">
                      <span className="capitalize">{type}</span>
                      <Badge variant="secondary">{count as React.ReactNode}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Errors by Severity</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {Object.entries(stats.bySeverity).map(([severity, count]) => (
                    <div key={severity} className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getSeverityIcon(severity)}
                        <span className="capitalize">{severity}</span>
                      </div>
                      <Badge variant={getSeverityColor(severity) as "default" | "secondary" | "destructive" | "outline"}>{count as React.ReactNode}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="reports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Error Reports</CardTitle>
              <CardDescription>Latest error reports from the application</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {reports.slice(0, 10).map((report) => (
                  <div key={report.errorId} className="border rounded-lg p-4">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getSeverityIcon(report.severity)}
                        <Badge variant="outline" className="font-mono text-xs">
                          {report.errorId}
                        </Badge>
                        <Badge variant="secondary">{report.errorType}</Badge>
                        {report.moduleName && (
                          <Badge variant="outline">{report.moduleName}</Badge>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {new Date(report.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <div className="text-sm font-medium mb-1">{report.message}</div>
                    {report.context && (
                      <div className="text-xs text-muted-foreground">
                        Context: {JSON.stringify(report.context)}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Error Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              {Object.keys(stats.bySeverity).length > 0 && (
                <Plot
                  data={[{
                    values: Object.values(stats.bySeverity),
                    labels: Object.keys(stats.bySeverity),
                    type: 'pie',
                    marker: {
                      colors: ['#ef4444', '#f97316', '#eab308', '#3b82f6']
                    }
                  }]}
                  layout={{
                    width: 400,
                    height: 300,
                    title: 'Errors by Severity'
                  }}
                  config={{ responsive: true }}
                />
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Error Trends</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">
                Error trend analysis will be available with more data points.
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default ErrorReportingDashboard;