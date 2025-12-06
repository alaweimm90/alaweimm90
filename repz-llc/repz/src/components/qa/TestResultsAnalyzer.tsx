import React, { useState, useMemo } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line } from 'recharts';
import { 
  TestTube, 
  CheckCircle, 
  XCircle, 
  Clock, 
  AlertTriangle,
  TrendingUp,
  Filter,
  Download,
  Calendar,
  Bug,
  Target
} from 'lucide-react';

interface TestResult {
  id: string;
  suite: string;
  name: string;
  status: 'passed' | 'failed' | 'skipped' | 'pending';
  duration: number;
  timestamp: string;
  category: 'unit' | 'integration' | 'e2e' | 'performance';
  error?: string;
  retries: number;
}

interface TestSuiteStats {
  suite: string;
  total: number;
  passed: number;
  failed: number;
  skipped: number;
  duration: number;
  coverage: number;
}

export const TestResultsAnalyzer = () => {
  const [selectedSuite, setSelectedSuite] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d'>('7d');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const testResults: TestResult[] = useMemo(() => [
    {
      id: '1',
      suite: 'Auth Components',
      name: 'should render login form correctly',
      status: 'passed',
      duration: 120,
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      category: 'unit',
      retries: 0
    },
    {
      id: '2',
      suite: 'Dashboard Tests',
      name: 'should load user dashboard',
      status: 'failed',
      duration: 850,
      timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000).toISOString(),
      category: 'integration',
      error: 'Timeout waiting for dashboard to load',
      retries: 2
    },
    {
      id: '3',
      suite: 'API Integration',
      name: 'should handle payment processing',
      status: 'passed',
      duration: 2340,
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      category: 'e2e',
      retries: 1
    },
    {
      id: '4',
      suite: 'Performance Tests',
      name: 'should load page under 2s',
      status: 'skipped',
      duration: 0,
      timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      category: 'performance',
      retries: 0
    },
    {
      id: '5',
      suite: 'Security Tests',
      name: 'should prevent XSS attacks',
      status: 'passed',
      duration: 567,
      timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      category: 'unit',
      retries: 0
    }
  ], []);

  const suiteStats: TestSuiteStats[] = useMemo(() => [
    {
      suite: 'Auth Components',
      total: 24,
      passed: 22,
      failed: 1,
      skipped: 1,
      duration: 4560,
      coverage: 94
    },
    {
      suite: 'Dashboard Tests',
      total: 18,
      passed: 15,
      failed: 2,
      skipped: 1,
      duration: 12340,
      coverage: 87
    },
    {
      suite: 'API Integration',
      total: 32,
      passed: 28,
      failed: 3,
      skipped: 1,
      duration: 45600,
      coverage: 91
    },
    {
      suite: 'Performance Tests',
      total: 12,
      passed: 8,
      failed: 1,
      skipped: 3,
      duration: 23400,
      coverage: 78
    },
    {
      suite: 'Security Tests',
      total: 16,
      passed: 15,
      failed: 0,
      skipped: 1,
      duration: 8900,
      coverage: 96
    }
  ], []);

  const filteredResults = useMemo(() => {
    return testResults.filter(result => {
      if (selectedSuite !== 'all' && result.suite !== selectedSuite) return false;
      if (statusFilter !== 'all' && result.status !== statusFilter) return false;
      return true;
    });
  }, [testResults, selectedSuite, statusFilter]);

  const overallStats = useMemo(() => {
    const total = suiteStats.reduce((acc, suite) => acc + suite.total, 0);
    const passed = suiteStats.reduce((acc, suite) => acc + suite.passed, 0);
    const failed = suiteStats.reduce((acc, suite) => acc + suite.failed, 0);
    const skipped = suiteStats.reduce((acc, suite) => acc + suite.skipped, 0);
    const duration = suiteStats.reduce((acc, suite) => acc + suite.duration, 0);
    const coverage = suiteStats.reduce((acc, suite) => acc + suite.coverage, 0) / suiteStats.length;

    return { total, passed, failed, skipped, duration, coverage };
  }, [suiteStats]);

  const pieData = [
    { name: 'Passed', value: overallStats.passed, color: '#22c55e' },
    { name: 'Failed', value: overallStats.failed, color: '#ef4444' },
    { name: 'Skipped', value: overallStats.skipped, color: '#f59e0b' }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'skipped': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'pending': return <AlertTriangle className="h-4 w-4 text-orange-500" />;
      default: return <TestTube className="h-4 w-4" />;
    }
  };

  const getCategoryBadgeColor = (category: string) => {
    switch (category) {
      case 'unit': return 'default';
      case 'integration': return 'secondary';
      case 'e2e': return 'outline';
      case 'performance': return 'destructive';
      default: return 'default';
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Test Results Analyzer</h2>
          <p className="text-muted-foreground">Analyze test results and identify patterns</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => console.log("TestResultsAnalyzer button clicked")}>
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="outline" size="sm" onClick={() => console.log("TestResultsAnalyzer button clicked")}>
            <Calendar className="h-4 w-4 mr-2" />
            Schedule
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-4 flex-wrap">
        <div className="flex gap-2">
          <Filter className="h-4 w-4 self-center text-muted-foreground" />
          <span className="text-sm font-medium">Filters:</span>
        </div>
        
        <select 
          value={selectedSuite} 
          onChange={(e) => setSelectedSuite(e.target.value)}
          className="px-3 py-1 border rounded-md bg-background"
        >
          <option value="all">All Suites</option>
          {suiteStats.map(suite => (
            <option key={suite.suite} value={suite.suite}>{suite.suite}</option>
          ))}
        </select>

        <select 
          value={statusFilter} 
          onChange={(e) => setStatusFilter(e.target.value)}
          className="px-3 py-1 border rounded-md bg-background"
        >
          <option value="all">All Status</option>
          <option value="passed">Passed</option>
          <option value="failed">Failed</option>
          <option value="skipped">Skipped</option>
        </select>

        <select 
          value={timeRange} 
          onChange={(e) => setTimeRange(e.target.value as '24h' | '7d' | '30d')}
          className="px-3 py-1 border rounded-md bg-background"
        >
          <option value="24h">Last 24 Hours</option>
          <option value="7d">Last 7 Days</option>
          <option value="30d">Last 30 Days</option>
        </select>
      </div>

      {/* Overview Stats */}
      <div className="grid gap-4 md:grid-cols-5">
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{overallStats.total}</div>
            <p className="text-xs text-muted-foreground">Total Tests</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-green-600">{overallStats.passed}</div>
            <p className="text-xs text-muted-foreground">Passed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold text-red-600">{overallStats.failed}</div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{overallStats.coverage.toFixed(1)}%</div>
            <p className="text-xs text-muted-foreground">Coverage</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="text-2xl font-bold">{formatDuration(overallStats.duration)}</div>
            <p className="text-xs text-muted-foreground">Total Duration</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="suites">Test Suites</TabsTrigger>
          <TabsTrigger value="results">Test Results</TabsTrigger>
          <TabsTrigger value="trends">Trends</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Test Status Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="flex justify-center gap-4 mt-4">
                  {pieData.map((item) => (
                    <div key={item.name} className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm">{item.name}: {item.value}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Suite Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={suiteStats}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="suite" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="duration" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="suites">
          <div className="space-y-4">
            {suiteStats.map((suite) => (
              <Card key={suite.suite}>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">{suite.suite}</h3>
                    <div className="flex gap-2">
                      <Badge variant="outline">{suite.total} tests</Badge>
                      <Badge variant="outline">{formatDuration(suite.duration)}</Badge>
                    </div>
                  </div>
                  
                  <div className="grid gap-4 md:grid-cols-4">
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span className="text-sm">Passed: {suite.passed}</span>
                      </div>
                      <Progress value={(suite.passed / suite.total) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <XCircle className="h-4 w-4 text-red-500" />
                        <span className="text-sm">Failed: {suite.failed}</span>
                      </div>
                      <Progress value={(suite.failed / suite.total) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Clock className="h-4 w-4 text-yellow-500" />
                        <span className="text-sm">Skipped: {suite.skipped}</span>
                      </div>
                      <Progress value={(suite.skipped / suite.total) * 100} className="h-2" />
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Target className="h-4 w-4 text-blue-500" />
                        <span className="text-sm">Coverage: {suite.coverage}%</span>
                      </div>
                      <Progress value={suite.coverage} className="h-2" />
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle>Recent Test Results</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {filteredResults.map((result) => (
                  <div key={result.id} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(result.status)}
                      <div>
                        <p className="font-medium">{result.name}</p>
                        <p className="text-sm text-muted-foreground">{result.suite}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-3">
                      <Badge variant={getCategoryBadgeColor(result.category)}>
                        {result.category}
                      </Badge>
                      
                      <span className="text-sm text-muted-foreground">
                        {formatDuration(result.duration)}
                      </span>
                      
                      {result.retries > 0 && (
                        <Badge variant="outline">
                          {result.retries} retries
                        </Badge>
                      )}
                      
                      <span className="text-xs text-muted-foreground">
                        {new Date(result.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="trends">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Test Success Rate Trends
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={suiteStats}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="suite" />
                    <YAxis />
                    <Tooltip />
                    <Line 
                      type="monotone" 
                      dataKey="coverage" 
                      stroke="hsl(var(--primary))" 
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};