import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  Play, 
  Pause, 
  RefreshCw,
  Target,
  TestTube,
  Monitor,
  Smartphone
} from 'lucide-react';
import { supabase } from '@/integrations/supabase/client';

interface TestResult {
  id: string;
  test_type: 'unit' | 'integration' | 'e2e';
  test_suite: string;
  test_name: string;
  status: 'passed' | 'failed' | 'running' | 'skipped';
  duration_ms: number;
  error_message?: string;
  created_at: string;
}

interface TestMetrics {
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  coverage_percentage: number;
  average_duration: number;
  last_run: string;
}

const TestingDashboard: React.FC = () => {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testMetrics, setTestMetrics] = useState<TestMetrics | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [selectedTestType, setSelectedTestType] = useState<'all' | 'unit' | 'integration' | 'e2e'>('all');

  // Mock data for demonstration
  const mockTestResults: TestResult[] = [
    {
      id: '1',
      test_type: 'unit',
      test_suite: 'TierCard',
      test_name: 'displays correct pricing for each tier',
      status: 'passed',
      duration_ms: 45,
      created_at: new Date().toISOString()
    },
    {
      id: '2',
      test_type: 'unit',
      test_suite: 'AuthModal',
      test_name: 'validates email format',
      status: 'passed',
      duration_ms: 32,
      created_at: new Date().toISOString()
    },
    {
      id: '3',
      test_type: 'integration',
      test_suite: 'Stripe Integration',
      test_name: 'creates subscription with correct tier',
      status: 'passed',
      duration_ms: 1250,
      created_at: new Date().toISOString()
    },
    {
      id: '4',
      test_type: 'e2e',
      test_suite: 'User Journey',
      test_name: 'complete signup to dashboard flow',
      status: 'failed',
      duration_ms: 15000,
      error_message: 'Element not found: [data-testid="ai-assistant-module"]',
      created_at: new Date().toISOString()
    },
    {
      id: '5',
      test_type: 'unit',
      test_suite: 'Analytics Components',
      test_name: 'renders charts with correct data',
      status: 'passed',
      duration_ms: 67,
      created_at: new Date().toISOString()
    }
  ];

  const mockMetrics: TestMetrics = {
    total_tests: 247,
    passed_tests: 235,
    failed_tests: 8,
    coverage_percentage: 85.7,
    average_duration: 1200,
    last_run: new Date().toISOString()
  };

  useEffect(() => {
    setTestResults(mockTestResults);
    setTestMetrics(mockMetrics);
  }, []);

  const runTests = async (testType: string) => {
    setIsRunning(true);
    console.log(`Running ${testType} tests...`);
    
    // Simulate test execution
    setTimeout(() => {
      setIsRunning(false);
      // Refresh results
      setTestResults([...mockTestResults]);
    }, 3000);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
    }
  };

  const filteredResults = selectedTestType === 'all' 
    ? testResults 
    : testResults.filter(result => result.test_type === selectedTestType);

  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Testing Dashboard</h1>
            <p className="text-muted-foreground">
              Enterprise Testing Standards - Google Testing Pyramid
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => runTests('unit')}
              disabled={isRunning}
              variant="outline"
            >
              {isRunning ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : <TestTube className="h-4 w-4 mr-2" />}
              Run Unit Tests
            </Button>
            <Button
              onClick={() => runTests('all')}
              disabled={isRunning}
            >
              {isRunning ? <RefreshCw className="h-4 w-4 animate-spin mr-2" /> : <Play className="h-4 w-4 mr-2" />}
              Run All Tests
            </Button>
          </div>
        </div>

        {/* Test Metrics Overview */}
        {testMetrics && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Total Tests</CardTitle>
                <Target className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{testMetrics.total_tests}</div>
                <p className="text-xs text-muted-foreground">
                  Across all test types
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
                <CheckCircle className="h-4 w-4 text-green-500" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {((testMetrics.passed_tests / testMetrics.total_tests) * 100).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  {testMetrics.passed_tests} passed, {testMetrics.failed_tests} failed
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Code Coverage</CardTitle>
                <Monitor className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{testMetrics.coverage_percentage}%</div>
                <Progress value={testMetrics.coverage_percentage} className="mt-2" />
                <p className="text-xs text-muted-foreground mt-1">
                  Target: 80%+ coverage
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Avg Duration</CardTitle>
                <RefreshCw className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {(testMetrics.average_duration / 1000).toFixed(1)}s
                </div>
                <p className="text-xs text-muted-foreground">
                  Per test execution
                </p>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Test Types Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Testing Pyramid Distribution</CardTitle>
            <p className="text-sm text-muted-foreground">
              Following Google Testing Standards: 70% Unit, 20% Integration, 10% E2E
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 border rounded-lg">
                <TestTube className="h-8 w-8 mx-auto mb-2 text-blue-500" />
                <h3 className="font-semibold">Unit Tests</h3>
                <p className="text-2xl font-bold text-blue-500">70%</p>
                <p className="text-sm text-muted-foreground">
                  Components, Hooks, Utils
                </p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <Monitor className="h-8 w-8 mx-auto mb-2 text-purple-500" />
                <h3 className="font-semibold">Integration Tests</h3>
                <p className="text-2xl font-bold text-purple-500">20%</p>
                <p className="text-sm text-muted-foreground">
                  API, Database, Auth
                </p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <Smartphone className="h-8 w-8 mx-auto mb-2 text-green-500" />
                <h3 className="font-semibold">E2E Tests</h3>
                <p className="text-2xl font-bold text-green-500">10%</p>
                <p className="text-sm text-muted-foreground">
                  User Journeys, Cross-browser
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Test Results */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Test Results</CardTitle>
              <Tabs value={selectedTestType} onValueChange={(value) => setSelectedTestType(value as any)}>
                <TabsList>
                  <TabsTrigger value="all">All</TabsTrigger>
                  <TabsTrigger value="unit">Unit</TabsTrigger>
                  <TabsTrigger value="integration">Integration</TabsTrigger>
                  <TabsTrigger value="e2e">E2E</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {filteredResults.map((result) => (
                <div
                  key={result.id}
                  className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50"
                >
                  <div className="flex items-center gap-3">
                    {getStatusIcon(result.status)}
                    <div>
                      <div className="font-medium">
                        {result.test_suite} - {result.test_name}
                      </div>
                      {result.error_message && (
                        <div className="text-sm text-red-500 mt-1">
                          {result.error_message}
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <Badge variant={result.test_type === 'unit' ? 'secondary' : 
                                   result.test_type === 'integration' ? 'outline' : 'default'}>
                      {result.test_type}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {result.duration_ms}ms
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Testing Standards Checklist */}
        <Card>
          <CardHeader>
            <CardTitle>Testing Standards Compliance</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>80%+ Code Coverage (Current: {testMetrics?.coverage_percentage}%)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>All React Components Tested</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>API Integration Tests Implemented</span>
              </div>
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-500" />
                <span>E2E User Journey Tests (In Progress)</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Cross-browser Testing Setup</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Performance Metrics Monitoring</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default TestingDashboard;