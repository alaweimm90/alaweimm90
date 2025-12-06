import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/ui/atoms/Badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { 
  Play, 
  Square, 
  CheckCircle, 
  XCircle, 
  Clock,
  AlertTriangle,
  FileText,
  BarChart3,
  Settings
} from 'lucide-react';

interface TestSuite {
  id: string;
  name: string;
  type: 'unit' | 'integration' | 'e2e' | 'performance';
  status: 'idle' | 'running' | 'passed' | 'failed' | 'skipped';
  duration: number;
  passed: number;
  failed: number;
  total: number;
  coverage?: number;
}

interface TestResult {
  suite: string;
  test: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
}

export const ComprehensiveTestRunner = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedSuites, setSelectedSuites] = useState<string[]>(['all']);
  const [testSuites, setTestSuites] = useState<TestSuite[]>([
    {
      id: 'unit',
      name: 'Unit Tests',
      type: 'unit',
      status: 'passed',
      duration: 1250,
      passed: 142,
      failed: 0,
      total: 142,
      coverage: 87
    },
    {
      id: 'integration',
      name: 'Integration Tests',
      type: 'integration',
      status: 'passed',
      duration: 3200,
      passed: 28,
      failed: 1,
      total: 29,
      coverage: 76
    },
    {
      id: 'e2e',
      name: 'End-to-End Tests',
      type: 'e2e',
      status: 'failed',
      duration: 8500,
      passed: 12,
      failed: 3,
      total: 15
    },
    {
      id: 'performance',
      name: 'Performance Tests',
      type: 'performance',
      status: 'passed',
      duration: 5600,
      passed: 8,
      failed: 0,
      total: 8
    }
  ]);

  const [recentResults, setRecentResults] = useState<TestResult[]>([
    { suite: 'Unit Tests', test: 'AuthService.login', status: 'passed', duration: 45 },
    { suite: 'Unit Tests', test: 'PaymentService.process', status: 'passed', duration: 32 },
    { suite: 'Integration', test: 'API.createUser', status: 'failed', duration: 120, error: 'Connection timeout' },
    { suite: 'E2E', test: 'User Registration Flow', status: 'passed', duration: 2100 },
    { suite: 'E2E', test: 'Payment Flow', status: 'failed', duration: 1800, error: 'Element not found' }
  ]);

  const runTests = async () => {
    setIsRunning(true);
    
    // Simulate test execution
    const suitesToRun = selectedSuites.includes('all') ? testSuites : 
      testSuites.filter(suite => selectedSuites.includes(suite.id));

    for (const suite of suitesToRun) {
      setTestSuites(prev => prev.map(s => 
        s.id === suite.id ? { ...s, status: 'running' } : s
      ));
      
      // Simulate test duration
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Random pass/fail for demo
      const success = Math.random() > 0.2;
      setTestSuites(prev => prev.map(s => 
        s.id === suite.id ? { ...s, status: success ? 'passed' : 'failed' } : s
      ));
    }
    
    setIsRunning(false);
  };

  const stopTests = () => {
    setIsRunning(false);
    setTestSuites(prev => prev.map(suite => 
      suite.status === 'running' ? { ...suite, status: 'idle' } : suite
    ));
  };

  const getStatusIcon = (status: TestSuite['status']) => {
    switch (status) {
      case 'running': return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'passed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'skipped': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default: return <div className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: TestSuite['status']) => {
    switch (status) {
      case 'passed': return 'default';
      case 'failed': return 'destructive';
      case 'running': return 'secondary';
      case 'skipped': return 'outline';
      default: return 'outline';
    }
  };

  const totalTests = testSuites.reduce((sum, suite) => sum + suite.total, 0);
  const totalPassed = testSuites.reduce((sum, suite) => sum + suite.passed, 0);
  const totalFailed = testSuites.reduce((sum, suite) => sum + suite.failed, 0);
  const overallCoverage = testSuites
    .filter(s => s.coverage)
    .reduce((sum, suite) => sum + (suite.coverage || 0), 0) / 
    testSuites.filter(s => s.coverage).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Comprehensive Test Runner</h2>
          <p className="text-muted-foreground">Execute and monitor all test suites</p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSelectedSuites(['all'])}
            className={selectedSuites.includes('all') ? 'bg-primary text-primary-foreground' : ''}
          >
            All Suites
          </Button>
          {!isRunning ? (
            <Button onClick={runTests} className="gap-2">
              <Play className="h-4 w-4" />
              Run Tests
            </Button>
          ) : (
            <Button onClick={stopTests} variant="destructive" className="gap-2">
              <Square className="h-4 w-4" />
              Stop Tests
            </Button>
          )}
        </div>
      </div>

      {/* Test Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{totalTests}</div>
            <p className="text-xs text-muted-foreground">Total Tests</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-green-600">{totalPassed}</div>
            <p className="text-xs text-muted-foreground">Passed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold text-red-600">{totalFailed}</div>
            <p className="text-xs text-muted-foreground">Failed</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-4">
            <div className="text-2xl font-bold">{Math.round(overallCoverage)}%</div>
            <p className="text-xs text-muted-foreground">Coverage</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="suites" className="w-full">
        <TabsList>
          <TabsTrigger value="suites">Test Suites</TabsTrigger>
          <TabsTrigger value="results">Recent Results</TabsTrigger>
          <TabsTrigger value="coverage">Coverage Report</TabsTrigger>
        </TabsList>

        <TabsContent value="suites">
          <div className="grid gap-4">
            {testSuites.map((suite) => (
              <Card key={suite.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(suite.status)}
                      <CardTitle className="text-lg">{suite.name}</CardTitle>
                      <Badge variant={getStatusColor(suite.status)}>
                        {suite.status}
                      </Badge>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        if (selectedSuites.includes(suite.id)) {
                          setSelectedSuites(prev => prev.filter(id => id !== suite.id));
                        } else {
                          setSelectedSuites(prev => [...prev.filter(id => id !== 'all'), suite.id]);
                        }
                      }}
                      className={selectedSuites.includes(suite.id) || selectedSuites.includes('all') ? 
                        'bg-primary text-primary-foreground' : ''}
                    >
                      {selectedSuites.includes(suite.id) || selectedSuites.includes('all') ? 'Selected' : 'Select'}
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-4">
                    <div>
                      <p className="text-sm font-medium">Duration</p>
                      <p className="text-2xl font-bold">{(suite.duration / 1000).toFixed(1)}s</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Passed</p>
                      <p className="text-2xl font-bold text-green-600">{suite.passed}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Failed</p>
                      <p className="text-2xl font-bold text-red-600">{suite.failed}</p>
                    </div>
                    {suite.coverage && (
                      <div>
                        <p className="text-sm font-medium">Coverage</p>
                        <p className="text-2xl font-bold">{suite.coverage}%</p>
                      </div>
                    )}
                  </div>
                  
                  {suite.total > 0 && (
                    <div className="mt-4">
                      <div className="flex justify-between text-sm mb-2">
                        <span>Test Progress</span>
                        <span>{suite.passed}/{suite.total}</span>
                      </div>
                      <Progress value={(suite.passed / suite.total) * 100} />
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="results">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Recent Test Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentResults.map((result, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(result.status as 'passed' | 'failed' | 'pending')}
                      <div>
                        <p className="font-medium">{result.test}</p>
                        <p className="text-sm text-muted-foreground">{result.suite}</p>
                        {result.error && (
                          <p className="text-sm text-red-500">{result.error}</p>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium">{result.duration}ms</p>
                      <Badge variant={result.status === 'passed' ? 'default' : 'destructive'}>
                        {result.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="coverage">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Code Coverage Report
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {testSuites.filter(s => s.coverage).map((suite) => (
                  <div key={suite.id}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">{suite.name}</span>
                      <span className="text-sm">{suite.coverage}%</span>
                    </div>
                    <Progress value={suite.coverage} />
                  </div>
                ))}
                <div className="mt-6 p-4 bg-muted rounded-lg">
                  <h4 className="font-medium mb-2">Coverage Summary</h4>
                  <div className="grid gap-2 md:grid-cols-3">
                    <div>
                      <p className="text-sm text-muted-foreground">Lines</p>
                      <p className="font-bold">{Math.round(overallCoverage)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Functions</p>
                      <p className="font-bold">{Math.round(overallCoverage - 5)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Branches</p>
                      <p className="font-bold">{Math.round(overallCoverage - 8)}%</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};