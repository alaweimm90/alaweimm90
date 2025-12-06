import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  FileText,
  Shield,
  Smartphone,
  Globe,
  Database,
  Zap
} from 'lucide-react';

interface TestResult {
  id: string;
  name: string;
  category: string;
  status: 'passed' | 'failed' | 'pending' | 'running';
  duration?: number;
  error?: string;
  coverage?: number;
}

const TestSuite: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [testResults, setTestResults] = useState<TestResult[]>([
    // Unit Tests
    { id: '1', name: 'useTierAccess Hook', category: 'unit', status: 'passed', duration: 45, coverage: 95 },
    { id: '2', name: 'useAnalytics Hook', category: 'unit', status: 'passed', duration: 32, coverage: 88 },
    { id: '3', name: 'TierCard Component', category: 'unit', status: 'passed', duration: 67, coverage: 92 },
    { id: '4', name: 'SubscriptionModal Component', category: 'unit', status: 'failed', error: 'Stripe integration test failed' },
    
    // Integration Tests
    { id: '5', name: 'Auth Flow Integration', category: 'integration', status: 'passed', duration: 156, coverage: 85 },
    { id: '6', name: 'Payment Processing', category: 'integration', status: 'pending' },
    { id: '7', name: 'Dashboard Data Loading', category: 'integration', status: 'passed', duration: 89, coverage: 90 },
    
    // E2E Tests
    { id: '8', name: 'User Registration Flow', category: 'e2e', status: 'passed', duration: 2340 },
    { id: '9', name: 'Subscription Upgrade', category: 'e2e', status: 'pending' },
    { id: '10', name: 'Workout Completion', category: 'e2e', status: 'passed', duration: 1890 },
    
    // Security Tests
    { id: '11', name: 'RLS Policy Validation', category: 'security', status: 'passed', duration: 234 },
    { id: '12', name: 'Auth Token Security', category: 'security', status: 'passed', duration: 123 },
    { id: '13', name: 'API Endpoint Protection', category: 'security', status: 'pending' },
    
    // Mobile Tests
    { id: '14', name: 'Mobile Gesture Recognition', category: 'mobile', status: 'passed', duration: 78 },
    { id: '15', name: 'Offline Data Sync', category: 'mobile', status: 'pending' },
    { id: '16', name: 'Push Notification Delivery', category: 'mobile', status: 'pending' },
    
    // Performance Tests
    { id: '17', name: 'Dashboard Load Time', category: 'performance', status: 'passed', duration: 1200 },
    { id: '18', name: 'Database Query Performance', category: 'performance', status: 'passed', duration: 890 },
    { id: '19', name: 'Bundle Size Analysis', category: 'performance', status: 'passed', duration: 456 }
  ]);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'unit': return <FileText className="h-4 w-4" />;
      case 'integration': return <Zap className="h-4 w-4" />;
      case 'e2e': return <Globe className="h-4 w-4" />;
      case 'security': return <Shield className="h-4 w-4" />;
      case 'mobile': return <Smartphone className="h-4 w-4" />;
      case 'performance': return <Database className="h-4 w-4" />;
      default: return <FileText className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passed': return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'failed': return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'running': return <Clock className="h-4 w-4 text-blue-500 animate-spin" />;
      default: return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const categoryStats = testResults.reduce((acc, test) => {
    if (!acc[test.category]) {
      acc[test.category] = { total: 0, passed: 0, failed: 0, pending: 0 };
    }
    acc[test.category].total++;
    acc[test.category][test.status]++;
    return acc;
  }, {} as Record<string, { total: number; passed: number; failed: number; pending: number }>);

  const totalTests = testResults.length;
  const passedTests = testResults.filter(t => t.status === 'passed').length;
  const failedTests = testResults.filter(t => t.status === 'failed').length;
  const pendingTests = testResults.filter(t => t.status === 'pending').length;
  const overallProgress = (passedTests / totalTests) * 100;

  const runAllTests = async () => {
    setIsRunning(true);
    
    // Simulate running pending tests
    const pendingTestIds = testResults.filter(t => t.status === 'pending').map(t => t.id);
    
    for (const testId of pendingTestIds) {
      setTestResults(prev => prev.map(test => 
        test.id === testId ? { ...test, status: 'running' } : test
      ));
      
      // Simulate test execution time
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
      
      // Randomly pass or fail (90% pass rate)
      const passed = Math.random() > 0.1;
      setTestResults(prev => prev.map(test => 
        test.id === testId ? { 
          ...test, 
          status: passed ? 'passed' : 'failed',
          duration: Math.floor(100 + Math.random() * 1000),
          error: passed ? undefined : 'Simulated test failure'
        } : test
      ));
    }
    
    setIsRunning(false);
  };

  return (
    <div className="space-y-6">
      {/* Overview */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Test Suite Overview</CardTitle>
            <Button 
              onClick={runAllTests} 
              disabled={isRunning}
              className="flex items-center gap-2"
            >
              <Play className="h-4 w-4" />
              {isRunning ? 'Running Tests...' : 'Run All Tests'}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-500">{passedTests}</div>
              <div className="text-sm text-muted-foreground">Passed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-500">{failedTests}</div>
              <div className="text-sm text-muted-foreground">Failed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-500">{pendingTests}</div>
              <div className="text-sm text-muted-foreground">Pending</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold">{totalTests}</div>
              <div className="text-sm text-muted-foreground">Total</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Overall Progress</span>
              <span>{Math.round(overallProgress)}%</span>
            </div>
            <Progress value={overallProgress} className="h-2" />
          </div>
        </CardContent>
      </Card>

      {/* Category Breakdown */}
      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(categoryStats).map(([category, stats]) => (
          <Card key={category}>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-base">
                {getCategoryIcon(category)}
                {category.charAt(0).toUpperCase() + category.slice(1)} Tests
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Passed:</span>
                  <span className="font-medium text-green-500">{stats.passed}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Failed:</span>
                  <span className="font-medium text-red-500">{stats.failed}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span>Pending:</span>
                  <span className="font-medium text-yellow-500">{stats.pending}</span>
                </div>
                <Progress 
                  value={(stats.passed / stats.total) * 100} 
                  className="h-1 mt-2" 
                />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Test Results */}
      <Card>
        <CardHeader>
          <CardTitle>Test Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {testResults.map((test) => (
              <div 
                key={test.id} 
                className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50"
              >
                <div className="flex items-center gap-3">
                  {getStatusIcon(test.status)}
                  <div>
                    <div className="font-medium text-sm">{test.name}</div>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      {getCategoryIcon(test.category)}
                      <span>{test.category}</span>
                      {test.duration && (
                        <>
                          <span>•</span>
                          <span>{test.duration}ms</span>
                        </>
                      )}
                      {test.coverage && (
                        <>
                          <span>•</span>
                          <span>{test.coverage}% coverage</span>
                        </>
                      )}
                    </div>
                    {test.error && (
                      <div className="text-xs text-red-500 mt-1">{test.error}</div>
                    )}
                  </div>
                </div>
                <Badge 
                  variant={
                    test.status === 'passed' ? 'default' : 
                    test.status === 'failed' ? 'destructive' : 'secondary'
                  }
                >
                  {test.status}
                </Badge>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TestSuite;