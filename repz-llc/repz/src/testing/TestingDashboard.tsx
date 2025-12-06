import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/ui/molecules/Tabs';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { Progress } from '@/components/ui/progress';
import { TestTube, Shield, Accessibility, Zap, Target, Play, CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

interface TestSuite {
  name: string;
  type: 'unit' | 'integration' | 'e2e' | 'security' | 'accessibility' | 'performance';
  status: 'passing' | 'failing' | 'warning' | 'running';
  coverage: number;
  tests: number;
  passed: number;
  failed: number;
  duration: number;
  lastRun: string;
}

interface SecurityTest {
  name: string;
  category: 'authentication' | 'authorization' | 'data-protection' | 'network' | 'input-validation';
  status: 'pass' | 'fail' | 'warning';
  risk: 'low' | 'medium' | 'high' | 'critical';
  description: string;
}

interface AccessibilityTest {
  name: string;
  rule: string;
  level: 'A' | 'AA' | 'AAA';
  status: 'pass' | 'fail' | 'warning';
  impact: 'minor' | 'moderate' | 'serious' | 'critical';
  description: string;
}

export const TestingDashboard: React.FC = () => {
  const [testSuites, setTestSuites] = useState<TestSuite[]>([
    {
      name: 'Unit Tests',
      type: 'unit',
      status: 'passing',
      coverage: 87,
      tests: 156,
      passed: 154,
      failed: 2,
      duration: 2.3,
      lastRun: '2024-08-03T10:30:00Z',
    },
    {
      name: 'Integration Tests',
      type: 'integration',
      status: 'passing',
      coverage: 79,
      tests: 45,
      passed: 43,
      failed: 2,
      duration: 8.7,
      lastRun: '2024-08-03T10:25:00Z',
    },
    {
      name: 'E2E Tests',
      type: 'e2e',
      status: 'warning',
      coverage: 65,
      tests: 23,
      passed: 20,
      failed: 1,
      duration: 45.2,
      lastRun: '2024-08-03T09:45:00Z',
    },
    {
      name: 'Security Tests',
      type: 'security',
      status: 'passing',
      coverage: 95,
      tests: 38,
      passed: 36,
      failed: 0,
      duration: 12.8,
      lastRun: '2024-08-03T08:30:00Z',
    },
    {
      name: 'Accessibility Tests',
      type: 'accessibility',
      status: 'passing',
      coverage: 92,
      tests: 67,
      passed: 65,
      failed: 0,
      duration: 3.1,
      lastRun: '2024-08-03T10:15:00Z',
    },
    {
      name: 'Performance Tests',
      type: 'performance',
      status: 'passing',
      coverage: 88,
      tests: 15,
      passed: 15,
      failed: 0,
      duration: 89.4,
      lastRun: '2024-08-03T07:00:00Z',
    },
  ]);

  const [securityTests] = useState<SecurityTest[]>([
    {
      name: 'SQL Injection Prevention',
      category: 'input-validation',
      status: 'pass',
      risk: 'critical',
      description: 'All database queries use parameterized statements',
    },
    {
      name: 'XSS Protection',
      category: 'input-validation',
      status: 'pass',
      risk: 'high',
      description: 'Content Security Policy configured correctly',
    },
    {
      name: 'Authentication Security',
      category: 'authentication',
      status: 'pass',
      risk: 'critical',
      description: 'Strong password requirements and secure session management',
    },
    {
      name: 'Data Encryption',
      category: 'data-protection',
      status: 'pass',
      risk: 'high',
      description: 'Sensitive data encrypted in transit and at rest',
    },
    {
      name: 'Rate Limiting',
      category: 'network',
      status: 'warning',
      risk: 'medium',
      description: 'Rate limiting implemented but could be more restrictive',
    },
  ]);

  const [accessibilityTests] = useState<AccessibilityTest[]>([
    {
      name: 'Color Contrast',
      rule: 'color-contrast',
      level: 'AA',
      status: 'pass',
      impact: 'serious',
      description: 'All text meets WCAG AA contrast ratios',
    },
    {
      name: 'Keyboard Navigation',
      rule: 'keyboard',
      level: 'A',
      status: 'pass',
      impact: 'critical',
      description: 'All interactive elements are keyboard accessible',
    },
    {
      name: 'Screen Reader Support',
      rule: 'aria-labels',
      level: 'A',
      status: 'pass',
      impact: 'critical',
      description: 'Proper ARIA labels and roles implemented',
    },
    {
      name: 'Focus Management',
      rule: 'focus-order',
      level: 'A',
      status: 'pass',
      impact: 'moderate',
      description: 'Logical focus order maintained throughout app',
    },
    {
      name: 'Alternative Text',
      rule: 'image-alt',
      level: 'A',
      status: 'warning',
      impact: 'serious',
      description: 'Most images have alt text, 2 decorative images flagged',
    },
  ]);

  const [isRunningTests, setIsRunningTests] = useState(false);

  const runAllTests = async () => {
    setIsRunningTests(true);
    
    // Simulate running tests
    for (let i = 0; i < testSuites.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setTestSuites(prev => prev.map((suite, index) => 
        index === i ? { ...suite, status: 'running' as const } : suite
      ));
    }

    // Complete tests
    await new Promise(resolve => setTimeout(resolve, 1000));
    setTestSuites(prev => prev.map(suite => ({
      ...suite,
      status: suite.failed === 0 ? 'passing' as const : 'failing' as const,
      lastRun: new Date().toISOString(),
    })));

    setIsRunningTests(false);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'passing':
      case 'pass':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'failing':
      case 'fail':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-yellow-600" />;
      case 'running':
        return <div className="h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />;
      default:
        return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'passing':
      case 'pass':
        return 'default';
      case 'failing':
      case 'fail':
        return 'destructive';
      case 'warning':
        return 'secondary';
      case 'running':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'critical':
        return 'destructive';
      case 'high':
        return 'destructive';
      case 'medium':
        return 'secondary';
      case 'low':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical':
        return 'destructive';
      case 'serious':
        return 'destructive';
      case 'moderate':
        return 'secondary';
      case 'minor':
        return 'outline';
      default:
        return 'outline';
    }
  };

  const totalTests = testSuites.reduce((sum, suite) => sum + suite.tests, 0);
  const totalPassed = testSuites.reduce((sum, suite) => sum + suite.passed, 0);
  const totalFailed = testSuites.reduce((sum, suite) => sum + suite.failed, 0);
  const averageCoverage = Math.round(testSuites.reduce((sum, suite) => sum + suite.coverage, 0) / testSuites.length);

  return (
    <div className="space-y-6">
      {/* Testing Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Tests</CardTitle>
            <TestTube className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{totalTests}</div>
            <p className="text-xs text-muted-foreground">
              Across {testSuites.length} test suites
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pass Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{Math.round((totalPassed / totalTests) * 100)}%</div>
            <p className="text-xs text-muted-foreground">
              {totalPassed} passed, {totalFailed} failed
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Coverage</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{averageCoverage}%</div>
            <p className="text-xs text-muted-foreground">
              Average across all suites
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Run</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round((Date.now() - new Date(testSuites[0].lastRun).getTime()) / 60000)}m
            </div>
            <p className="text-xs text-muted-foreground">
              Minutes ago
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Run Tests Button */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            Test Execution Control
            <Button 
              onClick={runAllTests} 
              disabled={isRunningTests}
              className="flex items-center"
            >
              <Play className="h-4 w-4 mr-2" />
              {isRunningTests ? 'Running Tests...' : 'Run All Tests'}
            </Button>
          </CardTitle>
          <CardDescription>
            Execute comprehensive test suite across all categories
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">
            <TestTube className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="security">
            <Shield className="h-4 w-4 mr-2" />
            Security
          </TabsTrigger>
          <TabsTrigger value="accessibility">
            <Accessibility className="h-4 w-4 mr-2" />
            Accessibility
          </TabsTrigger>
          <TabsTrigger value="performance">
            <Zap className="h-4 w-4 mr-2" />
            Performance
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Test Suite Results</CardTitle>
              <CardDescription>
                Comprehensive overview of all test categories
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {testSuites.map((suite) => (
                  <div key={suite.name} className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {getStatusIcon(suite.status)}
                        <div>
                          <h4 className="font-medium">{suite.name}</h4>
                          <p className="text-sm text-muted-foreground">
                            {suite.passed}/{suite.tests} tests passed • {suite.duration}s duration
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="text-sm font-medium">{suite.coverage}%</div>
                          <div className="text-xs text-muted-foreground">Coverage</div>
                        </div>
                        <Badge variant={getStatusColor(suite.status)}>
                          {suite.status}
                        </Badge>
                      </div>
                    </div>
                    <Progress value={suite.coverage} className="h-2" />
                    <div className="text-xs text-muted-foreground">
                      Last run: {new Date(suite.lastRun).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Security Test Results</CardTitle>
              <CardDescription>
                OWASP Top 10 and enterprise security standards validation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {securityTests.map((test, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(test.status)}
                      <div>
                        <h4 className="font-medium">{test.name}</h4>
                        <p className="text-sm text-muted-foreground">{test.description}</p>
                        <div className="text-xs text-muted-foreground mt-1">
                          Category: {test.category}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getRiskColor(test.risk)} className="text-xs">
                        {test.risk} risk
                      </Badge>
                      <Badge variant={getStatusColor(test.status)}>
                        {test.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="accessibility" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Accessibility Test Results</CardTitle>
              <CardDescription>
                WCAG 2.1 compliance testing across all accessibility levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {accessibilityTests.map((test, index) => (
                  <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getStatusIcon(test.status)}
                      <div>
                        <h4 className="font-medium">{test.name}</h4>
                        <p className="text-sm text-muted-foreground">{test.description}</p>
                        <div className="text-xs text-muted-foreground mt-1">
                          Rule: {test.rule} • Level: WCAG {test.level}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant={getImpactColor(test.impact)} className="text-xs">
                        {test.impact} impact
                      </Badge>
                      <Badge variant={getStatusColor(test.status)}>
                        {test.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance Test Results</CardTitle>
              <CardDescription>
                Load testing and performance benchmarks
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium">Load Testing Results</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Concurrent Users</span>
                      <span className="font-medium">1,000</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Average Response Time</span>
                      <span className="font-medium">245ms</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>95th Percentile</span>
                      <span className="font-medium">420ms</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Error Rate</span>
                      <span className="font-medium text-green-600">0.02%</span>
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <h4 className="font-medium">Throughput Metrics</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Requests/Second</span>
                      <span className="font-medium">4,200</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Data Transfer</span>
                      <span className="font-medium">1.2 GB/s</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>CPU Usage</span>
                      <span className="font-medium">68%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Memory Usage</span>
                      <span className="font-medium">45%</span>
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