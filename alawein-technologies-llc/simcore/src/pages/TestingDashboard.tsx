/**
 * Testing Dashboard - Comprehensive testing interface for SimCore
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { TestRunner } from '@/components/testing/TestRunner';
import { allTestSuites } from '@/components/testing/PhysicsTestSuites';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useToast } from '@/hooks/use-toast';
import { TestTube, Zap, Timer, MemoryStick, Activity } from 'lucide-react';
import { useSEO } from '@/hooks/use-seo';

const TestingDashboard = () => {
  const { toast } = useToast();
  const [recentResults, setRecentResults] = useState<any[]>([]);

  useSEO({ title: 'Testing Dashboard – SimCore', description: 'Run physics test suites, view recent results, and inspect coverage.' });

  const schemaLD = useMemo(() => ({
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'Testing Dashboard',
    description: 'Run physics test suites, view recent results, and inspect coverage.',
    inLanguage: 'en',
    author: { '@type': 'Organization', name: 'SimCore' },
    publisher: { '@type': 'Organization', name: 'SimCore' },
    keywords: 'testing dashboard, physics tests, coverage',
    mainEntityOfPage: typeof window !== 'undefined' ? window.location.href : ''
  }), []);

  const handleTestComplete = (result: any) => {
    setRecentResults(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 results
    
    if (result.status === 'failed') {
      toast({
        title: "Test Failed",
        description: `${result.name}: ${result.error}`,
        variant: "destructive"
      });
    }
  };

  const handleSuiteComplete = (suiteId: string, results: any[]) => {
    const passed = results.filter(r => r.status === 'passed').length;
    const total = results.length;
    
    toast({
      title: "Test Suite Complete",
      description: `${suiteId}: ${passed}/${total} tests passed`,
      variant: passed === total ? "default" : "destructive"
    });
  };

return (
    <div className="min-h-screen bg-gradient-cosmic relative overflow-hidden">
      <script type="application/ld+json" dangerouslySetInnerHTML={{ __html: JSON.stringify(schemaLD) }} />
      <div className="container mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent">
            SimCore Testing Dashboard
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Comprehensive testing framework for physics simulations, performance monitoring, and quality assurance
          </p>
        </div>

        {/* Testing Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Test Suites</CardTitle>
              <TestTube className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{allTestSuites.length}</div>
              <p className="text-xs text-muted-foreground">Physics simulation tests</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Tests</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {allTestSuites.reduce((total, suite) => total + suite.tests.length, 0)}
              </div>
              <p className="text-xs text-muted-foreground">Individual test cases</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Performance Tests</CardTitle>
              <Timer className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {allTestSuites.reduce((total, suite) => 
                  total + suite.tests.filter(t => t.category === 'performance').length, 0
                )}
              </div>
              <p className="text-xs text-muted-foreground">Benchmark tests</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Critical Tests</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {allTestSuites.reduce((total, suite) => 
                  total + suite.tests.filter(t => t.priority === 'critical').length, 0
                )}
              </div>
              <p className="text-xs text-muted-foreground">High priority tests</p>
            </CardContent>
          </Card>
        </div>

        {/* Main Testing Interface */}
        <Tabs defaultValue="runner" className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="runner">Test Runner</TabsTrigger>
            <TabsTrigger value="results">Recent Results</TabsTrigger>
            <TabsTrigger value="coverage">Test Coverage</TabsTrigger>
          </TabsList>

          <TabsContent value="runner" className="space-y-4">
            <TestRunner 
              suites={allTestSuites}
              onTestComplete={handleTestComplete}
              onSuiteComplete={handleSuiteComplete}
            />
          </TabsContent>

          <TabsContent value="results" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Recent Test Results</CardTitle>
                <CardDescription>
                  Last 10 test executions with results and timing
                </CardDescription>
              </CardHeader>
              <CardContent>
                {recentResults.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No test results yet. Run some tests to see results here.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {recentResults.map((result, index) => (
                      <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                        <div>
                          <div className="font-medium">{result.name}</div>
                          <div className="text-sm text-muted-foreground">
                            {new Date().toLocaleTimeString()}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={result.status === 'passed' ? 'default' : 'destructive'}>
                            {result.status}
                          </Badge>
                          <span className="text-sm text-muted-foreground">
                            {Math.round(result.duration)}ms
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="coverage" className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {allTestSuites.map(suite => (
                <Card key={suite.id}>
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <span>{suite.name}</span>
                      <Badge variant="outline">
                        {suite.tests.length} tests
                      </Badge>
                    </CardTitle>
                    <CardDescription>{suite.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Test Categories</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Array.from(new Set(suite.tests.map(t => t.category))).map(category => (
                          <Badge key={category} variant="outline">
                            {category}
                          </Badge>
                        ))}
                      </div>
                      
                      <div className="flex justify-between text-sm mt-4">
                        <span>Priority Distribution</span>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {Array.from(new Set(suite.tests.map(t => t.priority))).map(priority => (
                          <Badge 
                            key={priority} 
                            variant={priority === 'critical' ? 'destructive' : 'outline'}
                          >
                            {priority}: {suite.tests.filter(t => t.priority === priority).length}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default TestingDashboard;