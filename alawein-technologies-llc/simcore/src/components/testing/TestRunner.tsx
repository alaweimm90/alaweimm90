/**
 * Comprehensive testing framework for physics simulations
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Play, Pause, Square, CheckCircle, XCircle, AlertCircle, Zap } from 'lucide-react';

export interface TestResult {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
  details?: any;
}

export interface TestSuite {
  id: string;
  name: string;
  description: string;
  tests: TestCase[];
}

export interface TestCase {
  id: string;
  name: string;
  description: string;
  category: 'unit' | 'integration' | 'performance' | 'visual';
  priority: 'low' | 'medium' | 'high' | 'critical';
  timeout?: number;
  setup?: () => Promise<void> | void;
  teardown?: () => Promise<void> | void;
  run: () => Promise<void> | void;
  validate?: (result: any) => boolean;
}

interface TestRunnerProps {
  suites: TestSuite[];
  onTestComplete?: (result: TestResult) => void;
  onSuiteComplete?: (suiteId: string, results: TestResult[]) => void;
}

export const TestRunner: React.FC<TestRunnerProps> = ({
  suites,
  onTestComplete,
  onSuiteComplete
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState<string | null>(null);
  const [results, setResults] = useState<Map<string, TestResult>>(new Map());
  const [selectedSuite, setSelectedSuite] = useState<string>(suites[0]?.id || '');
  const [filter, setFilter] = useState<'all' | 'passed' | 'failed' | 'pending'>('all');

  const totalTests = useMemo(() => 
    suites.reduce((total, suite) => total + suite.tests.length, 0)
  , [suites]);

  const completedTests = useMemo(() => 
    Array.from(results.values()).filter(r => r.status !== 'pending').length
  , [results]);

  const passedTests = useMemo(() => 
    Array.from(results.values()).filter(r => r.status === 'passed').length
  , [results]);

  const failedTests = useMemo(() => 
    Array.from(results.values()).filter(r => r.status === 'failed').length
  , [results]);

  const progress = totalTests > 0 ? (completedTests / totalTests) * 100 : 0;

  const runTest = useCallback(async (test: TestCase): Promise<TestResult> => {
    const startTime = performance.now();
    
    const result: TestResult = {
      id: test.id,
      name: test.name,
      status: 'running',
      duration: 0
    };

    setCurrentTest(test.id);
    setResults(prev => new Map(prev).set(test.id, result));

    try {
      // Setup
      if (test.setup) {
        await test.setup();
      }

      // Run test with timeout
      const timeoutMs = test.timeout || 10000;
      await Promise.race([
        test.run(),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Test timeout')), timeoutMs)
        )
      ]);

      // Validation
      if (test.validate) {
        const isValid = test.validate(result.details);
        if (!isValid) {
          throw new Error('Validation failed');
        }
      }

      const endTime = performance.now();
      const finalResult: TestResult = {
        ...result,
        status: 'passed',
        duration: endTime - startTime
      };

      setResults(prev => new Map(prev).set(test.id, finalResult));
      onTestComplete?.(finalResult);
      return finalResult;

    } catch (error) {
      const endTime = performance.now();
      const finalResult: TestResult = {
        ...result,
        status: 'failed',
        duration: endTime - startTime,
        error: error instanceof Error ? error.message : String(error)
      };

      setResults(prev => new Map(prev).set(test.id, finalResult));
      onTestComplete?.(finalResult);
      return finalResult;

    } finally {
      // Teardown
      if (test.teardown) {
        try {
          await test.teardown();
        } catch (error) {
          console.warn('Test teardown failed:', error);
        }
      }
      setCurrentTest(null);
    }
  }, [onTestComplete]);

  const runSuite = useCallback(async (suite: TestSuite) => {
    const suiteResults: TestResult[] = [];
    
    for (const test of suite.tests) {
      if (!isRunning) break; // Allow cancellation
      
      const result = await runTest(test);
      suiteResults.push(result);
    }

    onSuiteComplete?.(suite.id, suiteResults);
  }, [isRunning, runTest, onSuiteComplete]);

  const runAllTests = useCallback(async () => {
    setIsRunning(true);
    setResults(new Map());
    
    try {
      for (const suite of suites) {
        if (!isRunning) break;
        await runSuite(suite);
      }
    } finally {
      setIsRunning(false);
      setCurrentTest(null);
    }
  }, [suites, runSuite, isRunning]);

  const stopTests = useCallback(() => {
    setIsRunning(false);
    setCurrentTest(null);
  }, []);

  const clearResults = useCallback(() => {
    setResults(new Map());
  }, []);

  const getStatusIcon = (status: TestResult['status']) => {
    switch (status) {
      case 'passed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'running':
        return <Zap className="w-4 h-4 text-blue-500 animate-pulse" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const filteredSuite = suites.find(s => s.id === selectedSuite);
  const filteredResults = filteredSuite?.tests
    .map(test => results.get(test.id))
    .filter(result => {
      if (!result) return filter === 'all' || filter === 'pending';
      if (filter === 'all') return true;
      if (filter === 'pending') return result.status === 'pending';
      return result.status === filter;
    }) || [];

  return (
    <div className="space-y-6">
      {/* Test Control Panel */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>Physics Test Runner</span>
            <div className="flex items-center gap-2">
              <Badge variant={passedTests > failedTests ? 'default' : 'destructive'}>
                {passedTests}/{totalTests} passed
              </Badge>
              <Badge variant="outline">
                {suites.length} suites
              </Badge>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Progress */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2" />
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <Button
              onClick={runAllTests}
              disabled={isRunning}
              className="flex items-center gap-2"
            >
              <Play className="w-4 h-4" />
              Run All Tests
            </Button>
            
            <Button
              variant="outline"
              onClick={stopTests}
              disabled={!isRunning}
              className="flex items-center gap-2"
            >
              <Square className="w-4 h-4" />
              Stop
            </Button>

            <Button
              variant="ghost"
              onClick={clearResults}
              disabled={isRunning}
              className="flex items-center gap-2"
            >
              Clear Results
            </Button>
          </div>

          {/* Current Test */}
          {currentTest && (
            <div className="p-3 bg-muted rounded-lg">
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-blue-500 animate-pulse" />
                <span className="text-sm font-medium">
                  Running: {filteredSuite?.tests.find(t => t.id === currentTest)?.name}
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Test Results */}
      <Card>
        <CardContent className="p-0">
          <Tabs value={selectedSuite} onValueChange={setSelectedSuite}>
            <div className="p-6 pb-0">
              <TabsList className="grid w-full grid-cols-auto">
                {suites.map(suite => (
                  <TabsTrigger key={suite.id} value={suite.id}>
                    {suite.name}
                  </TabsTrigger>
                ))}
              </TabsList>
            </div>

            {suites.map(suite => (
              <TabsContent key={suite.id} value={suite.id} className="p-6">
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-lg font-semibold">{suite.name}</h3>
                      <p className="text-sm text-muted-foreground">{suite.description}</p>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setFilter('all')}
                        className={filter === 'all' ? 'bg-primary text-primary-foreground' : ''}
                      >
                        All
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setFilter('passed')}
                        className={filter === 'passed' ? 'bg-green-500 text-white' : ''}
                      >
                        Passed
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setFilter('failed')}
                        className={filter === 'failed' ? 'bg-red-500 text-white' : ''}
                      >
                        Failed
                      </Button>
                    </div>
                  </div>

                  <ScrollArea className="h-[400px]">
                    <div className="space-y-2">
                      {suite.tests.map(test => {
                        const result = results.get(test.id);
                        const isVisible = filter === 'all' || 
                          (filter === 'pending' && !result) ||
                          (result && result.status === filter);

                        if (!isVisible) return null;

                        return (
                          <div
                            key={test.id}
                            className="flex items-center justify-between p-3 border rounded-lg"
                          >
                            <div className="flex items-center gap-3">
                              {getStatusIcon(result?.status || 'pending')}
                              <div>
                                <div className="font-medium">{test.name}</div>
                                <div className="text-sm text-muted-foreground">
                                  {test.description}
                                </div>
                              </div>
                            </div>

                            <div className="flex items-center gap-2">
                              <Badge variant="outline">
                                {test.category}
                              </Badge>
                              <Badge 
                                variant={test.priority === 'critical' ? 'destructive' : 'outline'}
                              >
                                {test.priority}
                              </Badge>
                              {result && result.duration > 0 && (
                                <span className="text-xs text-muted-foreground">
                                  {Math.round(result.duration)}ms
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </ScrollArea>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default TestRunner;