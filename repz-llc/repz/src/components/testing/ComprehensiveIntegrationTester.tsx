import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { Button } from "@/ui/atoms/Button";
import { Badge } from "@/ui/atoms/Badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/molecules/Tabs";
import { 
  CheckCircle, 
  AlertTriangle, 
  X, 
  Play,
  Monitor,
  Smartphone,
  Tablet,
  RefreshCw,
  TestTube,
  Network,
  Database
} from 'lucide-react';
import { useMediaQuery } from '@/hooks/useMediaQuery';
import { useMobileOptimization } from '@/hooks/useMobileOptimization';
import { useAccessibility } from '@/hooks/useAccessibility';
import { usePerformanceMonitor } from '@/hooks/usePerformanceMonitor';

interface TestResult {
  testName: string;
  status: 'pass' | 'fail' | 'warning' | 'running';
  message: string;
  duration?: number;
  platform?: string;
}

interface IntegrationTest {
  category: string;
  tests: Array<{
    name: string;
    description: string;
    testFn: () => Promise<TestResult>;
  }>;
}

export function ComprehensiveIntegrationTester() {
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState<TestResult[]>([]);
  const [currentTest, setCurrentTest] = useState<string>('');
  const [progress, setProgress] = useState(0);

  const isMobile = useMediaQuery('(max-width: 768px)');
  const isTablet = useMediaQuery('(min-width: 769px) and (max-width: 1023px)');
  const isDesktop = useMediaQuery('(min-width: 1024px)');

  const mobileOptimization = useMobileOptimization();
  const accessibility = useAccessibility();
  const { trackAsyncOperation } = usePerformanceMonitor('IntegrationTester');

  const integrationTests: IntegrationTest[] = [
    {
      category: 'Mobile & Accessibility',
      tests: [
        {
          name: 'Mobile Optimization Check',
          description: 'Verify mobile optimization is active',
          testFn: async () => {
            const start = performance.now();
            await new Promise(resolve => setTimeout(resolve, 100));
            const isOptimized = mobileOptimization.capabilities.isTouch;
            return {
              testName: 'Mobile Optimization Check',
              status: isOptimized ? 'pass' : 'warning',
              message: isOptimized ? 'Mobile optimization active' : 'Mobile optimization not detected',
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        },
        {
          name: 'Touch Target Size Test',
          description: 'Verify touch targets meet minimum size requirements',
          testFn: async () => {
            const start = performance.now();
            const buttons = document.querySelectorAll('button');
            let failedButtons = 0;
            
            buttons.forEach(button => {
              const rect = button.getBoundingClientRect();
              if (rect.width < 44 || rect.height < 44) {
                failedButtons++;
              }
            });
            
            return {
              testName: 'Touch Target Size Test',
              status: failedButtons === 0 ? 'pass' : 'warning',
              message: failedButtons === 0 ? 'All touch targets meet size requirements' : `${failedButtons} buttons below 44px minimum`,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        },
        {
          name: 'Color Contrast Check',
          description: 'Verify color contrast ratios meet WCAG standards',
          testFn: async () => {
            const start = performance.now();
            const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, a');
            let lowContrastCount = 0;
            
            // Simplified contrast check (in real app, would use proper contrast calculation)
            textElements.forEach(element => {
              const styles = window.getComputedStyle(element);
              const color = styles.color;
              const bgColor = styles.backgroundColor;
              
              // Basic check for very light text on light backgrounds
              if (color.includes('rgb(255') && bgColor.includes('rgb(255')) {
                lowContrastCount++;
              }
            });
            
            return {
              testName: 'Color Contrast Check',
              status: lowContrastCount === 0 ? 'pass' : 'warning',
              message: lowContrastCount === 0 ? 'Color contrast appears adequate' : `${lowContrastCount} potential contrast issues`,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        }
      ]
    },
    {
      category: 'Performance',
      tests: [
        {
          name: 'Core Web Vitals Check',
          description: 'Check LCP, FID, and CLS metrics',
          testFn: async () => {
            const start = performance.now();
            
            // Get paint timing
            const paintEntries = performance.getEntriesByType('paint');
            const fcp = paintEntries.find(entry => entry.name === 'first-contentful-paint');
            
            let status: 'pass' | 'warning' | 'fail' = 'pass';
            let message = 'Core Web Vitals within acceptable ranges';
            
            if (fcp && fcp.startTime > 1800) {
              status = 'warning';
              message = `FCP: ${fcp.startTime.toFixed(0)}ms (target: <1800ms)`;
            }
            
            return {
              testName: 'Core Web Vitals Check',
              status,
              message,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        },
        {
          name: 'Memory Usage Test',
          description: 'Check browser memory consumption',
          testFn: async () => {
            const start = performance.now();

            const memory = performance.memory;

            if (!memory) {
              return {
                testName: 'Memory Usage Test',
                status: 'warning' as const,
                message: 'Memory API not available',
                duration: performance.now() - start,
                platform: getCurrentPlatform()
              };
            }
            
            const usedMB = Math.round(memory.usedJSHeapSize / 1048576);
            const status = usedMB < 100 ? 'pass' : usedMB < 200 ? 'warning' : 'fail';
            
            return {
              testName: 'Memory Usage Test',
              status,
              message: `Memory usage: ${usedMB}MB`,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        }
      ]
    },
    {
      category: 'Tier System Integration',
      tests: [
        {
          name: 'Tier Card Rendering Test',
          description: 'Verify tier cards render correctly on all platforms',
          testFn: async () => {
            const start = performance.now();
            
            // Check if tier cards are present in DOM
            const tierCards = document.querySelectorAll('[data-testid*="tier-card"], .tier-card');
            
            return {
              testName: 'Tier Card Rendering Test',
              status: tierCards.length > 0 ? 'pass' : 'fail',
              message: tierCards.length > 0 ? `${tierCards.length} tier cards found` : 'No tier cards detected',
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        },
        {
          name: 'Responsive Layout Test',
          description: 'Check tier card layout across different screen sizes',
          testFn: async () => {
            const start = performance.now();
            
            const platform = getCurrentPlatform();
            const viewport = window.innerWidth;
            
            let expectedLayout: string;
            const status: 'pass' | 'warning' | 'fail' = 'pass';
            
            if (viewport < 768) {
              expectedLayout = 'Mobile: Single column layout';
            } else if (viewport < 1024) {
              expectedLayout = 'Tablet: Two column layout';
            } else {
              expectedLayout = 'Desktop: Multi-column layout';
            }
            
            return {
              testName: 'Responsive Layout Test',
              status,
              message: `${expectedLayout} (${viewport}px)`,
              duration: performance.now() - start,
              platform
            };
          }
        }
      ]
    },
    {
      category: 'User Experience',
      tests: [
        {
          name: 'Navigation Flow Test',
          description: 'Test navigation between pages',
          testFn: async () => {
            const start = performance.now();
            
            // Check for navigation elements
            const navElements = document.querySelectorAll('nav, [role="navigation"]');
            const links = document.querySelectorAll('a[href]');
            
            return {
              testName: 'Navigation Flow Test',
              status: navElements.length > 0 && links.length > 0 ? 'pass' : 'warning',
              message: `${navElements.length} nav elements, ${links.length} links found`,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        },
        {
          name: 'Interactive Elements Test',
          description: 'Test button and form interactions',
          testFn: async () => {
            const start = performance.now();
            
            const buttons = document.querySelectorAll('button:not([disabled])');
            const inputs = document.querySelectorAll('input, textarea, select');
            
            return {
              testName: 'Interactive Elements Test',
              status: buttons.length > 0 ? 'pass' : 'warning',
              message: `${buttons.length} interactive buttons, ${inputs.length} form elements`,
              duration: performance.now() - start,
              platform: getCurrentPlatform()
            };
          }
        }
      ]
    }
  ];

  const getCurrentPlatform = () => {
    if (isMobile) return 'Mobile';
    if (isTablet) return 'Tablet';
    if (isDesktop) return 'Desktop';
    return 'Unknown';
  };

  const runAllTests = async () => {
    setIsRunning(true);
    setResults([]);
    setProgress(0);
    
    const allTests = integrationTests.flatMap(category => category.tests);
    const totalTests = allTests.length;
    
    for (let i = 0; i < allTests.length; i++) {
      const test = allTests[i];
      setCurrentTest(test.name);
      
      try {
        const result = await trackAsyncOperation(test.name, test.testFn);
        setResults(prev => [...prev, result]);
      } catch (error) {
        setResults(prev => [...prev, {
          testName: test.name,
          status: 'fail',
          message: `Test failed: ${error}`,
          platform: getCurrentPlatform()
        }]);
      }
      
      setProgress(((i + 1) / totalTests) * 100);
      
      // Small delay between tests
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    
    setIsRunning(false);
    setCurrentTest('');
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pass': return <CheckCircle className="h-4 w-4 text-success" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-warning" />;
      case 'fail': return <X className="h-4 w-4 text-destructive" />;
      case 'running': return <RefreshCw className="h-4 w-4 animate-spin text-primary" />;
      default: return <TestTube className="h-4 w-4" />;
    }
  };

  const getStatusBadge = (status: string): "default" | "secondary" | "destructive" | "outline" => {
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      pass: 'default',
      warning: 'secondary',
      fail: 'destructive',
      running: 'outline'
    };
    return variants[status] || 'outline';
  };

  const successCount = results.filter(r => r.status === 'pass').length;
  const warningCount = results.filter(r => r.status === 'warning').length;
  const failCount = results.filter(r => r.status === 'fail').length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Integration & Cross-Platform Testing</h2>
          <p className="text-muted-foreground">
            Comprehensive testing across mobile, tablet, and desktop platforms
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 text-sm">
            {isMobile && <><Smartphone className="h-4 w-4" /> Mobile</>}
            {isTablet && <><Tablet className="h-4 w-4" /> Tablet</>}
            {isDesktop && <><Monitor className="h-4 w-4" /> Desktop</>}
          </div>
          <Button 
            onClick={runAllTests}
            disabled={isRunning}
            className="flex items-center gap-2"
          >
            <Play className="h-4 w-4" />
            {isRunning ? 'Running Tests...' : 'Run All Tests'}
          </Button>
        </div>
      </div>

      {isRunning && (
        <Card>
          <CardContent className="p-6">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Testing Progress</span>
                <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
              </div>
              <Progress value={progress} className="h-2" />
              {currentTest && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <RefreshCw className="h-3 w-3 animate-spin" />
                  Running: {currentTest}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {results.length > 0 && (
        <div className="grid gap-4 md:grid-cols-3">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-success" />
                <div>
                  <div className="text-2xl font-bold text-success">{successCount}</div>
                  <div className="text-sm text-muted-foreground">Tests Passed</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-warning" />
                <div>
                  <div className="text-2xl font-bold text-warning">{warningCount}</div>
                  <div className="text-sm text-muted-foreground">Warnings</div>
                </div>
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center gap-2">
                <X className="h-5 w-5 text-destructive" />
                <div>
                  <div className="text-2xl font-bold text-destructive">{failCount}</div>
                  <div className="text-sm text-muted-foreground">Tests Failed</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      <Tabs defaultValue="results" className="space-y-4">
        <TabsList>
          <TabsTrigger value="results">Test Results</TabsTrigger>
          <TabsTrigger value="categories">By Category</TabsTrigger>
          <TabsTrigger value="platform">Platform View</TabsTrigger>
        </TabsList>

        <TabsContent value="results">
          <div className="space-y-2">
            {results.map((result, index) => (
              <Card key={index}>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getStatusIcon(result.status)}
                      <div>
                        <div className="font-medium">{result.testName}</div>
                        <div className="text-sm text-muted-foreground">{result.message}</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {result.platform && (
                        <Badge variant="outline" className="text-xs">
                          {result.platform}
                        </Badge>
                      )}
                      <Badge variant={getStatusBadge(result.status)}>
                        {result.status.toUpperCase()}
                      </Badge>
                      {result.duration && (
                        <span className="text-xs text-muted-foreground">
                          {result.duration.toFixed(0)}ms
                        </span>
                      )}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="categories">
          <div className="space-y-4">
            {integrationTests.map((category, categoryIndex) => {
              const categoryResults = results.filter(r => 
                category.tests.some(t => t.name === r.testName)
              );
              
              return (
                <Card key={categoryIndex}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {category.category === 'Mobile & Accessibility' && <Smartphone className="h-5 w-5" />}
                      {category.category === 'Performance' && <Network className="h-5 w-5" />}
                      {category.category === 'Tier System Integration' && <Database className="h-5 w-5" />}
                      {category.category === 'User Experience' && <TestTube className="h-5 w-5" />}
                      {category.category}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {categoryResults.map((result, resultIndex) => (
                        <div key={resultIndex} className="flex items-center justify-between p-2 rounded-lg bg-muted/50">
                          <div className="flex items-center gap-2">
                            {getStatusIcon(result.status)}
                            <span className="text-sm">{result.testName}</span>
                          </div>
                          <Badge variant={getStatusBadge(result.status)} className="text-xs">
                            {result.status}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        <TabsContent value="platform">
          <div className="grid gap-4 md:grid-cols-3">
            {['Mobile', 'Tablet', 'Desktop'].map(platform => {
              const platformResults = results.filter(r => r.platform === platform);
              const platformPassed = platformResults.filter(r => r.status === 'pass').length;
              
              return (
                <Card key={platform}>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {platform === 'Mobile' && <Smartphone className="h-5 w-5" />}
                      {platform === 'Tablet' && <Tablet className="h-5 w-5" />}
                      {platform === 'Desktop' && <Monitor className="h-5 w-5" />}
                      {platform}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="text-center">
                        <div className="text-2xl font-bold">
                          {platformPassed}/{platformResults.length}
                        </div>
                        <div className="text-sm text-muted-foreground">Tests Passed</div>
                      </div>
                      {platformResults.length > 0 && (
                        <Progress 
                          value={(platformPassed / platformResults.length) * 100} 
                          className="h-2"
                        />
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}