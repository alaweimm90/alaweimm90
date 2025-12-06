import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useIsMobile } from '@/hooks/use-mobile';
import { Smartphone, Tablet, Monitor, TouchpadIcon as Touch, RotateCcw, Wifi } from 'lucide-react';

interface MobileTestResult {
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  score?: number;
  details?: Record<string, any>;
  duration?: number;
}

interface TouchTestData {
  startTime: number;
  touches: Array<{
    x: number;
    y: number;
    timestamp: number;
    pressure?: number;
  }>;
}

interface OrientationData {
  portrait: boolean;
  landscape: boolean;
  changesDetected: number;
}

export const MobileCompatibilityTester: React.FC = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentTest, setCurrentTest] = useState<string>('');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<MobileTestResult[]>([]);
  const [touchTestActive, setTouchTestActive] = useState(false);
  const [touchData, setTouchData] = useState<TouchTestData | null>(null);
  const [orientationData, setOrientationData] = useState<OrientationData>({
    portrait: false,
    landscape: false,
    changesDetected: 0
  });
  const isMobile = useIsMobile();
  const touchAreaRef = useRef<HTMLDivElement>(null);

  // Initialize test results
  const mobileTests: MobileTestResult[] = [
    { name: 'Device Detection', status: 'pending' },
    { name: 'Touch Support', status: 'pending' },
    { name: 'Screen Responsiveness', status: 'pending' },
    { name: 'Orientation Support', status: 'pending' },
    { name: 'Viewport Handling', status: 'pending' },
    { name: 'Performance on Mobile', status: 'pending' },
    { name: 'PWA Capabilities', status: 'pending' },
    { name: 'Gesture Recognition', status: 'pending' }
  ];

  useEffect(() => {
    setResults(mobileTests);
  }, []);

  // Orientation change detection
  useEffect(() => {
    const handleOrientationChange = () => {
      setOrientationData(prev => ({
        portrait: window.innerHeight > window.innerWidth,
        landscape: window.innerWidth > window.innerHeight,
        changesDetected: prev.changesDetected + 1
      }));
    };

    window.addEventListener('orientationchange', handleOrientationChange);
    window.addEventListener('resize', handleOrientationChange);
    
    // Initial detection
    handleOrientationChange();

    return () => {
      window.removeEventListener('orientationchange', handleOrientationChange);
      window.removeEventListener('resize', handleOrientationChange);
    };
  }, []);

  // Device Detection Test
  const testDeviceDetection = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      const userAgent = navigator.userAgent;
      const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      const maxTouchPoints = navigator.maxTouchPoints || 0;
      const screenWidth = window.screen.width;
      const screenHeight = window.screen.height;
      const devicePixelRatio = window.devicePixelRatio || 1;
      
      // Device type detection
      const isPhone = /Android.*Mobile|iPhone|iPod/i.test(userAgent);
      const isTablet = /iPad|Android(?!.*Mobile)/i.test(userAgent);
      const isDesktop = !isPhone && !isTablet;
      
      let score = 100;
      let deviceType = 'Desktop';
      
      if (isPhone) {
        deviceType = 'Mobile Phone';
        score = screenWidth >= 360 ? 100 : 80; // Penalize very small screens
      } else if (isTablet) {
        deviceType = 'Tablet';
        score = 95;
      } else if (isTouchDevice) {
        deviceType = 'Touch Desktop';
        score = 90;
      }

      return {
        name: 'Device Detection',
        status: 'completed',
        score,
        duration: performance.now() - startTime,
        details: {
          deviceType,
          isPhone,
          isTablet,
          isTouchDevice,
          maxTouchPoints,
          screenWidth,
          screenHeight,
          devicePixelRatio,
          userAgent: userAgent.substring(0, 50) + '...'
        }
      };
    } catch (error) {
      return {
        name: 'Device Detection',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  // Touch Support Test
  const testTouchSupport = useCallback((): Promise<MobileTestResult> => {
    return new Promise((resolve) => {
      const startTime = performance.now();
      
      try {
        const hasTouch = 'ontouchstart' in window;
        const maxTouchPoints = navigator.maxTouchPoints || 0;
        const hasTouchEvents = typeof TouchEvent !== 'undefined';
        const hasPointerEvents = 'onpointerdown' in window;
        
        let score = 0;
        const details: Record<string, any> = {
          hasTouch,
          maxTouchPoints,
          hasTouchEvents,
          hasPointerEvents
        };

        if (hasTouch) score += 30;
        if (maxTouchPoints > 0) score += 25;
        if (hasTouchEvents) score += 25;
        if (hasPointerEvents) score += 20;

        // Test actual touch if available
        if (hasTouch && touchAreaRef.current) {
          setTouchTestActive(true);
          
          const touchStartHandler = (e: TouchEvent) => {
            const touch = e.touches[0];
            setTouchData({
              startTime: performance.now(),
              touches: [{
                x: touch.clientX,
                y: touch.clientY,
                timestamp: performance.now(),
                pressure: (touch as any).force || 1
              }]
            });
          };

          const touchEndHandler = () => {
            setTouchTestActive(false);
            score = Math.min(100, score + 20); // Bonus for successful touch
            
            resolve({
              name: 'Touch Support',
              status: 'completed',
              score,
              duration: performance.now() - startTime,
              details: {
                ...details,
                touchTestCompleted: true,
                touchData: touchData ? {
                  duration: performance.now() - touchData.startTime,
                  touchPoints: touchData.touches.length
                } : null
              }
            });
          };

          touchAreaRef.current.addEventListener('touchstart', touchStartHandler);
          touchAreaRef.current.addEventListener('touchend', touchEndHandler);

          // Auto-complete after 5 seconds if no touch
          setTimeout(() => {
            setTouchTestActive(false);
            resolve({
              name: 'Touch Support',
              status: 'completed',
              score,
              duration: performance.now() - startTime,
              details: {
                ...details,
                touchTestCompleted: false,
                note: 'Touch test timed out - touch the test area if available'
              }
            });
          }, 5000);
        } else {
          resolve({
            name: 'Touch Support',
            status: 'completed',
            score,
            duration: performance.now() - startTime,
            details
          });
        }
      } catch (error) {
        resolve({
          name: 'Touch Support',
          status: 'failed',
          duration: performance.now() - startTime,
          details: { error: error instanceof Error ? error.message : 'Unknown error' }
        });
      }
    });
  }, [touchData]);

  // Screen Responsiveness Test
  const testScreenResponsiveness = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const screenWidth = window.screen.width;
      const screenHeight = window.screen.height;
      const devicePixelRatio = window.devicePixelRatio || 1;
      
      // Test CSS media queries
      const isMobileSize = viewportWidth < 768;
      const isTabletSize = viewportWidth >= 768 && viewportWidth < 1024;
      const isDesktopSize = viewportWidth >= 1024;
      
      // Test flexible layouts
      const testElement = document.createElement('div');
      testElement.style.width = '50vw';
      testElement.style.height = '50vh';
      document.body.appendChild(testElement);
      
      const computedStyle = window.getComputedStyle(testElement);
      const actualWidth = parseFloat(computedStyle.width);
      const actualHeight = parseFloat(computedStyle.height);
      
      document.body.removeChild(testElement);
      
      const widthAccurate = Math.abs(actualWidth - viewportWidth * 0.5) < 5;
      const heightAccurate = Math.abs(actualHeight - viewportHeight * 0.5) < 5;
      
      let score = 70;
      if (widthAccurate) score += 15;
      if (heightAccurate) score += 15;
      
      return {
        name: 'Screen Responsiveness',
        status: 'completed',
        score,
        duration: performance.now() - startTime,
        details: {
          viewportWidth,
          viewportHeight,
          screenWidth,
          screenHeight,
          devicePixelRatio,
          isMobileSize,
          isTabletSize,
          isDesktopSize,
          viewportTestPassed: widthAccurate && heightAccurate,
          actualWidth: Math.round(actualWidth),
          actualHeight: Math.round(actualHeight)
        }
      };
    } catch (error) {
      return {
        name: 'Screen Responsiveness',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  // Orientation Support Test
  const testOrientationSupport = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      const hasOrientationAPI = 'orientation' in screen;
      const hasOrientationEvent = 'onorientationchange' in window;
      const currentOrientation = window.innerWidth > window.innerHeight ? 'landscape' : 'portrait';
      
      let score = 50;
      if (hasOrientationAPI) score += 25;
      if (hasOrientationEvent) score += 25;
      if (orientationData.changesDetected > 0) score = Math.min(100, score + 20);

      return {
        name: 'Orientation Support',
        status: 'completed',
        score,
        duration: performance.now() - startTime,
        details: {
          hasOrientationAPI,
          hasOrientationEvent,
          currentOrientation,
          orientationChangesDetected: orientationData.changesDetected,
          supportedOrientations: {
            portrait: orientationData.portrait,
            landscape: orientationData.landscape
          }
        }
      };
    } catch (error) {
      return {
        name: 'Orientation Support',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, [orientationData]);

  // Viewport Handling Test
  const testViewportHandling = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      // Check viewport meta tag
      const viewportMeta = document.querySelector('meta[name="viewport"]');
      const hasViewportMeta = !!viewportMeta;
      const viewportContent = viewportMeta?.getAttribute('content') || '';
      
      // Check visual viewport API
      const hasVisualViewport = 'visualViewport' in window;
      const visualViewportWidth = hasVisualViewport ? (window as any).visualViewport?.width : null;
      const visualViewportHeight = hasVisualViewport ? (window as any).visualViewport?.height : null;
      
      // Test zoom behavior
      const initialZoom = hasVisualViewport ? (window as any).visualViewport?.scale : 1;
      
      let score = 60;
      if (hasViewportMeta) score += 20;
      if (hasVisualViewport) score += 20;
      if (viewportContent.includes('width=device-width')) score += 10;
      if (viewportContent.includes('initial-scale=1')) score += 10;

      return {
        name: 'Viewport Handling',
        status: 'completed',
        score: Math.min(100, score),
        duration: performance.now() - startTime,
        details: {
          hasViewportMeta,
          viewportContent,
          hasVisualViewport,
          visualViewportWidth,
          visualViewportHeight,
          initialZoom,
          windowInnerWidth: window.innerWidth,
          windowInnerHeight: window.innerHeight
        }
      };
    } catch (error) {
      return {
        name: 'Viewport Handling',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  // PWA Capabilities Test
  const testPWACapabilities = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      const hasServiceWorker = 'serviceWorker' in navigator;
      const hasManifest = !!document.querySelector('link[rel="manifest"]');
      const hasBeforeInstallPrompt = 'onbeforeinstallprompt' in window;
      const isStandalone = window.matchMedia('(display-mode: standalone)').matches;
      const hasWebAppMeta = !!document.querySelector('meta[name="apple-mobile-web-app-capable"]');
      
      let score = 0;
      if (hasServiceWorker) score += 25;
      if (hasManifest) score += 25;
      if (hasBeforeInstallPrompt) score += 20;
      if (hasWebAppMeta) score += 15;
      if (isStandalone) score += 15;

      return {
        name: 'PWA Capabilities',
        status: 'completed',
        score,
        duration: performance.now() - startTime,
        details: {
          hasServiceWorker,
          hasManifest,
          hasBeforeInstallPrompt,
          isStandalone,
          hasWebAppMeta,
          userAgent: navigator.userAgent.includes('Mobile') ? 'Mobile' : 'Desktop'
        }
      };
    } catch (error) {
      return {
        name: 'PWA Capabilities',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  // Performance on Mobile Test (simplified)
  const testMobilePerformance = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      // Quick performance test on mobile
      const testStart = performance.now();
      
      // Simulate mobile-intensive operations
      for (let i = 0; i < 10000; i++) {
        Math.sin(i) * Math.cos(i);
      }
      
      const computeTime = performance.now() - testStart;
      
      // @ts-ignore - memory API may not be available
      const memoryInfo = (performance as any).memory;
      const memoryUsed = memoryInfo ? memoryInfo.usedJSHeapSize / (1024 * 1024) : 0;
      
      let score = 100;
      if (computeTime > 10) score -= 20; // Slow computation
      if (memoryUsed > 50) score -= 20; // High memory usage on mobile
      if (isMobile && computeTime > 5) score -= 20; // Extra penalty for mobile

      return {
        name: 'Performance on Mobile',
        status: 'completed',
        score: Math.max(0, score),
        duration: performance.now() - startTime,
        details: {
          computeTime: Math.round(computeTime * 100) / 100,
          memoryUsed: Math.round(memoryUsed),
          isMobileDevice: isMobile,
          hardwareAcceleration: 'webgl' in window
        }
      };
    } catch (error) {
      return {
        name: 'Performance on Mobile',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, [isMobile]);

  // Gesture Recognition Test (placeholder)
  const testGestureRecognition = useCallback(async (): Promise<MobileTestResult> => {
    const startTime = performance.now();
    
    try {
      const hasTouchEvents = 'ontouchstart' in window;
      const hasPointerEvents = 'onpointerdown' in window;
      const hasGestureEvents = 'ongesturestart' in window;
      
      let score = 50;
      if (hasTouchEvents) score += 20;
      if (hasPointerEvents) score += 20;
      if (hasGestureEvents) score += 10;

      return {
        name: 'Gesture Recognition',
        status: 'completed',
        score,
        duration: performance.now() - startTime,
        details: {
          hasTouchEvents,
          hasPointerEvents,
          hasGestureEvents,
          note: 'Basic gesture support detected'
        }
      };
    } catch (error) {
      return {
        name: 'Gesture Recognition',
        status: 'failed',
        duration: performance.now() - startTime,
        details: { error: error instanceof Error ? error.message : 'Unknown error' }
      };
    }
  }, []);

  // Run all mobile tests
  const runMobileTests = useCallback(async () => {
    setIsRunning(true);
    setProgress(0);
    
    const tests = [
      testDeviceDetection,
      testTouchSupport,
      testScreenResponsiveness,
      testOrientationSupport,
      testViewportHandling,
      testMobilePerformance,
      testPWACapabilities,
      testGestureRecognition
    ];

    const newResults: MobileTestResult[] = [...results];
    
    for (let i = 0; i < tests.length; i++) {
      setCurrentTest(newResults[i].name);
      newResults[i].status = 'running';
      setResults([...newResults]);
      
      try {
        const result = await tests[i]();
        newResults[i] = result;
      } catch (error) {
        newResults[i] = {
          ...newResults[i],
          status: 'failed',
          details: { error: error instanceof Error ? error.message : 'Unknown error' }
        };
      }
      
      setResults([...newResults]);
      setProgress(((i + 1) / tests.length) * 100);
    }
    
    setCurrentTest('');
    setIsRunning(false);
  }, [results, testDeviceDetection, testTouchSupport, testScreenResponsiveness, testOrientationSupport, testViewportHandling, testMobilePerformance, testPWACapabilities, testGestureRecognition]);

  const getDeviceIcon = () => {
    if (window.innerWidth < 768) return <Smartphone className="w-5 h-5" />;
    if (window.innerWidth < 1024) return <Tablet className="w-5 h-5" />;
    return <Monitor className="w-5 h-5" />;
  };

  const getStatusColor = (status: MobileTestResult['status']) => {
    switch (status) {
      case 'completed': return 'text-green-500';
      case 'failed': return 'text-red-500';
      case 'running': return 'text-blue-500';
      default: return 'text-muted-foreground';
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-500';
    if (score >= 60) return 'text-yellow-500';
    return 'text-red-500';
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {getDeviceIcon()}
            Mobile Compatibility Testing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {/* Device Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-muted/30 rounded-lg">
              <div className="text-center">
                <div className="text-sm font-medium">Device</div>
                <div className="text-xs text-muted-foreground">
                  {isMobile ? 'Mobile' : 'Desktop'}
                </div>
              </div>
              <div className="text-center">
                <div className="text-sm font-medium">{window.innerWidth}×{window.innerHeight}</div>
                <div className="text-xs text-muted-foreground">Viewport</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-medium">{orientationData.portrait ? 'Portrait' : 'Landscape'}</div>
                <div className="text-xs text-muted-foreground">Orientation</div>
              </div>
              <div className="text-center">
                <div className="text-sm font-medium">{navigator.maxTouchPoints || 0}</div>
                <div className="text-xs text-muted-foreground">Touch Points</div>
              </div>
            </div>

            {/* Test Controls */}
            <div className="flex gap-4">
              <Button 
                onClick={runMobileTests} 
                disabled={isRunning}
                className="bg-gradient-quantum hover:shadow-quantum"
              >
                {isRunning ? 'Running Tests...' : 'Start Mobile Tests'}
              </Button>
              {isRunning && (
                <div className="flex-1">
                  <div className="text-sm text-muted-foreground mb-2">
                    {currentTest && `Running: ${currentTest}`}
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}
            </div>

            {/* Touch Test Area */}
            {touchTestActive && (
              <div 
                ref={touchAreaRef}
                className="p-8 border-2 border-dashed border-accent rounded-lg text-center bg-accent/10"
              >
                <Touch className="w-8 h-8 mx-auto mb-2 text-accent" />
                <div className="font-medium">Touch Test Active</div>
                <div className="text-sm text-muted-foreground">
                  Touch this area to test touch support
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="results" className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="results">Test Results</TabsTrigger>
          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
        </TabsList>
        
        <TabsContent value="results">
          <Card>
            <CardContent className="p-6">
              <div className="space-y-3">
                {results.map((result, index) => (
                  <div key={index} className="flex items-center justify-between p-3 rounded-lg bg-card/50">
                    <div className="flex items-center gap-3">
                      <div className={`w-2 h-2 rounded-full ${
                        result.status === 'completed' ? 'bg-green-500' :
                        result.status === 'failed' ? 'bg-red-500' :
                        result.status === 'running' ? 'bg-blue-500 animate-pulse' :
                        'bg-muted-foreground'
                      }`} />
                      <span className="font-medium">{result.name}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      {result.score !== undefined && (
                        <Badge variant="outline" className={getScoreColor(result.score)}>
                          {result.score}/100
                        </Badge>
                      )}
                      {result.duration && (
                        <span className="text-sm text-muted-foreground">
                          {Math.round(result.duration)}ms
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
        
        <TabsContent value="recommendations">
          <Card>
            <CardContent className="p-6">
              <div className="space-y-4">
                <h4 className="font-semibold">Mobile Optimization Recommendations</h4>
                <div className="space-y-3 text-sm">
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary mt-2" />
                    <div>
                      <div className="font-medium">Touch-First Design</div>
                      <div className="text-muted-foreground">Ensure all interactive elements are at least 44px in size for touch accessibility</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary mt-2" />
                    <div>
                      <div className="font-medium">Responsive Physics Controls</div>
                      <div className="text-muted-foreground">Use gesture-friendly sliders and controls for physics parameter adjustment</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary mt-2" />
                    <div>
                      <div className="font-medium">Performance Optimization</div>
                      <div className="text-muted-foreground">Reduce computational complexity on mobile devices to maintain 30+ FPS</div>
                    </div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary mt-2" />
                    <div>
                      <div className="font-medium">Offline Capability</div>
                      <div className="text-muted-foreground">Implement PWA features for offline physics simulations</div>
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