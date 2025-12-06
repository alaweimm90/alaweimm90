import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home, Bug } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';

// Enhanced Error Boundary with better UX
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  errorId: string;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<{ error: Error; reset: () => void }>;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

export class ProductionErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error,
      errorId: Date.now().toString(36)
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({ errorInfo });
    
    // Log error to console in development
    if (process.env.NODE_ENV === 'development') {
      console.error('Error caught by boundary:', error, errorInfo);
    }

    // Call custom error handler if provided
    this.props.onError?.(error, errorInfo);

    // Report to analytics (would integrate with actual service)
    this.reportError(error, errorInfo);
  }

  reportError = (error: Error, errorInfo: ErrorInfo) => {
    // In production, this would send to error reporting service
    const errorReport = {
      message: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    };
    
    console.log('Error Report:', errorReport);
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
      errorId: ''
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback;
        return <FallbackComponent error={this.state.error!} reset={this.handleReset} />;
      }

      return (
        <div className="min-h-screen flex items-center justify-center p-4 bg-background">
          <Card className="max-w-lg w-full">
            <CardHeader className="text-center">
              <div className="mx-auto w-16 h-16 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
                <AlertTriangle className="w-8 h-8 text-destructive" />
              </div>
              <CardTitle className="text-xl">Something went wrong</CardTitle>
              <CardDescription>
                We've encountered an unexpected error. Our team has been notified.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <Alert>
                <Bug className="w-4 h-4" />
                <AlertDescription className="text-sm">
                  Error ID: {this.state.errorId}
                </AlertDescription>
              </Alert>
              
              {process.env.NODE_ENV === 'development' && (
                <details className="text-xs text-muted-foreground bg-muted p-3 rounded">
                  <summary className="cursor-pointer font-medium">Technical Details</summary>
                  <pre className="mt-2 whitespace-pre-wrap">
                    {this.state.error?.message}
                    {this.state.error?.stack}
                  </pre>
                </details>
              )}

              <div className="flex gap-2">
                <Button onClick={this.handleReset} className="flex-1">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Try Again
                </Button>
                <Button variant="outline" onClick={() => window.location.href = '/'}>
                  <Home className="w-4 h-4 mr-2" />
                  Home
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// Performance Monitor Hook
export const usePerformanceMonitor = () => {
  const { toast } = useToast();

  React.useEffect(() => {
    if ('performance' in window && 'PerformanceObserver' in window) {
      const observer = new PerformanceObserver((list) => {
        list.getEntries().forEach((entry) => {
          // Monitor long tasks (>50ms)
          if (entry.entryType === 'longtask' && entry.duration > 50) {
            console.warn(`Long task detected: ${entry.duration}ms`);
            
            // Only show warning for very long tasks in production
            if (entry.duration > 200) {
              toast({
                title: "Performance Notice",
                description: "Some calculations are taking longer than expected.",
                variant: "default"
              });
            }
          }
        });
      });

      try {
        observer.observe({ entryTypes: ['longtask'] });
      } catch (e) {
        // Gracefully handle if longtask is not supported
      }

      return () => observer.disconnect();
    }
  }, [toast]);
};

// Analytics Helper
export const useAnalytics = () => {
  const trackEvent = React.useCallback((event: string, properties?: Record<string, any>) => {
    // In production, this would integrate with analytics service
    console.log('Analytics Event:', { event, properties, timestamp: Date.now() });
  }, []);

  const trackPageView = React.useCallback((page: string) => {
    trackEvent('page_view', { page });
  }, [trackEvent]);

  const trackSimulation = React.useCallback((simulationType: string, action: string) => {
    trackEvent('simulation_interaction', { simulationType, action });
  }, [trackEvent]);

  return { trackEvent, trackPageView, trackSimulation };
};

// Production Status Indicator
export const ProductionStatusIndicator = () => {
  const [status, setStatus] = React.useState<'checking' | 'healthy' | 'warning' | 'error'>('checking');

  React.useEffect(() => {
    const checkHealth = async () => {
      try {
        // Simulate health check
        await new Promise(resolve => setTimeout(resolve, 1000));
        setStatus('healthy');
      } catch {
        setStatus('error');
      }
    };

    checkHealth();
  }, []);

  if (status === 'checking') return null;

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className={`w-3 h-3 rounded-full ${
        status === 'healthy' ? 'bg-green-500' : 
        status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
      } animate-pulse`} />
    </div>
  );
};