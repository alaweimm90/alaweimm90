import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { Button } from '@/ui/atoms/Button';
import { Badge } from '@/ui/atoms/Badge';
import { AlertTriangle, RefreshCw, Home, FileText } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: React.ErrorInfo | null;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; reset: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({
      error,
      errorInfo
    });

    // Call onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error to analytics/monitoring service
    this.logErrorToService(error, errorInfo);
  }

  logErrorToService = (error: Error, errorInfo: React.ErrorInfo) => {
    // In production, send to error monitoring service (Sentry, LogRocket, etc.)
    console.error('Error Boundary caught an error:', {
      error: error.message,
      stack: error.stack,
      componentStack: errorInfo.componentStack,
      timestamp: new Date().toISOString(),
      userAgent: navigator.userAgent,
      url: window.location.href
    });

    // Example: Send to analytics service
    // analytics.track('error_boundary_triggered', {
    //   error_message: error.message,
    //   error_stack: error.stack,
    //   component_stack: errorInfo.componentStack
    // });
  };

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      const { fallback: Fallback } = this.props;
      
      if (Fallback && this.state.error) {
        return <Fallback error={this.state.error} reset={this.handleReset} />;
      }

      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 flex items-center justify-center p-4">
          <Card className="w-full max-w-2xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-red-700">
                <AlertTriangle className="h-6 w-6" />
                Something went wrong
              </CardTitle>
              <CardDescription>
                We're sorry, but something unexpected happened. Our team has been notified.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Alert className="border-red-200 bg-red-50">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-red-800">
                  <strong>Error:</strong> {this.state.error?.message || 'Unknown error occurred'}
                </AlertDescription>
              </Alert>

              <div className="flex flex-col sm:flex-row gap-3">
                <Button 
                  onClick={this.handleReset}
                  className="flex items-center gap-2"
                >
                  <RefreshCw className="h-4 w-4" />
                  Try Again
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => window.location.href = '/'}
                  className="flex items-center gap-2"
                >
                  <Home className="h-4 w-4" />
                  Go Home
                </Button>
                <Button 
                  variant="outline"
                  onClick={() => window.location.href = '/support'}
                  className="flex items-center gap-2"
                >
                  <FileText className="h-4 w-4" />
                  Get Support
                </Button>
              </div>

              {process.env.NODE_ENV === 'development' && this.state.error && (
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm font-medium text-gray-600 hover:text-gray-900">
                    Technical Details (Development Only)
                  </summary>
                  <div className="mt-3 p-4 bg-gray-100 rounded-lg">
                    <div className="space-y-3">
                      <div>
                        <Badge variant="destructive" className="mb-2">Error Stack</Badge>
                        <pre className="text-xs bg-white p-2 rounded border overflow-auto max-h-32">
                          {this.state.error.stack}
                        </pre>
                      </div>
                      {this.state.errorInfo && (
                        <div>
                          <Badge variant="secondary" className="mb-2">Component Stack</Badge>
                          <pre className="text-xs bg-white p-2 rounded border overflow-auto max-h-32">
                            {this.state.errorInfo.componentStack}
                          </pre>
                        </div>
                      )}
                    </div>
                  </div>
                </details>
              )}
            </CardContent>
          </Card>
        </div>
      );
    }

    return this.props.children;
  }
}

// Higher-order component for wrapping components with error boundaries
export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  errorComponent?: React.ComponentType<{ error: Error; reset: () => void }>
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary fallback={errorComponent}>
      <Component {...props} />
    </ErrorBoundary>
  );
  
  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};

// Custom error fallback for specific components
export const DashboardErrorFallback: React.FC<{ error: Error; reset: () => void }> = ({ error, reset }) => (
  <Card className="border-red-200 bg-red-50">
    <CardHeader>
      <CardTitle className="flex items-center gap-2 text-red-700">
        <AlertTriangle className="h-5 w-5" />
        Dashboard Error
      </CardTitle>
    </CardHeader>
    <CardContent>
      <p className="text-sm text-red-600 mb-4">
        Unable to load dashboard. Please try refreshing or contact support.
      </p>
      <Button onClick={reset} size="sm" className="flex items-center gap-2">
        <RefreshCw className="h-4 w-4" />
        Retry
      </Button>
    </CardContent>
  </Card>
);

export default ErrorBoundary;