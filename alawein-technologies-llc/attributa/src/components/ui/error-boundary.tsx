import React from 'react';
import { AlertTriangle, RefreshCw, FileText, Download, Github } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardHeader, CardContent } from '@/components/ui/card';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error: Error; retry: () => void }>;
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void;
}

class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error Boundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
    this.props.onError?.(error, errorInfo);
    
    // Log to local storage for debugging
    try {
      const errorLog = {
        message: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString(),
        url: window.location.href
      };
      const logs = JSON.parse(localStorage.getItem('error-logs') || '[]');
      logs.push(errorLog);
      // Keep only last 10 errors
      if (logs.length > 10) logs.shift();
      localStorage.setItem('error-logs', JSON.stringify(logs));
    } catch (e) {
      console.error('Failed to log error:', e);
    }
  }

  getErrorMessage = (error: Error) => {
    if (error.message.includes('file size') || error.message.includes('too large')) {
      return {
        title: 'File too large',
        message: 'Please select a file under 2MB or try our text input option.',
        action: 'Try text input instead',
        actionUrl: '/scan'
      };
    }
    if (error.message.includes('network') || error.message.includes('fetch')) {
      return {
        title: 'Connection issue',
        message: 'Unable to connect to services. Local analysis is still available.',
        action: 'Continue locally',
        actionUrl: '/scan'
      };
    }
    if (error.message.includes('memory') || error.message.includes('allocation')) {
      return {
        title: 'Browser memory limit',
        message: 'Try analyzing smaller files or restart your browser.',
        action: 'Try smaller file',
        actionUrl: '/scan'
      };
    }
    return {
      title: 'Unexpected error',
      message: 'Something went wrong. Please try again or report this issue.',
      action: 'Report issue',
      actionUrl: 'https://github.com/alaweimm90/Attributa/issues/new'
    };
  };

  handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        const FallbackComponent = this.props.fallback;
        return <FallbackComponent error={this.state.error!} retry={this.handleRetry} />;
      }

      const errorDetails = this.getErrorMessage(this.state.error!);
      const isExternalAction = errorDetails.actionUrl.startsWith('http');

      return (
        <Card className="w-full max-w-md mx-auto mt-8">
          <CardHeader className="text-center">
            <AlertTriangle className="w-12 h-12 text-destructive mx-auto mb-4" />
            <h2 className="text-lg font-semibold">{errorDetails.title}</h2>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <p className="text-muted-foreground">
              {errorDetails.message}
            </p>
            
            <div className="flex flex-col gap-2">
              <Button onClick={this.handleRetry}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Try Again
              </Button>
              
              {isExternalAction ? (
                <Button variant="outline" asChild>
                  <a href={errorDetails.actionUrl} target="_blank" rel="noopener noreferrer">
                    <Github className="w-4 h-4 mr-2" />
                    {errorDetails.action}
                  </a>
                </Button>
              ) : (
                <Button variant="outline" asChild>
                  <a href={errorDetails.actionUrl}>
                    <FileText className="w-4 h-4 mr-2" />
                    {errorDetails.action}
                  </a>
                </Button>
              )}
            </div>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <details className="text-left">
                <summary className="cursor-pointer text-sm font-medium">
                  Error Details (Development Only)
                </summary>
                <pre className="mt-2 text-xs bg-muted p-2 rounded overflow-auto">
                  {this.state.error.toString()}
                  {this.state.errorInfo?.componentStack}
                </pre>
              </details>
            )}
          </CardContent>
        </Card>
      );
    }

    return this.props.children;
  }
}

export { ErrorBoundary };
export default ErrorBoundary;