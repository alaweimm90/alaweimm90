import React, { Component, ErrorInfo, ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { trackQuantumEvents } from "@/lib/analytics";
import { errorReporter } from "@/lib/monitoring";
import { logger } from "@/lib/logger";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error,
      errorInfo: null,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log error to monitoring system
    errorReporter.captureError(error, {
      componentStack: errorInfo.componentStack,
      errorBoundary: 'ErrorBoundary'
    });

    // Log error to analytics
    trackQuantumEvents.errorBoundary(
      error.message,
      errorInfo.componentStack,
      'ErrorBoundary'
    );
    
    // Update state with error details
    this.setState({
      error,
      errorInfo,
    });
    
    // Add breadcrumb for debugging
    errorReporter.addBreadcrumb({
      message: 'Error boundary triggered',
      category: 'error',
      level: 'error',
      data: {
        errorMessage: error.message,
        componentStack: errorInfo.componentStack
      }
    });
    
    // Log to console in development
    if (import.meta.env.MODE === 'development') {
      logger.error('Error caught by ErrorBoundary', { error, errorInfo });
    }
  }

  handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null,
    });
    
    // Track recovery attempt
    trackQuantumEvents.errorRecovery('error_boundary_reset');
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return <>{this.props.fallback}</>;
      }

      return (
        <div className="min-h-[400px] flex items-center justify-center p-8">
          <div className="max-w-md w-full space-y-6 text-center">
            <div className="flex justify-center">
              <div className="p-4 bg-destructive/10 rounded-full">
                <AlertCircle className="h-8 w-8 text-destructive" />
              </div>
            </div>
            
            <div className="space-y-2">
              <h2 className="text-2xl font-semibold">Something went wrong</h2>
              <p className="text-muted-foreground">
                An unexpected error occurred while rendering this component. 
                The issue has been logged and we'll work on fixing it.
              </p>
            </div>

            {import.meta.env.MODE === 'development' && this.state.error && (
              <details className="text-left p-4 bg-muted/50 rounded-lg">
                <summary className="cursor-pointer text-sm font-medium">
                  Error Details (Development Only)
                </summary>
                <pre className="mt-2 text-xs overflow-auto">
                  {this.state.error.toString()}
                  {this.state.errorInfo && this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}

            <div className="flex gap-4 justify-center">
              <Button
                onClick={this.handleReset}
                variant="primary"
                className="gap-2"
              >
                <RefreshCw className="h-4 w-4" />
                Try Again
              </Button>
              <Button
                onClick={() => window.location.href = '/'}
                variant="outline"
              >
                Go Home
              </Button>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}