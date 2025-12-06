import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ComponentType<{ error?: Error; retry: () => void }>;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  retry = () => {
    this.setState({ hasError: false, error: undefined });
  };

  render() {
    if (this.state.hasError) {
      const FallbackComponent = this.props.fallback || DefaultErrorFallback;
      return <FallbackComponent error={this.state.error} retry={this.retry} />;
    }

    return this.props.children;
  }
}

interface ErrorFallbackProps {
  error?: Error;
  retry: () => void;
}

const DefaultErrorFallback: React.FC<ErrorFallbackProps> = ({ error, retry }) => {
  return (
    <div className="min-h-[400px] flex items-center justify-center p-6" role="alert">
      <Card className="max-w-md w-full">
        <CardHeader className="text-center">
          <div className="mx-auto w-12 h-12 bg-destructive/10 rounded-full flex items-center justify-center mb-4">
            <AlertTriangle className="h-6 w-6 text-destructive" aria-hidden="true" />
          </div>
          <CardTitle className="text-xl">Something went wrong</CardTitle>
          <CardDescription>
            We encountered an unexpected error. Please try refreshing the page.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          {error && (
            <details className="text-sm text-muted-foreground">
              <summary className="cursor-pointer hover:text-foreground">
                Error details
              </summary>
              <pre className="mt-2 p-2 bg-muted rounded text-left whitespace-pre-wrap text-xs">
                {error.message}
              </pre>
            </details>
          )}
          <Button onClick={retry} className="w-full">
            <RefreshCw className="h-4 w-4 mr-2" />
            Try again
          </Button>
        </CardContent>
      </Card>
    </div>
  );
};

export const withErrorBoundary = <P extends object>(
  Component: React.ComponentType<P>,
  fallback?: React.ComponentType<ErrorFallbackProps>
) => {
  const WrappedComponent = (props: P) => (
    <ErrorBoundary fallback={fallback}>
      <Component {...props} />
    </ErrorBoundary>
  );

  WrappedComponent.displayName = `withErrorBoundary(${Component.displayName || Component.name})`;
  return WrappedComponent;
};