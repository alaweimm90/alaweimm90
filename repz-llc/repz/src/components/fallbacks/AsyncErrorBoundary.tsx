import React, { ReactNode, useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/ui/molecules/Card';
import { Button } from '@/ui/atoms/Button';
import { Alert, AlertDescription } from '@/ui/molecules/Alert';
import { AlertTriangle, RefreshCw, Wifi, WifiOff } from 'lucide-react';

interface AsyncErrorBoundaryProps {
  children: ReactNode;
  fallback?: React.ComponentType<AsyncErrorFallbackProps>;
  onError?: (error: Error) => void;
}

interface AsyncErrorFallbackProps {
  error: Error;
  resetError: () => void;
  isOnline: boolean;
}

// Hook to track online status
const useOnlineStatus = () => {
  const [isOnline, setIsOnline] = useState(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
};

// Default async error fallback
const DefaultAsyncErrorFallback: React.FC<AsyncErrorFallbackProps> = ({ 
  error, 
  resetError, 
  isOnline 
}) => {
  const isNetworkError = error.message.includes('fetch') || 
                        error.message.includes('network') ||
                        error.message.includes('Failed to fetch');

  return (
    <Card className="border-red-200 bg-red-50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-red-700">
          <AlertTriangle className="h-5 w-5" />
          {isNetworkError ? 'Connection Error' : 'Loading Error'}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <Alert className="border-red-200">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {isNetworkError && !isOnline ? (
              'You appear to be offline. Please check your internet connection.'
            ) : isNetworkError ? (
              'Unable to connect to our servers. Please try again.'
            ) : (
              `Something went wrong: ${error.message}`
            )}
          </AlertDescription>
        </Alert>

        <div className="flex items-center gap-2 text-sm text-gray-600">
          {isOnline ? (
            <><Wifi className="h-4 w-4 text-green-500" /> Online</>
          ) : (
            <><WifiOff className="h-4 w-4 text-red-500" /> Offline</>
          )}
        </div>

        <Button onClick={resetError} className="flex items-center gap-2">
          <RefreshCw className="h-4 w-4" />
          Try Again
        </Button>
      </CardContent>
    </Card>
  );
};

// React Query Error Boundary for async operations
export const AsyncErrorBoundary: React.FC<AsyncErrorBoundaryProps> = ({
  children,
  fallback: Fallback = DefaultAsyncErrorFallback,
  onError
}) => {
  const [error, setError] = useState<Error | null>(null);
  const isOnline = useOnlineStatus();

  useEffect(() => {
    // Listen for unhandled promise rejections
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const error = new Error(event.reason?.message || 'Unhandled promise rejection');
      setError(error);
      onError?.(error);
      
      // Prevent the default console error
      event.preventDefault();
    };

    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    return () => {
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, [onError]);

  const resetError = () => {
    setError(null);
  };

  if (error) {
    return <Fallback error={error} resetError={resetError} isOnline={isOnline} />;
  }

  return <>{children}</>;
};

// Specific fallbacks for different types of async errors
export const DataLoadingErrorFallback: React.FC<AsyncErrorFallbackProps> = ({ 
  error, 
  resetError 
}) => (
  <div className="text-center py-8">
    <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
    <h3 className="text-lg font-semibold text-gray-900 mb-2">
      Failed to Load Data
    </h3>
    <p className="text-gray-600 mb-4">
      We couldn't load the requested information. This might be a temporary issue.
    </p>
    <Button onClick={resetError} variant="outline">
      <RefreshCw className="h-4 w-4 mr-2" />
      Retry
    </Button>
  </div>
);

export const PaymentErrorFallback: React.FC<AsyncErrorFallbackProps> = ({ 
  error, 
  resetError 
}) => (
  <Alert className="border-red-200 bg-red-50">
    <AlertTriangle className="h-4 w-4" />
    <AlertDescription className="flex items-center justify-between">
      <div>
        <strong>Payment Error:</strong> {error.message}
      </div>
      <Button onClick={resetError} size="sm" variant="outline">
        Try Again
      </Button>
    </AlertDescription>
  </Alert>
);

export const AuthErrorFallback: React.FC<AsyncErrorFallbackProps> = ({ 
  error, 
  resetError 
}) => (
  <Card className="border-amber-200 bg-amber-50">
    <CardContent className="pt-6">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="font-semibold text-amber-800">Authentication Issue</h4>
          <p className="text-sm text-amber-700">Please sign in again to continue.</p>
        </div>
        <div className="space-x-2">
          <Button 
            onClick={() => window.location.href = '/login'} 
            size="sm"
          >
            Sign In
          </Button>
          <Button onClick={resetError} size="sm" variant="outline">
            Retry
          </Button>
        </div>
      </div>
    </CardContent>
  </Card>
);