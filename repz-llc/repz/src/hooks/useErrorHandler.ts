import { useCallback, useState } from 'react';
import { toast } from 'sonner';

interface ErrorState {
  error: Error | null;
  isError: boolean;
  isLoading: boolean;
}

interface ErrorHandlerOptions {
  showToast?: boolean;
  retryAttempts?: number;
  retryDelay?: number;
  onError?: (error: Error) => void;
  onRetry?: () => void;
}

export const useErrorHandler = (options: ErrorHandlerOptions = {}) => {
  const {
    showToast = true,
    retryAttempts = 3,
    retryDelay = 1000,
    onError,
    onRetry
  } = options;

  const [errorState, setErrorState] = useState<ErrorState>({
    error: null,
    isError: false,
    isLoading: false
  });

  const [retryCount, setRetryCount] = useState(0);

  // Handle errors with retry logic
  const handleError = useCallback((error: Error) => {
    setErrorState({
      error,
      isError: true,
      isLoading: false
    });

    if (showToast) {
      toast.error(error.message || 'An unexpected error occurred');
    }

    onError?.(error);
    
    // Log error for monitoring
    console.error('[ErrorHandler]', {
      message: error.message,
      stack: error.stack,
      timestamp: new Date().toISOString(),
      retryCount
    });
  }, [showToast, onError, retryCount]);

  // Clear error state
  const clearError = useCallback(() => {
    setErrorState({
      error: null,
      isError: false,
      isLoading: false
    });
    setRetryCount(0);
  }, []);

  // Retry failed operation
  const retry = useCallback(async <T>(asyncFn: () => Promise<T>): Promise<T | undefined> => {
    if (retryCount >= retryAttempts) {
      toast.error('Maximum retry attempts reached');
      return undefined;
    }

    setErrorState(prev => ({ ...prev, isLoading: true }));
    setRetryCount(prev => prev + 1);
    
    onRetry?.();

    try {
      // Wait before retry
      if (retryDelay > 0) {
        await new Promise(resolve => setTimeout(resolve, retryDelay * retryCount));
      }

      const result = await asyncFn();
      clearError();
      return result;
    } catch (error) {
      handleError(error as Error);
      throw error;
    }
  }, [retryCount, retryAttempts, retryDelay, onRetry, handleError, clearError]);

  // Wrap async operations with error handling
  const executeAsync = useCallback(async <T>(
    asyncFn: () => Promise<T>,
    options?: { skipErrorToast?: boolean }
  ): Promise<T | null> => {
    setErrorState(prev => ({ ...prev, isLoading: true, isError: false }));

    try {
      const result = await asyncFn();
      setErrorState({
        error: null,
        isError: false,
        isLoading: false
      });
      return result;
    } catch (error) {
      if (!options?.skipErrorToast) {
        handleError(error as Error);
      } else {
        setErrorState({
          error: error as Error,
          isError: true,
          isLoading: false
        });
      }
      return null;
    }
  }, [handleError]);

  // Handle synchronous operations
  const executeSafe = useCallback(<T>(
    syncFn: () => T,
    defaultValue: T
  ): T => {
    try {
      return syncFn();
    } catch (error) {
      handleError(error as Error);
      return defaultValue;
    }
  }, [handleError]);

  return {
    ...errorState,
    handleError,
    clearError,
    retry,
    executeAsync,
    executeSafe,
    retryCount,
    canRetry: retryCount < retryAttempts
  };
};

// Hook for handling API errors specifically
export const useAPIErrorHandler = () => {
  const errorHandler = useErrorHandler({
    showToast: true,
    retryAttempts: 2,
    retryDelay: 1000
  });

  const handleAPIError = useCallback((error: unknown) => {
    let message = 'An unexpected error occurred';

    const apiError = error as { response?: { status?: number; data?: { message?: string } }; message?: string };
    if (apiError?.response?.status) {
      const status = apiError.response.status;
      switch (status) {
        case 400:
          message = 'Invalid request. Please check your input.';
          break;
        case 401:
          message = 'Authentication required. Please sign in.';
          // Redirect to login
          window.location.href = '/login';
          break;
        case 403:
          message = 'You do not have permission for this action.';
          break;
        case 404:
          message = 'Requested resource not found.';
          break;
        case 429:
          message = 'Too many requests. Please try again later.';
          break;
        case 500:
          message = 'Server error. Our team has been notified.';
          break;
        default:
          message = error?.response?.data?.message || error.message || message;
      }
    } else if (error?.message) {
      message = error.message;
    }

    const apiError = new Error(message);
    errorHandler.handleError(apiError);
  }, [errorHandler]);

  return {
    ...errorHandler,
    handleAPIError
  };
};

// Hook for form error handling
export const useFormErrorHandler = () => {
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});

  const setFieldError = useCallback((field: string, message: string) => {
    setFieldErrors(prev => ({
      ...prev,
      [field]: message
    }));
  }, []);

  const clearFieldError = useCallback((field: string) => {
    setFieldErrors(prev => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
  }, []);

  const clearAllFieldErrors = useCallback(() => {
    setFieldErrors({});
  }, []);

  const hasFieldError = useCallback((field: string) => {
    return !!fieldErrors[field];
  }, [fieldErrors]);

  const getFieldError = useCallback((field: string) => {
    return fieldErrors[field];
  }, [fieldErrors]);

  return {
    fieldErrors,
    setFieldError,
    clearFieldError,
    clearAllFieldErrors,
    hasFieldError,
    getFieldError,
    hasErrors: Object.keys(fieldErrors).length > 0
  };
};