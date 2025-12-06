import { logger } from '@/utils/logger';

export class ErrorHandler {
  // Network error handling
  static handleNetworkError(error: unknown, context: string): string {
    logger.error(`Network error in ${context}`, error);

    if (!navigator.onLine) {
      return 'You appear to be offline. Please check your internet connection.';
    }

    if (error?.message?.includes('fetch')) {
      return 'Unable to connect to our servers. Please try again in a moment.';
    }

    if (error?.status) {
      switch (error.status) {
        case 400:
          return 'Invalid request. Please check your input and try again.';
        case 401:
          return 'Authentication required. Please log in again.';
        case 403:
          return 'Access denied. You don\'t have permission for this action.';
        case 404:
          return 'The requested resource was not found.';
        case 429:
          return 'Too many requests. Please wait a moment before trying again.';
        case 500:
          return 'Server error. Our team has been notified.';
        case 503:
          return 'Service temporarily unavailable. Please try again later.';
        default:
          return `An error occurred (${error.status}). Please try again.`;
      }
    }

    return 'An unexpected error occurred. Please try again.';
  }

  // Auth error handling
  static handleAuthError(error: unknown): string {
    logger.error('Authentication error', error);

    const errorMessage = error?.message?.toLowerCase() || '';

    if (errorMessage.includes('email not confirmed')) {
      return 'Please check your email and click the confirmation link before signing in.';
    }

    if (errorMessage.includes('invalid login credentials')) {
      return 'Invalid email or password. Please check your credentials and try again.';
    }

    if (errorMessage.includes('signup requires a valid password')) {
      return 'Please enter a valid password with at least 8 characters.';
    }

    if (errorMessage.includes('user already registered')) {
      return 'An account with this email already exists. Try signing in instead.';
    }

    if (errorMessage.includes('rate limit')) {
      return 'Too many login attempts. Please wait a few minutes before trying again.';
    }

    return 'Authentication failed. Please try again.';
  }

  // Supabase error handling
  static handleSupabaseError(error: unknown, operation: string): string {
    logger.error(`Supabase error during ${operation}`, error);

    if (error?.code) {
      switch (error.code) {
        case 'PGRST301':
          return 'Access denied. Please check your permissions.';
        case 'PGRST116':
          return 'The requested data was not found.';
        case '23505':
          return 'This record already exists.';
        case '23503':
          return 'Cannot complete operation due to data dependencies.';
        default:
          logger.error(`Unknown Supabase error code: ${error.code}`, error);
      }
    }

    return this.handleNetworkError(error, operation);
  }

  // Form validation error handling
  static formatValidationErrors(errors: string[]): string {
    if (errors.length === 0) return '';
    if (errors.length === 1) return errors[0];
    
    return `Please fix the following issues:\n• ${errors.join('\n• ')}`;
  }

  // Generic error boundary handler
  static handleUnexpectedError(error: Error, context: string): void {
    logger.error(`Unexpected error in ${context}`, {
      message: error.message,
      stack: error.stack,
      name: error.name
    });

    // In production, you could send this to an error tracking service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry.captureException(error);
    }
  }

  // Rate limiting handler
  static handleRateLimit(retryAfter?: number): string {
    const waitTime = retryAfter || 60;
    return `Too many requests. Please wait ${waitTime} seconds before trying again.`;
  }

  // Offline handler
  static handleOfflineError(): string {
    return 'You\'re currently offline. This action will be retried when you\'re back online.';
  }
}