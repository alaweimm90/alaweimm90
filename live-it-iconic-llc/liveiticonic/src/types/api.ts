/**
 * API Response and Error Types
 * Shared types for all API endpoints
 */

/**
 * Standard API error response
 */
export interface ApiError {
  error: string;
  details?: string;
  code?: string;
}

/**
 * Standard API success response
 */
export interface ApiResponse<T = unknown> {
  data?: T;
  message?: string;
  success: boolean;
}

/**
 * Pagination metadata
 */
export interface PaginationMeta {
  page: number;
  pageSize: number;
  totalPages: number;
  totalItems: number;
}

/**
 * Paginated API response
 */
export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  meta: PaginationMeta;
}

/**
 * Generic error type for catch blocks
 */
export interface ErrorWithMessage {
  message: string;
  code?: string;
  status?: number;
}

/**
 * Type guard to check if error has message property
 */
export function isErrorWithMessage(error: unknown): error is ErrorWithMessage {
  return (
    typeof error === 'object' &&
    error !== null &&
    'message' in error &&
    typeof (error as Record<string, unknown>).message === 'string'
  );
}

/**
 * Convert unknown error to ErrorWithMessage
 */
export function toErrorWithMessage(maybeError: unknown): ErrorWithMessage {
  if (isErrorWithMessage(maybeError)) return maybeError;

  try {
    return {
      message: JSON.stringify(maybeError),
    };
  } catch {
    // fallback in case there's an error stringify-ing
    return {
      message: String(maybeError),
    };
  }
}

/**
 * Get error message from unknown error
 */
export function getErrorMessage(error: unknown): string {
  return toErrorWithMessage(error).message;
}
