/**
 * Shared error handling utilities
 * Provides consistent error types across the monorepo
 */

export class MonorepoError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode: number = 500
  ) {
    super(message);
    this.name = this.constructor.name;
    Error.captureStackTrace(this, this.constructor);
  }
}

export class ValidationError extends MonorepoError {
  constructor(message: string, public details?: Record<string, unknown>) {
    super(message, 'VALIDATION_ERROR', 400);
  }
}

export class NotFoundError extends MonorepoError {
  constructor(resource: string) {
    super(`${resource} not found`, 'NOT_FOUND', 404);
  }
}

export class UnauthorizedError extends MonorepoError {
  constructor(message: string = 'Unauthorized') {
    super(message, 'UNAUTHORIZED', 401);
  }
}

export class ForbiddenError extends MonorepoError {
  constructor(message: string = 'Forbidden') {
    super(message, 'FORBIDDEN', 403);
  }
}

export class ConflictError extends MonorepoError {
  constructor(message: string) {
    super(message, 'CONFLICT', 409);
  }
}

/**
 * Type guard to check if error is a MonorepoError
 */
export function isMonorepoError(error: unknown): error is MonorepoError {
  return error instanceof MonorepoError;
}

/**
 * Extract error message from unknown error type
 */
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === 'string') {
    return error;
  }
  return 'An unknown error occurred';
}

/**
 * Extract stack trace from error
 */
export function getErrorStack(error: unknown): string | undefined {
  if (error instanceof Error) {
    return error.stack;
  }
  return undefined;
}
