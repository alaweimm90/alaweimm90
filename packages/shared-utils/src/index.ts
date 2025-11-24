/**
 * @monorepo/shared-utils
 *
 * Shared utilities for the entire monorepo including:
 * - Logging (Winston-based, colored output)
 * - Error handling (Custom error classes)
 * - Validation (Common validation functions)
 */

// Export logger
export { createLogger, logger, type LogLevel, type LoggerOptions } from './logger';

// Export errors
export {
  MonorepoError,
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  ForbiddenError,
  ConflictError,
  isMonorepoError,
  getErrorMessage,
  getErrorStack,
} from './errors';

// Export validation
export {
  required,
  isValidEmail,
  isValidUrl,
  isValidUUID,
  validateLength,
  validateRange,
  validateEnum,
  sanitizeString,
  validateObject,
} from './validation';
