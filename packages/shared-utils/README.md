# @monorepo/shared-utils

Shared utilities for the entire monorepo providing consistent logging, error handling, and validation across all packages.

## Installation

```bash
pnpm add @monorepo/shared-utils
```

## Features

### 📝 Logging

Winston-based logger with colored console output and consistent formatting.

```typescript
import { createLogger, logger } from '@monorepo/shared-utils';

// Use default logger
logger.info('Application started');
logger.warn('Deprecation warning');
logger.error('Something went wrong');

// Create custom logger for your service
const myLogger = createLogger({ service: 'api-gateway', level: 'debug' });
myLogger.debug('Detailed debug info');
```

### ❌ Error Handling

Custom error classes with HTTP status codes and error codes.

```typescript
import {
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  isMonorepoError,
  getErrorMessage,
} from '@monorepo/shared-utils';

// Throw validation errors
throw new ValidationError('Invalid email format', { email: 'bad@' });

// Throw not found errors
throw new NotFoundError('User');

// Check error type
if (isMonorepoError(error)) {
  console.log(error.code, error.statusCode);
}

// Safe error message extraction
const message = getErrorMessage(unknownError);
```

### ✅ Validation

Common validation functions for emails, URLs, UUIDs, and more.

```typescript
import {
  required,
  isValidEmail,
  isValidUrl,
  validateLength,
  validateRange,
  validateEnum,
} from '@monorepo/shared-utils';

// Require non-null values
const email = required(req.body.email, 'email');

// Validate email format
if (!isValidEmail(email)) {
  throw new ValidationError('Invalid email');
}

// Validate string length
validateLength(password, 8, 128, 'password');

// Validate number range
validateRange(age, 0, 120, 'age');

// Validate enum
validateEnum(status, ['active', 'inactive'], 'status');
```

## API Reference

### Logger

#### `createLogger(options?: LoggerOptions): winston.Logger`

Create a configured Winston logger instance.

**Options**:
- `service?: string` - Service name (default: 'monorepo')
- `level?: LogLevel` - Log level ('error' | 'warn' | 'info' | 'debug', default: 'info')
- `silent?: boolean` - Disable logging (default: false)

#### `logger: winston.Logger`

Default logger instance with service name 'shared'.

### Errors

#### `MonorepoError extends Error`

Base error class for the monorepo.

**Properties**:
- `code: string` - Error code
- `statusCode: number` - HTTP status code (default: 500)

#### `ValidationError extends MonorepoError`

Validation error (status 400).

#### `NotFoundError extends MonorepoError`

Resource not found error (status 404).

#### `UnauthorizedError extends MonorepoError`

Unauthorized error (status 401).

#### `ForbiddenError extends MonorepoError`

Forbidden error (status 403).

#### `ConflictError extends MonorepoError`

Conflict error (status 409).

#### `isMonorepoError(error: unknown): error is MonorepoError`

Type guard to check if error is a MonorepoError.

#### `getErrorMessage(error: unknown): string`

Extract error message from unknown error type.

#### `getErrorStack(error: unknown): string | undefined`

Extract stack trace from error.

### Validation

#### `required<T>(value: T | null | undefined, fieldName: string): T`

Validate that a value is not null or undefined. Throws ValidationError if invalid.

#### `isValidEmail(email: string): boolean`

Validate email format.

#### `isValidUrl(url: string): boolean`

Validate URL format.

#### `isValidUUID(uuid: string): boolean`

Validate UUID format.

#### `validateLength(value: string, min?: number, max?: number, fieldName?: string): void`

Validate string length. Throws ValidationError if invalid.

#### `validateRange(value: number, min?: number, max?: number, fieldName?: string): void`

Validate number range. Throws ValidationError if invalid.

#### `validateEnum<T>(value: T, allowedValues: T[], fieldName?: string): void`

Validate that value is one of allowed values. Throws ValidationError if invalid.

#### `sanitizeString(input: string): string`

Sanitize string by removing HTML tags.

#### `validateObject<T>(obj: unknown, schema: Record<keyof T, (value: unknown) => boolean>, fieldName?: string): void`

Validate and sanitize object against schema. Throws ValidationError if invalid.

## Development

```bash
# Build
pnpm build

# Watch mode
pnpm build:watch

# Run tests
pnpm test

# Lint
pnpm lint
```

## License

MIT
