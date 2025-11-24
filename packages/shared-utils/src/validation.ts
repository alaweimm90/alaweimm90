/**
 * Shared validation utilities
 * Provides common validation functions across the monorepo
 */

import { ValidationError } from './errors';

/**
 * Validate that a value is not null or undefined
 */
export function required<T>(value: T | null | undefined, fieldName: string): T {
  if (value === null || value === undefined) {
    throw new ValidationError(`${fieldName} is required`);
  }
  return value;
}

/**
 * Validate email format
 */
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate URL format
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate UUID format
 */
export function isValidUUID(uuid: string): boolean {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
  return uuidRegex.test(uuid);
}

/**
 * Validate string length
 */
export function validateLength(
  value: string,
  min?: number,
  max?: number,
  fieldName: string = 'Value'
): void {
  if (min !== undefined && value.length < min) {
    throw new ValidationError(`${fieldName} must be at least ${min} characters`);
  }
  if (max !== undefined && value.length > max) {
    throw new ValidationError(`${fieldName} must be at most ${max} characters`);
  }
}

/**
 * Validate number range
 */
export function validateRange(
  value: number,
  min?: number,
  max?: number,
  fieldName: string = 'Value'
): void {
  if (min !== undefined && value < min) {
    throw new ValidationError(`${fieldName} must be at least ${min}`);
  }
  if (max !== undefined && value > max) {
    throw new ValidationError(`${fieldName} must be at most ${max}`);
  }
}

/**
 * Validate that value is one of allowed values
 */
export function validateEnum<T>(
  value: T,
  allowedValues: T[],
  fieldName: string = 'Value'
): void {
  if (!allowedValues.includes(value)) {
    throw new ValidationError(
      `${fieldName} must be one of: ${allowedValues.join(', ')}`
    );
  }
}

/**
 * Sanitize string (remove HTML tags)
 */
export function sanitizeString(input: string): string {
  return input.replace(/<[^>]*>/g, '');
}

/**
 * Validate and sanitize object
 */
export function validateObject<T extends object>(
  obj: unknown,
  schema: Record<keyof T, (value: unknown) => boolean>,
  fieldName: string = 'Object'
): void {
  if (typeof obj !== 'object' || obj === null) {
    throw new ValidationError(`${fieldName} must be an object`);
  }

  const typedObj = obj as Record<string, unknown>;

  for (const [key, validator] of Object.entries(schema)) {
    if (typeof validator === 'function' && !validator(typedObj[key])) {
      throw new ValidationError(`${fieldName}.${key} is invalid`);
    }
  }
}
