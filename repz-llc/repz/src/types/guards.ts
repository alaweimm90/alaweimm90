// Type guards - Runtime type checking functions
// These functions provide type-safe runtime validation

import type { 
  TierType, 
  UserRole, 
  BillingCycle, 
  ApiResponse
} from './index';

// Define ValidationResult locally since it's not exported yet
type ValidationResult<T> = 
  | { success: true; data: T }
  | { success: false; error: string };

// Business type guards
export const isTierType = (value: unknown): value is TierType => {
  return typeof value === 'string' && 
    ['core', 'adaptive', 'performance', 'longevity'].includes(value);
};

export const isUserRole = (value: unknown): value is UserRole => {
  return typeof value === 'string' && 
    ['client', 'coach', 'admin'].includes(value);
};

export const isBillingCycle = (value: unknown): value is BillingCycle => {
  return typeof value === 'string' && 
    ['monthly', 'quarterly', 'semi-annual', 'annual'].includes(value);
};

// API response guards
export const isApiResponse = <T = unknown>(value: unknown): value is ApiResponse<T> => {
  return typeof value === 'object' &&
    value !== null &&
    'success' in value &&
    'data' in value &&
    'timestamp' in value &&
    'requestId' in value &&
    typeof (value as Record<string, unknown>).success === 'boolean';
};

export const isSuccessResponse = <T = unknown>(value: unknown): value is ApiResponse<T> & { success: true } => {
  return isApiResponse(value) && value.success === true;
};

export const isErrorResponse = (value: unknown): value is ApiResponse<never> & { success: false } => {
  return isApiResponse(value) && value.success === false;
};

// Validation guards
export const isValidationResult = <T = unknown>(value: unknown): value is ValidationResult<T> => {
  if (typeof value !== 'object' || value === null || !('success' in value)) {
    return false;
  }
  const obj = value as Record<string, unknown>;
  return typeof obj.success === 'boolean' &&
    (
      (obj.success === true && 'data' in obj) ||
      (obj.success === false && 'error' in obj)
    );
};

// Utility type guards
export const isNonEmptyArray = <T>(value: T[]): value is [T, ...T[]] => {
  return Array.isArray(value) && value.length > 0;
};

export const isNonNullable = <T>(value: T): value is NonNullable<T> => {
  return value !== null && value !== undefined;
};

export const isString = (value: unknown): value is string => {
  return typeof value === 'string';
};

export const isNumber = (value: unknown): value is number => {
  return typeof value === 'number' && !isNaN(value);
};

export const isBoolean = (value: unknown): value is boolean => {
  return typeof value === 'boolean';
};

export const isObject = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
};

export const isArray = (value: unknown): value is unknown[] => {
  return Array.isArray(value);
};

export const isFunction = (value: unknown): value is Function => {
  return typeof value === 'function';
};

export const isDate = (value: unknown): value is Date => {
  return value instanceof Date && !isNaN(value.getTime());
};

export const isPromise = <T = unknown>(value: unknown): value is Promise<T> => {
  return value instanceof Promise ||
    (typeof value === 'object' &&
     value !== null &&
     'then' in value &&
     typeof (value as Record<string, unknown>).then === 'function');
};

// Email validation guard
export const isEmail = (value: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(value);
};

// URL validation guard
export const isUrl = (value: string): boolean => {
  try {
    new URL(value);
    return true;
  } catch {
    return false;
  }
};

// UUID validation guard
export const isUuid = (value: string): boolean => {
  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  return uuidRegex.test(value);
};

// Phone number validation guard (basic)
export const isPhoneNumber = (value: string): boolean => {
  const phoneRegex = /^\+?[\d\s\-\(\)]+$/;
  return phoneRegex.test(value) && value.replace(/\D/g, '').length >= 10;
};

// Branded type guards
export const createBrandedTypeGuard = <T, B>(
  validator: (value: unknown) => value is T
) => {
  return (value: unknown): value is T => {
    return validator(value);
  };
};

// Complex object validation
export const hasRequiredProperties = <T extends Record<string, unknown>>(
  obj: unknown,
  properties: (keyof T)[]
): obj is T => {
  if (!isObject(obj)) return false;
  
  return properties.every(prop => 
    typeof prop === 'string' && prop in obj
  );
};

// Array validation with predicate
export const isArrayOf = <T>(
  value: unknown,
  predicate: (item: unknown) => item is T
): value is T[] => {
  return isArray(value) && value.every(predicate);
};

// Object validation with predicate
export const isObjectWith = <T extends Record<string, unknown>>(
  value: unknown,
  predicate: (obj: Record<string, unknown>) => obj is T
): value is T => {
  return isObject(value) && predicate(value);
};

// Nested property existence check
export const hasNestedProperty = (
  obj: unknown,
  path: string
): boolean => {
  if (!isObject(obj)) return false;

  const keys = path.split('.');
  let current: unknown = obj;
  
  for (const key of keys) {
    if (!isObject(current) || !(key in current)) {
      return false;
    }
    current = current[key];
  }
  
  return true;
};

// Type assertion with error message
export const assertType = <T>(
  value: unknown,
  guard: (value: unknown) => value is T,
  errorMessage?: string
): T => {
  if (!guard(value)) {
    throw new Error(errorMessage || `Type assertion failed`);
  }
  return value;
};

// Safe type casting with default
export const safeCast = <T>(
  value: unknown,
  guard: (value: unknown) => value is T,
  defaultValue: T
): T => {
  return guard(value) ? value : defaultValue;
};