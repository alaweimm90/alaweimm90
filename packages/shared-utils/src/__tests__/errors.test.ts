/**
 * Tests for error utilities
 */

import {
  MonorepoError,
  ValidationError,
  NotFoundError,
  UnauthorizedError,
  ForbiddenError,
  ConflictError,
  isMonorepoError,
  getErrorMessage,
  getErrorStack,
} from '../errors';

describe('Error Utilities', () => {
  describe('MonorepoError', () => {
    it('should create base error with correct properties', () => {
      const error = new MonorepoError('Test error', 'TEST_ERROR', 500);
      expect(error.message).toBe('Test error');
      expect(error.code).toBe('TEST_ERROR');
      expect(error.statusCode).toBe(500);
      expect(error.name).toBe('MonorepoError');
    });
  });

  describe('ValidationError', () => {
    it('should create validation error with status 400', () => {
      const error = new ValidationError('Invalid input');
      expect(error.message).toBe('Invalid input');
      expect(error.statusCode).toBe(400);
      expect(error.code).toBe('VALIDATION_ERROR');
    });

    it('should accept details object', () => {
      const details = { field: 'email', value: 'invalid' };
      const error = new ValidationError('Invalid email', details);
      expect(error.details).toEqual(details);
    });
  });

  describe('NotFoundError', () => {
    it('should create not found error with status 404', () => {
      const error = new NotFoundError('User');
      expect(error.message).toBe('User not found');
      expect(error.statusCode).toBe(404);
      expect(error.code).toBe('NOT_FOUND');
    });
  });

  describe('UnauthorizedError', () => {
    it('should create unauthorized error with status 401', () => {
      const error = new UnauthorizedError();
      expect(error.statusCode).toBe(401);
      expect(error.code).toBe('UNAUTHORIZED');
    });
  });

  describe('ForbiddenError', () => {
    it('should create forbidden error with status 403', () => {
      const error = new ForbiddenError();
      expect(error.statusCode).toBe(403);
      expect(error.code).toBe('FORBIDDEN');
    });
  });

  describe('ConflictError', () => {
    it('should create conflict error with status 409', () => {
      const error = new ConflictError('User');
      expect(error.message).toBe('User already exists');
      expect(error.statusCode).toBe(409);
      expect(error.code).toBe('CONFLICT');
    });
  });

  describe('isMonorepoError', () => {
    it('should return true for MonorepoError instances', () => {
      expect(isMonorepoError(new MonorepoError('test', 'TEST', 500))).toBe(true);
      expect(isMonorepoError(new ValidationError('test'))).toBe(true);
      expect(isMonorepoError(new NotFoundError('User'))).toBe(true);
    });

    it('should return false for non-MonorepoError instances', () => {
      expect(isMonorepoError(new Error('regular error'))).toBe(false);
      expect(isMonorepoError('string')).toBe(false);
      expect(isMonorepoError(null)).toBe(false);
      expect(isMonorepoError(undefined)).toBe(false);
    });
  });

  describe('getErrorMessage', () => {
    it('should extract message from Error instances', () => {
      expect(getErrorMessage(new Error('test error'))).toBe('test error');
      expect(getErrorMessage(new ValidationError('validation failed'))).toBe('validation failed');
    });

    it('should handle non-Error values', () => {
      expect(getErrorMessage('string error')).toBe('string error');
      expect(getErrorMessage({ message: 'object error' })).toBe('object error');
      expect(getErrorMessage(null)).toBe('Unknown error');
      expect(getErrorMessage(undefined)).toBe('Unknown error');
    });
  });

  describe('getErrorStack', () => {
    it('should extract stack from Error instances', () => {
      const error = new Error('test');
      expect(getErrorStack(error)).toBeDefined();
      expect(typeof getErrorStack(error)).toBe('string');
    });

    it('should return undefined for non-Error values', () => {
      expect(getErrorStack('string')).toBeUndefined();
      expect(getErrorStack(null)).toBeUndefined();
      expect(getErrorStack({})).toBeUndefined();
    });
  });
});
