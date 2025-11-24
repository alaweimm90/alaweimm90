/**
 * Tests for validation utilities
 */

import {
  required,
  isValidEmail,
  isValidUrl,
  isValidUUID,
  validateLength,
  validateRange,
  validateEnum,
  sanitizeString,
  ValidationError,
} from '../validation';

describe('Validation Utilities', () => {
  describe('required', () => {
    it('should return value if not null or undefined', () => {
      expect(required('test', 'field')).toBe('test');
      expect(required(0, 'field')).toBe(0);
      expect(required(false, 'field')).toBe(false);
    });

    it('should throw ValidationError if value is null or undefined', () => {
      expect(() => required(null, 'field')).toThrow(ValidationError);
      expect(() => required(undefined, 'field')).toThrow(ValidationError);
      expect(() => required(null, 'field')).toThrow('field is required');
    });
  });

  describe('isValidEmail', () => {
    it('should validate correct email formats', () => {
      expect(isValidEmail('test@example.com')).toBe(true);
      expect(isValidEmail('user.name@domain.co.uk')).toBe(true);
      expect(isValidEmail('user+tag@example.org')).toBe(true);
    });

    it('should reject invalid email formats', () => {
      expect(isValidEmail('invalid')).toBe(false);
      expect(isValidEmail('test@')).toBe(false);
      expect(isValidEmail('@example.com')).toBe(false);
      expect(isValidEmail('test @example.com')).toBe(false);
    });
  });

  describe('isValidUrl', () => {
    it('should validate correct URL formats', () => {
      expect(isValidUrl('https://example.com')).toBe(true);
      expect(isValidUrl('http://localhost:3000')).toBe(true);
      expect(isValidUrl('ftp://files.example.com')).toBe(true);
    });

    it('should reject invalid URL formats', () => {
      expect(isValidUrl('invalid')).toBe(false);
      expect(isValidUrl('htp://wrong')).toBe(false);
      expect(isValidUrl('')).toBe(false);
    });
  });

  describe('isValidUUID', () => {
    it('should validate correct UUID formats', () => {
      expect(isValidUUID('123e4567-e89b-12d3-a456-426614174000')).toBe(true);
      expect(isValidUUID('550e8400-e29b-41d4-a716-446655440000')).toBe(true);
    });

    it('should reject invalid UUID formats', () => {
      expect(isValidUUID('invalid')).toBe(false);
      expect(isValidUUID('123e4567-e89b-12d3-a456')).toBe(false);
      expect(isValidUUID('123e4567e89b12d3a456426614174000')).toBe(false);
    });
  });

  describe('validateLength', () => {
    it('should pass for valid string lengths', () => {
      expect(() => validateLength('test', 1, 10)).not.toThrow();
      expect(() => validateLength('hello', 5, 5)).not.toThrow();
    });

    it('should throw for strings too short', () => {
      expect(() => validateLength('hi', 5)).toThrow(ValidationError);
      expect(() => validateLength('hi', 5)).toThrow('at least 5 characters');
    });

    it('should throw for strings too long', () => {
      expect(() => validateLength('verylongstring', undefined, 5)).toThrow(ValidationError);
      expect(() => validateLength('verylongstring', undefined, 5)).toThrow('at most 5 characters');
    });
  });

  describe('validateRange', () => {
    it('should pass for valid number ranges', () => {
      expect(() => validateRange(5, 1, 10)).not.toThrow();
      expect(() => validateRange(0, 0, 0)).not.toThrow();
    });

    it('should throw for numbers too small', () => {
      expect(() => validateRange(3, 5)).toThrow(ValidationError);
      expect(() => validateRange(3, 5)).toThrow('at least 5');
    });

    it('should throw for numbers too large', () => {
      expect(() => validateRange(15, undefined, 10)).toThrow(ValidationError);
      expect(() => validateRange(15, undefined, 10)).toThrow('at most 10');
    });
  });

  describe('validateEnum', () => {
    it('should pass for allowed values', () => {
      expect(() => validateEnum('active', ['active', 'inactive'])).not.toThrow();
      expect(() => validateEnum(1, [1, 2, 3])).not.toThrow();
    });

    it('should throw for disallowed values', () => {
      expect(() => validateEnum('pending', ['active', 'inactive'])).toThrow(ValidationError);
      expect(() => validateEnum('pending', ['active', 'inactive'])).toThrow('must be one of');
    });
  });

  describe('sanitizeString', () => {
    it('should remove HTML tags', () => {
      expect(sanitizeString('<script>alert("xss")</script>')).toBe('alert("xss")');
      expect(sanitizeString('Hello <b>World</b>!')).toBe('Hello World!');
      expect(sanitizeString('<p>Test</p>')).toBe('Test');
    });

    it('should leave plain text unchanged', () => {
      expect(sanitizeString('plain text')).toBe('plain text');
      expect(sanitizeString('no tags here')).toBe('no tags here');
    });
  });
});
