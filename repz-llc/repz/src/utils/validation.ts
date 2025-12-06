import { logger } from '@/utils/logger';

export interface ValidationRule<T> {
  validate: (value: T) => boolean;
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
}

export class DataValidator {
  // Email validation
  static email(email: string): ValidationResult {
    const rules: ValidationRule<string>[] = [
      {
        validate: (val) => typeof val === 'string' && val.length > 0,
        message: 'Email is required'
      },
      {
        validate: (val) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val),
        message: 'Invalid email format'
      },
      {
        validate: (val) => val.length <= 254,
        message: 'Email too long'
      }
    ];

    return this.validateWithRules(email, rules);
  }

  // Password validation
  static password(password: string): ValidationResult {
    const rules: ValidationRule<string>[] = [
      {
        validate: (val) => typeof val === 'string' && val.length >= 8,
        message: 'Password must be at least 8 characters'
      },
      {
        validate: (val) => /(?=.*[a-z])/.test(val),
        message: 'Password must contain at least one lowercase letter'
      },
      {
        validate: (val) => /(?=.*[A-Z])/.test(val),
        message: 'Password must contain at least one uppercase letter'
      },
      {
        validate: (val) => /(?=.*\d)/.test(val),
        message: 'Password must contain at least one number'
      }
    ];

    return this.validateWithRules(password, rules);
  }

  // Fitness data validation
  static age(age: number): ValidationResult {
    const rules: ValidationRule<number>[] = [
      {
        validate: (val) => typeof val === 'number' && !isNaN(val),
        message: 'Age must be a number'
      },
      {
        validate: (val) => val >= 13 && val <= 120,
        message: 'Age must be between 13 and 120'
      }
    ];

    return this.validateWithRules(age, rules);
  }

  static weight(weight: number): ValidationResult {
    const rules: ValidationRule<number>[] = [
      {
        validate: (val) => typeof val === 'number' && !isNaN(val),
        message: 'Weight must be a number'
      },
      {
        validate: (val) => val >= 30 && val <= 300,
        message: 'Weight must be between 30 and 300 kg'
      }
    ];

    return this.validateWithRules(weight, rules);
  }

  static height(height: number): ValidationResult {
    const rules: ValidationRule<number>[] = [
      {
        validate: (val) => typeof val === 'number' && !isNaN(val),
        message: 'Height must be a number'
      },
      {
        validate: (val) => val >= 100 && val <= 250,
        message: 'Height must be between 100 and 250 cm'
      }
    ];

    return this.validateWithRules(height, rules);
  }

  static bodyFatPercentage(bodyFat: number): ValidationResult {
    const rules: ValidationRule<number>[] = [
      {
        validate: (val) => typeof val === 'number' && !isNaN(val),
        message: 'Body fat percentage must be a number'
      },
      {
        validate: (val) => val >= 1 && val <= 50,
        message: 'Body fat percentage must be between 1% and 50%'
      }
    ];

    return this.validateWithRules(bodyFat, rules);
  }

  // Generic validation helper
  private static validateWithRules<T>(
    value: T, 
    rules: ValidationRule<T>[]
  ): ValidationResult {
    const errors: string[] = [];

    for (const rule of rules) {
      try {
        if (!rule.validate(value)) {
          errors.push(rule.message);
        }
      } catch (error) {
        logger.error('Validation rule error', error);
        errors.push('Validation error occurred');
      }
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  // Validate multiple fields
  static validateForm(data: Record<string, unknown>, schema: Record<string, (value: unknown) => ValidationResult>): ValidationResult {
    const allErrors: string[] = [];

    for (const [field, validator] of Object.entries(schema)) {
      const value = data[field];
      const result = validator(value);
      
      if (!result.isValid) {
        allErrors.push(...result.errors.map(error => `${field}: ${error}`));
      }
    }

    return {
      isValid: allErrors.length === 0,
      errors: allErrors
    };
  }
}