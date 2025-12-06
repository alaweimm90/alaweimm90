import { useMemo } from 'react';
import { FormService } from '@/services/FormService';
import { getTierConfig } from '@/constants/tiers';

export interface ValidationRule {
  field: string;
  required?: boolean;
  type?: 'email' | 'phone' | 'date' | 'number' | 'text';
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  customValidator?: (value: unknown, formData: Record<string, unknown>) => string | null;
}

export interface ValidationResult {
  isValid: boolean;
  errors: Record<string, string>;
  hasErrors: boolean;
}

export const useFormValidation = () => {
  const validateStep = (stepNumber: number, formData: Record<string, unknown>, selectedTier?: string): boolean => {
    switch (stepNumber) {
      case 1:
        return !!(
          formData.fullName?.trim() &&
          formData.email?.trim() &&
          formData.password &&
          formData.confirmPassword &&
          formData.password === formData.confirmPassword &&
          formData.password.length >= 6 &&
          formData.termsAccepted
        );
      case 2:
        return !!(
          formData.phone?.trim() &&
          formData.dateOfBirth &&
          formData.location?.trim() &&
          formData.currentWeight &&
          formData.height
        );
      case 3:
        return !!(
          formData.activityLevel &&
          formData.sleepHours
        );
      case 4:
        return !!(
          formData.specificGoals?.trim() &&
          formData.fitnessLevel &&
          formData.trainingFrequency &&
          formData.availableTime
        );
      case 5:
        return !!(
          formData.currentDiet &&
          formData.mealsPerDay &&
          formData.cookingExperience &&
          formData.mealPrepTime
        );
      case 6:
        return !!(
          formData.communicationStyle &&
          formData.motivationStyle &&
          // Tier-specific validation
          (() => {
            if (!selectedTier) return true;
            const tierConfig = getTierConfig(selectedTier as string);
            switch (tierConfig.id) {
              case 'core':
                return formData.coreGoals?.trim();
              case 'adaptive':
                return formData.performanceGoals?.trim() && formData.trackingPreference;
              case 'performance':
                return formData.performanceAdvancedGoals?.trim() && formData.dataTracking;
              case 'longevity':
                return formData.longevityGoals?.trim() && formData.healthOptimization && formData.communicationFrequency;
              default:
                return true;
            }
          })()
        );
      case 7:
        return true; // Payment step validation handled internally
      default:
        return false;
    }
  };

  const getPasswordStrength = (password: string): string => {
    const strength = FormService.calculatePasswordStrength(password);
    if (strength === 0) return '';
    if (strength <= 2) return 'weak';
    if (strength >= 4) return 'strong';
    return 'medium';
  };

  const validateFormData = (formData: Record<string, unknown>, rules: ValidationRule[]): ValidationResult => {
    const errors: Record<string, string> = {};

    rules.forEach(rule => {
      const value = formData[rule.field];
      
      // Required field validation
      if (rule.required && (!value || (typeof value === 'string' && !value.trim()))) {
        // Convert camelCase to friendly names
        const friendlyName = rule.field
          .replace(/([A-Z])/g, ' $1')
          .replace(/^./, str => str.toUpperCase())
          .replace(/([a-z])([A-Z])/g, '$1 $2')
          .toLowerCase()
          .replace(/^./, str => str.toUpperCase());
        errors[rule.field] = `${friendlyName} is required`;
        return;
      }

      // Skip other validations if field is empty and not required
      if (!value && !rule.required) return;

      // Type-specific validation
      switch (rule.type) {
        case 'email':
          if (value && !FormService.validateEmail(value)) {
            errors[rule.field] = 'Please enter a valid email address';
          }
          break;
        
        case 'phone':
          if (value && !/^[\+]?[\d\s\-\(\)]{10,}$/.test(value)) {
            errors[rule.field] = 'Please enter a valid phone number';
          }
          break;
        
        case 'date':
          if (value && isNaN(Date.parse(value))) {
            errors[rule.field] = 'Please enter a valid date';
          }
          break;
        
        case 'number':
          if (value && (isNaN(Number(value)) || Number(value) < 0)) {
            errors[rule.field] = 'Please enter a valid number';
          }
          break;
      }

      // Length validation
      if (rule.minLength && value && value.length < rule.minLength) {
        errors[rule.field] = `Minimum length is ${rule.minLength} characters`;
      }
      
      if (rule.maxLength && value && value.length > rule.maxLength) {
        errors[rule.field] = `Maximum length is ${rule.maxLength} characters`;
      }

      // Pattern validation
      if (rule.pattern && value && !rule.pattern.test(value)) {
        errors[rule.field] = `Invalid format for ${rule.field}`;
      }

      // Custom validation
      if (rule.customValidator) {
        const customError = rule.customValidator(value, formData);
        if (customError) {
          errors[rule.field] = customError;
        }
      }
    });

    return {
      isValid: Object.keys(errors).length === 0,
      errors,
      hasErrors: Object.keys(errors).length > 0
    };
  };

  return {
    validateStep,
    getPasswordStrength,
    validateFormData
  };
};

// Predefined validation rule sets for each step
export const stepValidationRules = {
  step1: [
    { field: 'fullName', required: true, type: 'text' as const, minLength: 2 },
    { field: 'email', required: true, type: 'email' as const },
    { field: 'password', required: true, type: 'text' as const, minLength: 8 },
    { 
      field: 'confirmPassword',
      required: true,
      customValidator: (value: string, formData: Record<string, unknown>) => {
        if (value !== formData.password) {
          return 'Passwords do not match';
        }
        return null;
      }
    },
    { field: 'termsAccepted', required: true }
  ],
  
  step2: [
    { field: 'phone', required: true, type: 'phone' as const },
    { field: 'dateOfBirth', required: true, type: 'date' as const },
    { field: 'location', required: true, type: 'text' as const, minLength: 2 },
    { field: 'currentWeight', required: true, type: 'number' as const },
    { field: 'height', required: true, type: 'text' as const }
  ],
  
  step3: [
    { field: 'activityLevel', required: true },
    { field: 'sleepHours', required: true }
  ],
  
  step4: [
    { field: 'specificGoals', required: true, type: 'text' as const, minLength: 10 },
    { field: 'fitnessLevel', required: true },
    { field: 'trainingFrequency', required: true },
    { field: 'availableTime', required: true }
  ],
  
  step5: [
    { field: 'currentDiet', required: true },
    { field: 'mealsPerDay', required: true },
    { field: 'cookingExperience', required: true },
    { field: 'mealPrepTime', required: true }
  ],
  
  step6: [
    { field: 'communicationStyle', required: true },
    { field: 'motivationStyle', required: true }
  ]
};