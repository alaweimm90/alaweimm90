import { DomainSMEValidator, ValidationRule, ValidationResult } from './interfaces';

export class ConfigurableSMEValidator implements DomainSMEValidator {
  private domain: string;
  private rules: ValidationRule[];

  constructor(domain: string, rules: ValidationRule[]) {
    this.domain = domain;
    this.rules = rules;
  }

  validate(data: Record<string, unknown>): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    for (const rule of this.rules) {
      try {
        const isValid = this.evaluateCondition(rule.condition, data);

        if (!isValid) {
          if (rule.severity === 'error') {
            errors.push(rule.message);
          } else if (rule.severity === 'warning') {
            warnings.push(rule.message);
          }
        }
      } catch (error) {
        errors.push(`Rule evaluation failed for ${rule.id}: ${error}`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings,
      domain: this.domain
    };
  }

  private evaluateCondition(condition: string, data: Record<string, unknown>): boolean {
    // Simple condition evaluator - in production, use a proper expression parser
    const context = { ...data };

    try {
      // Basic condition evaluation (simplified for demo)
      if (condition.includes('> 0') && typeof context.amount === 'number') {
        return context.amount > 0;
      }
      if (condition.includes('in [') && Array.isArray(context.currency)) {
        const allowedCurrencies = ['USD', 'EUR', 'GBP'];
        return allowedCurrencies.includes(context.currency as string);
      }
      if (condition.includes('length > 0') && Array.isArray(context.args)) {
        return context.args.length > 0;
      }
      return true; // Default pass for unrecognized conditions
    } catch {
      return false;
    }
  }

  addRule(rule: ValidationRule): void {
    this.rules.push(rule);
  }

  removeRule(ruleId: string): void {
    this.rules = this.rules.filter(rule => rule.id !== ruleId);
  }

  getRules(): ValidationRule[] {
    return [...this.rules];
  }
}