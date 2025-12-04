/**
 * ATLAS → KILO Integration Bridge (A2K Bridge)
 *
 * Enables ATLAS operations to leverage KILO validation and governance.
 * Routes ATLAS refactoring operations through KILO validation, provides
 * access to KILO DevOps templates, and validates ATLAS-generated code
 * against KILO policies.
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
// Bridge handles dynamic configuration and cross-system data structures

import { readFileSync, existsSync } from 'fs';
import { join, resolve } from 'path';

// ============================================================================
// INTERFACES & TYPES
// ============================================================================

export interface RefactoringOperation {
  id: string;
  type:
    | 'extract_function'
    | 'rename_variable'
    | 'simplify_conditional'
    | 'remove_duplication'
    | 'reduce_complexity'
    | 'improve_naming'
    | 'add_type_hints'
    | 'extract_constant';
  filePath: string;
  changes: CodeChange[];
  metadata: {
    riskLevel: 'low' | 'medium' | 'high';
    estimatedImpact: number;
    requiresValidation: boolean;
  };
}

export interface CodeChange {
  type: 'add' | 'remove' | 'modify';
  startLine: number;
  endLine: number;
  content: string;
  originalContent?: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
  rollbackRequired: boolean;
  validationMetadata: {
    validatorVersion: string;
    validationTime: number;
    checksPerformed: string[];
  };
}

export interface TemplateRequest {
  category: 'cicd' | 'db' | 'iac' | 'k8s' | 'logging' | 'monitoring' | 'ui';
  name: string;
  version?: string;
  parameters: Record<string, any>;
}

export interface TemplateResponse {
  templateId: string;
  files: TemplateFile[];
  metadata: {
    category: string;
    name: string;
    version: string;
    description: string;
    dependencies: string[];
  };
}

export interface TemplateFile {
  path: string;
  content: string;
  placeholders: string[];
}

export interface ComplianceCheck {
  code: string;
  language: string;
  context: {
    repository: string;
    filePath: string;
    framework?: string;
  };
  policies: string[];
}

export interface ComplianceResult {
  isCompliant: boolean;
  violations: ComplianceViolation[];
  recommendations: string[];
  metadata: {
    checkedPolicies: string[];
    complianceScore: number;
    checkTime: number;
  };
}

export interface ComplianceViolation {
  policy: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  location?: {
    line: number;
    column: number;
  };
  suggestion: string;
}

export interface A2KBridgeConfig {
  validation: {
    strictness: 'lenient' | 'standard' | 'strict';
    timeoutMs: number;
    enableRollback: boolean;
  };
  templates: {
    cacheEnabled: boolean;
    cacheTtlMs: number;
    basePath: string;
  };
  compliance: {
    enabledPolicies: string[];
    failOnWarning: boolean;
  };
}

// ============================================================================
// VALIDATION PROXY
// ============================================================================

export class ValidationProxy {
  private config: A2KBridgeConfig['validation'];

  constructor(config: A2KBridgeConfig['validation']) {
    this.config = config;
  }

  async validateRefactoring(operation: RefactoringOperation): Promise<ValidationResult> {
    const startTime = Date.now();

    try {
      // Synchronous validation for immediate feedback
      const syntaxCheck = this.validateSyntax(operation);
      const typeCheck = await this.validateTypes(operation);
      const safetyCheck = this.validateSafety(operation);

      const allErrors = [...syntaxCheck.errors, ...typeCheck.errors, ...safetyCheck.errors];
      const allWarnings = [...syntaxCheck.warnings, ...typeCheck.warnings, ...safetyCheck.warnings];

      const isValid = this.determineValidity(allErrors, allWarnings);

      return {
        isValid,
        errors: allErrors,
        warnings: allWarnings,
        rollbackRequired: !isValid && this.config.enableRollback,
        validationMetadata: {
          validatorVersion: '1.0.0',
          validationTime: Date.now() - startTime,
          checksPerformed: ['syntax', 'types', 'safety'],
        },
      };
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      return {
        isValid: false,
        errors: [`Validation failed: ${errorMessage}`],
        warnings: [],
        rollbackRequired: this.config.enableRollback,
        validationMetadata: {
          validatorVersion: '1.0.0',
          validationTime: Date.now() - startTime,
          checksPerformed: ['error_handling'],
        },
      };
    }
  }

  private validateSyntax(operation: RefactoringOperation): {
    errors: string[];
    warnings: string[];
  } {
    // Basic syntax validation - in real implementation would use AST parsing
    const errors: string[] = [];
    const warnings: string[] = [];

    for (const change of operation.changes) {
      if (change.type === 'add' || change.type === 'modify') {
        // Check for basic syntax issues
        if (change.content.includes('undefined') && !change.content.includes('typeof')) {
          warnings.push(`Potential undefined usage in change at line ${change.startLine}`);
        }
      }
    }

    return { errors, warnings };
  }

  private async validateTypes(
    operation: RefactoringOperation
  ): Promise<{ errors: string[]; warnings: string[] }> {
    // Asynchronous type checking - would integrate with TypeScript compiler or similar
    const errors: string[] = [];
    const warnings: string[] = [];

    // Simulate async validation
    await new Promise((resolve) => setTimeout(resolve, 10));

    if (operation.type === 'add_type_hints') {
      // Type hint validation would be more thorough
      warnings.push('Type hints validation completed');
    }

    return { errors, warnings };
  }

  private validateSafety(operation: RefactoringOperation): {
    errors: string[];
    warnings: string[];
  } {
    const errors: string[] = [];
    const warnings: string[] = [];

    // Risk-based safety checks
    if (operation.metadata.riskLevel === 'high' && this.config.strictness === 'strict') {
      errors.push('High-risk refactoring blocked in strict mode');
    }

    // Check for breaking changes
    if (
      operation.changes.some(
        (change) => change.type === 'remove' && change.content.includes('export')
      )
    ) {
      warnings.push('Potential breaking change: removing exported member');
    }

    return { errors, warnings };
  }

  private determineValidity(errors: string[], warnings: string[]): boolean {
    switch (this.config.strictness) {
      case 'strict':
        return errors.length === 0;
      case 'standard':
        return errors.length === 0 && warnings.length <= 2;
      case 'lenient':
        return errors.length === 0;
      default:
        return errors.length === 0;
    }
  }
}

// ============================================================================
// TEMPLATE ACCESSOR
// ============================================================================

export class TemplateAccessor {
  private config: A2KBridgeConfig['templates'];
  private cache: Map<string, { data: TemplateResponse; timestamp: number }> = new Map();

  constructor(config: A2KBridgeConfig['templates']) {
    this.config = config;
  }

  async getTemplates(request: TemplateRequest): Promise<TemplateResponse> {
    const cacheKey = `${request.category}/${request.name}/${request.version || 'latest'}`;

    // Check cache first
    if (this.config.cacheEnabled) {
      const cached = this.cache.get(cacheKey);
      if (cached && Date.now() - cached.timestamp < this.config.cacheTtlMs) {
        return this.applyParameters(cached.data, request.parameters);
      }
    }

    // Load template from filesystem
    const template = await this.loadTemplate(request);

    // Cache the result
    if (this.config.cacheEnabled) {
      this.cache.set(cacheKey, { data: template, timestamp: Date.now() });
    }

    return this.applyParameters(template, request.parameters);
  }

  private async loadTemplate(request: TemplateRequest): Promise<TemplateResponse> {
    const templatePath = join(this.config.basePath, request.category, request.name);
    const templateJsonPath = join(templatePath, 'template.json');

    if (!existsSync(templateJsonPath)) {
      throw new Error(`Template not found: ${request.category}/${request.name}`);
    }

    const templateJson = JSON.parse(readFileSync(templateJsonPath, 'utf-8'));
    const files: TemplateFile[] = [];

    // Load all required files
    for (const fileName of templateJson.requiredFiles) {
      const filePath = join(templatePath, fileName);
      if (existsSync(filePath)) {
        const content = readFileSync(filePath, 'utf-8');
        files.push({
          path: fileName,
          content,
          placeholders: this.extractPlaceholders(content),
        });
      }
    }

    return {
      templateId: `${request.category}/${request.name}/${templateJson.version}`,
      files,
      metadata: {
        category: request.category,
        name: request.name,
        version: templateJson.version,
        description: templateJson.description,
        dependencies: templateJson.dependencies || [],
      },
    };
  }

  private applyParameters(
    template: TemplateResponse,
    parameters: Record<string, any>
  ): TemplateResponse {
    const processedFiles = template.files.map((file) => ({
      ...file,
      content: this.replacePlaceholders(file.content, parameters),
    }));

    return {
      ...template,
      files: processedFiles,
    };
  }

  private extractPlaceholders(content: string): string[] {
    const placeholderRegex = /\{\{([^}]+)\}\}/g;
    const placeholders: string[] = [];
    let match;

    while ((match = placeholderRegex.exec(content)) !== null) {
      placeholders.push(match[1]);
    }

    return [...new Set(placeholders)]; // Remove duplicates
  }

  private replacePlaceholders(content: string, parameters: Record<string, any>): string {
    let result = content;
    for (const [key, value] of Object.entries(parameters)) {
      const regex = new RegExp(`\\{\\{${key}\\}\\}`, 'g');
      result = result.replace(regex, String(value));
    }
    return result;
  }
}

// ============================================================================
// COMPLIANCE CHECKER
// ============================================================================

export class ComplianceChecker {
  private config: A2KBridgeConfig['compliance'];

  constructor(config: A2KBridgeConfig['compliance']) {
    this.config = config;
  }

  async checkCompliance(check: ComplianceCheck): Promise<ComplianceResult> {
    const startTime = Date.now();
    const violations: ComplianceViolation[] = [];
    const recommendations: string[] = [];

    // Run enabled policy checks
    for (const policy of this.config.enabledPolicies) {
      const policyViolations = await this.checkPolicy(check, policy);
      violations.push(...policyViolations);
    }

    // Generate recommendations based on violations
    recommendations.push(...this.generateRecommendations(violations));

    const complianceScore = this.calculateComplianceScore(violations);

    return {
      isCompliant:
        violations.filter((v) => v.severity === 'critical' || v.severity === 'high').length === 0,
      violations,
      recommendations,
      metadata: {
        checkedPolicies: this.config.enabledPolicies,
        complianceScore,
        checkTime: Date.now() - startTime,
      },
    };
  }

  private async checkPolicy(
    check: ComplianceCheck,
    policy: string
  ): Promise<ComplianceViolation[]> {
    const violations: ComplianceViolation[] = [];

    switch (policy) {
      case 'security':
        violations.push(...this.checkSecurityPolicy(check));
        break;
      case 'code_quality':
        violations.push(...this.checkCodeQualityPolicy(check));
        break;
      case 'performance':
        violations.push(...this.checkPerformancePolicy(check));
        break;
      case 'maintainability':
        violations.push(...this.checkMaintainabilityPolicy(check));
        break;
    }

    return violations;
  }

  private checkSecurityPolicy(check: ComplianceCheck): ComplianceViolation[] {
    const violations: ComplianceViolation[] = [];

    // Check for hardcoded secrets
    if (
      check.code.includes('password') ||
      check.code.includes('secret') ||
      check.code.includes('token')
    ) {
      violations.push({
        policy: 'security',
        severity: 'high',
        message: 'Potential hardcoded secret detected',
        suggestion: 'Use environment variables or secure credential storage',
      });
    }

    // Check for SQL injection vulnerabilities
    if (check.code.includes('SELECT') && check.code.includes('+')) {
      violations.push({
        policy: 'security',
        severity: 'critical',
        message: 'Potential SQL injection vulnerability',
        suggestion: 'Use parameterized queries or prepared statements',
      });
    }

    return violations;
  }

  private checkCodeQualityPolicy(check: ComplianceCheck): ComplianceViolation[] {
    const violations: ComplianceViolation[] = [];

    // Check for TODO comments
    if (check.code.includes('TODO') || check.code.includes('FIXME')) {
      violations.push({
        policy: 'code_quality',
        severity: 'low',
        message: 'TODO/FIXME comment found',
        suggestion: 'Address technical debt or remove comment',
      });
    }

    // Check for console.log in production code
    if (check.code.includes('console.log') && !check.context.filePath.includes('test')) {
      violations.push({
        policy: 'code_quality',
        severity: 'medium',
        message: 'Console.log found in non-test code',
        suggestion: 'Use proper logging framework',
      });
    }

    return violations;
  }

  private checkPerformancePolicy(check: ComplianceCheck): ComplianceViolation[] {
    const violations: ComplianceViolation[] = [];

    // Check for inefficient loops
    const loopMatches = check.code.match(/for\s*\([^)]*\)\s*\{[^}]*for\s*\(/g);
    if (loopMatches && loopMatches.length > 0) {
      violations.push({
        policy: 'performance',
        severity: 'medium',
        message: 'Nested loops detected',
        suggestion: 'Consider optimizing nested loop performance',
      });
    }

    return violations;
  }

  private checkMaintainabilityPolicy(check: ComplianceCheck): ComplianceViolation[] {
    const violations: ComplianceViolation[] = [];

    // Check function length
    const functions = check.code.split(/function\s+\w+\s*\(|const\s+\w+\s*=\s*\(/);
    for (const func of functions) {
      const lines = func.split('\n').length;
      if (lines > 50) {
        violations.push({
          policy: 'maintainability',
          severity: 'medium',
          message: 'Function exceeds recommended length',
          suggestion: 'Consider breaking down into smaller functions',
        });
        break; // Only report once per file
      }
    }

    return violations;
  }

  private generateRecommendations(violations: ComplianceViolation[]): string[] {
    const recommendations: string[] = [];

    const policyGroups = violations.reduce(
      (acc, v) => {
        if (!acc[v.policy]) acc[v.policy] = [];
        acc[v.policy].push(v);
        return acc;
      },
      {} as Record<string, ComplianceViolation[]>
    );

    for (const [policy, policyViolations] of Object.entries(policyGroups)) {
      const highSeverity = policyViolations.filter(
        (v) => v.severity === 'high' || v.severity === 'critical'
      );
      if (highSeverity.length > 0) {
        recommendations.push(`Address ${highSeverity.length} high-severity ${policy} violations`);
      }
    }

    return recommendations;
  }

  private calculateComplianceScore(violations: ComplianceViolation[]): number {
    const weights = {
      critical: 10,
      high: 5,
      medium: 2,
      low: 1,
    };

    const totalPenalty = violations.reduce((sum, v) => sum + weights[v.severity], 0);
    return Math.max(0, 100 - totalPenalty);
  }
}

// ============================================================================
// A2K BRIDGE MAIN CLASS
// ============================================================================

export interface A2KBridge {
  validateRefactoring(operation: RefactoringOperation): Promise<ValidationResult>;
  getTemplates(request: TemplateRequest): Promise<TemplateResponse>;
  checkCompliance(check: ComplianceCheck): Promise<ComplianceResult>;
}

export class AtlasKiloBridge implements A2KBridge {
  private validationProxy: ValidationProxy;
  private templateAccessor: TemplateAccessor;
  private complianceChecker: ComplianceChecker;
  private config: A2KBridgeConfig;

  constructor(config: Partial<A2KBridgeConfig> = {}) {
    this.config = {
      validation: {
        strictness: 'standard',
        timeoutMs: 30000,
        enableRollback: true,
        ...config.validation,
      },
      templates: {
        cacheEnabled: true,
        cacheTtlMs: 3600000, // 1 hour
        basePath: resolve(__dirname, '../../../infrastructure/templates/devops'),
        ...config.templates,
      },
      compliance: {
        enabledPolicies: ['security', 'code_quality', 'performance', 'maintainability'],
        failOnWarning: false,
        ...config.compliance,
      },
    };

    this.validationProxy = new ValidationProxy(this.config.validation);
    this.templateAccessor = new TemplateAccessor(this.config.templates);
    this.complianceChecker = new ComplianceChecker(this.config.compliance);
  }

  async validateRefactoring(operation: RefactoringOperation): Promise<ValidationResult> {
    return this.validationProxy.validateRefactoring(operation);
  }

  async getTemplates(request: TemplateRequest): Promise<TemplateResponse> {
    return this.templateAccessor.getTemplates(request);
  }

  async checkCompliance(check: ComplianceCheck): Promise<ComplianceResult> {
    return this.complianceChecker.checkCompliance(check);
  }

  // Bridge status and management
  getBridgeStatus(): any {
    return {
      isActive: true,
      version: '1.0.0',
      config: this.config,
      components: {
        validation: 'active',
        templates: 'active',
        compliance: 'active',
      },
    };
  }

  updateConfig(newConfig: Partial<A2KBridgeConfig>): void {
    this.config = { ...this.config, ...newConfig };
    // Reinitialize components with new config
    this.validationProxy = new ValidationProxy(this.config.validation);
    this.templateAccessor = new TemplateAccessor(this.config.templates);
    this.complianceChecker = new ComplianceChecker(this.config.compliance);
  }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

export class A2KBridgeError extends Error {
  constructor(
    message: string,
    public code: string,
    public component: 'validation' | 'templates' | 'compliance' | 'bridge',
    public details?: any
  ) {
    super(message);
    this.name = 'A2KBridgeError';
  }
}

export class ValidationError extends A2KBridgeError {
  constructor(message: string, details?: any) {
    super(message, 'VALIDATION_FAILED', 'validation', details);
  }
}

export class TemplateError extends A2KBridgeError {
  constructor(message: string, details?: any) {
    super(message, 'TEMPLATE_ERROR', 'templates', details);
  }
}

export class ComplianceError extends A2KBridgeError {
  constructor(message: string, details?: any) {
    super(message, 'COMPLIANCE_ERROR', 'compliance', details);
  }
}

// ============================================================================
// DEFAULT EXPORT
// ============================================================================

export default AtlasKiloBridge;
