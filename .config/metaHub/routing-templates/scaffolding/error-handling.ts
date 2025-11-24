/**
 * TRAE Error Handling Patterns Template
 * Comprehensive Error Management for Production-Hardened Routing Systems
 *
 * This template provides complete error handling patterns that can be adapted
 * to any programming language or framework, ensuring robust error management,
 * proper logging, recovery strategies, and compliance with error reporting
 * requirements.
 *
 * Features:
 * - Hierarchical error classification
 * - Framework-agnostic error handling
 * - Recovery strategies and circuit breakers
 * - Compliance-aware error reporting
 * - Production monitoring and alerting
 */

// ============================================================================
// ERROR CLASSIFICATION SYSTEM
// ============================================================================

export enum ErrorCategory {
  NETWORK = 'network',
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  RATE_LIMITING = 'rate_limiting',
  QUOTA_EXCEEDED = 'quota_exceeded',
  MODEL_UNAVAILABLE = 'model_unavailable',
  INVALID_REQUEST = 'invalid_request',
  TIMEOUT = 'timeout',
  COST_EXCEEDED = 'cost_exceeded',
  COMPLIANCE_VIOLATION = 'compliance_violation',
  SYSTEM_ERROR = 'system_error',
  UNKNOWN = 'unknown',
}

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export enum ErrorSource {
  CLIENT = 'client',
  NETWORK = 'network',
  PROVIDER = 'provider',
  SYSTEM = 'system',
  CONFIGURATION = 'configuration',
}

// ============================================================================
// BASE ERROR INTERFACES
// ============================================================================

export interface RoutingError {
  id: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  source: ErrorSource;
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
  requestId?: string;
  sessionId?: string;
  userId?: string;
  model?: string;
  provider?: string;
  region?: string;
  recoverable: boolean;
  retryAfter?: number;
  suggestedActions?: string[];
  complianceImpact?: ComplianceImpact;
  metadata?: ErrorMetadata;
}

export interface ComplianceImpact {
  framework: string;
  violation: boolean;
  dataClassification: string;
  auditRequired: boolean;
  notificationRequired: boolean;
}

export interface ErrorMetadata {
  component: string;
  version: string;
  environment: string;
  stackTrace?: string;
  userAgent?: string;
  ipAddress?: string;
  requestHeaders?: Record<string, string>;
  responseHeaders?: Record<string, string>;
  latency?: number;
  attempts?: number;
}

// ============================================================================
// ERROR HANDLING STRATEGIES
// ============================================================================

export interface ErrorHandlingStrategy {
  category: ErrorCategory;
  severity: ErrorSeverity;
  maxRetries: number;
  backoffStrategy: BackoffStrategy;
  circuitBreaker?: CircuitBreakerConfig;
  fallbackActions: FallbackAction[];
  notificationRequired: boolean;
  complianceReporting: boolean;
}

export interface BackoffStrategy {
  type: 'fixed' | 'linear' | 'exponential' | 'fibonacci';
  initialDelay: number;
  maxDelay: number;
  multiplier?: number;
  jitter?: boolean;
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  recoveryTimeout: number;
  monitoringWindow: number;
  halfOpenMaxRequests?: number;
}

export interface FallbackAction {
  type:
    | 'retry'
    | 'switch_model'
    | 'switch_provider'
    | 'switch_region'
    | 'degrade_service'
    | 'return_error';
  priority: number;
  conditions?: FallbackCondition[];
  config?: any;
}

export interface FallbackCondition {
  type: 'error_count' | 'time_window' | 'resource_available' | 'compliance_check';
  operator: 'gt' | 'gte' | 'lt' | 'lte' | 'eq' | 'contains';
  value: any;
}

// ============================================================================
// ERROR HANDLER IMPLEMENTATION
// ============================================================================

export class ErrorHandler {
  private strategies: Map<string, ErrorHandlingStrategy> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private errorHistory: RoutingError[] = new Map();
  private maxHistorySize: number;
  private notificationHandler?: NotificationHandler;
  private complianceReporter?: ComplianceReporter;

  constructor(config?: {
    maxHistorySize?: number;
    notificationHandler?: NotificationHandler;
    complianceReporter?: ComplianceReporter;
  }) {
    this.maxHistorySize = config?.maxHistorySize || 10000;
    this.notificationHandler = config?.notificationHandler;
    this.complianceReporter = config?.complianceReporter;

    this.initializeDefaultStrategies();
  }

  /**
   * Handle an error with appropriate strategy
   */
  async handleError(error: RoutingError): Promise<ErrorHandlingResult> {
    // Record error in history
    this.recordError(error);

    // Get handling strategy
    const strategy = this.getStrategy(error);

    // Check circuit breaker
    if (strategy.circuitBreaker) {
      const circuitBreaker = this.getCircuitBreaker(error.category);
      if (circuitBreaker.isOpen()) {
        return {
          action: 'circuit_open',
          retryAfter: circuitBreaker.getRetryAfter(),
          error: this.createCircuitOpenError(error),
        };
      }
    }

    // Execute fallback actions
    for (const fallbackAction of strategy.fallbackActions) {
      if (await this.shouldExecuteFallback(fallbackAction, error)) {
        try {
          const result = await this.executeFallbackAction(fallbackAction, error);
          return result;
        } catch (fallbackError) {
          // Continue to next fallback action
          continue;
        }
      }
    }

    // No fallback succeeded, return final error
    return {
      action: 'return_error',
      error: this.enrichError(error),
    };
  }

  /**
   * Create a standardized routing error
   */
  createError(params: {
    category: ErrorCategory;
    severity: ErrorSeverity;
    source: ErrorSource;
    code: string;
    message: string;
    details?: any;
    requestId?: string;
    sessionId?: string;
    userId?: string;
    model?: string;
    provider?: string;
    region?: string;
    recoverable?: boolean;
    retryAfter?: number;
    suggestedActions?: string[];
    complianceImpact?: ComplianceImpact;
    originalError?: Error;
  }): RoutingError {
    const error: RoutingError = {
      id: this.generateErrorId(),
      timestamp: new Date(),
      recoverable: params.recoverable ?? this.isRecoverable(params.category),
      ...params,
      metadata: {
        component: 'routing_system',
        version: '1.0.0',
        environment: process.env.NODE_ENV || 'development',
        stackTrace: params.originalError?.stack,
      },
    };

    return error;
  }

  /**
   * Get error statistics
   */
  getErrorStats(timeRange?: { start: Date; end: Date }): ErrorStats {
    const errors = timeRange
      ? Array.from(this.errorHistory.values()).filter(
          e => e.timestamp >= timeRange.start && e.timestamp <= timeRange.end
        )
      : Array.from(this.errorHistory.values());

    const stats: ErrorStats = {
      totalErrors: errors.length,
      errorsByCategory: this.groupBy(errors, 'category'),
      errorsBySeverity: this.groupBy(errors, 'severity'),
      errorsBySource: this.groupBy(errors, 'source'),
      topErrors: this.getTopErrors(errors),
      errorRate: this.calculateErrorRate(errors),
      recoveryRate: this.calculateRecoveryRate(errors),
      averageResolutionTime: this.calculateAverageResolutionTime(errors),
    };

    return stats;
  }

  /**
   * Add custom error handling strategy
   */
  addStrategy(key: string, strategy: ErrorHandlingStrategy): void {
    this.strategies.set(key, strategy);
  }

  /**
   * Get error handling strategy for an error
   */
  getStrategy(error: RoutingError): ErrorHandlingStrategy {
    // Try specific strategy first
    const specificKey = `${error.category}_${error.severity}`;
    if (this.strategies.has(specificKey)) {
      return this.strategies.get(specificKey)!;
    }

    // Try category strategy
    if (this.strategies.has(error.category)) {
      return this.strategies.get(error.category)!;
    }

    // Return default strategy
    return this.getDefaultStrategy(error);
  }

  // Private methods

  private initializeDefaultStrategies(): void {
    // Network errors
    this.strategies.set(ErrorCategory.NETWORK, {
      category: ErrorCategory.NETWORK,
      severity: ErrorSeverity.MEDIUM,
      maxRetries: 3,
      backoffStrategy: {
        type: 'exponential',
        initialDelay: 1000,
        maxDelay: 30000,
        multiplier: 2,
        jitter: true,
      },
      circuitBreaker: {
        failureThreshold: 5,
        recoveryTimeout: 60000,
        monitoringWindow: 300000, // 5 minutes
      },
      fallbackActions: [
        {
          type: 'retry',
          priority: 1,
          conditions: [{ type: 'error_count', operator: 'lt', value: 3 }],
        },
        {
          type: 'switch_region',
          priority: 2,
          conditions: [{ type: 'error_count', operator: 'gte', value: 3 }],
        },
      ],
      notificationRequired: false,
      complianceReporting: false,
    });

    // Authentication errors
    this.strategies.set(ErrorCategory.AUTHENTICATION, {
      category: ErrorCategory.AUTHENTICATION,
      severity: ErrorSeverity.HIGH,
      maxRetries: 0,
      backoffStrategy: {
        type: 'fixed',
        initialDelay: 0,
        maxDelay: 0,
      },
      fallbackActions: [
        {
          type: 'return_error',
          priority: 1,
        },
      ],
      notificationRequired: true,
      complianceReporting: true,
    });

    // Rate limiting errors
    this.strategies.set(ErrorCategory.RATE_LIMITING, {
      category: ErrorCategory.RATE_LIMITING,
      severity: ErrorSeverity.MEDIUM,
      maxRetries: 2,
      backoffStrategy: {
        type: 'exponential',
        initialDelay: 2000,
        maxDelay: 60000,
        multiplier: 2,
      },
      fallbackActions: [
        {
          type: 'retry',
          priority: 1,
        },
        {
          type: 'switch_provider',
          priority: 2,
        },
      ],
      notificationRequired: false,
      complianceReporting: false,
    });

    // Cost exceeded errors
    this.strategies.set(ErrorCategory.COST_EXCEEDED, {
      category: ErrorCategory.COST_EXCEEDED,
      severity: ErrorSeverity.HIGH,
      maxRetries: 0,
      backoffStrategy: {
        type: 'fixed',
        initialDelay: 0,
        maxDelay: 0,
      },
      fallbackActions: [
        {
          type: 'switch_model',
          priority: 1,
        },
        {
          type: 'return_error',
          priority: 2,
        },
      ],
      notificationRequired: true,
      complianceReporting: true,
    });

    // Compliance violations
    this.strategies.set(ErrorCategory.COMPLIANCE_VIOLATION, {
      category: ErrorCategory.COMPLIANCE_VIOLATION,
      severity: ErrorSeverity.CRITICAL,
      maxRetries: 0,
      backoffStrategy: {
        type: 'fixed',
        initialDelay: 0,
        maxDelay: 0,
      },
      fallbackActions: [
        {
          type: 'return_error',
          priority: 1,
        },
      ],
      notificationRequired: true,
      complianceReporting: true,
    });
  }

  private getDefaultStrategy(error: RoutingError): ErrorHandlingStrategy {
    return {
      category: error.category,
      severity: error.severity,
      maxRetries: error.recoverable ? 1 : 0,
      backoffStrategy: {
        type: 'exponential',
        initialDelay: 1000,
        maxDelay: 10000,
        multiplier: 2,
      },
      fallbackActions: [
        {
          type: error.recoverable ? 'retry' : 'return_error',
          priority: 1,
        },
      ],
      notificationRequired: error.severity === ErrorSeverity.CRITICAL,
      complianceReporting: false,
    };
  }

  private async shouldExecuteFallback(
    action: FallbackAction,
    error: RoutingError
  ): Promise<boolean> {
    if (!action.conditions) return true;

    for (const condition of action.conditions) {
      if (!(await this.evaluateCondition(condition, error))) {
        return false;
      }
    }

    return true;
  }

  private async evaluateCondition(
    condition: FallbackCondition,
    error: RoutingError
  ): Promise<boolean> {
    switch (condition.type) {
      case 'error_count':
        const errorCount = this.getErrorCount(error.category, 300000); // Last 5 minutes
        return this.compareValues(errorCount, condition.operator, condition.value);

      case 'time_window':
        const timeSinceError = Date.now() - error.timestamp.getTime();
        return this.compareValues(timeSinceError, condition.operator, condition.value);

      case 'resource_available':
        // Check if alternative resources are available
        return true; // Simplified

      case 'compliance_check':
        // Check compliance requirements
        return !error.complianceImpact?.violation;

      default:
        return false;
    }
  }

  private compareValues(actual: any, operator: string, expected: any): boolean {
    switch (operator) {
      case 'gt':
        return actual > expected;
      case 'gte':
        return actual >= expected;
      case 'lt':
        return actual < expected;
      case 'lte':
        return actual <= expected;
      case 'eq':
        return actual === expected;
      case 'contains':
        return String(actual).includes(String(expected));
      default:
        return false;
    }
  }

  private async executeFallbackAction(
    action: FallbackAction,
    error: RoutingError
  ): Promise<ErrorHandlingResult> {
    switch (action.type) {
      case 'retry':
        return {
          action: 'retry',
          retryAfter: this.calculateBackoff(error),
        };

      case 'switch_model':
        return {
          action: 'switch_model',
          alternativeModel: this.findAlternativeModel(error),
        };

      case 'switch_provider':
        return {
          action: 'switch_provider',
          alternativeProvider: this.findAlternativeProvider(error),
        };

      case 'switch_region':
        return {
          action: 'switch_region',
          alternativeRegion: this.findAlternativeRegion(error),
        };

      case 'degrade_service':
        return {
          action: 'degrade_service',
          degraded: true,
        };

      case 'return_error':
      default:
        return {
          action: 'return_error',
          error: this.enrichError(error),
        };
    }
  }

  private calculateBackoff(error: RoutingError): number {
    const strategy = this.getStrategy(error);
    const attempt = error.metadata?.attempts || 0;

    switch (strategy.backoffStrategy.type) {
      case 'fixed':
        return strategy.backoffStrategy.initialDelay;

      case 'linear':
        return strategy.backoffStrategy.initialDelay * (attempt + 1);

      case 'exponential':
        const delay =
          strategy.backoffStrategy.initialDelay *
          Math.pow(strategy.backoffStrategy.multiplier || 2, attempt);
        const jitter = strategy.backoffStrategy.jitter ? Math.random() * 0.1 * delay : 0;
        return Math.min(delay + jitter, strategy.backoffStrategy.maxDelay);

      case 'fibonacci':
        const fib = this.fibonacci(attempt + 1);
        return Math.min(
          fib * strategy.backoffStrategy.initialDelay,
          strategy.backoffStrategy.maxDelay
        );

      default:
        return strategy.backoffStrategy.initialDelay;
    }
  }

  private fibonacci(n: number): number {
    if (n <= 1) return 1;
    return this.fibonacci(n - 1) + this.fibonacci(n - 2);
  }

  private findAlternativeModel(error: RoutingError): string {
    // Simplified model selection logic
    const alternatives: Record<string, string[]> = {
      'gpt-4': ['claude-3-opus', 'gpt-3.5-turbo'],
      'claude-3-opus': ['gpt-4', 'claude-3-sonnet'],
      'gpt-3.5-turbo': ['claude-3-sonnet', 'local-llm'],
    };

    return alternatives[error.model || '']?.[0] || 'local-llm';
  }

  private findAlternativeProvider(error: RoutingError): string {
    const alternatives: Record<string, string[]> = {
      openai: ['anthropic', 'gemini'],
      anthropic: ['openai', 'gemini'],
      gemini: ['openai', 'anthropic'],
    };

    return alternatives[error.provider || '']?.[0] || 'openrouter';
  }

  private findAlternativeRegion(error: RoutingError): string {
    const alternatives: Record<string, string[]> = {
      north_america: ['europe', 'asia_pacific'],
      europe: ['north_america', 'asia_pacific'],
      asia_pacific: ['north_america', 'europe'],
    };

    return alternatives[error.region || '']?.[0] || 'global';
  }

  private createCircuitOpenError(originalError: RoutingError): RoutingError {
    return this.createError({
      category: ErrorCategory.SYSTEM_ERROR,
      severity: ErrorSeverity.HIGH,
      source: ErrorSource.SYSTEM,
      code: 'CIRCUIT_BREAKER_OPEN',
      message: 'Circuit breaker is open due to repeated failures',
      details: { originalError: originalError.message },
      recoverable: true,
      retryAfter: 60000,
      suggestedActions: ['Wait for circuit breaker to recover', 'Check system health'],
    });
  }

  private enrichError(error: RoutingError): RoutingError {
    // Add additional context and suggestions
    const enriched = { ...error };

    if (!enriched.suggestedActions) {
      enriched.suggestedActions = this.generateSuggestedActions(error);
    }

    return enriched;
  }

  private generateSuggestedActions(error: RoutingError): string[] {
    const actions: string[] = [];

    switch (error.category) {
      case ErrorCategory.NETWORK:
        actions.push('Check network connectivity');
        actions.push('Verify API endpoints are accessible');
        break;

      case ErrorCategory.AUTHENTICATION:
        actions.push('Verify API keys are valid');
        actions.push('Check token expiration');
        break;

      case ErrorCategory.RATE_LIMITING:
        actions.push('Implement request throttling');
        actions.push('Consider upgrading API plan');
        break;

      case ErrorCategory.COST_EXCEEDED:
        actions.push('Review cost optimization settings');
        actions.push('Increase budget limits or reduce usage');
        break;

      case ErrorCategory.COMPLIANCE_VIOLATION:
        actions.push('Review compliance requirements');
        actions.push('Contact compliance officer');
        break;
    }

    return actions;
  }

  private recordError(error: RoutingError): void {
    this.errorHistory.set(error.id, error);

    // Maintain history size limit
    if (this.errorHistory.size > this.maxHistorySize) {
      // Remove oldest errors
      const entries = Array.from(this.errorHistory.entries());
      entries.sort(([, a], [, b]) => a.timestamp.getTime() - b.timestamp.getTime());

      const toRemove = entries.slice(0, this.errorHistory.size - this.maxHistorySize);
      toRemove.forEach(([id]) => this.errorHistory.delete(id));
    }

    // Update circuit breaker
    const strategy = this.getStrategy(error);
    if (strategy.circuitBreaker) {
      const circuitBreaker = this.getCircuitBreaker(error.category);
      circuitBreaker.recordFailure();
    }

    // Send notifications if required
    if (strategy.notificationRequired && this.notificationHandler) {
      this.notificationHandler.notify(error);
    }

    // Report compliance issues
    if (error.complianceImpact?.violation && this.complianceReporter) {
      this.complianceReporter.reportViolation(error);
    }
  }

  private getCircuitBreaker(category: ErrorCategory): CircuitBreaker {
    if (!this.circuitBreakers.has(category)) {
      const strategy = this.strategies.get(category);
      if (strategy?.circuitBreaker) {
        this.circuitBreakers.set(category, new CircuitBreaker(strategy.circuitBreaker));
      } else {
        // Default circuit breaker
        this.circuitBreakers.set(
          category,
          new CircuitBreaker({
            failureThreshold: 5,
            recoveryTimeout: 60000,
            monitoringWindow: 300000,
          })
        );
      }
    }

    return this.circuitBreakers.get(category)!;
  }

  private getErrorCount(category: ErrorCategory, timeWindow: number): number {
    const cutoff = new Date(Date.now() - timeWindow);
    return Array.from(this.errorHistory.values()).filter(
      error => error.category === category && error.timestamp > cutoff
    ).length;
  }

  private groupBy<T>(items: T[], key: keyof T): Record<string, number> {
    return items.reduce(
      (acc, item) => {
        const value = String(item[key]);
        acc[value] = (acc[value] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }

  private getTopErrors(
    errors: RoutingError[]
  ): Array<{ code: string; count: number; message: string }> {
    const grouped = this.groupBy(errors, 'code');
    return Object.entries(grouped)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([code, count]) => ({
        code,
        count,
        message: errors.find(e => e.code === code)?.message || 'Unknown error',
      }));
  }

  private calculateErrorRate(errors: RoutingError[]): number {
    if (errors.length === 0) return 0;

    const timeSpan = Date.now() - Math.min(...errors.map(e => e.timestamp.getTime()));
    const hours = timeSpan / (1000 * 60 * 60);

    return errors.length / Math.max(hours, 1);
  }

  private calculateRecoveryRate(errors: RoutingError[]): number {
    const recoverableErrors = errors.filter(e => e.recoverable);
    const recoveredErrors = recoverableErrors.filter(e => {
      // Check if error was followed by successful operation
      // This is a simplified check
      return true; // Assume recovery for now
    });

    return recoverableErrors.length > 0 ? recoveredErrors.length / recoverableErrors.length : 0;
  }

  private calculateAverageResolutionTime(errors: RoutingError[]): number {
    // Simplified calculation - would need actual resolution tracking
    return 300000; // 5 minutes average
  }

  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private isRecoverable(category: ErrorCategory): boolean {
    const nonRecoverable = [
      ErrorCategory.AUTHENTICATION,
      ErrorCategory.AUTHORIZATION,
      ErrorCategory.COMPLIANCE_VIOLATION,
      ErrorCategory.COST_EXCEEDED,
    ];

    return !nonRecoverable.includes(category);
  }
}

// ============================================================================
// SUPPORTING CLASSES
// ============================================================================

export class CircuitBreaker {
  private config: CircuitBreakerConfig;
  private state: 'closed' | 'open' | 'half_open' = 'closed';
  private failureCount = 0;
  private lastFailureTime = 0;
  private successCount = 0;

  constructor(config: CircuitBreakerConfig) {
    this.config = config;
  }

  isOpen(): boolean {
    this.updateState();
    return this.state === 'open';
  }

  getRetryAfter(): number {
    if (this.state !== 'open') return 0;

    const timeSinceLastFailure = Date.now() - this.lastFailureTime;
    return Math.max(0, this.config.recoveryTimeout - timeSinceLastFailure);
  }

  recordFailure(): void {
    this.failureCount++;
    this.lastFailureTime = Date.now();
    this.updateState();
  }

  recordSuccess(): void {
    if (this.state === 'half_open') {
      this.successCount++;
      if (this.successCount >= (this.config.halfOpenMaxRequests || 3)) {
        this.state = 'closed';
        this.failureCount = 0;
        this.successCount = 0;
      }
    } else if (this.state === 'closed') {
      this.failureCount = Math.max(0, this.failureCount - 1);
    }
  }

  private updateState(): void {
    const now = Date.now();
    const timeSinceLastFailure = now - this.lastFailureTime;

    if (this.state === 'open' && timeSinceLastFailure > this.config.recoveryTimeout) {
      this.state = 'half_open';
      this.successCount = 0;
    } else if (this.failureCount >= this.config.failureThreshold) {
      this.state = 'open';
    }
  }
}

export interface NotificationHandler {
  notify(error: RoutingError): Promise<void>;
}

export interface ComplianceReporter {
  reportViolation(error: RoutingError): Promise<void>;
}

export interface ErrorHandlingResult {
  action:
    | 'retry'
    | 'switch_model'
    | 'switch_provider'
    | 'switch_region'
    | 'degrade_service'
    | 'return_error'
    | 'circuit_open';
  retryAfter?: number;
  alternativeModel?: string;
  alternativeProvider?: string;
  alternativeRegion?: string;
  degraded?: boolean;
  error?: RoutingError;
}

export interface ErrorStats {
  totalErrors: number;
  errorsByCategory: Record<string, number>;
  errorsBySeverity: Record<string, number>;
  errorsBySource: Record<string, number>;
  topErrors: Array<{ code: string; count: number; message: string }>;
  errorRate: number;
  recoveryRate: number;
  averageResolutionTime: number;
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Error Handler Setup
 *
 * const errorHandler = new ErrorHandler({
 *   maxHistorySize: 10000,
 *   notificationHandler: myNotificationService,
 *   complianceReporter: myComplianceService
 * });
 */

/**
 * Example: Handling Routing Errors
 *
 * try {
 *   const result = await router.routeRequest(prompt, context);
 *   return result;
 * } catch (error) {
 *   const routingError = errorHandler.createError({
 *     category: ErrorCategory.NETWORK,
 *     severity: ErrorSeverity.MEDIUM,
 *     source: ErrorSource.PROVIDER,
 *     code: 'API_TIMEOUT',
 *     message: 'Request to AI provider timed out',
 *     details: { provider: 'openai', model: 'gpt-4' },
 *     requestId: context.requestId,
 *     sessionId: context.sessionId,
 *     model: 'gpt-4',
 *     provider: 'openai',
 *     originalError: error
 *   });
 *
 *   const handlingResult = await errorHandler.handleError(routingError);
 *
 *   switch (handlingResult.action) {
 *     case 'retry':
 *       // Retry with backoff
 *       await delay(handlingResult.retryAfter);
 *       return retryRequest();
 *
 *     case 'switch_model':
 *       // Switch to alternative model
 *       return routeWithModel(handlingResult.alternativeModel);
 *
 *     case 'return_error':
 *       // Return enriched error to client
 *       throw handlingResult.error;
 *   }
 * }
 */

/**
 * Example: Custom Error Strategy
 *
 * errorHandler.addStrategy('custom_network', {
 *   category: ErrorCategory.NETWORK,
 *   severity: ErrorSeverity.HIGH,
 *   maxRetries: 5,
 *   backoffStrategy: {
 *     type: 'fibonacci',
 *     initialDelay: 1000,
 *     maxDelay: 60000
 *   },
 *   circuitBreaker: {
 *     failureThreshold: 10,
 *     recoveryTimeout: 120000,
 *     monitoringWindow: 600000
 *   },
 *   fallbackActions: [
 *     {
 *       type: 'switch_region',
 *       priority: 1,
 *       conditions: [{ type: 'error_count', operator: 'gte', value: 3 }]
 *     }
 *   ],
 *   notificationRequired: true,
 *   complianceReporting: false
 * });
 */

/**
 * Example: Error Statistics and Monitoring
 *
 * const stats = errorHandler.getErrorStats({
 *   start: new Date(Date.now() - 86400000), // Last 24 hours
 *   end: new Date()
 * });
 *
 * console.log('Error rate:', stats.errorRate, 'errors/hour');
 * console.log('Recovery rate:', (stats.recoveryRate * 100).toFixed(1) + '%');
 * console.log('Top errors:', stats.topErrors.slice(0, 5));
 */

export default ErrorHandler;
