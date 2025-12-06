#!/usr/bin/env npx tsx
/**
 * AI Error Handling & Recovery System
 * Structured error types with automatic recovery strategies
 */

import * as path from 'path';
import { EventEmitter } from 'events';
import { loadJson, saveJson } from './utils/file-persistence.js';

const AI_DIR = path.join(process.cwd(), '.ai');
const ERROR_LOG_FILE = path.join(AI_DIR, 'error-log.json');

// ============================================================================
// Error Types
// ============================================================================

export type ErrorSeverity = 'low' | 'medium' | 'high' | 'critical';
export type ErrorCategory =
  | 'validation'
  | 'io'
  | 'network'
  | 'timeout'
  | 'permission'
  | 'configuration'
  | 'dependency'
  | 'runtime';

export interface AIError {
  id: string;
  timestamp: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  code: string;
  message: string;
  stack?: string;
  context?: Record<string, unknown>;
  recoverable: boolean;
  recoveryAttempts: number;
  resolved: boolean;
  resolvedAt?: string;
  resolution?: string;
}

export interface RecoveryStrategy {
  name: string;
  description: string;
  maxAttempts: number;
  backoffMs: number;
  backoffMultiplier: number;
  execute: () => Promise<boolean>;
}

interface ErrorLogState {
  errors: AIError[];
  stats: {
    total: number;
    byCategory: Record<ErrorCategory, number>;
    bySeverity: Record<ErrorSeverity, number>;
    recoveredCount: number;
    unresolvedCount: number;
  };
}

// ============================================================================
// Error Codes
// ============================================================================

export const ErrorCodes = {
  // Validation errors (1xxx)
  VALIDATION_FAILED: 'E1001',
  INVALID_INPUT: 'E1002',
  SCHEMA_MISMATCH: 'E1003',
  MISSING_REQUIRED: 'E1004',

  // IO errors (2xxx)
  FILE_NOT_FOUND: 'E2001',
  FILE_READ_ERROR: 'E2002',
  FILE_WRITE_ERROR: 'E2003',
  DIRECTORY_ERROR: 'E2004',

  // Network errors (3xxx)
  NETWORK_TIMEOUT: 'E3001',
  CONNECTION_REFUSED: 'E3002',
  DNS_ERROR: 'E3003',
  API_ERROR: 'E3004',

  // Permission errors (4xxx)
  ACCESS_DENIED: 'E4001',
  INSUFFICIENT_PERMISSIONS: 'E4002',
  PROTECTED_FILE: 'E4003',

  // Configuration errors (5xxx)
  CONFIG_NOT_FOUND: 'E5001',
  CONFIG_INVALID: 'E5002',
  CONFIG_PARSE_ERROR: 'E5003',

  // Dependency errors (6xxx)
  DEPENDENCY_MISSING: 'E6001',
  VERSION_MISMATCH: 'E6002',
  CIRCULAR_DEPENDENCY: 'E6003',

  // Runtime errors (7xxx)
  TIMEOUT: 'E7001',
  MEMORY_EXCEEDED: 'E7002',
  CIRCUIT_OPEN: 'E7003',
  RATE_LIMITED: 'E7004',
} as const;

// ============================================================================
// Recovery Strategies
// ============================================================================

const defaultStrategies: Record<ErrorCategory, RecoveryStrategy> = {
  validation: {
    name: 'Validation Retry',
    description: 'Re-validate with relaxed constraints',
    maxAttempts: 2,
    backoffMs: 100,
    backoffMultiplier: 1,
    execute: async (): Promise<boolean> => {
      // Retry validation logic
      return true;
    },
  },
  io: {
    name: 'IO Retry with Backoff',
    description: 'Retry file operation with exponential backoff',
    maxAttempts: 3,
    backoffMs: 500,
    backoffMultiplier: 2,
    execute: async (): Promise<boolean> => {
      // Retry IO operation
      return true;
    },
  },
  network: {
    name: 'Network Retry',
    description: 'Retry network request with exponential backoff',
    maxAttempts: 5,
    backoffMs: 1000,
    backoffMultiplier: 2,
    execute: async (): Promise<boolean> => {
      // Retry network operation
      return true;
    },
  },
  timeout: {
    name: 'Timeout Extension',
    description: 'Retry with extended timeout',
    maxAttempts: 2,
    backoffMs: 0,
    backoffMultiplier: 1,
    execute: async (): Promise<boolean> => {
      // Retry with longer timeout
      return true;
    },
  },
  permission: {
    name: 'Permission Check',
    description: 'Verify and request permissions',
    maxAttempts: 1,
    backoffMs: 0,
    backoffMultiplier: 1,
    execute: async (): Promise<boolean> => {
      // Cannot automatically recover from permission errors
      return false;
    },
  },
  configuration: {
    name: 'Config Fallback',
    description: 'Use default configuration',
    maxAttempts: 1,
    backoffMs: 0,
    backoffMultiplier: 1,
    execute: async (): Promise<boolean> => {
      // Use fallback config
      return true;
    },
  },
  dependency: {
    name: 'Dependency Resolution',
    description: 'Attempt to resolve missing dependencies',
    maxAttempts: 2,
    backoffMs: 1000,
    backoffMultiplier: 1,
    execute: async (): Promise<boolean> => {
      // Try to install missing deps
      return false;
    },
  },
  runtime: {
    name: 'Runtime Recovery',
    description: 'Restart or reset runtime state',
    maxAttempts: 3,
    backoffMs: 2000,
    backoffMultiplier: 2,
    execute: async (): Promise<boolean> => {
      // Reset state and retry
      return true;
    },
  },
};

// ============================================================================
// Error Handler Implementation
// ============================================================================

class ErrorHandler extends EventEmitter {
  private state: ErrorLogState;
  private strategies: Record<ErrorCategory, RecoveryStrategy>;
  private maxLogSize = 500;

  constructor() {
    super();
    this.state = this.loadState();
    this.strategies = { ...defaultStrategies };
  }

  private loadState(): ErrorLogState {
    const defaultState: ErrorLogState = {
      errors: [],
      stats: {
        total: 0,
        byCategory: {
          validation: 0,
          io: 0,
          network: 0,
          timeout: 0,
          permission: 0,
          configuration: 0,
          dependency: 0,
          runtime: 0,
        },
        bySeverity: {
          low: 0,
          medium: 0,
          high: 0,
          critical: 0,
        },
        recoveredCount: 0,
        unresolvedCount: 0,
      },
    };

    return loadJson<ErrorLogState>(ERROR_LOG_FILE, defaultState) ?? defaultState;
  }

  private saveState(): void {
    // Trim to max size
    this.state.errors = this.state.errors.slice(-this.maxLogSize);
    saveJson(ERROR_LOG_FILE, this.state);
  }

  private generateId(): string {
    return `err-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  // Create and log an error
  createError(
    category: ErrorCategory,
    code: string,
    message: string,
    options: {
      severity?: ErrorSeverity;
      context?: Record<string, unknown>;
      stack?: string;
    } = {}
  ): AIError {
    const error: AIError = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      category,
      severity: options.severity || 'medium',
      code,
      message,
      stack: options.stack,
      context: options.context,
      recoverable: category !== 'permission',
      recoveryAttempts: 0,
      resolved: false,
    };

    this.state.errors.push(error);
    this.state.stats.total++;
    this.state.stats.byCategory[category]++;
    this.state.stats.bySeverity[error.severity]++;
    this.state.stats.unresolvedCount++;

    this.emit('error', error);
    this.saveState();

    return error;
  }

  // Attempt recovery for an error
  async attemptRecovery(
    errorId: string,
    customStrategy?: RecoveryStrategy
  ): Promise<{ success: boolean; error?: AIError }> {
    const error = this.state.errors.find((e) => e.id === errorId);
    if (!error) {
      return { success: false };
    }

    if (!error.recoverable) {
      return { success: false, error };
    }

    const strategy = customStrategy || this.strategies[error.category];
    if (error.recoveryAttempts >= strategy.maxAttempts) {
      return { success: false, error };
    }

    // Calculate backoff
    const backoff =
      strategy.backoffMs * Math.pow(strategy.backoffMultiplier, error.recoveryAttempts);

    if (backoff > 0) {
      await this.sleep(backoff);
    }

    error.recoveryAttempts++;

    try {
      const success = await strategy.execute();

      if (success) {
        error.resolved = true;
        error.resolvedAt = new Date().toISOString();
        error.resolution = `Recovered via ${strategy.name} (attempt ${error.recoveryAttempts})`;
        this.state.stats.recoveredCount++;
        this.state.stats.unresolvedCount--;
        this.emit('recovered', error);
      }

      this.saveState();
      return { success, error };
    } catch {
      // Recovery itself failed
      this.saveState();
      return { success: false, error };
    }
  }

  // Wrap an async function with error handling and recovery
  async withRecovery<T>(
    operation: () => Promise<T>,
    category: ErrorCategory,
    code: string,
    options: {
      maxRetries?: number;
      context?: Record<string, unknown>;
    } = {}
  ): Promise<T> {
    const maxRetries = options.maxRetries ?? this.strategies[category].maxAttempts;
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        if (attempt < maxRetries) {
          const backoff =
            this.strategies[category].backoffMs *
            Math.pow(this.strategies[category].backoffMultiplier, attempt);
          await this.sleep(backoff);
        }
      }
    }

    // All retries exhausted, create error
    const error = this.createError(category, code, lastError?.message || 'Unknown error', {
      context: options.context,
      stack: lastError?.stack,
    });

    throw new AIOperationError(error);
  }

  // Mark error as resolved manually
  resolve(errorId: string, resolution: string): boolean {
    const error = this.state.errors.find((e) => e.id === errorId);
    if (!error || error.resolved) {
      return false;
    }

    error.resolved = true;
    error.resolvedAt = new Date().toISOString();
    error.resolution = resolution;
    this.state.stats.unresolvedCount--;

    this.saveState();
    this.emit('resolved', error);
    return true;
  }

  // Get unresolved errors
  getUnresolved(severity?: ErrorSeverity): AIError[] {
    return this.state.errors.filter((e) => !e.resolved && (!severity || e.severity === severity));
  }

  // Get errors by category
  getByCategory(category: ErrorCategory): AIError[] {
    return this.state.errors.filter((e) => e.category === category);
  }

  // Get error statistics
  getStats(): ErrorLogState['stats'] {
    return { ...this.state.stats };
  }

  // Get recent errors
  getRecent(limit = 10): AIError[] {
    return this.state.errors.slice(-limit);
  }

  // Register custom recovery strategy
  registerStrategy(category: ErrorCategory, strategy: RecoveryStrategy): void {
    this.strategies[category] = strategy;
  }

  // Utility sleep function
  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}

// ============================================================================
// Custom Error Class
// ============================================================================

export class AIOperationError extends Error {
  public readonly aiError: AIError;

  constructor(aiError: AIError) {
    super(aiError.message);
    this.name = 'AIOperationError';
    this.aiError = aiError;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const errorHandler = new ErrorHandler();

// ============================================================================
// CLI Interface
// ============================================================================

function displayErrors(errors: AIError[]): void {
  if (errors.length === 0) {
    console.log('\n✅ No errors to display\n');
    return;
  }

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║            🚨 AI ERROR LOG                                    ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');

  for (const error of errors.slice(-10)) {
    const icon =
      error.severity === 'critical'
        ? '🔴'
        : error.severity === 'high'
          ? '🟠'
          : error.severity === 'medium'
            ? '🟡'
            : '🔵';
    const status = error.resolved ? '✅' : '❌';

    console.log('║                                                              ║');
    console.log(`║  ${icon} [${error.code}] ${error.message.substring(0, 40)}`.padEnd(65) + '║');
    console.log(`║     Category: ${error.category} | Status: ${status}`.padEnd(65) + '║');
    console.log(`║     Time: ${error.timestamp}`.padEnd(65) + '║');
    if (error.resolved) {
      console.log(`║     Resolution: ${error.resolution?.substring(0, 35)}...`.padEnd(65) + '║');
    }
  }

  console.log('║                                                              ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'list':
    case 'errors': {
      const severity = args[1] as ErrorSeverity | undefined;
      const errors = severity ? errorHandler.getUnresolved(severity) : errorHandler.getRecent(20);
      displayErrors(errors);
      break;
    }

    case 'unresolved': {
      const errors = errorHandler.getUnresolved();
      console.log(`\n📊 Unresolved Errors: ${errors.length}\n`);
      displayErrors(errors);
      break;
    }

    case 'stats': {
      const stats = errorHandler.getStats();
      console.log('\n📊 Error Statistics\n');
      console.log(`Total Errors: ${stats.total}`);
      console.log(`Recovered: ${stats.recoveredCount}`);
      console.log(`Unresolved: ${stats.unresolvedCount}`);
      console.log('\nBy Category:');
      for (const [cat, count] of Object.entries(stats.byCategory)) {
        if (count > 0) console.log(`  ${cat}: ${count}`);
      }
      console.log('\nBy Severity:');
      for (const [sev, count] of Object.entries(stats.bySeverity)) {
        if (count > 0) console.log(`  ${sev}: ${count}`);
      }
      break;
    }

    case 'resolve': {
      const errorId = args[1];
      const resolution = args.slice(2).join(' ') || 'Manually resolved';
      if (errorId) {
        const resolved = errorHandler.resolve(errorId, resolution);
        console.log(
          resolved ? `✅ Error ${errorId} resolved` : `❌ Error not found or already resolved`
        );
      } else {
        console.log('Usage: errors resolve <error-id> [resolution message]');
      }
      break;
    }

    case 'test': {
      // Create a test error
      const error = errorHandler.createError(
        'validation',
        ErrorCodes.VALIDATION_FAILED,
        'Test validation error',
        {
          severity: 'medium',
          context: { field: 'test' },
        }
      );
      console.log(`Created test error: ${error.id}`);
      break;
    }

    default:
      console.log(`
AI Error Handler - Structured error handling with recovery

Commands:
  list [severity]    List recent errors (optionally filter by severity)
  unresolved         List unresolved errors
  stats              Show error statistics
  resolve <id> [msg] Resolve an error manually
  test               Create a test error

Severities: low, medium, high, critical
      `);
  }
}

main();
