/**
 * ATLAS Configuration Loader
 * Loads, validates, and manages optimization service configurations
 */

import * as fs from 'fs';
import * as path from 'path';
import { EventEmitter } from 'events';

export interface OptimizationConfig {
  version: string;
  description?: string;
  optimizer: {
    schedule: {
      enabled: boolean;
      intervalMinutes: number;
      maxConcurrentJobs: number;
      workingHours?: {
        enabled: boolean;
        start: string;
        end: string;
        timezone: string;
      };
    };
    thresholds: {
      chaosThreshold: number;
      complexityThreshold: number;
      minConfidence: number;
      maxFileSize: number;
      maxChangesPerCommit: number;
    };
    safety: {
      rateLimitPerHour: number;
      circuitBreakerThreshold: number;
      rollbackEnabled: boolean;
      manualOverride: boolean;
      dryRunByDefault: boolean;
      maxExecutionTimeMinutes: number;
    };
    repositories: RepositoryConfig[];
  };
  monitor: {
    polling: {
      enabled: boolean;
      intervalSeconds: number;
    };
    filesystem: {
      watchEnabled: boolean;
      debounceMs: number;
      ignorePatterns: string[];
    };
    triggers: {
      onFileChange: boolean;
      onCommit: boolean;
      onPush: boolean;
      minChangesThreshold: number;
      maxFrequencyMs: number;
      cooldownPeriodMs: number;
    };
    analysis: {
      incremental: boolean;
      cacheResults: boolean;
      cacheTtlMinutes: number;
      parallelAnalysis: boolean;
      maxConcurrency: number;
    };
  };
  dashboard: {
    port: number;
    host: string;
    enableWebSocket: boolean;
    enableREST: boolean;
    telemetry: {
      retentionPeriodDays: number;
      maxEvents: number;
      enableMetrics: boolean;
      enableTracing: boolean;
    };
    security: {
      enableAuth: boolean;
      apiKeys: string[];
      corsOrigins: string[];
    };
    ui: {
      theme: string;
      refreshIntervalMs: number;
      maxChartPoints: number;
    };
  };
  refactoring: {
    enabledRules: string[];
    disabledRules: string[];
    safetyChecks: {
      requireTests: boolean;
      checkBuild: boolean;
      validateTypes: boolean;
      performanceRegressionCheck: boolean;
    };
    limits: {
      maxFilesPerOperation: number;
      maxChangesPerFile: number;
      maxTotalChanges: number;
    };
  };
  agents: {
    defaultAgent: string;
    fallbackAgents: string[];
    timeouts: {
      analysis: number;
      refactoring: number;
      validation: number;
    };
    retryPolicy: {
      maxRetries: number;
      backoffMultiplier: number;
      initialDelayMs: number;
    };
  };
  logging: {
    level: 'debug' | 'info' | 'warn' | 'error';
    format: 'json' | 'text';
    outputs: string[];
    file?: {
      path: string;
      maxSize: string;
      maxFiles: number;
    };
    remote?: {
      enabled: boolean;
      endpoint: string;
      apiKey: string;
    };
  };
  notifications: {
    enabled: boolean;
    channels: {
      console: boolean;
      file: boolean;
      webhook: boolean;
      email: boolean;
    };
    triggers: {
      onOptimizationComplete: boolean;
      onOptimizationFailure: boolean;
      onHighChaosDetected: boolean;
      onSystemError: boolean;
      onCircuitBreakerOpen: boolean;
    };
    webhook?: {
      url: string;
      headers: Record<string, string>;
      timeoutMs: number;
    };
    email?: {
      smtp: {
        host: string;
        port: number;
        secure: boolean;
        auth: {
          user: string;
          pass: string;
        };
      };
      from: string;
      to: string[];
    };
  };
  backup: {
    enabled: boolean;
    strategy: string;
    preOptimizationBackup: boolean;
    postOptimizationBackup: boolean;
    backupBranch: string;
    retention: {
      maxBackups: number;
      maxAgeDays: number;
    };
  };
  performance: {
    memoryLimit: string;
    cpuLimit: string;
    concurrencyLimits: {
      analysis: number;
      refactoring: number;
      fileOperations: number;
    };
    timeouts: {
      fileOperation: number;
      networkRequest: number;
      analysis: number;
    };
  };
  experimental: {
    aiPoweredSuggestions: boolean;
    predictiveOptimization: boolean;
    autoScaling: boolean;
    federatedLearning: boolean;
  };
}

export interface RepositoryConfig {
  name: string;
  path: string;
  enabled: boolean;
  branch: string;
  priority: 'low' | 'medium' | 'high';
  excludedPaths: string[];
  includedExtensions: string[];
  customRules: {
    maxComplexity?: number;
    maxLinesPerFunction?: number;
    maxFileSize?: number;
  };
}

export class ConfigLoader extends EventEmitter {
  private configPath: string;
  private config: OptimizationConfig | null = null;
  private configMtime: number = 0;
  private watchTimer?: NodeJS.Timeout;

  constructor(configPath: string = 'tools/atlas/config/optimization.json') {
    super();
    this.configPath = path.resolve(configPath);
  }

  /**
   * Load configuration from file
   */
  async load(): Promise<OptimizationConfig> {
    try {
      const stats = fs.statSync(this.configPath);
      const mtime = stats.mtime.getTime();

      // Check if file has changed
      if (this.config && mtime === this.configMtime) {
        return this.config;
      }

      const content = fs.readFileSync(this.configPath, 'utf8');
      const parsed = JSON.parse(content);

      // Validate configuration
      const validated = this.validateConfig(parsed);

      this.config = validated;
      this.configMtime = mtime;

      this.emit('config:loaded', { config: this.config, path: this.configPath });
      return this.config;
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      this.emit('config:error', {
        error: errorMsg,
        path: this.configPath,
      });
      throw new Error(`Failed to load configuration: ${errorMsg}`);
    }
  }

  /**
   * Save configuration to file
   */
  async save(config: OptimizationConfig): Promise<void> {
    try {
      // Validate before saving
      const validated = this.validateConfig(config);

      const content = JSON.stringify(validated, null, 2);
      fs.writeFileSync(this.configPath, content, 'utf8');

      this.config = validated;
      this.configMtime = Date.now();

      this.emit('config:saved', { config: this.config, path: this.configPath });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      this.emit('config:error', {
        error: errorMsg,
        path: this.configPath,
      });
      throw new Error(`Failed to save configuration: ${errorMsg}`);
    }
  }

  /**
   * Update specific configuration values
   */
  async update(updates: Partial<OptimizationConfig>): Promise<OptimizationConfig> {
    const current = await this.load();
    const updated = this.deepMerge(current, updates);
    await this.save(updated);
    return updated;
  }

  /**
   * Get current configuration (loads if not cached)
   */
  async get(): Promise<OptimizationConfig> {
    if (!this.config) {
      return this.load();
    }
    return this.config;
  }

  /**
   * Watch configuration file for changes
   */
  watch(intervalMs: number = 5000): void {
    if (this.watchTimer) {
      clearInterval(this.watchTimer);
    }

    this.watchTimer = setInterval(async () => {
      try {
        const newConfig = await this.load();
        if (this.config !== newConfig) {
          this.emit('config:changed', {
            oldConfig: this.config,
            newConfig,
            path: this.configPath,
          });
        }
      } catch {
        // Ignore errors during watch - they'll be logged via events
      }
    }, intervalMs);
  }

  /**
   * Stop watching configuration file
   */
  unwatch(): void {
    if (this.watchTimer) {
      clearInterval(this.watchTimer);
      this.watchTimer = undefined;
    }
  }

  /**
   * Validate configuration structure and values
   */
  private validateConfig(config: any): OptimizationConfig {
    if (!config || typeof config !== 'object') {
      throw new Error('Configuration must be a valid object');
    }

    // Required fields validation
    this.validateRequired(config, 'version', 'string');
    this.validateRequired(config.optimizer, 'optimizer', 'object');
    this.validateRequired(config.monitor, 'monitor', 'object');
    this.validateRequired(config.dashboard, 'dashboard', 'object');

    // Version compatibility check
    if (!this.isCompatibleVersion(config.version)) {
      throw new Error(`Configuration version ${config.version} is not compatible`);
    }

    // Validate nested objects
    this.validateOptimizerConfig(config.optimizer);
    this.validateMonitorConfig(config.monitor);
    this.validateDashboardConfig(config.dashboard);
    this.validateRefactoringConfig(config.refactoring);
    this.validateAgentsConfig(config.agents);
    this.validateLoggingConfig(config.logging);
    this.validateNotificationsConfig(config.notifications);
    this.validateBackupConfig(config.backup);
    this.validatePerformanceConfig(config.performance);

    return config as OptimizationConfig;
  }

  private validateRequired(obj: any, field: string, type: string): void {
    if (!(field in obj)) {
      throw new Error(`Missing required field: ${field}`);
    }
    if (typeof obj[field] !== type) {
      throw new Error(`Field ${field} must be of type ${type}`);
    }
  }

  private validateOptimizerConfig(optimizer: any): void {
    this.validateRequired(optimizer.schedule, 'schedule', 'object');
    this.validateRequired(optimizer.thresholds, 'thresholds', 'object');
    this.validateRequired(optimizer.safety, 'safety', 'object');
    this.validateRequired(optimizer.repositories, 'repositories', 'object');

    // Validate numeric ranges
    this.validateRange(optimizer.thresholds.chaosThreshold, 0, 1, 'chaosThreshold');
    this.validateRange(optimizer.thresholds.complexityThreshold, 0, 1, 'complexityThreshold');
    this.validateRange(optimizer.thresholds.minConfidence, 0, 1, 'minConfidence');

    // Validate repositories
    if (!Array.isArray(optimizer.repositories)) {
      throw new Error('repositories must be an array');
    }

    for (const repo of optimizer.repositories) {
      this.validateRepositoryConfig(repo);
    }
  }

  private validateRepositoryConfig(repo: any): void {
    this.validateRequired(repo, 'name', 'string');
    this.validateRequired(repo, 'path', 'string');
    this.validateRequired(repo, 'enabled', 'boolean');
    this.validateRequired(repo, 'branch', 'string');

    if (!['low', 'medium', 'high'].includes(repo.priority)) {
      throw new Error(`Invalid repository priority: ${repo.priority}`);
    }
  }

  private validateMonitorConfig(monitor: any): void {
    this.validateRequired(monitor.polling, 'polling', 'object');
    this.validateRequired(monitor.filesystem, 'filesystem', 'object');
    this.validateRequired(monitor.triggers, 'triggers', 'object');
    this.validateRequired(monitor.analysis, 'analysis', 'object');
  }

  private validateDashboardConfig(dashboard: any): void {
    this.validateRequired(dashboard, 'port', 'number');
    this.validateRequired(dashboard, 'host', 'string');
    this.validateRequired(dashboard.telemetry, 'telemetry', 'object');
    this.validateRequired(dashboard.security, 'security', 'object');
  }

  private validateRefactoringConfig(refactoring: any): void {
    if (refactoring) {
      this.validateRequired(refactoring, 'enabledRules', 'object');
      this.validateRequired(refactoring, 'safetyChecks', 'object');
      this.validateRequired(refactoring, 'limits', 'object');
    }
  }

  private validateAgentsConfig(agents: any): void {
    if (agents) {
      this.validateRequired(agents, 'defaultAgent', 'string');
      this.validateRequired(agents.timeouts, 'timeouts', 'object');
      this.validateRequired(agents.retryPolicy, 'retryPolicy', 'object');
    }
  }

  private validateLoggingConfig(logging: any): void {
    if (logging) {
      if (!['debug', 'info', 'warn', 'error'].includes(logging.level)) {
        throw new Error(`Invalid log level: ${logging.level}`);
      }
      if (!['json', 'text'].includes(logging.format)) {
        throw new Error(`Invalid log format: ${logging.format}`);
      }
    }
  }

  private validateNotificationsConfig(notifications: any): void {
    if (notifications) {
      this.validateRequired(notifications, 'channels', 'object');
      this.validateRequired(notifications, 'triggers', 'object');
    }
  }

  private validateBackupConfig(backup: any): void {
    if (backup) {
      this.validateRequired(backup, 'retention', 'object');
    }
  }

  private validatePerformanceConfig(performance: any): void {
    if (performance) {
      this.validateRequired(performance, 'concurrencyLimits', 'object');
      this.validateRequired(performance, 'timeouts', 'object');
    }
  }

  private validateRange(value: number, min: number, max: number, field: string): void {
    if (typeof value !== 'number' || value < min || value > max) {
      throw new Error(`Field ${field} must be a number between ${min} and ${max}`);
    }
  }

  private isCompatibleVersion(version: string): boolean {
    // Simple version compatibility check
    const major = parseInt(version.split('.')[0]);
    return major === 1; // Only support v1.x for now
  }

  private deepMerge(target: any, source: any): any {
    const result = { ...target };

    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }

    return result;
  }
}

/**
 * Create default configuration
 */
export function createDefaultConfig(): OptimizationConfig {
  return {
    version: '1.0.0',
    description: 'Default ATLAS Optimization Configuration',

    optimizer: {
      schedule: {
        enabled: true,
        intervalMinutes: 60,
        maxConcurrentJobs: 3,
      },
      thresholds: {
        chaosThreshold: 0.7,
        complexityThreshold: 0.8,
        minConfidence: 0.6,
        maxFileSize: 1000000,
        maxChangesPerCommit: 50,
      },
      safety: {
        rateLimitPerHour: 10,
        circuitBreakerThreshold: 5,
        rollbackEnabled: true,
        manualOverride: false,
        dryRunByDefault: false,
        maxExecutionTimeMinutes: 30,
      },
      repositories: [],
    },

    monitor: {
      polling: {
        enabled: true,
        intervalSeconds: 300,
      },
      filesystem: {
        watchEnabled: true,
        debounceMs: 2000,
        ignorePatterns: ['**/node_modules/**', '**/dist/**', '**/build/**', '**/*.log'],
      },
      triggers: {
        onFileChange: true,
        onCommit: true,
        onPush: false,
        minChangesThreshold: 5,
        maxFrequencyMs: 300000,
        cooldownPeriodMs: 60000,
      },
      analysis: {
        incremental: true,
        cacheResults: true,
        cacheTtlMinutes: 30,
        parallelAnalysis: true,
        maxConcurrency: 2,
      },
    },

    dashboard: {
      port: 8080,
      host: 'localhost',
      enableWebSocket: true,
      enableREST: true,
      telemetry: {
        retentionPeriodDays: 30,
        maxEvents: 10000,
        enableMetrics: true,
        enableTracing: false,
      },
      security: {
        enableAuth: false,
        apiKeys: [],
        corsOrigins: ['*'],
      },
      ui: {
        theme: 'dark',
        refreshIntervalMs: 5000,
        maxChartPoints: 100,
      },
    },

    refactoring: {
      enabledRules: ['remove-unused-imports', 'simplify-conditionals', 'extract-methods'],
      disabledRules: ['aggressive-renaming'],
      safetyChecks: {
        requireTests: false,
        checkBuild: true,
        validateTypes: true,
        performanceRegressionCheck: false,
      },
      limits: {
        maxFilesPerOperation: 10,
        maxChangesPerFile: 20,
        maxTotalChanges: 100,
      },
    },

    agents: {
      defaultAgent: 'atlas-optimizer',
      fallbackAgents: ['atlas-refactor'],
      timeouts: {
        analysis: 300000,
        refactoring: 600000,
        validation: 120000,
      },
      retryPolicy: {
        maxRetries: 3,
        backoffMultiplier: 2,
        initialDelayMs: 1000,
      },
    },

    logging: {
      level: 'info',
      format: 'json',
      outputs: ['console'],
    },

    notifications: {
      enabled: true,
      channels: {
        console: true,
        file: false,
        webhook: false,
        email: false,
      },
      triggers: {
        onOptimizationComplete: true,
        onOptimizationFailure: true,
        onHighChaosDetected: true,
        onSystemError: true,
        onCircuitBreakerOpen: true,
      },
    },

    backup: {
      enabled: true,
      strategy: 'git-commit',
      preOptimizationBackup: true,
      postOptimizationBackup: true,
      backupBranch: 'atlas-backup-{timestamp}',
      retention: {
        maxBackups: 10,
        maxAgeDays: 7,
      },
    },

    performance: {
      memoryLimit: '512m',
      cpuLimit: '50%',
      concurrencyLimits: {
        analysis: 2,
        refactoring: 1,
        fileOperations: 5,
      },
      timeouts: {
        fileOperation: 30000,
        networkRequest: 10000,
        analysis: 300000,
      },
    },

    experimental: {
      aiPoweredSuggestions: false,
      predictiveOptimization: false,
      autoScaling: false,
      federatedLearning: false,
    },
  };
}
