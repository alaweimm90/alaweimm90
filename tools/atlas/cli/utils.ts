/**
 * ATLAS CLI Utils - Helper functions and utilities for CLI operations
 */

import { AtlasServices, initializeAtlasServices } from '../services/index.js';
import { ConfigLoader } from '../config/loader.js';
import ora from 'ora';
import chalk from 'chalk';

export const NO_COLOR = Boolean(process.env.NO_COLOR);
export const getDefaultStyle = (): string => process.env.ATLAS_OUTPUT_STYLE || 'compact';

const paint = {
  green: (s: string) => (NO_COLOR ? s : chalk.green(s)),
  red: (s: string) => (NO_COLOR ? s : chalk.red(s)),
  yellow: (s: string) => (NO_COLOR ? s : chalk.yellow(s)),
  blue: (s: string) => (NO_COLOR ? s : chalk.blue(s)),
  boldCyan: (s: string) => (NO_COLOR ? s : chalk.bold.cyan(s)),
  gray: (s: string) => (NO_COLOR ? s : chalk.gray(s)),
  white: (s: string) => (NO_COLOR ? s : chalk.white(s)),
};

/**
 * CLI context for managing services and configuration
 */
export class CLIContext {
  private services?: AtlasServices;
  private configPath?: string;

  constructor(configPath?: string) {
    this.configPath = configPath;
  }

  /**
   * Get or initialize ATLAS services
   */
  async getServices(): Promise<AtlasServices> {
    if (!this.services) {
      this.services = await initializeAtlasServices(this.configPath);
    }
    return this.services;
  }

  /**
   * Get configuration loader
   */
  async getConfig(): Promise<ConfigLoader> {
    const services = await this.getServices();
    return services.config;
  }
}

/**
 * Global CLI context instance
 */
export const cliContext = new CLIContext();

/**
 * Output formatting utilities
 */
export const output = {
  success: (message: string) => console.log(paint.green(`✅ ${message}`)),
  error: (message: string) => console.error(paint.red(`❌ ${message}`)),
  warning: (message: string) => console.warn(paint.yellow(`⚠️  ${message}`)),
  info: (message: string) => console.log(paint.blue(`ℹ️  ${message}`)),
  header: (message: string) => console.log(paint.boldCyan(`\n${message}\n`)),
  table: (headers: string[], rows: string[][]) => {
    const colWidths = headers.map((header, i) =>
      Math.max(header.length, ...rows.map((row) => row[i]?.length || 0))
    );

    // Print headers
    console.log(headers.map((h, i) => paint.gray(h.padEnd(colWidths[i]))).join(' │ '));
    console.log(colWidths.map((w) => '─'.repeat(w)).join('─┼─'));

    // Print rows
    rows.forEach((row) => {
      console.log(row.map((cell, i) => (cell || '').padEnd(colWidths[i])).join(' │ '));
    });
  },
};

/**
 * Progress indicator utilities
 */
export const progress = {
  start: (message: string) => ora(message).start(),
  succeed: (spinner: any, message?: string) => spinner.succeed(message),
  fail: (spinner: any, message?: string) => spinner.fail(message),
  stop: (spinner: any) => spinner.stop(),
};

/**
 * Error handling utilities
 */
export const errorHandler = {
  handle: (error: any, context?: string) => {
    const message = error instanceof Error ? error.message : String(error);
    const fullMessage = context ? `${context}: ${message}` : message;
    output.error(fullMessage);

    if (error.stack && process.env.DEBUG) {
      console.error(error.stack);
    }

    process.exit(1);
  },

  handleAsync: async (fn: () => Promise<void>, context?: string) => {
    try {
      await fn();
    } catch (error) {
      errorHandler.handle(error, context);
    }
  },
};

/**
 * Validation utilities
 */
export const validate = {
  path: (path: string, name: string = 'path') => {
    if (!path || typeof path !== 'string') {
      throw new Error(`${name} is required`);
    }
  },

  positiveNumber: (value: number, name: string = 'value') => {
    if (typeof value !== 'number' || value <= 0) {
      throw new Error(`${name} must be a positive number`);
    }
  },

  booleanString: (value: string, name: string = 'value') => {
    if (value !== 'true' && value !== 'false') {
      throw new Error(`${name} must be 'true' or 'false'`);
    }
    return value === 'true';
  },
};

/**
 * Interactive prompt utilities (for future enhancement)
 */
export const prompt = {
  confirm: async (message: string, defaultValue: boolean = false): Promise<boolean> => {
    // For now, return default value - can be enhanced with inquirer later
    output.info(`${message} (default: ${defaultValue ? 'yes' : 'no'})`);
    return defaultValue;
  },

  select: async <T>(message: string, choices: T[], defaultIndex: number = 0): Promise<T> => {
    // For now, return default choice - can be enhanced with inquirer later
    output.info(`${message}`);
    choices.forEach((choice, i) => {
      console.log(`  ${i === defaultIndex ? '>' : ' '} ${choice}`);
    });
    return choices[defaultIndex];
  },
};

/**
 * Format utilities for displaying data
 */
export const format = {
  bytes: (bytes: number): string => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unitIndex = 0;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(1)} ${units[unitIndex]}`;
  },

  duration: (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    if (ms < 3600000) return `${(ms / 60000).toFixed(1)}m`;
    return `${(ms / 3600000).toFixed(1)}h`;
  },

  percentage: (value: number, total: number): string => {
    if (total === 0) return '0%';
    return `${((value / total) * 100).toFixed(1)}%`;
  },

  timestamp: (date: Date): string => {
    return date.toISOString().replace('T', ' ').slice(0, -5);
  },
};

/**
 * Service status formatting
 */
export const formatStatus = {
  health: (status: string): string => {
    switch (status.toLowerCase()) {
      case 'healthy':
      case 'ok':
        return chalk.green(status);
      case 'warning':
      case 'degraded':
        return chalk.yellow(status);
      case 'error':
      case 'critical':
        return chalk.red(status);
      default:
        return status;
    }
  },

  jobStatus: (status: string): string => {
    switch (status.toLowerCase()) {
      case 'completed':
      case 'success':
        return chalk.green(status);
      case 'running':
      case 'in_progress':
        return chalk.blue(status);
      case 'failed':
      case 'error':
        return chalk.red(status);
      case 'pending':
        return chalk.gray(status);
      default:
        return status;
    }
  },
};
