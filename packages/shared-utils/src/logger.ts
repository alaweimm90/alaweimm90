import winston from 'winston';
import chalk from 'chalk';

/**
 * Shared logger configuration for the monorepo
 * Provides consistent logging across all packages
 */

export type LogLevel = 'error' | 'warn' | 'info' | 'debug';

export interface LoggerOptions {
  service?: string;
  level?: LogLevel;
  silent?: boolean;
}

/**
 * Create a configured Winston logger instance
 */
export function createLogger(options: LoggerOptions = {}): winston.Logger {
  const { service = 'monorepo', level = 'info', silent = false } = options;

  return winston.createLogger({
    level,
    silent,
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.printf(({ level, message, timestamp, stack }) => {
        const coloredLevel = colorizeLevel(level);
        const time = chalk.gray(timestamp);
        const svc = chalk.cyan(`[${service}]`);

        if (stack) {
          return `${time} ${coloredLevel} ${svc} ${message}\n${chalk.gray(stack)}`;
        }
        return `${time} ${coloredLevel} ${svc} ${message}`;
      })
    ),
    transports: [new winston.transports.Console()],
  });
}

function colorizeLevel(level: string): string {
  switch (level) {
    case 'error':
      return chalk.red.bold('ERROR');
    case 'warn':
      return chalk.yellow.bold('WARN ');
    case 'info':
      return chalk.blue.bold('INFO ');
    case 'debug':
      return chalk.gray.bold('DEBUG');
    default:
      return level.toUpperCase();
  }
}

// Default logger instance
export const logger = createLogger({ service: 'shared' });
