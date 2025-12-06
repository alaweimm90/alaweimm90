// Global console manager for production-ready logging
import * as Sentry from '@sentry/react';

class ConsoleManager {
  private readonly isDevelopment = (
    typeof import.meta !== 'undefined' &&
    ((import.meta as unknown as { env?: { MODE?: string } }).env?.MODE === 'development')
  );

  log(message: string, data?: unknown) {
    console.log(`[REPZ] ${message}`, data ?? '');
  }

  error(message: string, error?: unknown) {
    console.error(`[REPZ ERROR] ${message}`, error ?? '');
    if (error instanceof Error) {
      Sentry.captureException(error, { extra: { message } });
    } else {
      Sentry.captureMessage(message, { level: 'error', extra: { error } });
    }
  }

  warn(message: string, data?: unknown) {
    console.warn(`[REPZ WARN] ${message}`, data ?? '');
    Sentry.captureMessage(message, { level: 'warning', extra: { data } });
  }

  info(message: string, data?: unknown) {
    if (this.isDevelopment) {
      console.info(`[REPZ INFO] ${message}`, data ?? '');
    }
  }
}

export const logger = new ConsoleManager();
