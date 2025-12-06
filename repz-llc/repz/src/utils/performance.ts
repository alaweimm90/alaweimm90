import { logger } from '@/utils/logger';

// Performance monitoring utilities
export const performanceMonitor = {
  startTimer: (label: string): number => {
    const start = performance.now();
    logger.info(`Timer started: ${label}`);
    return start;
  },

  endTimer: (label: string, startTime: number): number => {
    const duration = performance.now() - startTime;
    logger.info(`Timer ended: ${label}`, `${duration.toFixed(2)}ms`);
    return duration;
  },

  measureAsync: async <T>(label: string, fn: () => Promise<T>): Promise<T> => {
    const start = performanceMonitor.startTimer(label);
    try {
      const result = await fn();
      performanceMonitor.endTimer(label, start);
      return result;
    } catch (error) {
      performanceMonitor.endTimer(label, start);
      logger.error(`Performance measure failed: ${label}`, error);
      throw error;
    }
  },

  measureSync: <T>(label: string, fn: () => T): T => {
    const start = performanceMonitor.startTimer(label);
    try {
      const result = fn();
      performanceMonitor.endTimer(label, start);
      return result;
    } catch (error) {
      performanceMonitor.endTimer(label, start);
      logger.error(`Performance measure failed: ${label}`, error);
      throw error;
    }
  }
};

// Memory usage monitoring
export const memoryMonitor = {
  getUsage: () => {
    if ('memory' in performance) {
      const memory = (performance as Performance & { memory?: MemoryInfo }).memory;
      if (!memory) return null;
      return {
        used: Math.round(memory.usedJSHeapSize / 1024 / 1024),
        total: Math.round(memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  },

  logUsage: (context: string) => {
    const usage = memoryMonitor.getUsage();
    if (usage) {
      logger.info(`Memory usage [${context}]`, `${usage.used}MB / ${usage.total}MB (limit: ${usage.limit}MB)`);
    }
  }
};

// Network performance monitoring
export const networkMonitor = {
  measureApiCall: async <T>(
    url: string, 
    apiCall: () => Promise<T>
  ): Promise<T> => {
    return performanceMonitor.measureAsync(`API: ${url}`, apiCall);
  }
};