/**
 * Performance Monitoring Module
 * Tracks and reports performance metrics
 */

export interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: Date;
}

export interface PerformanceReport {
  metrics: PerformanceMetric[];
  startTime: Date;
  endTime: Date;
  duration: number;
}

class PerformanceMonitorClass {
  private metrics: PerformanceMetric[] = [];
  private startTime: Date | null = null;

  start(): void {
    this.startTime = new Date();
    this.metrics = [];
  }

  record(name: string, value: number, unit: string = 'ms'): void {
    this.metrics.push({
      name,
      value,
      unit,
      timestamp: new Date(),
    });
  }

  stop(): PerformanceReport {
    const endTime = new Date();
    const duration = this.startTime
      ? endTime.getTime() - this.startTime.getTime()
      : 0;

    return {
      metrics: [...this.metrics],
      startTime: this.startTime || endTime,
      endTime,
      duration,
    };
  }

  getMetrics(): PerformanceMetric[] {
    return [...this.metrics];
  }

  clear(): void {
    this.metrics = [];
    this.startTime = null;
  }

  // Measure a function execution time
  async measure<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    try {
      const result = await fn();
      const duration = performance.now() - start;
      this.record(name, duration, 'ms');
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.record(`${name}_error`, duration, 'ms');
      throw error;
    }
  }

  // Measure sync function execution time
  measureSync<T>(name: string, fn: () => T): T {
    const start = performance.now();
    try {
      const result = fn();
      const duration = performance.now() - start;
      this.record(name, duration, 'ms');
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.record(`${name}_error`, duration, 'ms');
      throw error;
    }
  }
}

// Singleton instance
let instance: PerformanceMonitorClass | null = null;

export const PerformanceMonitor = {
  getInstance(): PerformanceMonitorClass {
    if (!instance) {
      instance = new PerformanceMonitorClass();
    }
    return instance;
  },

  // Direct access methods for convenience
  start: () => PerformanceMonitor.getInstance().start(),
  record: (name: string, value: number, unit?: string) => PerformanceMonitor.getInstance().record(name, value, unit),
  stop: () => PerformanceMonitor.getInstance().stop(),
  getMetrics: () => PerformanceMonitor.getInstance().getMetrics(),
  clear: () => PerformanceMonitor.getInstance().clear(),
  measure: <T>(name: string, fn: () => Promise<T>) => PerformanceMonitor.getInstance().measure(name, fn),
  measureSync: <T>(name: string, fn: () => T) => PerformanceMonitor.getInstance().measureSync(name, fn),
};

export default PerformanceMonitor;
