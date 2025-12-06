import { useEffect, useRef, useCallback } from 'react';

interface MemoryStats {
  usedJSHeapSize: number;
  totalJSHeapSize: number;
  jsHeapSizeLimit: number;
  usedMB: number;
  totalMB: number;
  limitMB: number;
  usagePercentage: number;
}

interface MemoryMonitorConfig {
  interval?: number;
  threshold?: number;
  onThresholdExceeded?: (stats: MemoryStats) => void;
  onMemoryLeak?: (stats: MemoryStats) => void;
  enabled?: boolean;
}

export const useMemoryMonitor = (config: MemoryMonitorConfig = {}) => {
  const {
    interval = 5000,
    threshold = 80,
    onThresholdExceeded,
    onMemoryLeak,
    enabled = process.env.NODE_ENV === 'development'
  } = config;

  const previousStats = useRef<MemoryStats | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const leakDetectionCount = useRef(0);

  const getMemoryStats = useCallback((): MemoryStats | null => {
    if (!('memory' in performance)) {
      return null;
    }

    const memory = (performance as Performance & { memory?: MemoryInfo }).memory;
    if (!memory) return null;

    const usedMB = Math.round(memory.usedJSHeapSize / 1024 / 1024);
    const totalMB = Math.round(memory.totalJSHeapSize / 1024 / 1024);
    const limitMB = Math.round(memory.jsHeapSizeLimit / 1024 / 1024);
    const usagePercentage = Math.round((memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100);

    return {
      usedJSHeapSize: memory.usedJSHeapSize,
      totalJSHeapSize: memory.totalJSHeapSize,
      jsHeapSizeLimit: memory.jsHeapSizeLimit,
      usedMB,
      totalMB,
      limitMB,
      usagePercentage
    };
  }, []);

  const checkMemoryLeak = useCallback((currentStats: MemoryStats) => {
    if (!previousStats.current) {
      previousStats.current = currentStats;
      return;
    }

    const growth = currentStats.usedMB - previousStats.current.usedMB;
    
    if (growth > 5) { // 5MB growth threshold
      leakDetectionCount.current++;
      
      if (leakDetectionCount.current >= 3) { // 3 consecutive increases
        console.warn('🚨 Potential memory leak detected:', {
          previousUsage: `${previousStats.current.usedMB}MB`,
          currentUsage: `${currentStats.usedMB}MB`,
          growth: `+${growth}MB`
        });
        
        onMemoryLeak?.(currentStats);
        leakDetectionCount.current = 0; // Reset counter
      }
    } else {
      leakDetectionCount.current = 0; // Reset counter on stable/decreased usage
    }

    previousStats.current = currentStats;
  }, [onMemoryLeak]);

  const monitor = useCallback(() => {
    const stats = getMemoryStats();
    
    if (!stats) return;

    console.log(`🧠 Memory usage: ${stats.usedMB}MB (${stats.usagePercentage}%)`);

    // Check threshold
    if (stats.usagePercentage > threshold) {
      console.warn(`⚠️ Memory usage exceeded ${threshold}%:`, stats);
      onThresholdExceeded?.(stats);
    }

    // Check for memory leaks
    checkMemoryLeak(stats);
  }, [getMemoryStats, threshold, onThresholdExceeded, checkMemoryLeak]);

  const forceGarbageCollection = useCallback(() => {
    const windowWithGC = window as Window & { gc?: () => void };
    if ('gc' in window && typeof windowWithGC.gc === 'function') {
      console.log('🗑️ Forcing garbage collection');
      windowWithGC.gc();
    } else {
      console.warn('Garbage collection not available in this environment');
    }
  }, []);

  const getCurrentStats = useCallback(() => {
    return getMemoryStats();
  }, [getMemoryStats]);

  useEffect(() => {
    if (!enabled) return;

    // Initial check
    monitor();

    // Set up interval monitoring
    intervalRef.current = setInterval(monitor, interval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, monitor, interval]);

  return {
    getCurrentStats,
    forceGarbageCollection,
    isSupported: 'memory' in performance
  };
};