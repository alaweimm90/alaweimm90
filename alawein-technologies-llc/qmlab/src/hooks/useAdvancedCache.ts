import { useState, useEffect, useCallback } from 'react';
import { cacheManager, CacheManager } from '@/lib/cache-manager';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface CacheStats {
  memoryUsage: number;
  memoryLimit: number;
  utilizationPercent: number;
  size: number;
  initialized: boolean;
  quantumCacheAvailable: boolean;
}

interface CacheHookReturn {
  stats: CacheStats;
  get: (key: string, category?: string) => Promise<any | null>;
  set: (key: string, data: any, category?: string) => Promise<boolean>;
  delete: (key: string) => Promise<boolean>;
  clear: () => Promise<void>;
  cacheQuantumResult: (circuitConfig: any, result: any, metadata?: any) => Promise<boolean>;
  getQuantumResult: (circuitConfig: any) => Promise<any | null>;
  isInitialized: boolean;
}

export const useAdvancedCache = (): CacheHookReturn => {
  const [stats, setStats] = useState<CacheStats>({
    memoryUsage: 0,
    memoryLimit: 0,
    utilizationPercent: 0,
    size: 0,
    initialized: false,
    quantumCacheAvailable: false
  });

  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize cache manager
  useEffect(() => {
    const initCache = async () => {
      try {
        const success = await cacheManager.init();
        setIsInitialized(success);
        
        if (success) {
          updateStats();
          trackQuantumEvents.featureDiscovery('advanced_cache_initialized');
        }
      } catch (error) {
        logger.error('Failed to initialize cache manager', { error });
        trackQuantumEvents.errorBoundary(
          'Cache manager initialization failed',
          (error as Error).stack || 'No stack trace',
          'cache-manager'
        );
      }
    };

    initCache();
  }, []);

  // Update cache statistics
  const updateStats = useCallback(() => {
    const currentStats = cacheManager.getStats();
    setStats({
      memoryUsage: currentStats.memory.memoryUsage,
      memoryLimit: currentStats.memory.memoryLimit,
      utilizationPercent: currentStats.memory.utilizationPercent,
      size: currentStats.memory.size,
      initialized: currentStats.initialized,
      quantumCacheAvailable: currentStats.quantumCacheAvailable
    });
  }, []);

  // Periodically update stats
  useEffect(() => {
    if (!isInitialized) return;

    const interval = setInterval(updateStats, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [isInitialized, updateStats]);

  // Cache operations with analytics
  const get = useCallback(async (key: string, category: string = 'general'): Promise<any | null> => {
    const startTime = performance.now();
    
    try {
      const result = await cacheManager.get(key, category);
      const duration = performance.now() - startTime;
      
      trackQuantumEvents.componentLoad('cache_get_operation', duration);
      updateStats();
      
      return result;
    } catch (error) {
      logger.error('Cache get operation failed', { error });
      return null;
    }
  }, [updateStats]);

  const set = useCallback(async (
    key: string, 
    data: any, 
    category: string = 'general'
  ): Promise<boolean> => {
    const startTime = performance.now();
    
    try {
      const success = await cacheManager.set(key, data, category);
      const duration = performance.now() - startTime;
      
      trackQuantumEvents.componentLoad('cache_set_operation', duration);
      updateStats();
      
      return success;
    } catch (error) {
      logger.error('Cache set operation failed', { error });
      return false;
    }
  }, [updateStats]);

  const deleteItem = useCallback(async (key: string): Promise<boolean> => {
    try {
      const success = await cacheManager.delete(key);
      updateStats();
      return success;
    } catch (error) {
      logger.error('Cache delete operation failed', { error });
      return false;
    }
  }, [updateStats]);

  const clear = useCallback(async (): Promise<void> => {
    try {
      await cacheManager.clear();
      updateStats();
      trackQuantumEvents.featureDiscovery('cache_cleared');
    } catch (error) {
      logger.error('Cache clear operation failed', { error });
    }
  }, [updateStats]);

  // Quantum-specific operations
  const cacheQuantumResult = useCallback(async (
    circuitConfig: any,
    result: any,
    metadata?: any
  ): Promise<boolean> => {
    const startTime = performance.now();
    
    try {
      const success = await cacheManager.cacheQuantumResult(circuitConfig, result, metadata);
      const duration = performance.now() - startTime;
      
      trackQuantumEvents.componentLoad('quantum_cache_set', duration);
      updateStats();
      
      return success;
    } catch (error) {
      logger.error('Quantum result caching failed', { error });
      return false;
    }
  }, [updateStats]);

  const getQuantumResult = useCallback(async (circuitConfig: any): Promise<any | null> => {
    const startTime = performance.now();
    
    try {
      const result = await cacheManager.getQuantumResult(circuitConfig);
      const duration = performance.now() - startTime;
      
      if (result) {
        trackQuantumEvents.componentLoad('quantum_cache_hit', duration);
      } else {
        trackQuantumEvents.componentLoad('quantum_cache_miss', duration);
      }
      
      return result;
    } catch (error) {
      logger.error('Quantum result retrieval failed', { error });
      return null;
    }
  }, []);

  return {
    stats,
    get,
    set,
    delete: deleteItem,
    clear,
    cacheQuantumResult,
    getQuantumResult,
    isInitialized
  };
};

// Hook for automatic quantum simulation caching
export const useQuantumSimulationCache = () => {
  const { cacheQuantumResult, getQuantumResult, isInitialized } = useAdvancedCache();

  const simulateWithCache = useCallback(async (
    circuitConfig: any,
    simulationFunction: (config: any) => Promise<any>
  ): Promise<any> => {
    if (!isInitialized) {
      // No caching available, run simulation directly
      return await simulationFunction(circuitConfig);
    }

    // Check cache first
    const cachedResult = await getQuantumResult(circuitConfig);
    if (cachedResult) {
      logger.debug('Using cached quantum simulation result');
      return cachedResult;
    }

    // Run simulation and cache result
    const startTime = performance.now();
    
    try {
      const result = await simulationFunction(circuitConfig);
      const duration = performance.now() - startTime;
      
      // Cache the result if it took significant time
      if (duration > 100) { // Cache results that took more than 100ms
        await cacheQuantumResult(circuitConfig, result, {
          computationTime: duration,
          timestamp: Date.now()
        });
      }
      
      trackQuantumEvents.componentLoad('quantum_simulation_computed', duration);
      return result;
      
    } catch (error) {
      logger.error('Quantum simulation failed', { error });
      throw error;
    }
  }, [cacheQuantumResult, getQuantumResult, isInitialized]);

  return {
    simulateWithCache,
    isInitialized
  };
};

// Hook for cache-aware data fetching
export const useCachedFetch = () => {
  const { get, set, isInitialized } = useAdvancedCache();

  const fetchWithCache = useCallback(async (
    url: string,
    options: RequestInit = {},
    cacheOptions: { ttl?: number; category?: string } = {}
  ): Promise<any> => {
    const { ttl = 5 * 60 * 1000, category = 'api' } = cacheOptions; // Default 5 minutes TTL
    const cacheKey = `fetch:${url}:${JSON.stringify(options)}`;

    // Check cache first
    if (isInitialized) {
      const cachedData = await get(cacheKey, category);
      if (cachedData && Date.now() - cachedData.timestamp < ttl) {
        trackQuantumEvents.componentLoad('api_cache_hit', 0);
        return cachedData.data;
      }
    }

    // Fetch from network
    const startTime = performance.now();
    
    try {
      const response = await fetch(url, options);
      const data = await response.json();
      const duration = performance.now() - startTime;
      
      // Cache successful responses
      if (response.ok && isInitialized) {
        await set(cacheKey, {
          data,
          timestamp: Date.now(),
          status: response.status,
          headers: Object.fromEntries(response.headers.entries())
        }, category);
      }
      
      trackQuantumEvents.componentLoad('api_fetch_completed', duration);
      return data;
      
    } catch (error) {
      // Try to return stale cached data on network error
      if (isInitialized) {
        const staleData = await get(cacheKey, category);
        if (staleData) {
          logger.info('Using stale cached data due to network error');
          trackQuantumEvents.componentLoad('api_stale_cache_hit', 0);
          return staleData.data;
        }
      }
      
      throw error;
    }
  }, [get, set, isInitialized]);

  return { fetchWithCache };
};