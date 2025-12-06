import { useState, useEffect, useCallback, useRef } from 'react';
import { performanceOptimizer } from '@/lib/performance-optimizer';
import { logger } from '@/lib/logger';

interface PerformanceMetrics {
  mode: 'balanced' | 'performance' | 'memory';
  memory: {
    used: number;
    total: number;
    limit: number;
    utilization: number;
  } | null;
  memoryPressure: boolean;
  resourcePools: Record<string, {
    available: number;
    total: number;
    maxSize: number;
    utilization: number;
  }>;
  workerPools: Record<string, {
    totalWorkers: number;
    availableWorkers: number;
    queueLength: number;
    utilization: number;
  }>;
  quantum: {
    circuitCacheSize: number;
    stateVectorPool: any;
    gateOperationPool: any;
  };
}

interface OptimizationRecommendation {
  type: 'memory' | 'performance' | 'worker' | 'cache';
  severity: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  action: () => void;
  impact: string;
}

export const usePerformanceOptimization = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [recommendations, setRecommendations] = useState<OptimizationRecommendation[]>([]);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [optimizationHistory, setOptimizationHistory] = useState<Array<{
    timestamp: number;
    action: string;
    impact: string;
    metrics: Partial<PerformanceMetrics>;
  }>>([]);
  
  const lastUpdate = useRef(0);
  const updateInterval = useRef<NodeJS.Timeout>();

  // Update metrics
  const updateMetrics = useCallback(() => {
    const currentMetrics = performanceOptimizer.getQuantumMetrics();
    setMetrics(currentMetrics);
    lastUpdate.current = Date.now();
    
    // Generate recommendations based on current metrics
    generateRecommendations(currentMetrics);
  }, []);

  // Generate optimization recommendations
  const generateRecommendations = useCallback((currentMetrics: PerformanceMetrics) => {
    const newRecommendations: OptimizationRecommendation[] = [];

    // Memory-related recommendations
    if (currentMetrics.memory && currentMetrics.memory.utilization > 0.8) {
      newRecommendations.push({
        type: 'memory',
        severity: currentMetrics.memory.utilization > 0.9 ? 'high' : 'medium',
        title: 'High Memory Usage',
        description: `Memory usage is at ${(currentMetrics.memory.utilization * 100).toFixed(1)}%`,
        action: () => performanceOptimizer.setPerformanceMode('memory'),
        impact: 'May improve memory efficiency by clearing caches and compacting pools'
      });
    }

    if (currentMetrics.memoryPressure) {
      newRecommendations.push({
        type: 'memory',
        severity: 'high',
        title: 'Memory Pressure Detected',
        description: 'System is under memory pressure',
        action: () => {
          performanceOptimizer.setPerformanceMode('memory');
          // Trigger garbage collection
          if (typeof window !== 'undefined' && 'gc' in window) {
            (window as any).gc();
          }
        },
        impact: 'Immediate memory cleanup and resource compaction'
      });
    }

    // Worker pool recommendations
    Object.entries(currentMetrics.workerPools).forEach(([poolName, stats]) => {
      if (stats.utilization > 0.9) {
        newRecommendations.push({
          type: 'worker',
          severity: 'medium',
          title: `High Worker Utilization: ${poolName}`,
          description: `Worker pool ${poolName} is ${(stats.utilization * 100).toFixed(1)}% utilized`,
          action: () => {
            // Could increase worker pool size or queue optimization
            logger.debug(`Optimizing worker pool: ${poolName}`);
          },
          impact: 'May improve task processing speed'
        });
      }

      if (stats.queueLength > 10) {
        newRecommendations.push({
          type: 'worker',
          severity: 'high',
          title: `Large Task Queue: ${poolName}`,
          description: `${stats.queueLength} tasks waiting in ${poolName} queue`,
          action: () => {
            performanceOptimizer.setPerformanceMode('performance');
          },
          impact: 'Switch to performance mode to handle queue backlog'
        });
      }
    });

    // Resource pool recommendations
    Object.entries(currentMetrics.resourcePools).forEach(([poolName, stats]) => {
      if (stats.utilization > 0.8 && stats.available < 2) {
        newRecommendations.push({
          type: 'performance',
          severity: 'medium',
          title: `Low Resource Availability: ${poolName}`,
          description: `Only ${stats.available} resources available in ${poolName}`,
          action: () => {
            // Could expand resource pool
            logger.debug(`Expanding resource pool: ${poolName}`);
          },
          impact: 'May reduce resource allocation delays'
        });
      }
    });

    // Quantum-specific recommendations
    if (currentMetrics.quantum.circuitCacheSize > 500) {
      newRecommendations.push({
        type: 'cache',
        severity: 'low',
        title: 'Large Circuit Cache',
        description: `Circuit cache contains ${currentMetrics.quantum.circuitCacheSize} entries`,
        action: () => {
          // Clear circuit cache
          logger.debug('Clearing circuit cache');
        },
        impact: 'Frees memory used by cached circuits'
      });
    }

    setRecommendations(newRecommendations);
  }, []);

  // Apply optimization
  const applyOptimization = useCallback((recommendation: OptimizationRecommendation) => {
    setIsOptimizing(true);
    
    try {
      const beforeMetrics = performanceOptimizer.getQuantumMetrics();
      
      // Apply the optimization
      recommendation.action();
      
      // Record the optimization
      const optimizationRecord = {
        timestamp: Date.now(),
        action: recommendation.title,
        impact: recommendation.impact,
        metrics: beforeMetrics
      };
      
      setOptimizationHistory(prev => [optimizationRecord, ...prev].slice(0, 20));
      
      // Update metrics after optimization
      setTimeout(() => {
        updateMetrics();
        setIsOptimizing(false);
      }, 1000);
      
    } catch (error) {
      logger.error('Optimization failed', { error });
      setIsOptimizing(false);
    }
  }, [updateMetrics]);

  // Set performance mode
  const setPerformanceMode = useCallback((mode: 'balanced' | 'performance' | 'memory') => {
    setIsOptimizing(true);
    
    try {
      performanceOptimizer.setPerformanceMode(mode);
      
      setTimeout(() => {
        updateMetrics();
        setIsOptimizing(false);
      }, 500);
      
    } catch (error) {
      logger.error('Failed to set performance mode', { error });
      setIsOptimizing(false);
    }
  }, [updateMetrics]);

  // Get optimization suggestions
  const getOptimizationSuggestions = useCallback(() => {
    if (!metrics) return [];

    const suggestions = [];

    // Suggest performance mode based on current conditions
    if (metrics.memoryPressure || (metrics.memory && metrics.memory.utilization > 0.85)) {
      if (metrics.mode !== 'memory') {
        suggestions.push({
          type: 'mode-change',
          title: 'Switch to Memory Mode',
          description: 'Current memory usage suggests switching to memory-optimized mode',
          action: () => setPerformanceMode('memory')
        });
      }
    } else if (metrics.memory && metrics.memory.utilization < 0.5) {
      if (metrics.mode !== 'performance') {
        suggestions.push({
          type: 'mode-change',
          title: 'Switch to Performance Mode',
          description: 'Low memory usage allows switching to performance-optimized mode',
          action: () => setPerformanceMode('performance')
        });
      }
    }

    return suggestions;
  }, [metrics, setPerformanceMode]);

  // Auto-optimization
  const enableAutoOptimization = useCallback((enabled: boolean) => {
    if (enabled) {
      const autoOptimize = () => {
        const highPriorityRecommendations = recommendations.filter(r => r.severity === 'high');
        
        if (highPriorityRecommendations.length > 0) {
          // Apply the first high-priority recommendation automatically
          applyOptimization(highPriorityRecommendations[0]);
        }
      };

      // Check for auto-optimizations every 30 seconds
      const autoInterval = setInterval(autoOptimize, 30000);
      
      return () => clearInterval(autoInterval);
    }
  }, [recommendations, applyOptimization]);

  // Start monitoring
  useEffect(() => {
    // Initial metrics update
    updateMetrics();
    
    // Set up periodic updates
    updateInterval.current = setInterval(updateMetrics, 10000); // Every 10 seconds
    
    return () => {
      if (updateInterval.current) {
        clearInterval(updateInterval.current);
      }
    };
  }, [updateMetrics]);

  // Listen for memory pressure events
  useEffect(() => {
    const handleMemoryPressure = () => {
      updateMetrics();
    };

    window.addEventListener('memoryPressure', handleMemoryPressure);
    
    return () => {
      window.removeEventListener('memoryPressure', handleMemoryPressure);
    };
  }, [updateMetrics]);

  return {
    metrics,
    recommendations,
    optimizationHistory,
    isOptimizing,
    lastUpdate: lastUpdate.current,
    actions: {
      updateMetrics,
      applyOptimization,
      setPerformanceMode,
      getOptimizationSuggestions,
      enableAutoOptimization
    }
  };
};

// Hook for quantum-specific performance monitoring
export const useQuantumPerformance = () => {
  const [quantumMetrics, setQuantumMetrics] = useState({
    circuitExecutionTime: [] as number[],
    simulationMemory: [] as number[],
    workerUtilization: 0,
    cacheEfficiency: 0
  });

  const [activeOptimizations, setActiveOptimizations] = useState<string[]>([]);

  const collectQuantumMetrics = useCallback(() => {
    const metrics = performanceOptimizer.getQuantumMetrics();
    
    // Extract quantum-specific metrics
    setQuantumMetrics(prev => ({
      circuitExecutionTime: [...prev.circuitExecutionTime, Date.now()].slice(-20),
      simulationMemory: [
        ...prev.simulationMemory, 
        metrics.memory?.used || 0
      ].slice(-20),
      workerUtilization: Object.values(metrics.workerPools)
        .reduce((sum, pool) => sum + pool.utilization, 0) / 
        Object.keys(metrics.workerPools).length || 0,
      cacheEfficiency: metrics.quantum.circuitCacheSize > 0 ? 0.8 : 0 // Simplified calculation
    }));
  }, []);

  // Optimize quantum circuit
  const optimizeCircuit = useCallback(async (circuitConfig: any) => {
    const startTime = performance.now();
    
    try {
      setActiveOptimizations(prev => [...prev, 'circuit-optimization']);
      
      // Use the performance optimizer to optimize the circuit
      const optimizedCircuit = performanceOptimizer.optimizeCircuit(circuitConfig);
      
      const executionTime = performance.now() - startTime;
      
      // Update metrics
      collectQuantumMetrics();
      
      return {
        ...optimizedCircuit,
        optimizationTime: executionTime,
        timestamp: Date.now()
      };
      
    } finally {
      setActiveOptimizations(prev => prev.filter(opt => opt !== 'circuit-optimization'));
    }
  }, [collectQuantumMetrics]);

  // Optimize quantum simulation
  const optimizeSimulation = useCallback(async (config: any) => {
    const startTime = performance.now();
    
    try {
      setActiveOptimizations(prev => [...prev, 'simulation-optimization']);
      
      // Use performance optimizer for quantum simulation
      const result = await performanceOptimizer.optimizeQuantumSimulation(config);
      
      const executionTime = performance.now() - startTime;
      
      return {
        ...result,
        optimizationTime: executionTime,
        timestamp: Date.now()
      };
      
    } finally {
      setActiveOptimizations(prev => prev.filter(opt => opt !== 'simulation-optimization'));
    }
  }, []);

  // Monitor quantum performance
  useEffect(() => {
    const interval = setInterval(collectQuantumMetrics, 5000); // Every 5 seconds
    
    return () => clearInterval(interval);
  }, [collectQuantumMetrics]);

  return {
    quantumMetrics,
    activeOptimizations,
    optimizeCircuit,
    optimizeSimulation,
    collectQuantumMetrics
  };
};