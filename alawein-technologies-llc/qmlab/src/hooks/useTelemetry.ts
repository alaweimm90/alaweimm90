import { useEffect, useCallback, useRef } from 'react';
import { telemetry, withTelemetry } from '@/lib/telemetry';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface CircuitConfig {
  gates?: Array<Record<string, unknown>>;
  qubits?: number;
  id?: string;
  [key: string]: unknown;
}

interface TelemetryHookOptions {
  autoTrackInteractions?: boolean;
  autoTrackPerformance?: boolean;
  samplingRate?: number;
}

export const useTelemetry = (options: TelemetryHookOptions = {}) => {
  const {
    autoTrackInteractions = true,
    autoTrackPerformance = true,
    samplingRate = 0.1
  } = options;

  const interactionTimeouts = useRef(new Map<string, number>());

  // Track quantum operations with telemetry
  const trackQuantumOperation = useCallback(async <T>(
    operation: 'circuit_build' | 'simulation' | 'training' | 'visualization',
    fn: () => Promise<T>,
    circuitConfig?: CircuitConfig
  ): Promise<T> => {
    const startTime = performance.now();
    
    // Calculate quantum metrics
    const gates = circuitConfig?.gates || [];
    const qubits = circuitConfig?.qubits || 1;
    const metrics = {
      circuitComplexity: gates.length * qubits || 1,
      gateCount: gates.length || 0,
      qubitCount: qubits,
      circuitDepth: calculateCircuitDepth(gates),
      compilationTime: 0,
      executionTime: 0,
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0
    };

    const spanId = telemetry.trackQuantumOperation(operation, metrics, {
      circuit_id: circuitConfig?.id,
      user_initiated: true
    });

    try {
      const result = await fn();
      
      const endTime = performance.now();
      metrics.executionTime = endTime - startTime;
      
      // Update span with final metrics
      telemetry.endSpan(spanId, 'ok', {
        'quantum.execution_time': metrics.executionTime,
        'quantum.success': true
      });

      return result;
    } catch (error) {
      telemetry.endSpan(spanId, 'error', {
        'quantum.error': (error as Error).message
      });
      throw error;
    }
  }, []);

  // Track user interactions automatically
  useEffect(() => {
    if (!autoTrackInteractions) return;

    const handleInteraction = (event: Event) => {
      const target = event.target as HTMLElement;
      const elementInfo = getElementInfo(target);
      
      // Sample interactions to reduce overhead
      if (Math.random() > samplingRate) return;

      const interactionType = event.type as any;
      
      telemetry.trackInteraction(
        interactionType,
        elementInfo.selector,
        elementInfo.value,
        getEventPosition(event)
      );

      // Track interaction duration for certain events
      if (event.type === 'mousedown' || event.type === 'touchstart') {
        const startTime = Date.now();
        const key = `${elementInfo.selector}-${startTime}`;
        
        const endHandler = () => {
          const duration = Date.now() - startTime;
          interactionTimeouts.current.delete(key);
          
          telemetry.recordMetric('interaction.duration', duration, {
            element: elementInfo.selector,
            type: event.type
          });
          
          target.removeEventListener('mouseup', endHandler);
          target.removeEventListener('touchend', endHandler);
        };
        
        target.addEventListener('mouseup', endHandler, { once: true });
        target.addEventListener('touchend', endHandler, { once: true });
        
        interactionTimeouts.current.set(key, window.setTimeout(() => {
          endHandler();
        }, 10000)); // Max 10 second interaction
      }
    };

    // Track various interaction types
    const eventTypes = ['click', 'mousedown', 'mouseup', 'touchstart', 'touchend', 'keydown'];
    
    eventTypes.forEach(eventType => {
      document.addEventListener(eventType, handleInteraction, { passive: true });
    });

    return () => {
      eventTypes.forEach(eventType => {
        document.removeEventListener(eventType, handleInteraction);
      });
      
      // Clear any pending timeouts
      interactionTimeouts.current.forEach(timeout => clearTimeout(timeout));
      interactionTimeouts.current.clear();
    };
  }, [autoTrackInteractions, samplingRate]);

  // Track performance metrics
  useEffect(() => {
    if (!autoTrackPerformance) return;

    const performanceObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        // Track paint metrics
        if (entry.entryType === 'paint') {
          telemetry.recordMetric(`performance.${entry.name}`, entry.startTime);
        }
        
        // Track layout shifts
        if (entry.entryType === 'layout-shift' && (entry as any).hadRecentInput === false) {
          telemetry.recordMetric('performance.layout_shift', (entry as any).value);
        }
        
        // Track largest contentful paint
        if (entry.entryType === 'largest-contentful-paint') {
          telemetry.recordMetric('performance.lcp', entry.startTime);
        }
      }
    });

    try {
      performanceObserver.observe({ 
        entryTypes: ['paint', 'layout-shift', 'largest-contentful-paint'] 
      });
    } catch (error) {
      logger.warn('Performance observer not fully supported', { error });
    }

    return () => {
      performanceObserver.disconnect();
    };
  }, [autoTrackPerformance]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clear any pending timeouts
      interactionTimeouts.current.forEach(timeout => clearTimeout(timeout));
      interactionTimeouts.current.clear();
    };
  }, []);

  return {
    trackQuantumOperation,
    telemetry
  };
};

// Hook for quantum circuit telemetry
export const useQuantumCircuitTelemetry = () => {
  const { trackQuantumOperation } = useTelemetry();

  const trackCircuitBuild = useCallback(async (
    circuit: CircuitConfig,
    buildFunction: () => Promise<unknown>
  ) => {
    return await trackQuantumOperation('circuit_build', buildFunction, circuit);
  }, [trackQuantumOperation]);

  const trackSimulation = useCallback(async (
    circuit: CircuitConfig,
    simulationFunction: () => Promise<unknown>
  ) => {
    return await trackQuantumOperation('simulation', simulationFunction, circuit);
  }, [trackQuantumOperation]);

  const trackTraining = useCallback(async (
    circuit: CircuitConfig,
    trainingFunction: () => Promise<unknown>
  ) => {
    return await trackQuantumOperation('training', trainingFunction, circuit);
  }, [trackQuantumOperation]);

  const trackVisualization = useCallback(async (
    circuit: CircuitConfig,
    visualizationFunction: () => Promise<unknown>
  ) => {
    return await trackQuantumOperation('visualization', visualizationFunction, circuit);
  }, [trackQuantumOperation]);

  return {
    trackCircuitBuild,
    trackSimulation,
    trackTraining,
    trackVisualization
  };
};

// Hook for performance-critical operations
export const usePerformanceTelemetry = () => {
  const measureOperation = useCallback(async <T>(
    operationName: string,
    operation: () => Promise<T> | T,
    context?: Record<string, unknown>
  ): Promise<T> => {
    return await withTelemetry(operationName, async () => {
      const startMemory = (performance as any).memory?.usedJSHeapSize || 0;
      const result = await operation();
      const endMemory = (performance as any).memory?.usedJSHeapSize || 0;
      
      telemetry.recordMetric(`${operationName}.memory_delta`, endMemory - startMemory);
      
      return result;
    }, context);
  }, []);

  const measureRender = useCallback((componentName: string) => {
    const startTime = performance.now();
    
    return () => {
      const renderTime = performance.now() - startTime;
      telemetry.recordMetric(`render.${componentName}`, renderTime);
      
      if (renderTime > 16) { // > 1 frame at 60fps
        telemetry.trackError('javascript', `Slow render detected: ${componentName}`, {
          render_time: renderTime,
          component: componentName
        });
      }
    };
  }, []);

  return {
    measureOperation,
    measureRender
  };
};

// Utility functions
function getElementInfo(element: HTMLElement): { selector: string; value?: unknown } {
  const selector = generateSelector(element);
  let value;

  if (element instanceof HTMLInputElement) {
    value = element.type === 'password' ? '[REDACTED]' : element.value;
  } else if (element instanceof HTMLSelectElement) {
    value = element.value;
  } else if (element instanceof HTMLTextAreaElement) {
    value = element.value.substring(0, 100); // Limit length
  }

  return { selector, value };
}

function generateSelector(element: HTMLElement): string {
  if (element.id) {
    return `#${element.id}`;
  }
  
  if (element.className) {
    const classes = element.className.split(' ').filter(c => c.length > 0);
    if (classes.length > 0) {
      return `.${classes[0]}`;
    }
  }
  
  const tagName = element.tagName.toLowerCase();
  const role = element.getAttribute('role');
  const dataTestId = element.getAttribute('data-testid');
  
  if (dataTestId) {
    return `[data-testid="${dataTestId}"]`;
  }
  
  if (role) {
    return `${tagName}[role="${role}"]`;
  }
  
  return tagName;
}

function getEventPosition(event: Event): { x: number; y: number } | undefined {
  if ('clientX' in event && 'clientY' in event) {
    return { x: (event as MouseEvent).clientX, y: (event as MouseEvent).clientY };
  }
  
  if ('touches' in event && (event as TouchEvent).touches.length > 0) {
    const touch = (event as TouchEvent).touches[0];
    return { x: touch.clientX, y: touch.clientY };
  }
  
  return undefined;
}

function calculateCircuitDepth(gates: Array<Record<string, unknown>>): number {
  if (!gates || gates.length === 0) return 0;
  
  // Simplified circuit depth calculation
  const qubitTimeline = new Map<number, number>();
  
  for (const gate of gates) {
    const qubits = Array.isArray(gate.qubits) ? gate.qubits : [gate.qubit];
    const maxTime = Math.max(...qubits.map(q => qubitTimeline.get(q) || 0));
    
    qubits.forEach(q => {
      qubitTimeline.set(q, maxTime + 1);
    });
  }
  
  return Math.max(...qubitTimeline.values());
}