/**
 * React Hook for WebWorker Physics Integration
 * Provides easy-to-use interface for physics calculations using WebWorkers
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { WebWorkerPhysicsEngine, type WorkerTask, type WorkerResult } from '@/lib/webworker-physics-engine';

export interface UseWebWorkerPhysicsOptions {
  autoInitialize?: boolean;
  maxWorkers?: number;
  onError?: (error: Error) => void;
  onProgress?: (progress: number) => void;
}

export interface PhysicsCalculationState<T = any> {
  data: T | null;
  loading: boolean;
  error: string | null;
  progress: number;
  computationTime?: number;
}

export function useWebWorkerPhysics(options: UseWebWorkerPhysicsOptions = {}) {
  const [isInitialized, setIsInitialized] = useState(false);
  const engineRef = useRef<WebWorkerPhysicsEngine | null>(null);
  const { autoInitialize = true, maxWorkers, onError, onProgress } = options;

  // Initialize the physics engine
  useEffect(() => {
    if (autoInitialize && !engineRef.current) {
      try {
        engineRef.current = WebWorkerPhysicsEngine.getInstance();
        if (maxWorkers) {
          engineRef.current.scaleWorkers(maxWorkers);
        }
        setIsInitialized(true);
      } catch (error) {
        onError?.(error as Error);
      }
    }

    return () => {
      if (engineRef.current) {
        engineRef.current.shutdown();
        engineRef.current = null;
      }
    };
  }, [autoInitialize, maxWorkers, onError]);

  // Generic calculation hook
  const useCalculation = <T,>(
    calculationType: 'graphene' | 'montecarlo' | 'matrix' | 'quantum',
    dependencies: any[] = []
  ) => {
    const [state, setState] = useState<PhysicsCalculationState<T>>({
      data: null,
      loading: false,
      error: null,
      progress: 0
    });

    const calculate = useCallback(async (params: any) => {
      if (!engineRef.current || !isInitialized) {
        setState(prev => ({ ...prev, error: 'Science engine not initialized' }));
        return;
      }

      setState(prev => ({ ...prev, loading: true, error: null, progress: 0 }));

      try {
        let result: T;
        const startTime = performance.now();

        switch (calculationType) {
          case 'graphene':
            result = await engineRef.current.calculateGrapheneBandStructure(
              params.kPoints,
              params.parameters,
              { 
                priority: 'medium',
                timeout: 30000
              }
            ) as T;
            break;

          case 'montecarlo':
            result = await engineRef.current.runMonteCarloSimulation(
              params.parameters.systemSize || 50,
              params.parameters.temperature || 2.5,
              params.steps || 1000
            ) as T;
            break;

          case 'matrix':
            result = await engineRef.current.diagonalizeMatrix(
              params.matrix,
              { priority: 'high', timeout: 15000 }
            ) as T;
            break;

          case 'quantum':
            // Placeholder for quantum evolution calculations
            result = null as T;
            break;

          default:
            throw new Error(`Unknown calculation type: ${calculationType}`);
        }

        const computationTime = performance.now() - startTime;

        setState({
          data: result,
          loading: false,
          error: null,
          progress: 100,
          computationTime
        });

      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Calculation failed';
        setState(prev => ({
          ...prev,
          loading: false,
          error: errorMessage,
          progress: 0
        }));
        onError?.(error as Error);
      }
    }, [calculationType, isInitialized, ...dependencies]);

    const reset = useCallback(() => {
      setState({
        data: null,
        loading: false,
        error: null,
        progress: 0
      });
    }, []);

    return { ...state, calculate, reset };
  };

  // Specific calculation hooks
  const useGrapheneBandStructure = (dependencies: any[] = []) => 
    useCalculation<{
      energyPlus: number[];
      energyMinus: number[];
      computationTime: number;
    }>('graphene', dependencies);

  const useMonteCarloSimulation = (dependencies: any[] = []) =>
    useCalculation<{
      trajectory: Array<{ step: number; energy: number; magnetization: number }>;
      averageEnergy: number;
      averageMagnetization: number;
      computationTime: number;
    }>('montecarlo', dependencies);

  const useMatrixDiagonalization = (dependencies: any[] = []) =>
    useCalculation<{
      eigenvalues: number[];
      eigenvectors: number[][];
      computationTime: number;
    }>('matrix', dependencies);

  // Get engine statistics
  const getStats = useCallback(() => {
    return engineRef.current?.getStats() || null;
  }, []);

  // Scale workers
  const scaleWorkers = useCallback((count: number) => {
    if (engineRef.current) {
      engineRef.current.scaleWorkers(count);
    }
  }, []);

  return {
    isInitialized,
    useGrapheneBandStructure,
    useMonteCarloSimulation,
    useMatrixDiagonalization,
    getStats,
    scaleWorkers
  };
}

export default useWebWorkerPhysics;