import { useState, useEffect } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface ChunkLoadingState {
  isLoading: boolean;
  error: Error | null;
  retryCount: number;
}

export const useChunkLoading = () => {
  const [loadingStates, setLoadingStates] = useState<Record<string, ChunkLoadingState>>({});

  const trackChunkLoad = (chunkName: string, startTime: number) => {
    const loadTime = performance.now() - startTime;
    trackQuantumEvents.componentLoad(chunkName, loadTime);
  };

  const trackChunkError = (chunkName: string, error: Error, retryCount: number) => {
    trackQuantumEvents.errorBoundary(
      `Chunk loading failed: ${chunkName}`,
      error.stack || 'No stack trace',
      'chunk-loader'
    );
  };

  const loadChunk = async (chunkName: string, importFn: () => Promise<any>) => {
    const startTime = performance.now();
    
    setLoadingStates(prev => ({
      ...prev,
      [chunkName]: { isLoading: true, error: null, retryCount: 0 }
    }));

    try {
      const module = await importFn();
      trackChunkLoad(chunkName, startTime);
      
      setLoadingStates(prev => ({
        ...prev,
        [chunkName]: { isLoading: false, error: null, retryCount: 0 }
      }));
      
      return module;
    } catch (error) {
      const currentState = loadingStates[chunkName];
      const retryCount = (currentState?.retryCount || 0) + 1;
      
      trackChunkError(chunkName, error as Error, retryCount);
      
      setLoadingStates(prev => ({
        ...prev,
        [chunkName]: { 
          isLoading: false, 
          error: error as Error, 
          retryCount 
        }
      }));
      
      throw error;
    }
  };

  const retryChunkLoad = (chunkName: string, importFn: () => Promise<any>) => {
    return loadChunk(chunkName, importFn);
  };

  return {
    loadingStates,
    loadChunk,
    retryChunkLoad
  };
};

// Preload critical chunks during idle time
export const preloadCriticalChunks = () => {
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      // Preload Three.js chunk when browser is idle
      import('three').catch(() => {
        logger.debug('Three.js preload failed - will load on demand');
      });
    });
  }
};