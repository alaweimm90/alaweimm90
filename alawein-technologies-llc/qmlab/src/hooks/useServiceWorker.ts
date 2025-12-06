import { useState, useEffect, useCallback } from 'react';
import { trackQuantumEvents } from '@/lib/analytics';
import { logger } from '@/lib/logger';

interface ServiceWorkerState {
  registration: ServiceWorkerRegistration | null;
  isSupported: boolean;
  isRegistered: boolean;
  isUpdateAvailable: boolean;
  isOffline: boolean;
  version: string | null;
}

interface ServiceWorkerActions {
  register: () => Promise<boolean>;
  update: () => Promise<boolean>;
  skipWaiting: () => void;
  clearCache: () => Promise<boolean>;
  cacheQuantumResult: (key: string, result: any) => Promise<boolean>;
}

export const useServiceWorker = (): ServiceWorkerState & ServiceWorkerActions => {
  const [state, setState] = useState<ServiceWorkerState>({
    registration: null,
    isSupported: 'serviceWorker' in navigator,
    isRegistered: false,
    isUpdateAvailable: false,
    isOffline: !navigator.onLine,
    version: null
  });

  // Register service worker
  const register = useCallback(async (): Promise<boolean> => {
    if (!state.isSupported) {
      logger.info('Service Worker not supported');
      return false;
    }

    try {
      const registration = await navigator.serviceWorker.register('/sw.js', {
        scope: '/'
      });

      setState(prev => ({
        ...prev,
        registration,
        isRegistered: true
      }));

      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              setState(prev => ({ ...prev, isUpdateAvailable: true }));
              trackQuantumEvents.featureDiscovery('service_worker_update_available');
            }
          });
        }
      });

      // Listen for controller change (new SW activated)
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
      });

      // Get current version
      const messageChannel = new MessageChannel();
      messageChannel.port1.onmessage = (event) => {
        setState(prev => ({ ...prev, version: event.data.version }));
      };

      if (registration.active) {
        registration.active.postMessage({ type: 'GET_VERSION' }, [messageChannel.port2]);
      }

      trackQuantumEvents.featureDiscovery('service_worker_registered');
      logger.info('Service Worker registered successfully');
      return true;

    } catch (error) {
      logger.error('Service Worker registration failed', { error });
      trackQuantumEvents.errorBoundary(
        'Service Worker registration failed',
        (error as Error).stack || 'No stack trace',
        'service-worker'
      );
      return false;
    }
  }, [state.isSupported]);

  // Update service worker
  const update = useCallback(async (): Promise<boolean> => {
    if (!state.registration) return false;

    try {
      await state.registration.update();
      trackQuantumEvents.featureDiscovery('service_worker_update_checked');
      return true;
    } catch (error) {
      logger.error('Failed to update service worker', { error });
      return false;
    }
  }, [state.registration]);

  // Skip waiting for new service worker
  const skipWaiting = useCallback(() => {
    if (state.registration?.waiting) {
      state.registration.waiting.postMessage({ type: 'SKIP_WAITING' });
      trackQuantumEvents.featureDiscovery('service_worker_skip_waiting');
    }
  }, [state.registration]);

  // Clear all caches
  const clearCache = useCallback(async (): Promise<boolean> => {
    if (!state.registration?.active) return false;

    try {
      const messageChannel = new MessageChannel();
      
      return new Promise((resolve) => {
        messageChannel.port1.onmessage = (event) => {
          resolve(event.data.success);
        };

        state.registration!.active!.postMessage(
          { type: 'CLEAR_CACHE' },
          [messageChannel.port2]
        );
      });
    } catch (error) {
      logger.error('Failed to clear cache', { error });
      return false;
    }
  }, [state.registration]);

  // Cache quantum computation result
  const cacheQuantumResult = useCallback(async (key: string, result: any): Promise<boolean> => {
    if (!state.registration?.active) return false;

    try {
      const messageChannel = new MessageChannel();
      
      return new Promise((resolve) => {
        messageChannel.port1.onmessage = (event) => {
          resolve(event.data.cached);
        };

        state.registration!.active!.postMessage(
          { 
            type: 'CACHE_QUANTUM_RESULT',
            data: { key, result }
          },
          [messageChannel.port2]
        );
      });
    } catch (error) {
      logger.error('Failed to cache quantum result', { error });
      return false;
    }
  }, [state.registration]);

  // Handle online/offline status
  useEffect(() => {
    const handleOnline = () => {
      setState(prev => ({ ...prev, isOffline: false }));
      trackQuantumEvents.preferenceChange('connectivity', 'online', 'offline');
    };

    const handleOffline = () => {
      setState(prev => ({ ...prev, isOffline: true }));
      trackQuantumEvents.preferenceChange('connectivity', 'offline', 'online');
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Listen for service worker messages
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const { type, version } = event.data;

      switch (type) {
        case 'SW_ACTIVATED':
          setState(prev => ({ ...prev, version }));
          trackQuantumEvents.featureDiscovery('service_worker_activated', JSON.stringify({ version }));
          break;
      }
    };

    if (state.isSupported) {
      navigator.serviceWorker.addEventListener('message', handleMessage);

      return () => {
        navigator.serviceWorker.removeEventListener('message', handleMessage);
      };
    }
  }, [state.isSupported]);

  // Auto-register on mount
  useEffect(() => {
    if (state.isSupported && !state.isRegistered) {
      register();
    }
  }, [state.isSupported, state.isRegistered, register]);

  return {
    ...state,
    register,
    update,
    skipWaiting,
    clearCache,
    cacheQuantumResult
  };
};

// Hook for quantum simulation caching
export const useQuantumCache = () => {
  const { cacheQuantumResult, isRegistered } = useServiceWorker();

  const cacheSimulation = useCallback(async (
    circuitConfig: any,
    simulationResult: any
  ): Promise<boolean> => {
    if (!isRegistered) return false;

    // Create cache key from circuit configuration
    const cacheKey = `/api/quantum-simulation?${new URLSearchParams({
      circuit: JSON.stringify(circuitConfig),
      timestamp: Date.now().toString()
    }).toString()}`;

    return await cacheQuantumResult(cacheKey, simulationResult);
  }, [cacheQuantumResult, isRegistered]);

  const getCachedSimulation = useCallback(async (
    circuitConfig: any
  ): Promise<any | null> => {
    if (!isRegistered) return null;

    try {
      const cacheKey = `/api/quantum-simulation?${new URLSearchParams({
        circuit: JSON.stringify(circuitConfig)
      }).toString()}`;

      const cache = await caches.open('qmlab-quantum-v1.0.0');
      const response = await cache.match(cacheKey);

      if (response) {
        const result = await response.json();
        trackQuantumEvents.componentLoad('quantum_cache_hit', 0);
        return result;
      }

      return null;
    } catch (error) {
      logger.error('Failed to get cached simulation', { error });
      return null;
    }
  }, [isRegistered]);

  return {
    cacheSimulation,
    getCachedSimulation,
    isAvailable: isRegistered
  };
};