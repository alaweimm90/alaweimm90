/**
 * Health check endpoint for production monitoring
 */

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: number;
  version: string;
  uptime: number;
  checks: {
    database: boolean;
    storage: boolean;
    network: boolean;
    performance: boolean;
  };
  metrics: {
    errors: number;
    performance: {
      summary: Record<string, number>;
      coreWebVitals: Record<string, number>;
    };
    user: {
      sessionDuration: number;
      interactions: number;
    };
  };
}

/**
 * Health check function that can be called by monitoring services
 */
export const getHealthStatus = async (): Promise<HealthStatus> => {
  try {
    // Get monitoring data if available
    const healthCheck = (window as unknown as { __health_check?: () => Promise<unknown> }).__health_check;
    let monitoringData = null;
    
    if (healthCheck) {
      try {
        monitoringData = await healthCheck();
      } catch (error) {
        console.warn('Failed to get monitoring data:', error);
      }
    }

    // Basic health checks
    const checks = {
      database: true, // Would check Supabase connection in real implementation
      storage: !!window.localStorage,
      network: navigator.onLine,
      performance: performance.now() > 0,
    };

    const allChecksPass = Object.values(checks).every(check => check);
    
    const status: HealthStatus = {
      status: allChecksPass ? 'healthy' : 'degraded',
      timestamp: Date.now(),
      version: '1.0.0',
      uptime: performance.now(),
      checks,
      metrics: {
        errors: monitoringData?.errors || 0,
        performance: monitoringData?.performance || {
          summary: {},
          coreWebVitals: {},
        },
        user: monitoringData?.user || {
          sessionDuration: 0,
          interactions: 0,
        },
      },
    };

    return status;
  } catch (error) {
    console.error('Health check failed:', error);
    
    return {
      status: 'unhealthy',
      timestamp: Date.now(),
      version: '1.0.0',
      uptime: performance.now(),
      checks: {
        database: false,
        storage: false,
        network: false,
        performance: false,
      },
      metrics: {
        errors: 1,
        performance: {
          summary: {},
          coreWebVitals: {},
        },
        user: {
          sessionDuration: 0,
          interactions: 0,
        },
      },
    };
  }
};

/**
 * Simple ping endpoint for basic availability checks
 */
export const ping = (): Promise<{ status: 'ok'; timestamp: number }> => {
  return Promise.resolve({
    status: 'ok',
    timestamp: Date.now(),
  });
};

/**
 * Expose health endpoints globally for monitoring services
 */
if (typeof window !== 'undefined') {
  (window as unknown as { __llm_works_health?: unknown }).__llm_works_health = {
    getHealthStatus,
    ping,
  };
}