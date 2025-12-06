import { useState, useEffect, useCallback, useRef } from 'react';
import { quantumMetrics, quantumAlerts } from '@/lib/monitoring';
import type { PerformanceMetric, Alert } from '@/lib/monitoring';
import { logger } from '@/lib/logger';

interface DashboardMetrics {
  quantumHealth: number;
  activeAlerts: Alert[];
  recentMetrics: {
    circuitExecutionTime: number[];
    trainingLoss: number[];
    simulationMemory: number[];
    errorRates: number[];
  };
  systemStats: {
    totalCircuits: number;
    totalTraining: number;
    totalSimulations: number;
    uptime: number;
  };
}

interface MonitoringDashboard {
  metrics: DashboardMetrics;
  isLoading: boolean;
  lastUpdate: number;
  refreshInterval: number;
  autoRefresh: boolean;
  refresh: () => void;
  setAutoRefresh: (enabled: boolean) => void;
  setRefreshInterval: (interval: number) => void;
  exportMetrics: (format: 'json' | 'prometheus') => string;
  acknowledgeAlert: (alertId: string) => void;
  getMetricHistory: (metricName: string, hours?: number) => PerformanceMetric[];
}

export const useMonitoringDashboard = (): MonitoringDashboard => {
  const [metrics, setMetrics] = useState<DashboardMetrics>({
    quantumHealth: 100,
    activeAlerts: [],
    recentMetrics: {
      circuitExecutionTime: [],
      trainingLoss: [],
      simulationMemory: [],
      errorRates: []
    },
    systemStats: {
      totalCircuits: 0,
      totalTraining: 0,
      totalSimulations: 0,
      uptime: 0
    }
  });
  
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(0);
  const [refreshInterval, setRefreshIntervalState] = useState(30000); // 30 seconds
  const [autoRefresh, setAutoRefreshState] = useState(true);
  
  const startTime = useRef(Date.now());
  const refreshTimeoutRef = useRef<NodeJS.Timeout>();

  // Collect dashboard metrics
  const collectDashboardMetrics = useCallback((): DashboardMetrics => {
    const now = Date.now();
    const fiveMinutesAgo = now - 5 * 60 * 1000;
    const oneHourAgo = now - 60 * 60 * 1000;

    // Get quantum health score
    const quantumHealth = quantumMetrics.getQuantumHealthScore();

    // Get active alerts
    const activeAlerts = quantumAlerts.getActiveAlerts();

    // Get recent metrics for visualization
    const circuitExecutionMetrics = quantumMetrics.getMetrics('circuit.execution_time', fiveMinutesAgo);
    const trainingLossMetrics = quantumMetrics.getMetrics('training.loss', fiveMinutesAgo);
    const simulationMemoryMetrics = quantumMetrics.getMetrics('simulation.memory_peak', fiveMinutesAgo);
    const errorRateMetrics = quantumMetrics.getMetrics('circuit.error_rate', fiveMinutesAgo);

    // Get system stats
    const allMetrics = quantumMetrics.getMetrics(undefined, oneHourAgo);
    const circuitMetrics = allMetrics.filter(m => m.name.startsWith('circuit.'));
    const trainingMetrics = allMetrics.filter(m => m.name.startsWith('training.'));
    const simulationMetrics = allMetrics.filter(m => m.name.startsWith('simulation.'));

    // Count unique sessions/circuits
    const uniqueCircuits = new Set(circuitMetrics.map(m => m.tags.circuit_id)).size;
    const uniqueTraining = new Set(trainingMetrics.map(m => m.tags.session_id)).size;
    const uniqueSimulations = new Set(simulationMetrics.map(m => m.tags.simulation_id)).size;

    return {
      quantumHealth,
      activeAlerts,
      recentMetrics: {
        circuitExecutionTime: circuitExecutionMetrics.map(m => m.value),
        trainingLoss: trainingLossMetrics.map(m => m.value),
        simulationMemory: simulationMemoryMetrics.map(m => m.value / (1024 * 1024)), // Convert to MB
        errorRates: errorRateMetrics.map(m => m.value * 100) // Convert to percentage
      },
      systemStats: {
        totalCircuits: uniqueCircuits,
        totalTraining: uniqueTraining,
        totalSimulations: uniqueSimulations,
        uptime: now - startTime.current
      }
    };
  }, []);

  // Refresh metrics
  const refresh = useCallback(() => {
    setIsLoading(true);
    
    try {
      const newMetrics = collectDashboardMetrics();
      setMetrics(newMetrics);
      setLastUpdate(Date.now());
    } catch (error) {
      logger.error('Failed to refresh monitoring dashboard', { error });
    } finally {
      setIsLoading(false);
    }
  }, [collectDashboardMetrics]);

  // Set up auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      const scheduleRefresh = () => {
        refreshTimeoutRef.current = setTimeout(() => {
          refresh();
          scheduleRefresh();
        }, refreshInterval);
      };

      scheduleRefresh();

      return () => {
        if (refreshTimeoutRef.current) {
          clearTimeout(refreshTimeoutRef.current);
        }
      };
    }
  }, [autoRefresh, refreshInterval, refresh]);

  // Initial refresh
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Set auto-refresh
  const setAutoRefresh = useCallback((enabled: boolean) => {
    setAutoRefreshState(enabled);
    
    if (!enabled && refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
    }
  }, []);

  // Set refresh interval
  const setRefreshInterval = useCallback((interval: number) => {
    setRefreshIntervalState(interval);
  }, []);

  // Export metrics
  const exportMetrics = useCallback((format: 'json' | 'prometheus' = 'json'): string => {
    return quantumMetrics.exportMetrics(format);
  }, []);

  // Acknowledge alert (mark as silenced)
  const acknowledgeAlert = useCallback((alertId: string) => {
    // Find the alert and mark it as acknowledged
    // This would typically update the alert status in the system
    logger.info('Alert acknowledged', { alertId });
    
    // Refresh to get updated alert status
    refresh();
  }, [refresh]);

  // Get metric history
  const getMetricHistory = useCallback((
    metricName: string, 
    hours: number = 1
  ): PerformanceMetric[] => {
    const startTime = Date.now() - (hours * 60 * 60 * 1000);
    return quantumMetrics.getMetrics(metricName, startTime);
  }, []);

  return {
    metrics,
    isLoading,
    lastUpdate,
    refreshInterval,
    autoRefresh,
    refresh,
    setAutoRefresh,
    setRefreshInterval,
    exportMetrics,
    acknowledgeAlert,
    getMetricHistory
  };
};

// Hook for real-time metric streaming
export const useMetricStream = (
  metricName: string,
  maxDataPoints: number = 50
) => {
  const [dataPoints, setDataPoints] = useState<Array<{ timestamp: number; value: number }>>([]);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const updateDataPoints = () => {
      const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
      const metrics = quantumMetrics.getMetrics(metricName, fiveMinutesAgo);
      
      const newDataPoints = metrics
        .map(m => ({ timestamp: m.timestamp, value: m.value }))
        .slice(-maxDataPoints)
        .sort((a, b) => a.timestamp - b.timestamp);

      setDataPoints(newDataPoints);
      setIsConnected(true);
    };

    // Initial update
    updateDataPoints();

    // Set up polling
    intervalId = setInterval(updateDataPoints, 5000); // Update every 5 seconds

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
      setIsConnected(false);
    };
  }, [metricName, maxDataPoints]);

  return { dataPoints, isConnected };
};

// Hook for alert notifications
export const useAlertNotifications = () => {
  const [notifications, setNotifications] = useState<Alert[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const checkAlerts = () => {
      const activeAlerts = quantumAlerts.getActiveAlerts();
      const newAlerts = activeAlerts.filter(alert => 
        !notifications.some(existing => existing.id === alert.id)
      );

      if (newAlerts.length > 0) {
        setNotifications(prev => [...prev, ...newAlerts].slice(-20)); // Keep last 20
        setUnreadCount(prev => prev + newAlerts.length);

        // Show browser notification if permitted
        if (Notification.permission === 'granted') {
          newAlerts.forEach(alert => {
            new Notification(`QMLab Alert: ${alert.rule.severity}`, {
              body: alert.message,
              icon: '/favicon.ico',
              tag: alert.id
            });
          });
        }
      }
    };

    checkAlerts();
    intervalId = setInterval(checkAlerts, 10000); // Check every 10 seconds

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [notifications]);

  const markAsRead = useCallback((alertId: string) => {
    setNotifications(prev => prev.filter(alert => alert.id !== alertId));
    setUnreadCount(prev => Math.max(0, prev - 1));
  }, []);

  const markAllAsRead = useCallback(() => {
    setNotifications([]);
    setUnreadCount(0);
  }, []);

  // Request notification permission
  const requestNotificationPermission = useCallback(async () => {
    if (Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      return permission === 'granted';
    }
    return Notification.permission === 'granted';
  }, []);

  return {
    notifications,
    unreadCount,
    markAsRead,
    markAllAsRead,
    requestNotificationPermission
  };
};

// Hook for performance insights and recommendations
export const usePerformanceInsights = () => {
  const [insights, setInsights] = useState<Array<{
    type: 'optimization' | 'warning' | 'info';
    title: string;
    description: string;
    impact: 'high' | 'medium' | 'low';
    recommendation: string;
    metric?: string;
    value?: number;
  }>>([]);

  useEffect(() => {
    const analyzePerformance = () => {
      const now = Date.now();
      const oneHourAgo = now - 60 * 60 * 1000;
      const metrics = quantumMetrics.getMetrics(undefined, oneHourAgo);

      const newInsights = [];

      // Analyze circuit execution times
      const executionTimeMetrics = metrics.filter(m => m.name === 'circuit.execution_time');
      if (executionTimeMetrics.length > 0) {
        const avgTime = executionTimeMetrics.reduce((sum, m) => sum + m.value, 0) / executionTimeMetrics.length;
        
        if (avgTime > 5000) {
          newInsights.push({
            type: 'optimization',
            title: 'Slow Circuit Execution',
            description: `Average circuit execution time is ${(avgTime/1000).toFixed(2)}s`,
            impact: 'high',
            recommendation: 'Consider optimizing circuit depth or using faster quantum simulation backends',
            metric: 'circuit.execution_time',
            value: avgTime
          });
        }
      }

      // Analyze memory usage
      const memoryMetrics = metrics.filter(m => m.name === 'simulation.memory_peak');
      if (memoryMetrics.length > 0) {
        const maxMemory = Math.max(...memoryMetrics.map(m => m.value));
        const maxMemoryGB = maxMemory / (1024 * 1024 * 1024);
        
        if (maxMemoryGB > 1.5) {
          newInsights.push({
            type: 'warning',
            title: 'High Memory Usage',
            description: `Peak memory usage reached ${maxMemoryGB.toFixed(2)}GB`,
            impact: 'medium',
            recommendation: 'Consider reducing qubit count or implementing memory-efficient algorithms',
            metric: 'simulation.memory_peak',
            value: maxMemory
          });
        }
      }

      // Analyze error rates
      const errorMetrics = metrics.filter(m => m.name === 'circuit.error_rate');
      if (errorMetrics.length > 0) {
        const avgErrorRate = errorMetrics.reduce((sum, m) => sum + m.value, 0) / errorMetrics.length;
        
        if (avgErrorRate > 0.05) {
          newInsights.push({
            type: 'optimization',
            title: 'High Error Rate',
            description: `Average error rate is ${(avgErrorRate * 100).toFixed(2)}%`,
            impact: 'high',
            recommendation: 'Review circuit implementation and consider error correction techniques',
            metric: 'circuit.error_rate',
            value: avgErrorRate
          });
        }
      }

      // Training insights
      const trainingMetrics = metrics.filter(m => m.name === 'training.loss');
      if (trainingMetrics.length > 5) {
        const recentLoss = trainingMetrics.slice(-5).map(m => m.value);
        const isStagnant = recentLoss.every(loss => Math.abs(loss - recentLoss[0]) < 0.01);
        
        if (isStagnant) {
          newInsights.push({
            type: 'info',
            title: 'Training Progress Stagnant',
            description: 'Loss has not improved significantly in recent epochs',
            impact: 'medium',
            recommendation: 'Consider adjusting learning rate or changing optimization strategy',
            metric: 'training.loss',
            value: recentLoss[recentLoss.length - 1]
          });
        }
      }

      setInsights(newInsights);
    };

    analyzePerformance();
    
    const intervalId = setInterval(analyzePerformance, 60000); // Analyze every minute

    return () => clearInterval(intervalId);
  }, []);

  return insights;
};