/**
 * Admin Monitoring API Routes
 * Provides endpoints for system monitoring, health checks, and observability
 */

import { performHealthCheck, getHealthCheckHistory, getRecentErrors } from '@/lib/healthCheck';
import { perfMonitor } from '@/lib/performance-monitor';
import { alertManager } from '@/lib/alerts';
import { uptimeMonitor } from '@/lib/uptime';

/**
 * GET /api/admin/monitoring/health
 * Get current system health status
 */
export async function getSystemHealth() {
  const health = await performHealthCheck();

  return {
    status: 200,
    data: {
      status: health.status,
      services: health.services,
      metrics: health.metrics,
      timestamp: health.timestamp,
      details: health.details,
    },
  };
}

/**
 * GET /api/admin/monitoring/metrics
 * Get performance metrics and Web Vitals
 */
export async function getPerformanceMetrics() {
  const metrics = perfMonitor.getMetrics();
  const webVitals = perfMonitor.getWebVitals();
  const summary = perfMonitor.getSummary();

  return {
    status: 200,
    data: {
      metrics,
      webVitals,
      summary,
      timestamp: Date.now(),
    },
  };
}

/**
 * GET /api/admin/monitoring/alerts
 * Get all alerts with optional filtering
 */
export async function getAlerts(query?: { severity?: string; resolved?: boolean }) {
  const alerts = alertManager.getAlerts({
    severity: query?.severity,
    resolved: query?.resolved,
  });

  const stats = alertManager.getStatistics();

  return {
    status: 200,
    data: {
      alerts: alerts.slice(0, 100), // Return latest 100 alerts
      statistics: stats,
      timestamp: Date.now(),
    },
  };
}

/**
 * POST /api/admin/monitoring/alerts
 * Create a new alert (internal use)
 */
export async function createAlert(alert: {
  severity: string;
  title: string;
  message: string;
  source: string;
  metadata?: Record<string, unknown>;
}) {
  try {
    const newAlert = alertManager.addAlert({
      severity: alert.severity as 'info' | 'warning' | 'error' | 'critical',
      title: alert.title,
      message: alert.message,
      source: alert.source,
      metadata: alert.metadata,
    });

    return {
      status: 201,
      data: newAlert,
    };
  } catch (error) {
    return {
      status: 400,
      error: error instanceof Error ? error.message : 'Failed to create alert',
    };
  }
}

/**
 * PUT /api/admin/monitoring/alerts/:alertId
 * Resolve an alert
 */
export async function resolveAlert(alertId: string) {
  try {
    const resolved = alertManager.resolveAlert(alertId);

    if (!resolved) {
      return {
        status: 404,
        error: 'Alert not found',
      };
    }

    return {
      status: 200,
      data: { message: 'Alert resolved successfully' },
    };
  } catch (error) {
    return {
      status: 400,
      error: error instanceof Error ? error.message : 'Failed to resolve alert',
    };
  }
}

/**
 * GET /api/admin/monitoring/uptime
 * Get uptime report and statistics
 */
export async function getUptimeReport() {
  const report = uptimeMonitor.getReport();
  const stats = uptimeMonitor.getDowntimeStats();
  const recentEvents = uptimeMonitor.getRecentDowntimeEvents(10);

  return {
    status: 200,
    data: {
      report,
      statistics: stats,
      recentDowntimeEvents: recentEvents,
      timestamp: Date.now(),
    },
  };
}

/**
 * GET /api/admin/monitoring/health-history
 * Get health check history
 */
export async function getHealthCheckHistory() {
  const history = getHealthCheckHistory();

  return {
    status: 200,
    data: history,
  };
}

/**
 * GET /api/admin/monitoring/errors
 * Get recent errors
 */
export async function getRecentErrorsEndpoint(limit: number = 20) {
  const errors = getRecentErrors(limit);

  return {
    status: 200,
    data: {
      errors,
      count: errors.length,
      timestamp: Date.now(),
    },
  };
}

/**
 * GET /api/admin/monitoring/report
 * Get comprehensive monitoring report
 */
export async function getMonitoringReport() {
  const health = await performHealthCheck();
  const metrics = perfMonitor.getMetrics();
  const webVitals = perfMonitor.getWebVitals();
  const perfSummary = perfMonitor.getSummary();
  const uptime = uptimeMonitor.getReport();
  const alerts = alertManager.getAlerts();
  const alertStats = alertManager.getStatistics();

  return {
    status: 200,
    data: {
      generatedAt: new Date().toISOString(),
      system: {
        health,
        uptime,
      },
      performance: {
        metrics,
        webVitals,
        summary: perfSummary,
      },
      alerts: {
        alerts: alerts.slice(0, 50),
        statistics: alertStats,
      },
    },
  };
}

/**
 * GET /api/admin/monitoring/export
 * Export monitoring data in requested format
 */
export async function exportMonitoringData(format: 'json' | 'csv' = 'json') {
  try {
    const report = await getMonitoringReport();

    if (format === 'csv') {
      // Convert to CSV format
      const csv = convertToCSV(report.data);
      return {
        status: 200,
        data: csv,
        headers: {
          'Content-Type': 'text/csv',
          'Content-Disposition': 'attachment; filename="monitoring-report.csv"',
        },
      };
    }

    return {
      status: 200,
      data: report.data,
      headers: {
        'Content-Type': 'application/json',
        'Content-Disposition': 'attachment; filename="monitoring-report.json"',
      },
    };
  } catch (error) {
    return {
      status: 400,
      error: error instanceof Error ? error.message : 'Failed to export data',
    };
  }
}

/**
 * POST /api/admin/monitoring/test-health
 * Test health check functionality
 */
export async function testHealthCheck() {
  try {
    const health = await performHealthCheck();

    return {
      status: 200,
      data: {
        message: 'Health check test completed',
        result: health,
      },
    };
  } catch (error) {
    return {
      status: 400,
      error: error instanceof Error ? error.message : 'Health check test failed',
    };
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert monitoring data to CSV format
 */
function convertToCSV(data: Record<string, unknown>): string {
  const rows: string[] = [];

  // Add header
  rows.push('Monitoring Report Export');
  rows.push(`Generated: ${new Date().toISOString()}`);
  rows.push('');

  // Add system health
  rows.push('System Health');
  rows.push(`Status,${data.system.health.status}`);
  rows.push(`Response Time (ms),${data.system.health.metrics.responseTime}`);
  rows.push(`Error Rate (%),${data.system.health.metrics.errorRate}`);
  rows.push(`Uptime (%),${data.system.health.metrics.uptime}`);
  rows.push('');

  // Add service status
  rows.push('Services');
  rows.push('Service,Status');
  Object.entries(data.system.health.services).forEach(([name, status]) => {
    rows.push(`${name},${status ? 'Operational' : 'Down'}`);
  });
  rows.push('');

  // Add uptime statistics
  rows.push('Uptime Statistics');
  rows.push(`Total Uptime (%),${data.system.uptime.uptime.toFixed(2)}`);
  rows.push(`Running Duration,${data.system.uptime.days}d ${data.system.uptime.hours}h`);
  rows.push('');

  // Add alerts
  rows.push('Active Alerts');
  rows.push(`Total,${data.alerts.statistics.activeAlerts}`);
  rows.push(`Critical,${data.alerts.statistics.criticalAlerts}`);
  rows.push('');

  return rows.join('\n');
}

export default {
  getSystemHealth,
  getPerformanceMetrics,
  getAlerts,
  createAlert,
  resolveAlert,
  getUptimeReport,
  getHealthCheckHistory,
  getRecentErrorsEndpoint,
  getMonitoringReport,
  exportMonitoringData,
  testHealthCheck,
};
