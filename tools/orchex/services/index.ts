/**
 * ORCHEX Continuous Optimization Service - Main Entry Point
 * Phase 8: Complete optimization service with monitoring and dashboard
 */

/* eslint-disable @typescript-eslint/no-explicit-any */
// Service index handles dynamic initialization and configuration

import { ContinuousOptimizer } from './optimizer';
import { RepositoryMonitor } from './monitor';
import { DashboardService } from './dashboard';
import { ConfigLoader, createDefaultConfig } from '@ORCHEX/config/loader';

export interface OrchexServices {
  optimizer: ContinuousOptimizer;
  monitor: RepositoryMonitor;
  dashboard: DashboardService;
  config: ConfigLoader;
}

/**
 * Initialize all ORCHEX optimization services
 */
export async function initializeOrchexServices(configPath?: string): Promise<OrchexServices> {
  console.log('🚀 Initializing ORCHEX Continuous Optimization Services...');

  // Load configuration
  const configLoader = new ConfigLoader(configPath);
  let config;

  try {
    config = await configLoader.load();
    console.log('✅ Configuration loaded successfully');
  } catch {
    console.warn('⚠️  Configuration file not found, creating default configuration...');
    config = createDefaultConfig();
    await configLoader.save(config);
    console.log('✅ Default configuration created');
  }

  // Initialize services
  const optimizer = new ContinuousOptimizer({
    schedule: {
      interval: config.optimizer.schedule.intervalMinutes,
      enabled: config.optimizer.schedule.enabled,
      maxConcurrent: config.optimizer.schedule.maxConcurrentJobs,
    },
    thresholds: config.optimizer.thresholds,
    safety: {
      rateLimit: config.optimizer.safety.rateLimitPerHour,
      circuitBreakerThreshold: config.optimizer.safety.circuitBreakerThreshold,
      rollbackEnabled: config.optimizer.safety.rollbackEnabled,
      manualOverride: config.optimizer.safety.manualOverride,
    },
    repositories: config.optimizer.repositories.map((repo: any) => ({
      ...repo,
      optimizationHistory: repo.optimizationHistory || [],
    })),
  });

  const monitor = new RepositoryMonitor(
    {
      repositories: config.optimizer.repositories.map((r: any) => ({
        path: r.path,
        name: r.name,
        enabled: r.enabled,
        branch: r.branch || 'main',
        changeCount: 0,
        watchers: [],
        watcherAbortControllers: new Map(),
      })),
      polling: {
        enabled: config.monitor.polling.enabled,
        interval: config.monitor.polling.intervalSeconds,
      },
      filesystem: config.monitor.filesystem,
      triggers: config.monitor.triggers,
      analysis: {
        incremental: config.monitor.analysis.incremental,
        cacheResults: config.monitor.analysis.cacheResults,
        cacheTtl: config.monitor.analysis.cacheTtlMinutes,
      },
    },
    optimizer
  );

  const dashboard = new DashboardService(
    {
      port: config.dashboard.port,
      host: config.dashboard.host,
      enableWebSocket: config.dashboard.enableWebSocket,
      enableREST: config.dashboard.enableREST,
      telemetry: {
        retentionPeriod: config.dashboard.telemetry.retentionPeriodDays,
        maxEvents: config.dashboard.telemetry.maxEvents,
        enableMetrics: config.dashboard.telemetry.enableMetrics,
      },
      security: config.dashboard.security,
    },
    optimizer,
    monitor
  );

  // Set up service event forwarding
  setupServiceEventForwarding(optimizer, monitor, dashboard);

  console.log('✅ All ORCHEX services initialized successfully');
  console.log(`📊 Dashboard available at http://${config.dashboard.host}:${config.dashboard.port}`);
  console.log(`📈 Monitoring ${config.optimizer.repositories.length} repositories`);
  console.log(
    `⏰ Optimization schedule: ${config.optimizer.schedule.enabled ? 'Enabled' : 'Disabled'}`
  );

  return {
    optimizer,
    monitor,
    dashboard,
    config: configLoader,
  };
}

/**
 * Start all ORCHEX services
 */
export async function startOrchexServices(services: OrchexServices): Promise<void> {
  console.log('▶️  Starting ORCHEX services...');

  try {
    // Start services in order
    await services.monitor.start();
    console.log('✅ Repository monitor started');

    await services.optimizer.start();
    console.log('✅ Continuous optimizer started');

    await services.dashboard.start();
    console.log('✅ Dashboard service started');

    console.log('🎉 ORCHEX Continuous Optimization Service is now running!');
    console.log('');
    console.log('Available endpoints:');
    console.log('  GET  /api/dashboard - Get current dashboard data');
    console.log('  GET  /api/events - Get telemetry events');
    console.log('  GET  /api/health - Get system health status');
    console.log('  WS   / - Real-time dashboard updates');
    console.log('');
    console.log('Service controls:');
    console.log('  optimizer.optimizeRepository(path) - Manual optimization');
    console.log('  monitor.triggerAnalysis(path) - Manual analysis');
    console.log('  services.stop() - Stop all services');
  } catch (error) {
    console.error('❌ Failed to start ORCHEX services:', error);
    throw error;
  }
}

/**
 * Stop all ORCHEX services
 */
export async function stopOrchexServices(services: OrchexServices): Promise<void> {
  console.log('⏹️  Stopping ORCHEX services...');

  try {
    await services.dashboard.stop();
    await services.optimizer.stop();
    await services.monitor.stop();

    console.log('✅ All ORCHEX services stopped');
  } catch (error) {
    console.error('❌ Error stopping services:', error);
    throw error;
  }
}

/**
 * Get comprehensive system status
 */
export function getOrchexStatus(services: OrchexServices): {
  services: {
    optimizer: any;
    monitor: any;
    dashboard: any;
  };
  health: any;
  uptime: number;
} {
  return {
    services: {
      optimizer: services.optimizer.getStatus(),
      monitor: services.monitor.getStatus(),
      dashboard: services.dashboard.getHealthStatus(),
    },
    health: services.dashboard.getHealthStatus(),
    uptime: process.uptime(),
  };
}

/**
 * Set up event forwarding between services
 */
function setupServiceEventForwarding(
  optimizer: ContinuousOptimizer,
  monitor: RepositoryMonitor,
  dashboard: DashboardService
): void {
  // Forward optimizer events to dashboard
  optimizer.on('job:start', (job) => {
    dashboard.recordEvent({
      type: 'optimization',
      source: 'optimizer',
      data: job,
      severity: 'low',
      tags: ['job', 'start'],
    });
  });

  optimizer.on('job:complete', (job) => {
    dashboard.recordEvent({
      type: 'optimization',
      source: 'optimizer',
      data: job,
      severity: 'low',
      tags: ['job', 'complete', 'success'],
    });
  });

  optimizer.on('job:fail', (job) => {
    dashboard.recordEvent({
      type: 'error',
      source: 'optimizer',
      data: job,
      severity: 'high',
      tags: ['job', 'fail', 'error'],
    });
  });

  // Forward monitor events to dashboard
  monitor.on('analysis:complete', (result) => {
    dashboard.recordEvent({
      type: 'analysis',
      source: 'monitor',
      data: result,
      severity: 'low',
      tags: ['analysis', 'complete'],
    });
  });

  monitor.on('analysis:error', (error) => {
    dashboard.recordEvent({
      type: 'error',
      source: 'monitor',
      data: error,
      severity: 'medium',
      tags: ['analysis', 'error'],
    });
  });

  monitor.on('repository:add', (data) => {
    dashboard.recordEvent({
      type: 'system',
      source: 'monitor',
      data,
      severity: 'low',
      tags: ['repository', 'add'],
    });
  });

  // Forward dashboard events for logging
  dashboard.on('telemetry:event', (event) => {
    // Could forward to external logging systems
    if (event.severity === 'high' || event.severity === 'critical') {
      console.warn(`🚨 ${event.type.toUpperCase()}:`, event.data);
    }
  });
}

/**
 * Quick start example
 */
export async function quickStart(): Promise<void> {
  try {
    const services = await initializeOrchexServices();
    await startOrchexServices(services);

    // Example: Add a repository to monitor
    // TODO: Add a public method to check if repositories are configured
    console.log('💡 To add repositories, use:');
    console.log('   services.monitor.addRepository({');
    console.log('     name: "my-repo",');
    console.log('     path: "/path/to/repo",');
    console.log('     enabled: true,');
    console.log('     branch: "main"');
    console.log('   });');

    // Graceful shutdown handling
    process.on('SIGINT', async () => {
      console.log('\n🛑 Received SIGINT, shutting down gracefully...');
      await stopOrchexServices(services);
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
      await stopOrchexServices(services);
      process.exit(0);
    });
  } catch (error) {
    console.error('❌ Failed to start ORCHEX:', error);
    process.exit(1);
  }
}

// Export individual services for advanced usage
export { ContinuousOptimizer } from './optimizer';
export { RepositoryMonitor } from './monitor';
export { DashboardService } from './dashboard';
export { ConfigLoader, createDefaultConfig } from '@ORCHEX/config/loader';

// CLI entry point
if (require.main === module) {
  quickStart().catch(console.error);
}
