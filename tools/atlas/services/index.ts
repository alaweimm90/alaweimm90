/**
 * ATLAS Continuous Optimization Service - Main Entry Point
 * Phase 8: Complete optimization service with monitoring and dashboard
 */

import { ContinuousOptimizer } from './optimizer';
import { RepositoryMonitor } from './monitor';
import { DashboardService } from './dashboard';
import { ConfigLoader, createDefaultConfig } from '../config/loader';

export interface AtlasServices {
  optimizer: ContinuousOptimizer;
  monitor: RepositoryMonitor;
  dashboard: DashboardService;
  config: ConfigLoader;
}

/**
 * Initialize all ATLAS optimization services
 */
export async function initializeAtlasServices(
  configPath?: string
): Promise<AtlasServices> {
  console.log('🚀 Initializing ATLAS Continuous Optimization Services...');

  // Load configuration
  const configLoader = new ConfigLoader(configPath);
  let config;

  try {
    config = await configLoader.load();
    console.log('✅ Configuration loaded successfully');
  } catch (error) {
    console.warn('⚠️  Configuration file not found, creating default configuration...');
    config = createDefaultConfig();
    await configLoader.save(config);
    console.log('✅ Default configuration created');
  }

  // Initialize services
  const optimizer = new ContinuousOptimizer({
    schedule: config.optimizer.schedule,
    thresholds: config.optimizer.thresholds,
    safety: config.optimizer.safety,
    repositories: config.optimizer.repositories
  });

  const monitor = new RepositoryMonitor({
    repositories: config.monitor.repositories || [],
    polling: config.monitor.polling,
    filesystem: config.monitor.filesystem,
    triggers: config.monitor.triggers,
    analysis: config.monitor.analysis
  }, optimizer);

  const dashboard = new DashboardService({
    port: config.dashboard.port,
    host: config.dashboard.host,
    enableWebSocket: config.dashboard.enableWebSocket,
    enableREST: config.dashboard.enableREST,
    telemetry: config.dashboard.telemetry,
    security: config.dashboard.security
  }, optimizer, monitor);

  // Set up service event forwarding
  setupServiceEventForwarding(optimizer, monitor, dashboard);

  console.log('✅ All ATLAS services initialized successfully');
  console.log(`📊 Dashboard available at http://${config.dashboard.host}:${config.dashboard.port}`);
  console.log(`📈 Monitoring ${config.optimizer.repositories.length} repositories`);
  console.log(`⏰ Optimization schedule: ${config.optimizer.schedule.enabled ? 'Enabled' : 'Disabled'}`);

  return {
    optimizer,
    monitor,
    dashboard,
    config: configLoader
  };
}

/**
 * Start all ATLAS services
 */
export async function startAtlasServices(services: AtlasServices): Promise<void> {
  console.log('▶️  Starting ATLAS services...');

  try {
    // Start services in order
    await services.monitor.start();
    console.log('✅ Repository monitor started');

    await services.optimizer.start();
    console.log('✅ Continuous optimizer started');

    await services.dashboard.start();
    console.log('✅ Dashboard service started');

    console.log('🎉 ATLAS Continuous Optimization Service is now running!');
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
    console.error('❌ Failed to start ATLAS services:', error);
    throw error;
  }
}

/**
 * Stop all ATLAS services
 */
export async function stopAtlasServices(services: AtlasServices): Promise<void> {
  console.log('⏹️  Stopping ATLAS services...');

  try {
    await services.dashboard.stop();
    await services.optimizer.stop();
    await services.monitor.stop();

    console.log('✅ All ATLAS services stopped');
  } catch (error) {
    console.error('❌ Error stopping services:', error);
    throw error;
  }
}

/**
 * Get comprehensive system status
 */
export function getAtlasStatus(services: AtlasServices): {
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
      dashboard: services.dashboard.getHealthStatus()
    },
    health: services.dashboard.getHealthStatus(),
    uptime: process.uptime()
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
      tags: ['job', 'start']
    });
  });

  optimizer.on('job:complete', (job) => {
    dashboard.recordEvent({
      type: 'optimization',
      source: 'optimizer',
      data: job,
      severity: 'low',
      tags: ['job', 'complete', 'success']
    });
  });

  optimizer.on('job:fail', (job) => {
    dashboard.recordEvent({
      type: 'error',
      source: 'optimizer',
      data: job,
      severity: 'high',
      tags: ['job', 'fail', 'error']
    });
  });

  // Forward monitor events to dashboard
  monitor.on('analysis:complete', (result) => {
    dashboard.recordEvent({
      type: 'analysis',
      source: 'monitor',
      data: result,
      severity: 'low',
      tags: ['analysis', 'complete']
    });
  });

  monitor.on('analysis:error', (error) => {
    dashboard.recordEvent({
      type: 'error',
      source: 'monitor',
      data: error,
      severity: 'medium',
      tags: ['analysis', 'error']
    });
  });

  monitor.on('repository:add', (data) => {
    dashboard.recordEvent({
      type: 'system',
      source: 'monitor',
      data,
      severity: 'low',
      tags: ['repository', 'add']
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
    const services = await initializeAtlasServices();
    await startAtlasServices(services);

    // Example: Add a repository to monitor
    if (services.optimizer.config.repositories.length === 0) {
      console.log('💡 No repositories configured. Add one with:');
      console.log('   services.monitor.addRepository({');
      console.log('     name: "my-repo",');
      console.log('     path: "/path/to/repo",');
      console.log('     enabled: true,');
      console.log('     branch: "main"');
      console.log('   });');
    }

    // Graceful shutdown handling
    process.on('SIGINT', async () => {
      console.log('\n🛑 Received SIGINT, shutting down gracefully...');
      await stopAtlasServices(services);
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
      await stopAtlasServices(services);
      process.exit(0);
    });

  } catch (error) {
    console.error('❌ Failed to start ATLAS:', error);
    process.exit(1);
  }
}

// Export individual services for advanced usage
export { ContinuousOptimizer } from './optimizer';
export { RepositoryMonitor } from './monitor';
export { DashboardService } from './dashboard';
export { ConfigLoader, createDefaultConfig } from '../config/loader';

// CLI entry point
if (require.main === module) {
  quickStart().catch(console.error);
}