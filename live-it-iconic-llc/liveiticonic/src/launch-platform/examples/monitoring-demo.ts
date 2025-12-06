#!/usr/bin/env tsx
/**
 * LiveItIconic Launch Platform - Monitoring Demo
 *
 * Demonstrates real-time agent performance monitoring
 */

import { AgentFactory } from '../agents';
import { agentMonitor } from '../monitoring/AgentMonitor';
import { Task, TaskStatus, AgentPriority } from '../types';

/**
 * Demo: Real-Time Agent Performance Monitoring
 */
async function runMonitoringDemo(): Promise<void> {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║    Agent Performance Monitoring - Live Demo               ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  // Create and register agents
  console.log('📋 Step 1: Creating and registering agents...\n');

  const agents = AgentFactory.createAllAgents();
  agentMonitor.registerAgents(agents);

  console.log(`✓ Registered ${agents.length} agents for monitoring\n`);

  // Take initial snapshot
  console.log('📊 Step 2: Taking initial performance snapshot...\n');

  const initialSnapshot = agentMonitor.takeSnapshot();
  console.log(`  Total Agents: ${initialSnapshot.systemMetrics.totalAgents}`);
  console.log(`  Idle Agents: ${initialSnapshot.systemMetrics.idleAgents}`);
  console.log(`  Active Alerts: ${initialSnapshot.alerts.length}\n`);

  // Simulate some agent activity
  console.log('⚡ Step 3: Simulating agent activity...\n');

  // Execute tasks on different agents
  const testAgents = agents.slice(0, 5);

  for (const agent of testAgents) {
    const task: Task = {
      id: `task-${agent.getId()}-${Date.now()}`,
      name: `Test Task for ${agent.getName()}`,
      description: 'Performance monitoring test task',
      assignedTo: agent.getId(),
      status: TaskStatus.PENDING,
      priority: AgentPriority.NORMAL,
      dependencies: [],
      estimatedDuration: Math.random() * 3000 + 1000,
      deliverables: ['test_output'],
      createdAt: new Date(),
      dueDate: new Date(Date.now() + 86400000),
    };

    try {
      console.log(`  Executing task on ${agent.getName()}...`);
      await agent.executeTask(task);
      console.log(`  ✓ Task completed`);
    } catch (error) {
      console.log(`  ✗ Task failed: ${error}`);
    }
  }

  console.log();

  // Take snapshot after activity
  console.log('📊 Step 4: Taking post-activity snapshot...\n');

  const postSnapshot = agentMonitor.takeSnapshot();
  console.log(`  Tasks Completed: ${postSnapshot.systemMetrics.totalTasksCompleted}`);
  console.log(`  Tasks Failed: ${postSnapshot.systemMetrics.totalTasksFailed}`);
  console.log(`  Success Rate: ${(postSnapshot.systemMetrics.averageSuccessRate * 100).toFixed(1)}%\n`);

  // Show top performers
  console.log('🏆 Step 5: Top Performing Agents\n');

  const topPerformers = agentMonitor.getTopPerformers(3);
  topPerformers.forEach((metrics, idx) => {
    console.log(`  ${idx + 1}. ${metrics.agentName}`);
    console.log(`     Success Rate: ${(metrics.successRate * 100).toFixed(1)}%`);
    console.log(`     Tasks Completed: ${metrics.tasksCompleted}`);
    console.log(`     Avg Duration: ${(metrics.averageDuration / 1000).toFixed(2)}s\n`);
  });

  // Check for alerts
  console.log('🚨 Step 6: Performance Alerts\n');

  const alerts = agentMonitor.checkAlerts();
  if (alerts.length > 0) {
    alerts.forEach(alert => {
      const icon =
        alert.level === 'critical' ? '🔴' :
        alert.level === 'error' ? '🟠' :
        alert.level === 'warning' ? '🟡' : '🔵';
      console.log(`  ${icon} ${alert.level.toUpperCase()}: ${alert.message}`);
      if (alert.agentName) {
        console.log(`     Agent: ${alert.agentName}`);
      }
    });
    console.log();
  } else {
    console.log('  ✅ No active alerts\n');
  }

  // Generate full report
  console.log('📈 Step 7: Generating Full Performance Report\n');

  const report = agentMonitor.generateReport();
  console.log(report);

  // Export metrics
  console.log('💾 Step 8: Exporting Metrics to JSON\n');

  const metricsJson = agentMonitor.exportMetrics();
  console.log('  Metrics exported (sample):');
  console.log('  ' + metricsJson.split('\n').slice(0, 5).join('\n  '));
  console.log('  ...(truncated)\n');

  // Show snapshot history
  console.log('📚 Step 9: Snapshot History\n');

  const history = agentMonitor.getSnapshotHistory();
  console.log(`  Total Snapshots: ${history.length}`);
  console.log(`  Earliest: ${history[0]?.timestamp.toLocaleString()}`);
  console.log(`  Latest: ${history[history.length - 1]?.timestamp.toLocaleString()}\n`);

  // Cleanup
  console.log('🧹 Cleanup: Shutting down agents...\n');

  agents.forEach(agent => agent.shutdown());

  console.log('═══════════════════════════════════════════════════════════\n');
  console.log('✅ Monitoring Demo Complete!\n');
  console.log('Key Features Demonstrated:');
  console.log('  ✓ Real-time agent registration');
  console.log('  ✓ Performance snapshot capture');
  console.log('  ✓ Task execution monitoring');
  console.log('  ✓ Top performer identification');
  console.log('  ✓ Alert detection');
  console.log('  ✓ Report generation');
  console.log('  ✓ Metrics export');
  console.log('  ✓ Historical tracking\n');
  console.log('═══════════════════════════════════════════════════════════\n');
}

/**
 * Demo: Live Monitoring Dashboard
 *
 * This starts a continuous monitoring dashboard that updates every 5 seconds.
 * Press Ctrl+C to stop.
 */
async function runLiveDashboard(): Promise<void> {
  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║         Live Agent Performance Dashboard                  ║');
  console.log('║         (Press Ctrl+C to stop)                            ║');
  console.log('╚════════════════════════════════════════════════════════════╝\n');

  // Create and register agents
  const agents = AgentFactory.createAllAgents();
  agentMonitor.registerAgents(agents);

  // Start background activity simulation
  const runBackgroundTasks = async () => {
    // Pick random agents and run tasks
    const agent = agents[Math.floor(Math.random() * agents.length)];

    const task: Task = {
      id: `bg-task-${Date.now()}`,
      name: `Background Task`,
      description: 'Simulated background activity',
      assignedTo: agent.getId(),
      status: TaskStatus.PENDING,
      priority: AgentPriority.LOW,
      dependencies: [],
      estimatedDuration: Math.random() * 5000,
      deliverables: [],
      createdAt: new Date(),
      dueDate: new Date(Date.now() + 86400000),
    };

    try {
      await agent.executeTask(task);
    } catch (error) {
      // Ignore errors in background tasks
    }
  };

  // Run background tasks periodically
  const bgInterval = setInterval(runBackgroundTasks, 3000);

  // Start monitoring dashboard
  const monitorInterval = agentMonitor.startMonitoring(5000);

  // Handle Ctrl+C
  process.on('SIGINT', () => {
    console.log('\n\n⏹️  Stopping dashboard...\n');
    clearInterval(bgInterval);
    agentMonitor.stopMonitoring(monitorInterval);
    agents.forEach(agent => agent.shutdown());
    process.exit(0);
  });
}

/**
 * Main entry point
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const mode = args[0] || 'demo';

  if (mode === 'live' || mode === 'dashboard') {
    await runLiveDashboard();
  } else {
    await runMonitoringDemo();
  }
}

// Run if called directly
if (require.main === module) {
  main()
    .then(() => {
      // Demo mode exits here
    })
    .catch(error => {
      console.error('Error running monitoring demo:', error);
      process.exit(1);
    });
}

export { runMonitoringDemo, runLiveDashboard };
