/**
 * LiveItIconic Launch Platform - Agent Performance Monitor
 *
 * Real-time monitoring and analytics for agent performance
 */

import { BaseAgent } from '../core/BaseAgent';
import { stateManager } from '../core/StateManager';
import { AgentStatus, AgentType } from '../types';

export interface AgentMetrics {
  agentId: string;
  agentName: string;
  agentType: AgentType;
  status: AgentStatus;
  tasksCompleted: number;
  tasksInProgress: number;
  tasksFailed: number;
  successRate: number;
  averageDuration: number;
  lastActivity: Date;
  uptime: number; // in milliseconds
}

export interface SystemMetrics {
  totalAgents: number;
  activeAgents: number;
  idleAgents: number;
  busyAgents: number;
  failedAgents: number;
  totalTasksCompleted: number;
  totalTasksFailed: number;
  averageSuccessRate: number;
  systemUptime: number;
}

export interface AgentPerformanceSnapshot {
  timestamp: Date;
  systemMetrics: SystemMetrics;
  agentMetrics: AgentMetrics[];
  topPerformers: AgentMetrics[];
  underperformers: AgentMetrics[];
  alerts: PerformanceAlert[];
}

export interface PerformanceAlert {
  level: 'info' | 'warning' | 'error' | 'critical';
  agentId?: string;
  agentName?: string;
  message: string;
  timestamp: Date;
  metric?: string;
  value?: number;
  threshold?: number;
}

export class AgentMonitor {
  private agents: Map<string, BaseAgent> = new Map();
  private startTime: Date = new Date();
  private snapshotHistory: AgentPerformanceSnapshot[] = [];
  private maxHistorySize: number = 100;
  private alertThresholds = {
    minSuccessRate: 0.7,
    maxAverageDuration: 30000, // 30 seconds
    maxTasksFailed: 10,
  };

  /**
   * Register agents for monitoring
   */
  registerAgent(agent: BaseAgent): void {
    this.agents.set(agent.getId(), agent);
  }

  /**
   * Register multiple agents
   */
  registerAgents(agents: BaseAgent[]): void {
    agents.forEach(agent => this.registerAgent(agent));
  }

  /**
   * Get metrics for a specific agent
   */
  getAgentMetrics(agentId: string): AgentMetrics | null {
    const agent = this.agents.get(agentId);
    if (!agent) return null;

    const state = agent.getState();

    return {
      agentId: agent.getId(),
      agentName: agent.getName(),
      agentType: agent.getType(),
      status: state.status,
      tasksCompleted: state.metrics.tasksCompleted,
      tasksInProgress: state.metrics.tasksInProgress,
      tasksFailed: state.metrics.tasksFailed,
      successRate: state.metrics.successRate,
      averageDuration: state.metrics.averageDuration,
      lastActivity: state.lastActivity,
      uptime: Date.now() - this.startTime.getTime(),
    };
  }

  /**
   * Get metrics for all agents
   */
  getAllAgentMetrics(): AgentMetrics[] {
    const metrics: AgentMetrics[] = [];

    for (const [agentId, agent] of this.agents) {
      const agentMetrics = this.getAgentMetrics(agentId);
      if (agentMetrics) {
        metrics.push(agentMetrics);
      }
    }

    return metrics;
  }

  /**
   * Get system-wide metrics
   */
  getSystemMetrics(): SystemMetrics {
    const allMetrics = this.getAllAgentMetrics();

    const activeAgents = allMetrics.filter(m =>
      m.status === AgentStatus.EXECUTING || m.status === AgentStatus.THINKING
    ).length;

    const idleAgents = allMetrics.filter(m => m.status === AgentStatus.IDLE).length;

    const busyAgents = allMetrics.filter(m =>
      m.status === AgentStatus.EXECUTING ||
      m.status === AgentStatus.THINKING ||
      m.status === AgentStatus.WAITING
    ).length;

    const failedAgents = allMetrics.filter(m => m.status === AgentStatus.FAILED).length;

    const totalTasksCompleted = allMetrics.reduce(
      (sum, m) => sum + m.tasksCompleted,
      0
    );

    const totalTasksFailed = allMetrics.reduce((sum, m) => sum + m.tasksFailed, 0);

    const averageSuccessRate =
      allMetrics.length > 0
        ? allMetrics.reduce((sum, m) => sum + m.successRate, 0) / allMetrics.length
        : 0;

    return {
      totalAgents: this.agents.size,
      activeAgents,
      idleAgents,
      busyAgents,
      failedAgents,
      totalTasksCompleted,
      totalTasksFailed,
      averageSuccessRate,
      systemUptime: Date.now() - this.startTime.getTime(),
    };
  }

  /**
   * Get top performing agents
   */
  getTopPerformers(count: number = 5): AgentMetrics[] {
    const allMetrics = this.getAllAgentMetrics();

    return allMetrics
      .filter(m => m.tasksCompleted > 0)
      .sort((a, b) => {
        // Sort by success rate first, then by tasks completed
        if (Math.abs(a.successRate - b.successRate) > 0.01) {
          return b.successRate - a.successRate;
        }
        return b.tasksCompleted - a.tasksCompleted;
      })
      .slice(0, count);
  }

  /**
   * Get underperforming agents
   */
  getUnderperformers(count: number = 5): AgentMetrics[] {
    const allMetrics = this.getAllAgentMetrics();

    return allMetrics
      .filter(m => m.tasksCompleted > 0)
      .sort((a, b) => {
        // Sort by success rate (ascending), then by average duration (descending)
        if (Math.abs(a.successRate - b.successRate) > 0.01) {
          return a.successRate - b.successRate;
        }
        return b.averageDuration - a.averageDuration;
      })
      .slice(0, count);
  }

  /**
   * Check for performance alerts
   */
  checkAlerts(): PerformanceAlert[] {
    const alerts: PerformanceAlert[] = [];
    const allMetrics = this.getAllAgentMetrics();

    allMetrics.forEach(metrics => {
      // Check success rate
      if (
        metrics.tasksCompleted > 5 &&
        metrics.successRate < this.alertThresholds.minSuccessRate
      ) {
        alerts.push({
          level: 'warning',
          agentId: metrics.agentId,
          agentName: metrics.agentName,
          message: `Low success rate: ${(metrics.successRate * 100).toFixed(1)}%`,
          timestamp: new Date(),
          metric: 'successRate',
          value: metrics.successRate,
          threshold: this.alertThresholds.minSuccessRate,
        });
      }

      // Check average duration
      if (
        metrics.tasksCompleted > 0 &&
        metrics.averageDuration > this.alertThresholds.maxAverageDuration
      ) {
        alerts.push({
          level: 'info',
          agentId: metrics.agentId,
          agentName: metrics.agentName,
          message: `High average duration: ${(metrics.averageDuration / 1000).toFixed(1)}s`,
          timestamp: new Date(),
          metric: 'averageDuration',
          value: metrics.averageDuration,
          threshold: this.alertThresholds.maxAverageDuration,
        });
      }

      // Check failed tasks
      if (metrics.tasksFailed > this.alertThresholds.maxTasksFailed) {
        alerts.push({
          level: 'error',
          agentId: metrics.agentId,
          agentName: metrics.agentName,
          message: `High failure count: ${metrics.tasksFailed} failed tasks`,
          timestamp: new Date(),
          metric: 'tasksFailed',
          value: metrics.tasksFailed,
          threshold: this.alertThresholds.maxTasksFailed,
        });
      }

      // Check for failed status
      if (metrics.status === AgentStatus.FAILED) {
        alerts.push({
          level: 'critical',
          agentId: metrics.agentId,
          agentName: metrics.agentName,
          message: `Agent in failed state`,
          timestamp: new Date(),
        });
      }
    });

    // Check system-wide metrics
    const systemMetrics = this.getSystemMetrics();

    if (systemMetrics.averageSuccessRate < 0.5 && systemMetrics.totalTasksCompleted > 10) {
      alerts.push({
        level: 'critical',
        message: `System-wide low success rate: ${(systemMetrics.averageSuccessRate * 100).toFixed(1)}%`,
        timestamp: new Date(),
        metric: 'systemSuccessRate',
        value: systemMetrics.averageSuccessRate,
        threshold: 0.5,
      });
    }

    return alerts;
  }

  /**
   * Take a performance snapshot
   */
  takeSnapshot(): AgentPerformanceSnapshot {
    const snapshot: AgentPerformanceSnapshot = {
      timestamp: new Date(),
      systemMetrics: this.getSystemMetrics(),
      agentMetrics: this.getAllAgentMetrics(),
      topPerformers: this.getTopPerformers(5),
      underperformers: this.getUnderperformers(5),
      alerts: this.checkAlerts(),
    };

    // Store in history
    this.snapshotHistory.push(snapshot);

    // Limit history size
    if (this.snapshotHistory.length > this.maxHistorySize) {
      this.snapshotHistory.shift();
    }

    return snapshot;
  }

  /**
   * Get snapshot history
   */
  getSnapshotHistory(count?: number): AgentPerformanceSnapshot[] {
    if (count) {
      return this.snapshotHistory.slice(-count);
    }
    return [...this.snapshotHistory];
  }

  /**
   * Generate performance report
   */
  generateReport(): string {
    const snapshot = this.takeSnapshot();
    const system = snapshot.systemMetrics;

    let report = '\n';
    report += '╔════════════════════════════════════════════════════════════╗\n';
    report += '║       Agent Performance Monitor - Real-Time Report        ║\n';
    report += '╚════════════════════════════════════════════════════════════╝\n\n';

    // System Overview
    report += '📊 SYSTEM OVERVIEW\n\n';
    report += `  Total Agents: ${system.totalAgents}\n`;
    report += `  Active: ${system.activeAgents} | Idle: ${system.idleAgents} | Busy: ${system.busyAgents} | Failed: ${system.failedAgents}\n`;
    report += `  Tasks Completed: ${system.totalTasksCompleted} | Failed: ${system.totalTasksFailed}\n`;
    report += `  Average Success Rate: ${(system.averageSuccessRate * 100).toFixed(1)}%\n`;
    report += `  System Uptime: ${(system.systemUptime / 1000 / 60).toFixed(1)} minutes\n\n`;

    // Top Performers
    if (snapshot.topPerformers.length > 0) {
      report += '🏆 TOP PERFORMERS\n\n';
      snapshot.topPerformers.forEach((agent, idx) => {
        report += `  ${idx + 1}. ${agent.agentName}\n`;
        report += `     Success Rate: ${(agent.successRate * 100).toFixed(1)}% | `;
        report += `Tasks: ${agent.tasksCompleted} | `;
        report += `Avg Duration: ${(agent.averageDuration / 1000).toFixed(2)}s\n`;
      });
      report += '\n';
    }

    // Underperformers
    if (snapshot.underperformers.length > 0) {
      report += '⚠️  NEEDS ATTENTION\n\n';
      snapshot.underperformers.forEach((agent, idx) => {
        report += `  ${idx + 1}. ${agent.agentName}\n`;
        report += `     Success Rate: ${(agent.successRate * 100).toFixed(1)}% | `;
        report += `Failed: ${agent.tasksFailed} | `;
        report += `Avg Duration: ${(agent.averageDuration / 1000).toFixed(2)}s\n`;
      });
      report += '\n';
    }

    // Alerts
    if (snapshot.alerts.length > 0) {
      report += '🚨 ACTIVE ALERTS\n\n';
      snapshot.alerts.forEach(alert => {
        const icon =
          alert.level === 'critical'
            ? '🔴'
            : alert.level === 'error'
              ? '🟠'
              : alert.level === 'warning'
                ? '🟡'
                : '🔵';
        report += `  ${icon} ${alert.level.toUpperCase()}: ${alert.message}\n`;
        if (alert.agentName) {
          report += `     Agent: ${alert.agentName}\n`;
        }
      });
      report += '\n';
    } else {
      report += '✅ NO ACTIVE ALERTS\n\n';
    }

    report += '═══════════════════════════════════════════════════════════\n';
    report += `Report generated at: ${snapshot.timestamp.toLocaleString()}\n\n`;

    return report;
  }

  /**
   * Print real-time dashboard to console
   */
  printDashboard(): void {
    console.clear();
    console.log(this.generateReport());
  }

  /**
   * Start monitoring with periodic updates
   */
  startMonitoring(intervalMs: number = 5000): NodeJS.Timeout {
    console.log('🎯 Starting Agent Performance Monitor...\n');

    // Initial snapshot
    this.printDashboard();

    // Set up periodic updates
    return setInterval(() => {
      this.printDashboard();
    }, intervalMs);
  }

  /**
   * Stop monitoring
   */
  stopMonitoring(intervalId: NodeJS.Timeout): void {
    clearInterval(intervalId);
    console.log('\n⏹️  Agent Performance Monitor stopped.\n');
  }

  /**
   * Export metrics to JSON
   */
  exportMetrics(): string {
    const snapshot = this.takeSnapshot();
    return JSON.stringify(snapshot, null, 2);
  }

  /**
   * Get agents by status
   */
  getAgentsByStatus(status: AgentStatus): AgentMetrics[] {
    return this.getAllAgentMetrics().filter(m => m.status === status);
  }

  /**
   * Get agents by type
   */
  getAgentsByType(type: AgentType): AgentMetrics[] {
    return this.getAllAgentMetrics().filter(m => m.agentType === type);
  }

  /**
   * Reset monitoring statistics
   */
  reset(): void {
    this.snapshotHistory = [];
    this.startTime = new Date();
    console.log('✓ Monitoring statistics reset\n');
  }
}

// Singleton instance
export const agentMonitor = new AgentMonitor();
