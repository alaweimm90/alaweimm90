#!/usr/bin/env npx tsx
/**
 * AI Telemetry & Observability
 * Comprehensive metrics collection, performance tracking, and alerting
 */

import * as path from 'path';
import { EventEmitter } from 'events';
import { loadJson, saveJson } from './utils/file-persistence.js';

const AI_DIR = path.join(process.cwd(), '.ai');
const TELEMETRY_FILE = path.join(AI_DIR, 'telemetry.json');

// ============================================================================
// Types
// ============================================================================

interface TelemetryEvent {
  id: string;
  timestamp: string;
  type: EventType;
  source: string;
  duration?: number;
  success?: boolean;
  metadata: Record<string, unknown>;
  tags: string[];
}

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  timestamp: string;
  tags: string[];
}

interface Alert {
  id: string;
  timestamp: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  source: string;
  message: string;
  resolved: boolean;
  resolvedAt?: string;
}

interface TelemetryState {
  events: TelemetryEvent[];
  metrics: PerformanceMetric[];
  alerts: Alert[];
  stats: {
    eventsTotal: number;
    eventsByType: Record<EventType, number>;
    avgLatency: Record<string, number>;
    errorRate: number;
    lastUpdated: string;
  };
}

interface AlertThreshold {
  metric: string;
  condition: 'gt' | 'lt' | 'eq';
  value: number;
  severity: Alert['severity'];
  message: string;
}

type EventType =
  | 'task.start'
  | 'task.complete'
  | 'task.fail'
  | 'cache.hit'
  | 'cache.miss'
  | 'sync.start'
  | 'sync.complete'
  | 'compliance.check'
  | 'monitor.trigger'
  | 'circuit.open'
  | 'circuit.close'
  | 'error';

// ============================================================================
// Default Thresholds
// ============================================================================

const DEFAULT_THRESHOLDS: AlertThreshold[] = [
  {
    metric: 'error_rate',
    condition: 'gt',
    value: 10,
    severity: 'warning',
    message: 'Error rate exceeds 10%',
  },
  {
    metric: 'error_rate',
    condition: 'gt',
    value: 25,
    severity: 'error',
    message: 'Error rate exceeds 25%',
  },
  {
    metric: 'avg_latency',
    condition: 'gt',
    value: 5000,
    severity: 'warning',
    message: 'Average latency exceeds 5s',
  },
  {
    metric: 'cache_hit_rate',
    condition: 'lt',
    value: 50,
    severity: 'info',
    message: 'Cache hit rate below 50%',
  },
  {
    metric: 'compliance_score',
    condition: 'lt',
    value: 70,
    severity: 'warning',
    message: 'Compliance score below 70%',
  },
];

// ============================================================================
// Telemetry Implementation
// ============================================================================

class Telemetry extends EventEmitter {
  private state: TelemetryState;
  private thresholds: AlertThreshold[] = DEFAULT_THRESHOLDS;
  private maxEvents = 1000;
  private maxMetrics = 500;
  private maxAlerts = 100;

  constructor() {
    super();
    this.state = this.loadState();
  }

  // Load state from disk
  private loadState(): TelemetryState {
    const defaultState: TelemetryState = {
      events: [],
      metrics: [],
      alerts: [],
      stats: {
        eventsTotal: 0,
        eventsByType: {} as Record<EventType, number>,
        avgLatency: {},
        errorRate: 0,
        lastUpdated: new Date().toISOString(),
      },
    };

    return loadJson<TelemetryState>(TELEMETRY_FILE, defaultState) ?? defaultState;
  }

  // Save state to disk
  private saveState(): void {
    // Trim to max sizes
    this.state.events = this.state.events.slice(-this.maxEvents);
    this.state.metrics = this.state.metrics.slice(-this.maxMetrics);
    this.state.alerts = this.state.alerts.slice(-this.maxAlerts);

    saveJson(TELEMETRY_FILE, this.state);
  }

  // Generate unique ID
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  // Record an event
  recordEvent(
    type: EventType,
    source: string,
    metadata: Record<string, unknown> = {},
    options: { duration?: number; success?: boolean; tags?: string[] } = {}
  ): string {
    const event: TelemetryEvent = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      type,
      source,
      duration: options.duration,
      success: options.success,
      metadata,
      tags: options.tags || [],
    };

    this.state.events.push(event);
    this.state.stats.eventsTotal++;
    this.state.stats.eventsByType[type] = (this.state.stats.eventsByType[type] || 0) + 1;
    this.state.stats.lastUpdated = new Date().toISOString();

    // Update latency tracking
    if (options.duration !== undefined) {
      const current = this.state.stats.avgLatency[source] || 0;
      const count = this.state.events.filter(
        (e) => e.source === source && e.duration !== undefined
      ).length;
      this.state.stats.avgLatency[source] = (current * (count - 1) + options.duration) / count;
    }

    // Check for errors
    if (type === 'error' || type.endsWith('.fail') || options.success === false) {
      this.updateErrorRate();
    }

    this.emit('event', event);
    this.checkThresholds();
    this.saveState();

    return event.id;
  }

  // Record a performance metric
  recordMetric(name: string, value: number, unit: string, tags: string[] = []): void {
    const metric: PerformanceMetric = {
      name,
      value,
      unit,
      timestamp: new Date().toISOString(),
      tags,
    };

    this.state.metrics.push(metric);
    this.emit('metric', metric);
    this.checkThresholds();
    this.saveState();
  }

  // Create an alert
  createAlert(severity: Alert['severity'], source: string, message: string): string {
    const alert: Alert = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      severity,
      source,
      message,
      resolved: false,
    };

    this.state.alerts.push(alert);
    this.emit('alert', alert);
    this.saveState();

    return alert.id;
  }

  // Resolve an alert
  resolveAlert(id: string): boolean {
    const alert = this.state.alerts.find((a) => a.id === id);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = new Date().toISOString();
      this.saveState();
      return true;
    }
    return false;
  }

  // Update error rate
  private updateErrorRate(): void {
    const recentEvents = this.state.events.slice(-100);
    const errors = recentEvents.filter(
      (e) => e.type === 'error' || e.type.endsWith('.fail') || e.success === false
    ).length;
    this.state.stats.errorRate = Math.round((errors / Math.max(recentEvents.length, 1)) * 100);
  }

  // Check thresholds and create alerts
  private checkThresholds(): void {
    for (const threshold of this.thresholds) {
      let currentValue: number | undefined;

      switch (threshold.metric) {
        case 'error_rate':
          currentValue = this.state.stats.errorRate;
          break;
        case 'avg_latency':
          currentValue = Math.max(...Object.values(this.state.stats.avgLatency));
          break;
        case 'cache_hit_rate': {
          const hits = this.state.stats.eventsByType['cache.hit'] || 0;
          const misses = this.state.stats.eventsByType['cache.miss'] || 0;
          currentValue = hits + misses > 0 ? Math.round((hits / (hits + misses)) * 100) : 100;
          break;
        }
        default:
          continue;
      }

      if (currentValue === undefined) continue;

      const violated =
        (threshold.condition === 'gt' && currentValue > threshold.value) ||
        (threshold.condition === 'lt' && currentValue < threshold.value) ||
        (threshold.condition === 'eq' && currentValue === threshold.value);

      if (violated) {
        // Check if similar alert already exists and is unresolved
        const existingAlert = this.state.alerts.find(
          (a) => !a.resolved && a.message === threshold.message
        );

        if (!existingAlert) {
          this.createAlert(threshold.severity, threshold.metric, threshold.message);
        }
      }
    }
  }

  // Get summary statistics
  getSummary(): {
    eventsTotal: number;
    eventsByType: Record<EventType, number>;
    errorRate: number;
    avgLatency: Record<string, number>;
    activeAlerts: number;
    recentEvents: TelemetryEvent[];
    topMetrics: PerformanceMetric[];
  } {
    return {
      eventsTotal: this.state.stats.eventsTotal,
      eventsByType: this.state.stats.eventsByType,
      errorRate: this.state.stats.errorRate,
      avgLatency: this.state.stats.avgLatency,
      activeAlerts: this.state.alerts.filter((a) => !a.resolved).length,
      recentEvents: this.state.events.slice(-10),
      topMetrics: this.state.metrics.slice(-5),
    };
  }

  // Get active alerts
  getActiveAlerts(): Alert[] {
    return this.state.alerts.filter((a) => !a.resolved);
  }

  // Get events by type
  getEventsByType(type: EventType, limit = 50): TelemetryEvent[] {
    return this.state.events.filter((e) => e.type === type).slice(-limit);
  }

  // Create a timer for measuring duration
  startTimer(_source: string): () => number {
    const start = Date.now();
    return () => Date.now() - start;
  }

  // Instrument a function
  async instrument<T>(
    name: string,
    fn: () => Promise<T>,
    options: { tags?: string[] } = {}
  ): Promise<T> {
    const timer = this.startTimer(name);
    let success = true;

    try {
      const result = await fn();
      return result;
    } catch (error) {
      success = false;
      this.recordEvent(
        'error',
        name,
        { error: String(error) },
        { success: false, tags: options.tags }
      );
      throw error;
    } finally {
      const duration = timer();
      this.recordEvent(
        success ? 'task.complete' : 'task.fail',
        name,
        { duration },
        { duration, success, tags: options.tags }
      );
    }
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const telemetry = new Telemetry();

// ============================================================================
// CLI Interface
// ============================================================================

function displaySummary(): void {
  const summary = telemetry.getSummary();

  console.log('\n╔══════════════════════════════════════════════════════════════╗');
  console.log('║            📡 AI TELEMETRY DASHBOARD                         ║');
  console.log('╠══════════════════════════════════════════════════════════════╣');
  console.log('║                                                              ║');

  // Overview
  console.log('║  📊 OVERVIEW                                                 ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  console.log(`║  Total Events: ${summary.eventsTotal}`.padEnd(65) + '║');
  console.log(`║  Error Rate: ${summary.errorRate}%`.padEnd(65) + '║');
  console.log(`║  Active Alerts: ${summary.activeAlerts}`.padEnd(65) + '║');
  console.log('║                                                              ║');

  // Events by Type
  console.log('║  📈 EVENTS BY TYPE                                           ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  const topTypes = Object.entries(summary.eventsByType)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);
  for (const [type, count] of topTypes) {
    console.log(`║  ${type.padEnd(20)} ${count}`.padEnd(65) + '║');
  }
  console.log('║                                                              ║');

  // Latency
  if (Object.keys(summary.avgLatency).length > 0) {
    console.log('║  ⏱️  AVERAGE LATENCY                                         ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');
    for (const [source, latency] of Object.entries(summary.avgLatency)) {
      console.log(`║  ${source.padEnd(20)} ${Math.round(latency)}ms`.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  // Active Alerts
  const activeAlerts = telemetry.getActiveAlerts();
  if (activeAlerts.length > 0) {
    console.log('║  🚨 ACTIVE ALERTS                                            ║');
    console.log('║  ─────────────────────────────────────────────────────────── ║');
    for (const alert of activeAlerts.slice(0, 3)) {
      const icon =
        alert.severity === 'critical'
          ? '🔴'
          : alert.severity === 'error'
            ? '🟠'
            : alert.severity === 'warning'
              ? '🟡'
              : '🔵';
      console.log(`║  ${icon} [${alert.severity.toUpperCase()}] ${alert.message}`.padEnd(65) + '║');
    }
    if (activeAlerts.length > 3) {
      console.log(`║  ... and ${activeAlerts.length - 3} more alerts`.padEnd(65) + '║');
    }
    console.log('║                                                              ║');
  }

  // Recent Events
  console.log('║  📝 RECENT EVENTS                                            ║');
  console.log('║  ─────────────────────────────────────────────────────────── ║');
  for (const event of summary.recentEvents.slice(-5)) {
    const icon = event.success === false ? '❌' : event.type.includes('complete') ? '✅' : '📌';
    const time = new Date(event.timestamp).toLocaleTimeString();
    console.log(`║  ${icon} ${time} ${event.type} (${event.source})`.padEnd(65) + '║');
  }
  console.log('║                                                              ║');

  console.log('╚══════════════════════════════════════════════════════════════╝\n');
}

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'summary':
    case 'status':
      displaySummary();
      break;

    case 'alerts': {
      const alerts = telemetry.getActiveAlerts();
      if (alerts.length === 0) {
        console.log('\n✅ No active alerts\n');
      } else {
        console.log('\n🚨 Active Alerts:\n');
        for (const alert of alerts) {
          console.log(`[${alert.severity.toUpperCase()}] ${alert.message}`);
          console.log(`  Source: ${alert.source}`);
          console.log(`  Time: ${alert.timestamp}`);
          console.log(`  ID: ${alert.id}\n`);
        }
      }
      break;
    }

    case 'resolve': {
      const alertId = args[1];
      if (alertId) {
        const resolved = telemetry.resolveAlert(alertId);
        console.log(resolved ? `✅ Alert ${alertId} resolved` : `❌ Alert ${alertId} not found`);
      } else {
        console.log('Usage: telemetry resolve <alert-id>');
      }
      break;
    }

    case 'record': {
      const type = args[1] as EventType;
      const source = args[2] || 'cli';
      if (type) {
        const id = telemetry.recordEvent(type, source);
        console.log(`📝 Recorded event: ${id}`);
      } else {
        console.log('Usage: telemetry record <type> [source]');
      }
      break;
    }

    case 'events': {
      const type = args[1] as EventType;
      const events = type ? telemetry.getEventsByType(type) : telemetry.getSummary().recentEvents;
      console.log(JSON.stringify(events, null, 2));
      break;
    }

    default:
      console.log(`
AI Telemetry - Observability and monitoring

Commands:
  summary           Show telemetry dashboard
  alerts            List active alerts
  resolve <id>      Resolve an alert
  record <type> [source]  Record an event
  events [type]     List recent events

Event Types:
  task.start, task.complete, task.fail
  cache.hit, cache.miss
  sync.start, sync.complete
  compliance.check
  monitor.trigger
  circuit.open, circuit.close
  error
      `);
  }
}

main();
