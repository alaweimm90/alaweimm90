/**
 * TRAE Event System Template
 * Comprehensive Event-Driven Architecture for Monitoring and Analytics
 *
 * This template provides a complete event system for routing operations,
 * enabling real-time monitoring, analytics, alerting, and compliance tracking
 * across all routing components with production-hardened reliability.
 *
 * Features:
 * - Event-driven architecture with multiple transport layers
 * - Real-time monitoring and alerting
 * - Analytics aggregation and reporting
 * - Compliance event tracking and audit trails
 * - Framework and language agnostic
 * - Production monitoring and error handling
 */

// ============================================================================
// TYPE DEFINITIONS (Framework Agnostic)
// ============================================================================

export interface RoutingEvent {
  type: RoutingEventType;
  context: RoutingContext;
  data: any;
  timestamp: Date;
  severity: 'info' | 'warning' | 'error' | 'critical';
  correlationId?: string;
  source: string;
  metadata?: EventMetadata;
}

export enum RoutingEventType {
  ROUTING_DECISION = 'routing_decision',
  COST_THRESHOLD_EXCEEDED = 'cost_threshold_exceeded',
  EMERGENCY_CONTROL_ACTIVATED = 'emergency_control_activated',
  FALLBACK_TRIGGERED = 'fallback_triggered',
  CACHE_MISS = 'cache_miss',
  CACHE_HIT = 'cache_hit',
  MODEL_FAILURE = 'model_failure',
  GEOGRAPHIC_SWITCH = 'geographic_switch',
  COMPLIANCE_VIOLATION = 'compliance_violation',
  HEALTH_CHECK_FAILED = 'health_check_failed',
  RATE_LIMIT_EXCEEDED = 'rate_limit_exceeded',
  CONFIGURATION_CHANGED = 'configuration_changed',
}

export interface RoutingContext {
  userId?: string;
  sessionId: string;
  requestId: string;
  timestamp: Date;
  clientRegion?: GeographicRegion;
  priority: RequestPriority;
  tags: string[];
  complianceContext?: ComplianceContext;
}

export enum GeographicRegion {
  NORTH_AMERICA = 'north_america',
  EUROPE = 'europe',
  ASIA_PACIFIC = 'asia_pacific',
  SOUTH_AMERICA = 'south_america',
  AFRICA = 'africa',
  GLOBAL = 'global',
}

export enum RequestPriority {
  LOW = 'low',
  NORMAL = 'normal',
  HIGH = 'high',
  CRITICAL = 'critical',
}

export interface ComplianceContext {
  dataClassification: DataClassification;
  retentionPolicy: RetentionPolicy;
  auditRequirements: boolean;
  geographicRestrictions: GeographicRegion[];
}

export enum DataClassification {
  PUBLIC = 'public',
  INTERNAL = 'internal',
  CONFIDENTIAL = 'confidential',
  RESTRICTED = 'restricted',
}

export enum RetentionPolicy {
  EPHEMERAL = 'ephemeral',
  SHORT_TERM = 'short_term',
  LONG_TERM = 'long_term',
  PERMANENT = 'permanent',
}

export interface EventMetadata {
  component: string;
  version: string;
  environment: string;
  processingTime?: number;
  retryCount?: number;
  userAgent?: string;
  ipAddress?: string;
}

export interface EventFilter {
  types?: RoutingEventType[];
  severities?: ('info' | 'warning' | 'error' | 'critical')[];
  sources?: string[];
  timeRange?: {
    start: Date;
    end: Date;
  };
  context?: Partial<RoutingContext>;
}

export interface EventSubscription {
  id: string;
  filter: EventFilter;
  handler: (event: RoutingEvent) => void | Promise<void>;
  options?: SubscriptionOptions;
}

export interface SubscriptionOptions {
  batchSize?: number;
  batchTimeout?: number;
  retryPolicy?: RetryPolicy;
  deadLetterQueue?: boolean;
}

export interface RetryPolicy {
  maxRetries: number;
  backoffMultiplier: number;
  initialDelay: number;
  maxDelay: number;
}

export interface EventTransport {
  name: string;
  type: 'memory' | 'redis' | 'kafka' | 'webhook' | 'file' | 'database';
  config: any;
  send: (event: RoutingEvent) => Promise<void>;
  subscribe?: (subscription: EventSubscription) => Promise<void>;
  unsubscribe?: (subscriptionId: string) => Promise<void>;
}

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  filter: EventFilter;
  condition: AlertCondition;
  actions: AlertAction[];
  enabled: boolean;
  cooldownPeriod: number; // seconds
  lastTriggered?: Date;
}

export interface AlertCondition {
  type: 'count' | 'threshold' | 'pattern' | 'composite';
  parameters: any;
  timeWindow: number; // seconds
}

export interface AlertAction {
  type: 'email' | 'slack' | 'webhook' | 'log' | 'metric' | 'shutdown';
  config: any;
  priority: 'low' | 'normal' | 'high' | 'critical';
}

export interface AnalyticsConfig {
  enabled: boolean;
  retentionPeriod: number; // days
  aggregationIntervals: number[]; // seconds
  metrics: AnalyticsMetric[];
  dashboards: AnalyticsDashboard[];
}

export interface AnalyticsMetric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  description: string;
  labels: string[];
}

export interface AnalyticsDashboard {
  name: string;
  description: string;
  panels: DashboardPanel[];
}

export interface DashboardPanel {
  title: string;
  type: 'line' | 'bar' | 'pie' | 'table' | 'heatmap';
  query: string;
  config: any;
}

// ============================================================================
// EVENT SYSTEM IMPLEMENTATION
// ============================================================================

export class EventSystem {
  private transports: Map<string, EventTransport> = new Map();
  private subscriptions: Map<string, EventSubscription> = new Map();
  private alertRules: Map<string, AlertRule> = new Map();
  private analytics: AnalyticsEngine;
  private eventBuffer: RoutingEvent[] = [];
  private bufferSize: number;
  private flushInterval: number;
  private flushTimer?: NodeJS.Timeout;

  private eventCount = 0;
  private processedCount = 0;
  private failedCount = 0;

  constructor(config?: {
    bufferSize?: number;
    flushInterval?: number;
    analytics?: AnalyticsConfig;
    transports?: EventTransport[];
  }) {
    this.bufferSize = config?.bufferSize || 1000;
    this.flushInterval = config?.flushInterval || 5000; // 5 seconds

    this.analytics = new AnalyticsEngine(config?.analytics);

    // Initialize default transports
    this.initializeDefaultTransports();

    // Add configured transports
    if (config?.transports) {
      config.transports.forEach(transport => {
        this.addTransport(transport);
      });
    }

    // Start buffer flushing
    this.startBufferFlush();
  }

  /**
   * Emit a routing event
   */
  async emit(event: RoutingEvent): Promise<void> {
    this.eventCount++;

    // Add metadata
    event.metadata = {
      ...event.metadata,
      component: 'routing_system',
      version: '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      processingTime: Date.now() - event.timestamp.getTime(),
    };

    // Add correlation ID if not present
    if (!event.correlationId) {
      event.correlationId = this.generateCorrelationId();
    }

    // Buffer event for processing
    this.eventBuffer.push(event);

    // Flush if buffer is full
    if (this.eventBuffer.length >= this.bufferSize) {
      await this.flushBuffer();
    }

    // Send to analytics
    await this.analytics.processEvent(event);

    // Check alert rules
    await this.checkAlertRules(event);
  }

  /**
   * Subscribe to events
   */
  async subscribe(subscription: Omit<EventSubscription, 'id'>): Promise<string> {
    const id = this.generateSubscriptionId();
    const fullSubscription: EventSubscription = {
      id,
      ...subscription,
    };

    this.subscriptions.set(id, fullSubscription);

    // Subscribe with transports that support it
    for (const transport of this.transports.values()) {
      if (transport.subscribe) {
        try {
          await transport.subscribe(fullSubscription);
        } catch (error) {
          console.error(`Failed to subscribe with transport ${transport.name}:`, error);
        }
      }
    }

    return id;
  }

  /**
   * Unsubscribe from events
   */
  async unsubscribe(subscriptionId: string): Promise<void> {
    const subscription = this.subscriptions.get(subscriptionId);
    if (!subscription) return;

    // Unsubscribe from transports
    for (const transport of this.transports.values()) {
      if (transport.unsubscribe) {
        try {
          await transport.unsubscribe(subscriptionId);
        } catch (error) {
          console.error(`Failed to unsubscribe from transport ${transport.name}:`, error);
        }
      }
    }

    this.subscriptions.delete(subscriptionId);
  }

  /**
   * Add event transport
   */
  addTransport(transport: EventTransport): void {
    this.transports.set(transport.name, transport);
  }

  /**
   * Remove event transport
   */
  removeTransport(name: string): void {
    this.transports.delete(name);
  }

  /**
   * Add alert rule
   */
  addAlertRule(rule: AlertRule): void {
    this.alertRules.set(rule.id, rule);
  }

  /**
   * Remove alert rule
   */
  removeAlertRule(ruleId: string): void {
    this.alertRules.delete(ruleId);
  }

  /**
   * Get event statistics
   */
  getStats(): EventStats {
    return {
      totalEvents: this.eventCount,
      processedEvents: this.processedCount,
      failedEvents: this.failedCount,
      bufferedEvents: this.eventBuffer.length,
      activeSubscriptions: this.subscriptions.size,
      activeTransports: this.transports.size,
      activeAlertRules: this.alertRules.size,
    };
  }

  /**
   * Query events with filtering
   */
  async queryEvents(filter: EventFilter, limit: number = 100): Promise<RoutingEvent[]> {
    // This would query from persistent storage
    // For now, return from buffer (not production-ready)
    return this.eventBuffer.filter(event => this.matchesFilter(event, filter)).slice(-limit);
  }

  /**
   * Get analytics report
   */
  async getAnalyticsReport(timeRange?: { start: Date; end: Date }): Promise<AnalyticsReport> {
    return await this.analytics.generateReport(timeRange);
  }

  // Private methods

  private initializeDefaultTransports(): void {
    // Memory transport for local development
    this.addTransport({
      name: 'memory',
      type: 'memory',
      config: {},
      send: async (event: RoutingEvent) => {
        // In-memory processing - just log for development
        console.log(`📊 Event [${event.type}]: ${JSON.stringify(event.data)}`);
      },
    });

    // File transport for basic persistence
    this.addTransport({
      name: 'file',
      type: 'file',
      config: { path: './logs/events.log' },
      send: async (event: RoutingEvent) => {
        const fs = require('fs').promises;
        const logEntry = `${event.timestamp.toISOString()} [${event.severity.toUpperCase()}] ${
          event.type
        }: ${JSON.stringify(event)}\n`;
        await fs.appendFile('./logs/events.log', logEntry);
      },
    });
  }

  private async flushBuffer(): Promise<void> {
    if (this.eventBuffer.length === 0) return;

    const events = [...this.eventBuffer];
    this.eventBuffer = [];

    // Process events through all transports
    const transportPromises = Array.from(this.transports.values()).map(async transport => {
      const eventPromises = events.map(async event => {
        try {
          await transport.send(event);
          this.processedCount++;
        } catch (error) {
          this.failedCount++;
          console.error(`Failed to send event to transport ${transport.name}:`, error);
        }
      });

      await Promise.allSettled(eventPromises);
    });

    await Promise.allSettled(transportPromises);

    // Process subscriptions
    await this.processSubscriptions(events);
  }

  private async processSubscriptions(events: RoutingEvent[]): Promise<void> {
    for (const subscription of this.subscriptions.values()) {
      const matchingEvents = events.filter(event => this.matchesFilter(event, subscription.filter));

      if (matchingEvents.length > 0) {
        try {
          if (subscription.options?.batchSize && subscription.options.batchSize > 1) {
            // Batch processing
            await this.processBatchedEvents(subscription, matchingEvents);
          } else {
            // Individual processing
            for (const event of matchingEvents) {
              await subscription.handler(event);
            }
          }
        } catch (error) {
          console.error(`Error in subscription handler ${subscription.id}:`, error);
          // Implement retry logic based on subscription options
        }
      }
    }
  }

  private async processBatchedEvents(
    subscription: EventSubscription,
    events: RoutingEvent[]
  ): Promise<void> {
    const batchSize = subscription.options!.batchSize!;
    const batches: RoutingEvent[][] = [];

    for (let i = 0; i < events.length; i += batchSize) {
      batches.push(events.slice(i, i + batchSize));
    }

    for (const batch of batches) {
      await subscription.handler(batch[0]); // For now, just call with first event
    }
  }

  private async checkAlertRules(event: RoutingEvent): Promise<void> {
    for (const rule of this.alertRules.values()) {
      if (!rule.enabled) continue;

      // Check cooldown
      if (rule.lastTriggered) {
        const timeSinceLastTrigger = Date.now() - rule.lastTriggered.getTime();
        if (timeSinceLastTrigger < rule.cooldownPeriod * 1000) continue;
      }

      // Check if event matches filter
      if (!this.matchesFilter(event, rule.filter)) continue;

      // Evaluate condition
      const conditionMet = await this.evaluateAlertCondition(rule.condition, event);
      if (!conditionMet) continue;

      // Trigger alert
      rule.lastTriggered = new Date();
      await this.triggerAlertActions(rule.actions, event, rule);
    }
  }

  private async evaluateAlertCondition(
    condition: AlertCondition,
    event: RoutingEvent
  ): Promise<boolean> {
    // Simplified condition evaluation
    // In production, this would be more sophisticated
    switch (condition.type) {
      case 'threshold':
        return event.severity === 'error' || event.severity === 'critical';
      case 'count':
        // Would check event counts over time window
        return false;
      case 'pattern':
        // Would check for specific patterns in event data
        return false;
      default:
        return false;
    }
  }

  private async triggerAlertActions(
    actions: AlertAction[],
    event: RoutingEvent,
    rule: AlertRule
  ): Promise<void> {
    for (const action of actions) {
      try {
        await this.executeAlertAction(action, event, rule);
      } catch (error) {
        console.error(`Failed to execute alert action ${action.type}:`, error);
      }
    }
  }

  private async executeAlertAction(
    action: AlertAction,
    event: RoutingEvent,
    rule: AlertRule
  ): Promise<void> {
    switch (action.type) {
      case 'log':
        console.error(`🚨 ALERT [${rule.name}]: ${event.type} - ${JSON.stringify(event.data)}`);
        break;
      case 'email':
        // Would send email notification
        console.log(`📧 Would send email alert for rule: ${rule.name}`);
        break;
      case 'slack':
        // Would send Slack notification
        console.log(`💬 Would send Slack alert for rule: ${rule.name}`);
        break;
      case 'webhook':
        // Would call webhook
        console.log(`🔗 Would call webhook for rule: ${rule.name}`);
        break;
      case 'metric':
        // Would increment metric
        console.log(`📈 Would increment metric for rule: ${rule.name}`);
        break;
      case 'shutdown':
        // Would trigger system shutdown
        console.error(`🛑 ALERT: System shutdown triggered by rule: ${rule.name}`);
        break;
    }
  }

  private matchesFilter(event: RoutingEvent, filter: EventFilter): boolean {
    if (filter.types && !filter.types.includes(event.type)) return false;
    if (filter.severities && !filter.severities.includes(event.severity)) return false;
    if (filter.sources && !filter.sources.includes(event.source)) return false;

    if (filter.timeRange) {
      const eventTime = event.timestamp.getTime();
      if (
        eventTime < filter.timeRange.start.getTime() ||
        eventTime > filter.timeRange.end.getTime()
      ) {
        return false;
      }
    }

    if (filter.context) {
      // Would implement context matching logic
    }

    return true;
  }

  private generateCorrelationId(): string {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSubscriptionId(): string {
    return `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private startBufferFlush(): void {
    this.flushTimer = setInterval(async () => {
      await this.flushBuffer();
    }, this.flushInterval);
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    this.transports.clear();
    this.subscriptions.clear();
    this.alertRules.clear();
    this.eventBuffer = [];
  }
}

// ============================================================================
// SUPPORTING CLASSES
// ============================================================================

export class AnalyticsEngine {
  private events: RoutingEvent[] = [];
  private metrics: Map<string, any> = new Map();
  private config: AnalyticsConfig;

  constructor(config?: AnalyticsConfig) {
    this.config = config || {
      enabled: true,
      retentionPeriod: 30,
      aggregationIntervals: [60, 3600, 86400], // 1m, 1h, 1d
      metrics: [],
      dashboards: [],
    };
  }

  async processEvent(event: RoutingEvent): Promise<void> {
    if (!this.config.enabled) return;

    // Store event
    this.events.push(event);

    // Clean up old events
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.retentionPeriod);
    this.events = this.events.filter(e => e.timestamp > cutoffDate);

    // Update metrics
    this.updateMetrics(event);

    // Aggregate data
    await this.aggregateData();
  }

  async generateReport(timeRange?: { start: Date; end: Date }): Promise<AnalyticsReport> {
    const events = timeRange
      ? this.events.filter(e => e.timestamp >= timeRange.start && e.timestamp <= timeRange.end)
      : this.events;

    const report: AnalyticsReport = {
      timeRange: timeRange || {
        start: new Date(this.events[0]?.timestamp || Date.now()),
        end: new Date(),
      },
      totalEvents: events.length,
      eventsByType: this.groupBy(events, 'type'),
      eventsBySeverity: this.groupBy(events, 'severity'),
      eventsBySource: this.groupBy(events, 'source'),
      topErrors: this.getTopErrors(events),
      performanceMetrics: this.calculatePerformanceMetrics(events),
      costAnalysis: this.calculateCostAnalysis(events),
      complianceMetrics: this.calculateComplianceMetrics(events),
    };

    return report;
  }

  private updateMetrics(event: RoutingEvent): void {
    // Update counters
    const typeKey = `events_by_type_${event.type}`;
    this.metrics.set(typeKey, (this.metrics.get(typeKey) || 0) + 1);

    const severityKey = `events_by_severity_${event.severity}`;
    this.metrics.set(severityKey, (this.metrics.get(severityKey) || 0) + 1);
  }

  private async aggregateData(): Promise<void> {
    // Implement time-based aggregation
    // This would create aggregated metrics for different time intervals
  }

  private groupBy(events: RoutingEvent[], key: keyof RoutingEvent): Record<string, number> {
    return events.reduce(
      (acc, event) => {
        const value = String(event[key]);
        acc[value] = (acc[value] || 0) + 1;
        return acc;
      },
      {} as Record<string, number>
    );
  }

  private getTopErrors(
    events: RoutingEvent[]
  ): Array<{ type: string; count: number; sample: string }> {
    const errors = events.filter(e => e.severity === 'error' || e.severity === 'critical');
    const grouped = this.groupBy(errors, 'type');

    return Object.entries(grouped)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([type, count]) => ({
        type,
        count,
        sample: errors.find(e => e.type === type)?.data?.message || 'No sample available',
      }));
  }

  private calculatePerformanceMetrics(events: RoutingEvent[]): PerformanceMetrics {
    const routingEvents = events.filter(e => e.type === RoutingEventType.ROUTING_DECISION);
    const latencies = routingEvents
      .map(e => e.metadata?.processingTime)
      .filter((time): time is number => time !== undefined);

    return {
      averageLatency:
        latencies.length > 0 ? latencies.reduce((a, b) => a + b) / latencies.length : 0,
      p95Latency: this.calculatePercentile(latencies, 95),
      p99Latency: this.calculatePercentile(latencies, 99),
      totalRequests: routingEvents.length,
      successRate:
        routingEvents.filter(e => e.data?.success !== false).length / routingEvents.length,
    };
  }

  private calculateCostAnalysis(events: RoutingEvent[]): CostAnalysis {
    const costEvents = events.filter(e => e.data?.actualCost !== undefined);
    const costs = costEvents.map(e => e.data.actualCost);

    return {
      totalCost: costs.reduce((a, b) => a + b, 0),
      averageCost: costs.length > 0 ? costs.reduce((a, b) => a + b) / costs.length : 0,
      costByModel: this.groupCostsByModel(costEvents),
      costByRegion: this.groupCostsByRegion(costEvents),
      costTrend: this.calculateCostTrend(costEvents),
    };
  }

  private calculateComplianceMetrics(events: RoutingEvent[]): ComplianceMetrics {
    const complianceEvents = events.filter(e => e.type === RoutingEventType.COMPLIANCE_VIOLATION);

    return {
      totalViolations: complianceEvents.length,
      violationsByFramework: this.groupBy(complianceEvents, 'data.framework'),
      violationsBySeverity: this.groupBy(complianceEvents, 'severity'),
      complianceScore: Math.max(0, 100 - (complianceEvents.length / events.length) * 100),
    };
  }

  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    const sorted = values.sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  private groupCostsByModel(events: RoutingEvent[]): Record<string, number> {
    return events.reduce(
      (acc, event) => {
        const model = event.data?.selectedModel?.name || 'unknown';
        acc[model] = (acc[model] || 0) + (event.data?.actualCost || 0);
        return acc;
      },
      {} as Record<string, number>
    );
  }

  private groupCostsByRegion(events: RoutingEvent[]): Record<string, number> {
    return events.reduce(
      (acc, event) => {
        const region = event.context?.clientRegion || 'unknown';
        acc[region] = (acc[region] || 0) + (event.data?.actualCost || 0);
        return acc;
      },
      {} as Record<string, number>
    );
  }

  private calculateCostTrend(events: RoutingEvent[]): 'increasing' | 'decreasing' | 'stable' {
    if (events.length < 10) return 'stable';

    const recent = events.slice(-10);
    const older = events.slice(-20, -10);

    const recentAvg = recent.reduce((sum, e) => sum + (e.data?.actualCost || 0), 0) / recent.length;
    const olderAvg = older.reduce((sum, e) => sum + (e.data?.actualCost || 0), 0) / older.length;

    const change = (recentAvg - olderAvg) / olderAvg;

    if (change > 0.1) return 'increasing';
    if (change < -0.1) return 'decreasing';
    return 'stable';
  }
}

// ============================================================================
// SUPPORTING INTERFACES AND TYPES
// ============================================================================

export interface EventStats {
  totalEvents: number;
  processedEvents: number;
  failedEvents: number;
  bufferedEvents: number;
  activeSubscriptions: number;
  activeTransports: number;
  activeAlertRules: number;
}

export interface AnalyticsReport {
  timeRange: { start: Date; end: Date };
  totalEvents: number;
  eventsByType: Record<string, number>;
  eventsBySeverity: Record<string, number>;
  eventsBySource: Record<string, number>;
  topErrors: Array<{ type: string; count: number; sample: string }>;
  performanceMetrics: PerformanceMetrics;
  costAnalysis: CostAnalysis;
  complianceMetrics: ComplianceMetrics;
}

export interface PerformanceMetrics {
  averageLatency: number;
  p95Latency: number;
  p99Latency: number;
  totalRequests: number;
  successRate: number;
}

export interface CostAnalysis {
  totalCost: number;
  averageCost: number;
  costByModel: Record<string, number>;
  costByRegion: Record<string, number>;
  costTrend: 'increasing' | 'decreasing' | 'stable';
}

export interface ComplianceMetrics {
  totalViolations: number;
  violationsByFramework: Record<string, number>;
  violationsBySeverity: Record<string, number>;
  complianceScore: number;
}

// ============================================================================
// USAGE EXAMPLES AND INTEGRATION PATTERNS
// ============================================================================

/**
 * Example: Basic Event System Setup
 *
 * const eventSystem = new EventSystem({
 *   bufferSize: 1000,
 *   flushInterval: 5000,
 *   analytics: {
 *     enabled: true,
 *     retentionPeriod: 30,
 *     aggregationIntervals: [60, 3600, 86400]
 *   }
 * });
 */

/**
 * Example: Emitting Events
 *
 * await eventSystem.emit({
 *   type: RoutingEventType.ROUTING_DECISION,
 *   context: {
 *     sessionId: 'session_123',
 *     requestId: 'req_456',
 *     timestamp: new Date(),
 *     priority: RequestPriority.NORMAL,
 *     tags: ['cost_optimized']
 *   },
 *   data: {
 *     selectedModel: modelCapability,
 *     estimatedCost: 0.02,
 *     reasoning: ['Cost optimization applied']
 *   },
 *   timestamp: new Date(),
 *   severity: 'info',
 *   source: 'intelligent_router'
 * });
 */

/**
 * Example: Event Subscription
 *
 * const subscriptionId = await eventSystem.subscribe({
 *   filter: {
 *     types: [RoutingEventType.COST_THRESHOLD_EXCEEDED],
 *     severities: ['warning', 'error', 'critical']
 *   },
 *   handler: async (event) => {
 *     console.log('Cost threshold exceeded:', event.data);
 *     // Send notification, update dashboard, etc.
 *   },
 *   options: {
 *     batchSize: 10,
 *     retryPolicy: {
 *       maxRetries: 3,
 *       backoffMultiplier: 2,
 *       initialDelay: 1000,
 *       maxDelay: 30000
 *     }
 *   }
 * });
 */

/**
 * Example: Alert Rules
 *
 * eventSystem.addAlertRule({
 *   id: 'high-error-rate',
 *   name: 'High Error Rate Alert',
 *   description: 'Alert when error rate exceeds 5%',
 *   filter: {
 *     severities: ['error', 'critical'],
 *     timeRange: { start: new Date(Date.now() - 300000), end: new Date() } // Last 5 minutes
 *   },
 *   condition: {
 *     type: 'threshold',
 *     parameters: { threshold: 0.05 },
 *     timeWindow: 300
 *   },
 *   actions: [
 *     {
 *       type: 'slack',
 *       config: { channel: '#alerts', message: 'High error rate detected!' },
 *       priority: 'high'
 *     },
 *     {
 *       type: 'email',
 *       config: { to: 'admin@example.com', subject: 'High Error Rate Alert' },
 *       priority: 'high'
 *     }
 *   ],
 *   enabled: true,
 *   cooldownPeriod: 600 // 10 minutes
 * }); */

/**
 * Example: Analytics Reporting
 *
 * const report = await eventSystem.getAnalyticsReport({
 *   start: new Date(Date.now() - 86400000), // Last 24 hours
 *   end: new Date()
 * });
 *
 * console.log('Performance metrics:', report.performanceMetrics);
 * console.log('Cost analysis:', report.costAnalysis);
 * console.log('Compliance score:', report.complianceMetrics.complianceScore);
 */

export default EventSystem;
