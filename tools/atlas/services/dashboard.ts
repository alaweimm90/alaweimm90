/**
 * ATLAS Dashboard Service
 * Real-time monitoring dashboard with telemetry collection and reporting
 */

import { ContinuousOptimizer } from './optimizer';
import { RepositoryMonitor } from './monitor';
import { DashboardWidget, TelemetryEventDTO } from '../types/index';
import { EventEmitter } from 'events';
import * as WebSocket from 'ws';
import * as http from 'http';

export interface DashboardConfig {
  port: number;
  host: string;
  enableWebSocket: boolean;
  enableREST: boolean;
  telemetry: {
    retentionPeriod: number; // days
    maxEvents: number;
    enableMetrics: boolean;
  };
  security: {
    enableAuth: boolean;
    apiKeys: string[];
  };
}

export interface TelemetryEvent {
  id: string;
  timestamp: Date;
  type: 'optimization' | 'analysis' | 'error' | 'metric' | 'system';
  source: string;
  data: Record<string, unknown>;
  severity: 'low' | 'medium' | 'high' | 'critical';
  tags: string[];
}

export interface DashboardMetrics {
  totalOptimizations: number;
  successfulOptimizations: number;
  failedOptimizations: number;
  averageOptimizationTime: number;
  activeJobs: number;
  repositoriesMonitored: number;
  systemHealth: 'healthy' | 'warning' | 'critical';
  uptime: number; // seconds
}

export interface RealTimeData {
  metrics: DashboardMetrics;
  recentEvents: TelemetryEvent[];
  activeJobs: JobStatus[];
  repositoryStatus: RepositoryStatus[];
  systemStatus: SystemStatus;
}

export interface JobStatus {
  id: string;
  repository: string;
  status: string;
  progress: number;
  startTime: Date;
  estimatedCompletion?: Date;
}

export interface RepositoryStatus {
  name: string;
  path: string;
  lastOptimization?: Date;
  chaosScore: number;
  complexityScore: number;
  issuesCount: number;
  status: 'healthy' | 'warning' | 'critical';
}

export interface SystemStatus {
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  services: {
    optimizer: 'running' | 'stopped' | 'error';
    monitor: 'running' | 'stopped' | 'error';
    dashboard: 'running' | 'stopped' | 'error';
  };
}

export class DashboardService extends EventEmitter {
  private config: DashboardConfig;
  private optimizer: ContinuousOptimizer;
  private monitor: RepositoryMonitor;
  private server?: http.Server;
  private wss?: WebSocket.Server;
  private telemetryEvents: TelemetryEvent[] = [];
  private metrics: DashboardMetrics;
  private startTime: Date;
  private connectedClients: Set<WebSocket> = new Set();

  constructor(config: DashboardConfig, optimizer: ContinuousOptimizer, monitor: RepositoryMonitor) {
    super();
    this.config = config;
    this.optimizer = optimizer;
    this.monitor = monitor;
    this.startTime = new Date();
    this.metrics = this.initializeMetrics();

    this.setupEventListeners();
  }

  /**
   * Start the dashboard service
   */
  async start(): Promise<void> {
    this.emit('dashboard:start', { timestamp: new Date() });

    if (this.config.enableREST || this.config.enableWebSocket) {
      await this.startWebServer();
    }

    this.emit('dashboard:ready', {
      port: this.config.port,
      websocket: this.config.enableWebSocket,
      rest: this.config.enableREST,
      timestamp: new Date(),
    });
  }

  /**
   * Stop the dashboard service
   */
  async stop(): Promise<void> {
    if (this.wss) {
      this.wss.close();
    }

    if (this.server) {
      this.server.close();
    }

    this.connectedClients.clear();
    this.emit('dashboard:stop', { timestamp: new Date() });
  }

  /**
   * Get current dashboard data
   */
  getDashboardData(): RealTimeData {
    return {
      metrics: this.metrics,
      recentEvents: this.getRecentEvents(50),
      activeJobs: this.getActiveJobs(),
      repositoryStatus: this.getRepositoryStatus(),
      systemStatus: this.getSystemStatus(),
    };
  }

  /**
   * Record a telemetry event
   */
  recordEvent(event: Omit<TelemetryEvent, 'id' | 'timestamp'>): void {
    const telemetryEvent: TelemetryEvent = {
      ...event,
      id: this.generateEventId(),
      timestamp: new Date(),
    };

    this.telemetryEvents.push(telemetryEvent);

    // Maintain event limit
    if (this.telemetryEvents.length > this.config.telemetry.maxEvents) {
      this.telemetryEvents = this.telemetryEvents.slice(-this.config.telemetry.maxEvents);
    }

    // Update metrics based on event type
    this.updateMetricsFromEvent(telemetryEvent);

    // Emit to connected WebSocket clients
    this.broadcastEvent(telemetryEvent);

    // Emit internal event
    this.emit('telemetry:event', telemetryEvent);
  }

  /**
   * Get telemetry events with optional filtering
   */
  getEvents(filter?: {
    type?: string;
    source?: string;
    severity?: string;
    tags?: string[];
    since?: Date;
    limit?: number;
  }): TelemetryEvent[] {
    let events = [...this.telemetryEvents];

    if (filter) {
      if (filter.type) {
        events = events.filter((e) => e.type === filter.type);
      }
      if (filter.source) {
        events = events.filter((e) => e.source === filter.source);
      }
      if (filter.severity) {
        events = events.filter((e) => e.severity === filter.severity);
      }
      if (filter.tags && filter.tags.length > 0) {
        events = events.filter((e) => filter.tags!.some((tag) => e.tags.includes(tag)));
      }
      if (filter.since) {
        events = events.filter((e) => e.timestamp >= filter.since!);
      }
    }

    // Sort by timestamp descending (newest first)
    events.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());

    return filter?.limit ? events.slice(0, filter.limit) : events;
  }

  /**
   * Get system health status
   */
  getHealthStatus(): {
    status: 'healthy' | 'warning' | 'critical';
    checks: {
      name: string;
      status: 'pass' | 'fail' | 'warn';
      message: string;
    }[];
  } {
    const checks = [
      this.checkOptimizerHealth(),
      this.checkMonitorHealth(),
      this.checkSystemResources(),
      this.checkEventBacklog(),
    ];

    const hasFailures = checks.some((check) => check.status === 'fail');
    const hasWarnings = checks.some((check) => check.status === 'warn');

    let status: 'healthy' | 'warning' | 'critical' = 'healthy';
    if (hasFailures) {
      status = 'critical';
    } else if (hasWarnings) {
      status = 'warning';
    }

    return { status, checks };
  }

  /**
   * Export telemetry data
   */
  exportTelemetry(format: 'json' | 'csv' = 'json', filter?: any): string {
    const events = this.getEvents(filter);

    if (format === 'csv') {
      return this.eventsToCSV(events);
    }

    return JSON.stringify(events, null, 2);
  }

  /**
   * Clear old telemetry data based on retention policy
   */
  cleanupTelemetry(): void {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - this.config.telemetry.retentionPeriod);

    this.telemetryEvents = this.telemetryEvents.filter((event) => event.timestamp >= cutoffDate);

    this.emit('telemetry:cleanup', {
      eventsRemoved: this.telemetryEvents.length,
      timestamp: new Date(),
    });
  }

  private async startWebServer(): Promise<void> {
    this.server = http.createServer((req, res) => {
      this.handleHTTPRequest(req, res);
    });

    return new Promise((resolve, reject) => {
      this.server!.listen(this.config.port, this.config.host, () => {
        if (this.config.enableWebSocket) {
          this.wss = new WebSocket.Server({ server: this.server });
          this.setupWebSocketHandlers();
        }
        resolve();
      });

      this.server!.on('error', reject);
    });
  }

  private handleHTTPRequest(req: http.IncomingMessage, res: http.ServerResponse): void {
    // Set CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    // Basic authentication check
    if (this.config.security.enableAuth && !this.checkAuth(req)) {
      res.writeHead(401, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Unauthorized' }));
      return;
    }

    const url = new URL(req.url || '', `http://${req.headers.host}`);

    try {
      switch (url.pathname) {
        case '/api/dashboard':
          this.handleDashboardAPI(req, res);
          break;
        case '/api/events':
          this.handleEventsAPI(req, res);
          break;
        case '/api/health':
          this.handleHealthAPI(req, res);
          break;
        case '/api/metrics':
          this.handleMetricsAPI(req, res);
          break;
        default:
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Not found' }));
      }
    } catch (error) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(
        JSON.stringify({
          error: 'Internal server error',
          message: error instanceof Error ? error.message : String(error),
        })
      );
    }
  }

  private handleDashboardAPI(req: http.IncomingMessage, res: http.ServerResponse): void {
    if (req.method !== 'GET') {
      res.writeHead(405);
      res.end();
      return;
    }

    const data = this.getDashboardData();
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  }

  private handleEventsAPI(req: http.IncomingMessage, res: http.ServerResponse): void {
    if (req.method !== 'GET') {
      res.writeHead(405);
      res.end();
      return;
    }

    const url = new URL(req.url || '', `http://${req.headers.host}`);
    const filter = {
      type: url.searchParams.get('type') || undefined,
      source: url.searchParams.get('source') || undefined,
      severity: url.searchParams.get('severity') || undefined,
      limit: url.searchParams.get('limit') ? parseInt(url.searchParams.get('limit')!) : undefined,
    };

    const events = this.getEvents(filter);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(events));
  }

  private handleHealthAPI(req: http.IncomingMessage, res: http.ServerResponse): void {
    const health = this.getHealthStatus();
    const statusCode = health.status === 'healthy' ? 200 : health.status === 'warning' ? 200 : 503;

    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health));
  }

  private handleMetricsAPI(req: http.IncomingMessage, res: http.ServerResponse): void {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(this.metrics));
  }

  private setupWebSocketHandlers(): void {
    this.wss!.on('connection', (ws: WebSocket) => {
      this.connectedClients.add(ws);

      // Send initial dashboard data
      ws.send(
        JSON.stringify({
          type: 'dashboard:init',
          data: this.getDashboardData(),
        })
      );

      ws.on('message', (message: string) => {
        try {
          const msg = JSON.parse(message);
          this.handleWebSocketMessage(ws, msg);
        } catch {
          ws.send(
            JSON.stringify({
              type: 'error',
              message: 'Invalid message format',
            })
          );
        }
      });

      ws.on('close', () => {
        this.connectedClients.delete(ws);
      });

      ws.on('error', (error) => {
        this.connectedClients.delete(ws);
        this.emit('websocket:error', error);
      });
    });
  }

  private handleWebSocketMessage(ws: WebSocket, message: any): void {
    switch (message.type) {
      case 'subscribe':
        // Client wants to subscribe to real-time updates
        // Already handled by adding to connectedClients
        break;
      case 'unsubscribe':
        this.connectedClients.delete(ws);
        break;
      default:
        ws.send(
          JSON.stringify({
            type: 'error',
            message: 'Unknown message type',
          })
        );
    }
  }

  private broadcastEvent(event: TelemetryEvent): void {
    const message = JSON.stringify({
      type: 'telemetry:event',
      data: event,
    });

    for (const client of this.connectedClients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(message);
      }
    }
  }

  private setupEventListeners(): void {
    // Listen to optimizer events
    this.optimizer.on('job:start', (job) => {
      this.recordEvent({
        type: 'optimization',
        source: 'optimizer',
        data: job,
        severity: 'low',
        tags: ['job', 'start'],
      });
    });

    this.optimizer.on('job:complete', (job) => {
      this.recordEvent({
        type: 'optimization',
        source: 'optimizer',
        data: job,
        severity: 'low',
        tags: ['job', 'complete', 'success'],
      });
    });

    this.optimizer.on('job:fail', (job) => {
      this.recordEvent({
        type: 'error',
        source: 'optimizer',
        data: job,
        severity: 'high',
        tags: ['job', 'fail', 'error'],
      });
    });

    // Listen to monitor events
    this.monitor.on('analysis:complete', (result) => {
      this.recordEvent({
        type: 'analysis',
        source: 'monitor',
        data: result,
        severity: 'low',
        tags: ['analysis', 'complete'],
      });
    });

    this.monitor.on('analysis:error', (error) => {
      this.recordEvent({
        type: 'error',
        source: 'monitor',
        data: error,
        severity: 'medium',
        tags: ['analysis', 'error'],
      });
    });
  }

  private initializeMetrics(): DashboardMetrics {
    return {
      totalOptimizations: 0,
      successfulOptimizations: 0,
      failedOptimizations: 0,
      averageOptimizationTime: 0,
      activeJobs: 0,
      repositoriesMonitored: 0,
      systemHealth: 'healthy',
      uptime: 0,
    };
  }

  private updateMetricsFromEvent(event: TelemetryEvent): void {
    switch (event.type) {
      case 'optimization':
        if (event.data.status === 'completed') {
          this.metrics.successfulOptimizations++;
        } else if (event.data.status === 'failed') {
          this.metrics.failedOptimizations++;
        }
        this.metrics.totalOptimizations =
          this.metrics.successfulOptimizations + this.metrics.failedOptimizations;
        break;
      case 'system':
        // Update system health based on system events
        break;
    }

    // Update uptime
    this.metrics.uptime = (Date.now() - this.startTime.getTime()) / 1000;

    // Update system health
    this.metrics.systemHealth = this.calculateSystemHealth();
  }

  private calculateSystemHealth(): 'healthy' | 'warning' | 'critical' {
    const health = this.getHealthStatus();
    return health.status;
  }

  private getRecentEvents(limit: number): TelemetryEvent[] {
    return this.telemetryEvents.slice(-limit);
  }

  private getActiveJobs(): JobStatus[] {
    // This would integrate with the optimizer to get active jobs
    // For now, return mock data
    return [];
  }

  private getRepositoryStatus(): RepositoryStatus[] {
    // This would integrate with the monitor to get repository status
    // For now, return mock data
    return [];
  }

  private getSystemStatus(): SystemStatus {
    // Mock system status - in real implementation would use system monitoring
    return {
      cpu: 45,
      memory: 60,
      disk: 30,
      network: 20,
      services: {
        optimizer: 'running',
        monitor: 'running',
        dashboard: 'running',
      },
    };
  }

  private checkOptimizerHealth(): {
    name: string;
    status: 'pass' | 'fail' | 'warn';
    message: string;
  } {
    try {
      const status = this.optimizer.getStatus();
      if (status.circuitBreaker === 'open') {
        return { name: 'Optimizer', status: 'fail', message: 'Circuit breaker is open' };
      }
      if (status.activeJobs > status.repositories.length) {
        return { name: 'Optimizer', status: 'warn', message: 'High number of active jobs' };
      }
      return { name: 'Optimizer', status: 'pass', message: 'Operating normally' };
    } catch {
      return { name: 'Optimizer', status: 'fail', message: 'Unable to get optimizer status' };
    }
  }

  private checkMonitorHealth(): {
    name: string;
    status: 'pass' | 'fail' | 'warn';
    message: string;
  } {
    try {
      const status = this.monitor.getStatus();
      if (status.repositories === 0) {
        return { name: 'Monitor', status: 'warn', message: 'No repositories being monitored' };
      }
      return {
        name: 'Monitor',
        status: 'pass',
        message: `Monitoring ${status.repositories} repositories`,
      };
    } catch {
      return { name: 'Monitor', status: 'fail', message: 'Unable to get monitor status' };
    }
  }

  private checkSystemResources(): {
    name: string;
    status: 'pass' | 'fail' | 'warn';
    message: string;
  } {
    const status = this.getSystemStatus();
    if (status.memory > 90 || status.cpu > 95) {
      return { name: 'System Resources', status: 'fail', message: 'High resource usage' };
    }
    if (status.memory > 80 || status.cpu > 80) {
      return { name: 'System Resources', status: 'warn', message: 'Elevated resource usage' };
    }
    return { name: 'System Resources', status: 'pass', message: 'Resources normal' };
  }

  private checkEventBacklog(): { name: string; status: 'pass' | 'fail' | 'warn'; message: string } {
    const backlogRatio = this.telemetryEvents.length / this.config.telemetry.maxEvents;
    if (backlogRatio > 0.95) {
      return { name: 'Event Backlog', status: 'fail', message: 'Event backlog near capacity' };
    }
    if (backlogRatio > 0.8) {
      return { name: 'Event Backlog', status: 'warn', message: 'High event backlog' };
    }
    return { name: 'Event Backlog', status: 'pass', message: 'Event backlog normal' };
  }

  private checkAuth(req: http.IncomingMessage): boolean {
    if (!this.config.security.enableAuth) {
      return true;
    }

    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return false;
    }

    const token = authHeader.substring(7);
    return this.config.security.apiKeys.includes(token);
  }

  private generateEventId(): string {
    return `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private eventsToCSV(events: TelemetryEvent[]): string {
    const headers = ['id', 'timestamp', 'type', 'source', 'severity', 'tags', 'data'];
    const rows = events.map((event) => [
      event.id,
      event.timestamp.toISOString(),
      event.type,
      event.source,
      event.severity,
      event.tags.join(';'),
      JSON.stringify(event.data),
    ]);

    return [headers, ...rows]
      .map((row) => row.map((field) => `"${field.replace(/"/g, '""')}"`).join(','))
      .join('\n');
  }
}
