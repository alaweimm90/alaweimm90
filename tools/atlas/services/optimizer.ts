/**
 * ATLAS Continuous Optimization Service
 * Main orchestration service for automated repository monitoring and improvement
 */

import { RepositoryAnalyzer } from '../analysis/analyzer';
import { RefactoringEngine } from '../refactoring/engine';
import { TaskRouter } from '../orchestration/router';
import { agentRegistry } from '../agents/registry';
import { OptimizationPlan, OptimizationSuggestion } from '../types/index';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import * as path from 'path';

export interface OptimizationConfig {
  schedule: {
    interval: number; // minutes
    enabled: boolean;
    maxConcurrent: number;
  };
  thresholds: {
    chaosThreshold: number;
    complexityThreshold: number;
    minConfidence: number;
  };
  safety: {
    rateLimit: number; // operations per hour
    circuitBreakerThreshold: number;
    rollbackEnabled: boolean;
    manualOverride: boolean;
  };
  repositories: RepositoryTarget[];
}

export interface RepositoryTarget {
  path: string;
  name: string;
  enabled: boolean;
  lastOptimized?: Date;
  optimizationHistory: OptimizationResult[];
}

export interface OptimizationResult {
  id: string;
  timestamp: Date;
  repository: string;
  success: boolean;
  changes: CodeChange[];
  metrics: {
    before: RepositoryMetrics;
    after: RepositoryMetrics;
  };
  rollbackAvailable: boolean;
  error?: string;
}

export interface CodeChange {
  file: string;
  type: 'refactor' | 'optimize' | 'cleanup';
  description: string;
  confidence: number;
  applied: boolean;
}

export interface RepositoryMetrics {
  chaosScore: number;
  complexityScore: number;
  fileCount: number;
  totalLines: number;
  issuesCount: number;
}

export interface OptimizationJob {
  id: string;
  repository: RepositoryTarget;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  startTime?: Date;
  endTime?: Date;
  result?: OptimizationResult;
}

export class ContinuousOptimizer extends EventEmitter {
  private config: OptimizationConfig;
  private analyzer: RepositoryAnalyzer;
  private refactoringEngine: RefactoringEngine;
  private taskRouter: TaskRouter;
  private activeJobs: Map<string, OptimizationJob> = new Map();
  private jobQueue: JobQueue;
  private rateLimiter: RateLimiter;
  private circuitBreaker: CircuitBreaker;
  private scheduler: OptimizationScheduler;
  private telemetry: TelemetryRecorder;
  private rollbackStore: Map<string, RollbackData> = new Map();
  private watchers: Map<string, fs.FSWatcher> = new Map();

  constructor(config: OptimizationConfig) {
    super();
    this.config = config;
    this.analyzer = new RepositoryAnalyzer();
    this.refactoringEngine = new RefactoringEngine();
    this.taskRouter = new TaskRouter({ maxConcurrency: config.schedule.maxConcurrent });
    this.jobQueue = new JobQueue(config.schedule.maxConcurrent);
    this.rateLimiter = new RateLimiter(config.safety.rateLimit);
    this.circuitBreaker = new CircuitBreaker(config.safety.circuitBreakerThreshold);
    this.scheduler = new OptimizationScheduler(this);
    this.telemetry = new TelemetryRecorder();

    this.setupEventHandlers();
  }

  /**
   * Start the continuous optimization service
   */
  async start(): Promise<void> {
    this.emit('service:start', { timestamp: new Date() });

    // Start scheduled optimization runs
    if (this.config.schedule.enabled) {
      this.scheduler.start(this.config.schedule.interval);
    }

    // Initialize monitoring for all enabled repositories
    for (const repo of this.config.repositories.filter((r) => r.enabled)) {
      await this.monitorRepository(repo);
    }

    this.emit('service:ready', { timestamp: new Date() });
  }

  /**
   * Stop the optimization service
   */
  async stop(): Promise<void> {
    this.scheduler.stop();
    await this.cancelAllJobs();
    this.emit('service:stop', { timestamp: new Date() });
  }

  /**
   * Manually trigger optimization for a specific repository
   */
  async optimizeRepository(
    repositoryPath: string,
    options: {
      force?: boolean;
      dryRun?: boolean;
      manual?: boolean;
    } = {}
  ): Promise<OptimizationResult> {
    const repository = this.config.repositories.find((r) => r.path === repositoryPath);
    if (!repository) {
      throw new Error(`Repository not found: ${repositoryPath}`);
    }

    // Check rate limiting
    if (!this.rateLimiter.allow()) {
      throw new Error('Rate limit exceeded');
    }

    // Check circuit breaker
    if (!this.circuitBreaker.allow()) {
      throw new Error('Circuit breaker open - service temporarily unavailable');
    }

    // Check manual override if not manual operation
    if (!options.manual && this.config.safety.manualOverride) {
      throw new Error('Manual override required for optimization');
    }

    const jobId = this.generateJobId();
    const job: OptimizationJob = {
      id: jobId,
      repository,
      status: 'running',
      progress: 0,
      startTime: new Date(),
    };

    this.activeJobs.set(jobId, job);
    this.emit('job:start', job);

    try {
      const result = await this.performOptimization(job, options);
      job.status = 'completed';
      job.endTime = new Date();
      job.result = result;
      job.progress = 100;

      this.emit('job:complete', job);
      this.circuitBreaker.recordSuccess();

      return result;
    } catch (error) {
      job.status = 'failed';
      job.endTime = new Date();
      job.result = {
        id: jobId,
        timestamp: new Date(),
        repository: repository.name,
        success: false,
        changes: [],
        metrics: {
          before: {
            chaosScore: 0,
            complexityScore: 0,
            fileCount: 0,
            totalLines: 0,
            issuesCount: 0,
          },
          after: {
            chaosScore: 0,
            complexityScore: 0,
            fileCount: 0,
            totalLines: 0,
            issuesCount: 0,
          },
        },
        rollbackAvailable: false,
        error: error.message,
      };

      this.emit('job:fail', job);
      this.circuitBreaker.recordFailure();

      throw error;
    } finally {
      this.activeJobs.delete(jobId);
    }
  }

  /**
   * Get current status of all jobs and service health
   */
  getStatus(): {
    service: 'running' | 'stopped';
    activeJobs: number;
    queuedJobs: number;
    circuitBreaker: 'closed' | 'open' | 'half-open';
    rateLimitRemaining: number;
    repositories: RepositoryTarget[];
  } {
    return {
      service: this.scheduler.isRunning() ? 'running' : 'stopped',
      activeJobs: this.activeJobs.size,
      queuedJobs: this.jobQueue.size(),
      circuitBreaker: this.circuitBreaker.getState(),
      rateLimitRemaining: this.rateLimiter.getRemaining(),
      repositories: this.config.repositories,
    };
  }

  /**
   * Cancel a running optimization job
   */
  async cancelJob(jobId: string): Promise<void> {
    const job = this.activeJobs.get(jobId);
    if (!job) {
      throw new Error(`Job not found: ${jobId}`);
    }

    job.status = 'cancelled';
    job.endTime = new Date();
    this.activeJobs.delete(jobId);
    this.emit('job:cancel', job);
  }

  /**
   * Rollback a completed optimization
   */
  async rollbackOptimization(resultId: string): Promise<void> {
    const rollbackData = this.rollbackStore.get(resultId);
    if (!rollbackData) {
      throw new Error(`No rollback data found for result: ${resultId}`);
    }

    this.emit('rollback:start', { resultId, timestamp: new Date() });

    try {
      // Restore each file from the backup
      for (const [filePath, originalContent] of rollbackData.fileBackups) {
        await fs.promises.writeFile(filePath, originalContent, 'utf8');
      }

      // Clean up rollback data after successful rollback
      this.rollbackStore.delete(resultId);
      this.emit('rollback:complete', { resultId, timestamp: new Date() });
    } catch (error) {
      this.emit('rollback:error', { resultId, error: error.message, timestamp: new Date() });
      throw error;
    }
  }

  /**
   * Store rollback data for a completed optimization
   */
  private storeRollbackData(resultId: string, changes: CodeChange[], backups: Map<string, string>): void {
    if (this.config.safety.rollbackEnabled) {
      this.rollbackStore.set(resultId, {
        resultId,
        timestamp: new Date(),
        changes,
        fileBackups: backups,
      });
    }
  }

  /**
   * Update optimization configuration
   */
  updateConfig(newConfig: Partial<OptimizationConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.rateLimiter.updateLimit(this.config.safety.rateLimit);
    this.circuitBreaker.updateThreshold(this.config.safety.circuitBreakerThreshold);

    if (newConfig.schedule) {
      this.scheduler.updateInterval(newConfig.schedule.interval);
    }

    this.emit('config:update', { config: this.config, timestamp: new Date() });
  }

  private async performOptimization(
    job: OptimizationJob,
    options: { force?: boolean; dryRun?: boolean } = {}
  ): Promise<OptimizationResult> {
    const { repository } = job;

    // Step 1: Analyze repository
    job.progress = 10;
    this.emit('job:progress', job);

    const analysis = await this.analyzer.analyze(repository.path);
    const metrics: RepositoryMetrics = {
      chaosScore: analysis.chaosScore || 0,
      complexityScore: analysis.complexityScore || 0,
      fileCount: analysis.files?.length || 0,
      totalLines: analysis.totalLines || 0,
      issuesCount: analysis.issues?.length || 0,
    };

    // Step 2: Check if optimization is needed
    if (!options.force && !this.shouldOptimize(metrics)) {
      return {
        id: job.id,
        timestamp: new Date(),
        repository: repository.name,
        success: true,
        changes: [],
        metrics: { before: metrics, after: metrics },
        rollbackAvailable: false,
      };
    }

    // Step 3: Generate refactoring suggestions
    job.progress = 30;
    this.emit('job:progress', job);

    const suggestions = await this.refactoringEngine.generateSuggestions(analysis);

    // Step 4: Route tasks to appropriate agents
    job.progress = 50;
    this.emit('job:progress', job);

    const routingDecisions = await Promise.all(
      suggestions.map((suggestion) =>
        this.taskRouter.route({
          type: 'refactoring',
          data: suggestion,
          priority: 'normal',
        })
      )
    );

    // Step 5: Execute optimizations
    job.progress = 70;
    this.emit('job:progress', job);

    const changes: CodeChange[] = [];
    if (!options.dryRun) {
      for (const [index, suggestion] of suggestions.entries()) {
        const decision = routingDecisions[index];
        if (decision.confidence >= this.config.thresholds.minConfidence) {
          try {
            const change = await this.executeRefactoring(suggestion, decision);
            changes.push(change);
          } catch (error) {
            // Log error but continue with other suggestions
            this.emit('refactoring:error', { suggestion, error: error.message });
          }
        }
      }
    }

    // Step 6: Re-analyze and collect final metrics
    job.progress = 90;
    this.emit('job:progress', job);

    const finalAnalysis = await this.analyzer.analyze(repository.path);
    const finalMetrics: RepositoryMetrics = {
      chaosScore: finalAnalysis.chaosScore || 0,
      complexityScore: finalAnalysis.complexityScore || 0,
      fileCount: finalAnalysis.files?.length || 0,
      totalLines: finalAnalysis.totalLines || 0,
      issuesCount: finalAnalysis.issues?.length || 0,
    };

    const result: OptimizationResult = {
      id: job.id,
      timestamp: new Date(),
      repository: repository.name,
      success: true,
      changes,
      metrics: { before: metrics, after: finalMetrics },
      rollbackAvailable: this.config.safety.rollbackEnabled,
    };

    // Update repository history
    repository.lastOptimized = new Date();
    repository.optimizationHistory.push(result);

    return result;
  }

  private shouldOptimize(metrics: RepositoryMetrics): boolean {
    return (
      metrics.chaosScore > this.config.thresholds.chaosThreshold ||
      metrics.complexityScore > this.config.thresholds.complexityThreshold
    );
  }

  private async executeRefactoring(suggestion: any, routingDecision: any): Promise<CodeChange> {
    // Route to appropriate agent via task router
    const agent = agentRegistry.getAgent(routingDecision.agentId);
    if (!agent) {
      throw new Error(`Agent not found: ${routingDecision.agentId}`);
    }

    // Execute the refactoring
    const result = await agent.execute({
      type: 'refactor',
      data: suggestion,
      context: routingDecision,
    });

    return {
      file: suggestion.file,
      type: suggestion.type,
      description: suggestion.description,
      confidence: routingDecision.confidence,
      applied: result.success,
    };
  }

  private async monitorRepository(repository: RepositoryTarget): Promise<void> {
    // Watch for file changes and trigger optimization when needed
    if (this.watchers.has(repository.path)) {
      return; // Already monitoring
    }

    try {
      const watcher = fs.watch(
        repository.path,
        { recursive: true },
        (eventType, filename) => {
          if (filename && this.shouldTriggerOptimization(filename)) {
            this.emit('repository:change', {
              repository: repository.name,
              eventType,
              filename,
              timestamp: new Date(),
            });

            // Debounce optimization triggers
            this.scheduleOptimization(repository);
          }
        }
      );

      this.watchers.set(repository.path, watcher);
      this.emit('monitor:start', { repository: repository.name, timestamp: new Date() });
    } catch (error) {
      this.emit('monitor:error', { repository: repository.name, error: error.message });
    }
  }

  private shouldTriggerOptimization(filename: string): boolean {
    // Only trigger for source code files
    const sourceExtensions = ['.ts', '.tsx', '.js', '.jsx', '.py', '.go', '.rs', '.java'];
    const ext = path.extname(filename).toLowerCase();
    return sourceExtensions.includes(ext);
  }

  private optimizationDebounce: Map<string, NodeJS.Timeout> = new Map();

  private scheduleOptimization(repository: RepositoryTarget): void {
    // Clear existing timeout for this repository
    const existing = this.optimizationDebounce.get(repository.path);
    if (existing) {
      clearTimeout(existing);
    }

    // Schedule optimization after 5 seconds of no changes
    const timeout = setTimeout(async () => {
      this.optimizationDebounce.delete(repository.path);
      try {
        await this.optimizeRepository(repository.path, { manual: true });
      } catch (error) {
        this.emit('optimization:auto-trigger-error', { repository: repository.name, error: error.message });
      }
    }, 5000);

    this.optimizationDebounce.set(repository.path, timeout);
  }

  /**
   * Stop monitoring a repository
   */
  stopMonitoring(repositoryPath: string): void {
    const watcher = this.watchers.get(repositoryPath);
    if (watcher) {
      watcher.close();
      this.watchers.delete(repositoryPath);
    }
  }

  private async cancelAllJobs(): Promise<void> {
    const jobs = Array.from(this.activeJobs.values());
    await Promise.all(jobs.map((job) => this.cancelJob(job.id)));
  }

  private generateJobId(): string {
    return `opt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private setupEventHandlers(): void {
    // Set up internal event handlers for telemetry and monitoring
    this.on('job:start', (job: OptimizationJob) => {
      this.telemetry.record({
        type: 'job_start',
        jobId: job.id,
        repository: job.repository.name,
        timestamp: new Date(),
      });
    });

    this.on('job:complete', (job: OptimizationJob) => {
      this.telemetry.record({
        type: 'job_complete',
        jobId: job.id,
        repository: job.repository.name,
        duration: job.endTime && job.startTime ? job.endTime.getTime() - job.startTime.getTime() : 0,
        changesApplied: job.result?.changes?.length || 0,
        timestamp: new Date(),
      });
    });

    this.on('job:fail', (job: OptimizationJob) => {
      this.telemetry.record({
        type: 'job_fail',
        jobId: job.id,
        repository: job.repository.name,
        error: job.result?.error,
        timestamp: new Date(),
      });

      // Trigger alert if configured
      if (this.circuitBreaker.getState() === 'open') {
        this.emit('alert:circuit-open', {
          message: 'Circuit breaker opened due to repeated failures',
          timestamp: new Date(),
        });
      }
    });
  }
}

/**
 * Rate limiter for controlling optimization frequency
 */
class RateLimiter {
  private tokens: number;
  private lastRefill: number;
  private maxTokens: number;
  private refillRate: number; // tokens per millisecond

  constructor(maxPerHour: number) {
    this.maxTokens = maxPerHour;
    this.tokens = maxPerHour;
    this.lastRefill = Date.now();
    this.refillRate = maxPerHour / (60 * 60 * 1000); // tokens per millisecond
  }

  allow(): boolean {
    this.refill();
    if (this.tokens >= 1) {
      this.tokens -= 1;
      return true;
    }
    return false;
  }

  getRemaining(): number {
    this.refill();
    return Math.floor(this.tokens);
  }

  updateLimit(newLimit: number): void {
    this.maxTokens = newLimit;
    this.refillRate = newLimit / (60 * 60 * 1000);
  }

  private refill(): void {
    const now = Date.now();
    const timePassed = now - this.lastRefill;
    const tokensToAdd = timePassed * this.refillRate;
    this.tokens = Math.min(this.maxTokens, this.tokens + tokensToAdd);
    this.lastRefill = now;
  }
}

/**
 * Circuit breaker for service protection
 */
class CircuitBreaker {
  private failures: number = 0;
  private lastFailureTime: number = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';
  private threshold: number;
  private timeout: number = 60000; // 1 minute

  constructor(threshold: number) {
    this.threshold = threshold;
  }

  allow(): boolean {
    switch (this.state) {
      case 'closed':
        return true;
      case 'open':
        if (Date.now() - this.lastFailureTime > this.timeout) {
          this.state = 'half-open';
          return true;
        }
        return false;
      case 'half-open':
        return true;
      default:
        return false;
    }
  }

  recordSuccess(): void {
    this.failures = 0;
    this.state = 'closed';
  }

  recordFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();

    if (this.failures >= this.threshold) {
      this.state = 'open';
    }
  }

  getState(): 'closed' | 'open' | 'half-open' {
    return this.state;
  }

  updateThreshold(newThreshold: number): void {
    this.threshold = newThreshold;
  }
}

/**
 * Scheduler for automated optimization runs
 */
class OptimizationScheduler {
  private intervalId?: NodeJS.Timeout;
  private optimizer: ContinuousOptimizer;
  private intervalMinutes: number;

  constructor(optimizer: ContinuousOptimizer) {
    this.optimizer = optimizer;
    this.intervalMinutes = 60; // default 1 hour
  }

  start(intervalMinutes: number): void {
    this.intervalMinutes = intervalMinutes;
    this.intervalId = setInterval(
      () => {
        this.runScheduledOptimization();
      },
      intervalMinutes * 60 * 1000
    );
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }

  updateInterval(intervalMinutes: number): void {
    this.stop();
    this.start(intervalMinutes);
  }

  isRunning(): boolean {
    return this.intervalId !== undefined;
  }

  private async runScheduledOptimization(): Promise<void> {
    const status = this.optimizer.getStatus();

    // Only run if we have capacity
    if (status.activeJobs < status.repositories.filter((r) => r.enabled).length) {
      // Find repositories that need optimization
      const reposToOptimize = status.repositories
        .filter((r) => r.enabled)
        .filter((r) => {
          if (!r.lastOptimized) return true;
          const hoursSinceLast = (Date.now() - r.lastOptimized.getTime()) / (1000 * 60 * 60);
          return hoursSinceLast >= 24; // At least 24 hours since last optimization
        });

      // Optimize repositories in parallel (respecting max concurrent limit)
      const promises = reposToOptimize
        .slice(0, status.repositories.length - status.activeJobs)
        .map((repo) => this.optimizer.optimizeRepository(repo.path));

      await Promise.allSettled(promises);
    }
  }
}

/**
 * Job queue for managing optimization tasks
 */
class JobQueue {
  private queue: Array<{ repository: RepositoryTarget; options: any }> = [];
  private maxConcurrent: number;

  constructor(maxConcurrent: number) {
    this.maxConcurrent = maxConcurrent;
  }

  enqueue(repository: RepositoryTarget, options: any): void {
    this.queue.push({ repository, options });
  }

  dequeue(): { repository: RepositoryTarget; options: any } | undefined {
    return this.queue.shift();
  }

  size(): number {
    return this.queue.length;
  }

  clear(): void {
    this.queue = [];
  }

  peek(): { repository: RepositoryTarget; options: any } | undefined {
    return this.queue[0];
  }
}

/**
 * Telemetry recorder for tracking optimization metrics
 */
interface TelemetryEvent {
  type: string;
  timestamp: Date;
  [key: string]: any;
}

class TelemetryRecorder {
  private events: TelemetryEvent[] = [];
  private maxEvents: number = 1000;

  record(event: TelemetryEvent): void {
    this.events.push(event);

    // Trim old events to prevent memory growth
    if (this.events.length > this.maxEvents) {
      this.events = this.events.slice(-this.maxEvents);
    }
  }

  getEvents(type?: string): TelemetryEvent[] {
    if (type) {
      return this.events.filter((e) => e.type === type);
    }
    return [...this.events];
  }

  getMetrics(): {
    totalJobs: number;
    completedJobs: number;
    failedJobs: number;
    averageDuration: number;
  } {
    const starts = this.events.filter((e) => e.type === 'job_start');
    const completes = this.events.filter((e) => e.type === 'job_complete');
    const fails = this.events.filter((e) => e.type === 'job_fail');

    const durations = completes.map((e) => e.duration || 0).filter((d) => d > 0);
    const avgDuration = durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;

    return {
      totalJobs: starts.length,
      completedJobs: completes.length,
      failedJobs: fails.length,
      averageDuration: avgDuration,
    };
  }

  clear(): void {
    this.events = [];
  }
}

/**
 * Rollback data structure for storing file backups
 */
interface RollbackData {
  resultId: string;
  timestamp: Date;
  changes: CodeChange[];
  fileBackups: Map<string, string>;
}
