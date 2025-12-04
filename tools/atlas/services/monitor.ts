/**
 * ATLAS Repository Monitor Service
 * Continuous monitoring of repositories for changes and optimization triggers
 */

import { RepositoryAnalyzer } from '@atlas/analysis/analyzer';
import { ContinuousOptimizer } from './optimizer';
import { EventEmitter } from 'events';
import * as fs from 'fs';
import { promises as fsPromises } from 'fs';
import * as path from 'path';
import chokidar from 'chokidar';

export interface MonitorConfig {
  repositories: MonitoredRepository[];
  polling: {
    enabled: boolean;
    interval: number; // seconds
  };
  filesystem: {
    watchEnabled: boolean;
    debounceMs: number;
  };
  triggers: {
    onFileChange: boolean;
    onCommit: boolean;
    onPush: boolean;
    minChangesThreshold: number;
    maxFrequencyMs: number; // minimum time between triggers
  };
  analysis: {
    incremental: boolean;
    cacheResults: boolean;
    cacheTtl: number; // minutes
  };
}

export interface MonitoredRepository {
  path: string;
  name: string;
  enabled: boolean;
  branch: string;
  lastCommit?: string;
  lastAnalysis?: Date;
  changeCount: number;
  watchers: chokidar.FSWatcher[];
  watcherAbortControllers: Map<string, AbortController>;
}

export interface RepositoryChange {
  repository: string;
  type: 'file' | 'commit' | 'push';
  files: ChangedFile[];
  timestamp: Date;
  commitHash?: string;
  author?: string;
}

export interface ChangedFile {
  path: string;
  type: 'added' | 'modified' | 'deleted' | 'renamed';
  oldPath?: string;
  size?: number;
  linesChanged?: number;
}

export interface AnalysisResult {
  repository: string;
  timestamp: Date;
  triggeredBy: RepositoryChange;
  metrics: RepositoryMetrics;
  recommendations: string[];
  cacheHit: boolean;
  duration: number; // milliseconds
}

export interface RepositoryMetrics {
  chaosScore: number;
  complexityScore: number;
  fileCount: number;
  totalLines: number;
  issuesCount: number;
  coverage?: number;
  testCount?: number;
}

export class RepositoryMonitor extends EventEmitter {
  private config: MonitorConfig;
  private analyzer: RepositoryAnalyzer;
  private optimizer: ContinuousOptimizer;
  private repositories: Map<string, MonitoredRepository> = new Map();
  private analysisCache: Map<string, { result: AnalysisResult; expires: Date }> = new Map();
  private pollingTimer?: NodeJS.Timeout;
  private changeBuffers: Map<string, RepositoryChange[]> = new Map();
  private debounceTimers: Map<string, NodeJS.Timeout> = new Map();
  private lastTriggerTimes: Map<string, number> = new Map();

  constructor(config: MonitorConfig, optimizer: ContinuousOptimizer) {
    super();
    this.config = config;
    this.analyzer = new RepositoryAnalyzer();
    this.optimizer = optimizer;

    this.setupEventHandlers();
  }

  /**
   * Start monitoring all configured repositories
   */
  async start(): Promise<void> {
    this.emit('monitor:start', { timestamp: new Date() });

    // Initialize repositories
    for (const repoConfig of this.config.repositories) {
      if (repoConfig.enabled) {
        await this.addRepository(repoConfig);
      }
    }

    // Start polling if enabled
    if (this.config.polling.enabled) {
      this.startPolling();
    }

    this.emit('monitor:ready', {
      repositories: this.repositories.size,
      timestamp: new Date(),
    });
  }

  /**
   * Stop monitoring and clean up resources
   */
  async stop(): Promise<void> {
    // Stop polling
    if (this.pollingTimer) {
      clearInterval(this.pollingTimer);
      this.pollingTimer = undefined;
    }

    // Clear debounce timers
    for (const timer of this.debounceTimers.values()) {
      clearTimeout(timer);
    }
    this.debounceTimers.clear();

    // Stop file watchers and abort controllers
    for (const repo of this.repositories.values()) {
      for (const watcher of repo.watchers) {
        await watcher.close();
      }
      repo.watchers = [];

      // Abort all pending operations
      for (const controller of repo.watcherAbortControllers.values()) {
        controller.abort();
      }
      repo.watcherAbortControllers.clear();
    }

    this.repositories.clear();
    this.analysisCache.clear();
    this.changeBuffers.clear();

    this.emit('monitor:stop', { timestamp: new Date() });
  }

  /**
   * Add a repository to monitoring
   */
  async addRepository(
    config: Omit<MonitoredRepository, 'watchers' | 'changeCount' | 'watcherAbortControllers'>
  ): Promise<void> {
    const repository: MonitoredRepository = {
      ...config,
      watchers: [],
      changeCount: 0,
      watcherAbortControllers: new Map(),
    };

    // Validate repository exists and is a git repo
    if (!fs.existsSync(path.join(repository.path, '.git'))) {
      throw new Error(`Not a git repository: ${repository.path}`);
    }

    this.repositories.set(repository.path, repository);

    // Start file system watching if enabled
    if (this.config.filesystem.watchEnabled) {
      await this.startFileWatching(repository);
    }

    // Get initial commit hash
    repository.lastCommit = await this.getCurrentCommit(repository.path);

    this.emit('repository:add', { repository: repository.name, path: repository.path });
  }

  /**
   * Remove a repository from monitoring
   */
  async removeRepository(repoPath: string): Promise<void> {
    const repository = this.repositories.get(repoPath);
    if (!repository) {
      return;
    }

    // Stop watchers
    for (const watcher of repository.watchers) {
      await watcher.close();
    }

    // Abort all pending operations
    for (const controller of repository.watcherAbortControllers.values()) {
      controller.abort();
    }
    repository.watcherAbortControllers.clear();

    this.repositories.delete(repoPath);
    this.changeBuffers.delete(repoPath);
    this.analysisCache.delete(repoPath);

    // Clear any pending timers
    const debounceTimer = this.debounceTimers.get(repoPath);
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      this.debounceTimers.delete(repoPath);
    }

    this.emit('repository:remove', { repository: repository.name, path: repoPath });
  }

  /**
   * Get current monitoring status
   */
  getStatus(): {
    repositories: number;
    activeWatchers: number;
    polling: boolean;
    cacheSize: number;
    pendingChanges: number;
  } {
    let activeWatchers = 0;
    for (const repo of this.repositories.values()) {
      activeWatchers += repo.watchers.length;
    }

    let pendingChanges = 0;
    for (const changes of this.changeBuffers.values()) {
      pendingChanges += changes.length;
    }

    return {
      repositories: this.repositories.size,
      activeWatchers,
      polling: this.pollingTimer !== undefined,
      cacheSize: this.analysisCache.size,
      pendingChanges,
    };
  }

  /**
   * Manually trigger analysis for a repository
   */
  async triggerAnalysis(repoPath: string, reason: string = 'manual'): Promise<AnalysisResult> {
    const repository = this.repositories.get(repoPath);
    if (!repository) {
      throw new Error(`Repository not found: ${repoPath}`);
    }

    const change: RepositoryChange = {
      repository: repository.name,
      type: 'file',
      files: [],
      timestamp: new Date(),
    };

    return this.performAnalysis(repository, change, reason);
  }

  /**
   * Get analysis cache for a repository
   */
  getCachedAnalysis(repoPath: string): AnalysisResult | null {
    const cached = this.analysisCache.get(repoPath);
    if (!cached || cached.expires < new Date()) {
      this.analysisCache.delete(repoPath);
      return null;
    }
    return cached.result;
  }

  /**
   * Clear analysis cache
   */
  clearCache(repoPath?: string): void {
    if (repoPath) {
      this.analysisCache.delete(repoPath);
    } else {
      this.analysisCache.clear();
    }
  }

  /**
   * Update monitoring configuration
   */
  updateConfig(newConfig: Partial<MonitorConfig>): void {
    const oldConfig = JSON.parse(JSON.stringify(this.config));
    this.config = { ...this.config, ...newConfig };

    // Deep compare polling config
    if (!this.deepEquals(newConfig.polling, oldConfig.polling)) {
      if (this.pollingTimer) {
        clearInterval(this.pollingTimer);
        this.startPolling();
      }
    }

    // Deep compare filesystem config
    if (!this.deepEquals(newConfig.filesystem, oldConfig.filesystem)) {
      this.restartFileWatching();
    }

    this.emit('config:update', { config: this.config, timestamp: new Date() });
  }

  private deepEquals(obj1: unknown, obj2: unknown): boolean {
    if (obj1 === obj2) return true;
    if (obj1 === null || obj2 === null) return obj1 === obj2;
    if (typeof obj1 !== 'object' || typeof obj2 !== 'object') return false;

    const keys1 = Object.keys(obj1 as Record<string, unknown>);
    const keys2 = Object.keys(obj2 as Record<string, unknown>);

    if (keys1.length !== keys2.length) return false;

    for (const key of keys1) {
      if (
        !this.deepEquals(
          (obj1 as Record<string, unknown>)[key],
          (obj2 as Record<string, unknown>)[key]
        )
      ) {
        return false;
      }
    }

    return true;
  }

  private async startFileWatching(repository: MonitoredRepository): Promise<void> {
    try {
      const ignorePatterns = [
        /\.git/,
        /node_modules/,
        /\.DS_Store/,
        /Thumbs\.db/,
        /\.(log|tmp|cache)$/,
        /dist/,
        /build/,
        /\.env/,
      ];

      const watcher = chokidar.watch(repository.path, {
        persistent: true,
        ignored: (path: string) => ignorePatterns.some((pattern) => pattern.test(path)),
        ignoreInitial: true,
        awaitWriteFinish: {
          stabilityThreshold: 100,
          pollInterval: 100,
        },
        usePolling: process.platform === 'linux',
        interval: 100,
      });

      watcher.on('change', (filePath: string) => {
        this.handleFileChange(repository, 'change', filePath);
      });

      watcher.on('add', (filePath: string) => {
        this.handleFileChange(repository, 'add', filePath);
      });

      watcher.on('unlink', (filePath: string) => {
        this.handleFileChange(repository, 'unlink', filePath);
      });

      watcher.on('error', (error: unknown) => {
        this.emit('watcher:error', {
          repository: repository.name,
          error: error instanceof Error ? error.message : String(error),
        });
      });

      repository.watchers.push(watcher);
    } catch (error) {
      this.emit('watcher:error', {
        repository: repository.name,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  private handleFileChange(
    repository: MonitoredRepository,
    eventType: string,
    filename: string
  ): void {
    const fullPath = path.join(repository.path, filename);

    // Determine change type from event
    let changeType: ChangedFile['type'] = 'modified';
    if (eventType === 'add') {
      changeType = 'added';
    } else if (eventType === 'unlink') {
      changeType = 'deleted';
    }

    // Use async I/O to get file stats (non-blocking)
    this.getFileStats(fullPath)
      .then((stats) => {
        const changedFile: ChangedFile = {
          path: filename,
          type: changeType,
          size: stats.size,
          linesChanged: stats.linesChanged,
        };

        // Buffer the change
        let buffer = this.changeBuffers.get(repository.path);
        if (!buffer) {
          buffer = [];
          this.changeBuffers.set(repository.path, buffer);
        }

        // Check buffer cap (1000 events)
        const MAX_BUFFER_SIZE = 1000;
        if (buffer.length >= MAX_BUFFER_SIZE) {
          this.emit('buffer:warn', {
            repository: repository.name,
            bufferSize: buffer.length,
            message: `Repository buffer cap (${MAX_BUFFER_SIZE}) reached, truncating`,
          });
          buffer = buffer.slice(-MAX_BUFFER_SIZE + 1);
          this.changeBuffers.set(repository.path, buffer);
        }

        buffer.push({
          repository: repository.name,
          type: 'file',
          files: [changedFile],
          timestamp: new Date(),
        });

        repository.changeCount++;

        // Debounce analysis trigger
        this.scheduleAnalysisTrigger(repository);
      })
      .catch((error) => {
        // Log error but don't block
        this.emit('file:stat:error', {
          repository: repository.name,
          file: filename,
          error: error instanceof Error ? error.message : String(error),
        });
      });
  }

  private async getFileStats(filePath: string): Promise<{ size: number; linesChanged: number }> {
    try {
      const stats = await fsPromises.stat(filePath);
      let linesChanged = 0;

      // Only read first few KB to estimate lines changed
      if (stats.size > 0 && stats.size < 102400) {
        // 100KB limit
        const content = await fsPromises.readFile(filePath, 'utf8');
        linesChanged = content.split('\n').length;
      } else if (stats.size >= 102400) {
        // For larger files, estimate based on size
        linesChanged = Math.ceil(stats.size / 50); // Rough estimate
      }

      return { size: stats.size, linesChanged };
    } catch {
      // File might have been deleted or inaccessible
      return { size: 0, linesChanged: 0 };
    }
  }

  private scheduleAnalysisTrigger(repository: MonitoredRepository): void {
    // Clear existing timer
    const existingTimer = this.debounceTimers.get(repository.path);
    if (existingTimer) {
      clearTimeout(existingTimer);
    }

    // Check frequency limit
    const lastTrigger = this.lastTriggerTimes.get(repository.path);
    const now = Date.now();
    const timeSinceLastTrigger = lastTrigger ? now - lastTrigger : Infinity;

    if (timeSinceLastTrigger < this.config.triggers.maxFrequencyMs) {
      // Too frequent - queue for later instead of dropping
      const delayMs = this.config.triggers.maxFrequencyMs - timeSinceLastTrigger;
      const timer = setTimeout(() => {
        this.triggerAnalysisForRepository(repository);
      }, delayMs + this.config.filesystem.debounceMs);

      this.debounceTimers.set(repository.path, timer);
      return;
    }

    // Schedule debounced analysis
    const timer = setTimeout(() => {
      this.triggerAnalysisForRepository(repository);
    }, this.config.filesystem.debounceMs);

    this.debounceTimers.set(repository.path, timer);
  }

  private async triggerAnalysisForRepository(repository: MonitoredRepository): Promise<void> {
    const buffer = this.changeBuffers.get(repository.path);
    if (!buffer || buffer.length === 0) {
      return;
    }

    // Check if we have enough changes to trigger analysis
    const totalChanges = buffer.reduce((sum, change) => sum + change.files.length, 0);
    if (totalChanges < this.config.triggers.minChangesThreshold) {
      return;
    }

    // Combine all changes into one trigger
    const combinedChange: RepositoryChange = {
      repository: repository.name,
      type: 'file',
      files: buffer.flatMap((change) => change.files),
      timestamp: new Date(),
    };

    // Clear buffer
    this.changeBuffers.delete(repository.path);

    // Update last trigger time
    this.lastTriggerTimes.set(repository.path, Date.now());

    try {
      const result = await this.performAnalysis(repository, combinedChange, 'file-change');
      this.emit('analysis:complete', result);

      // Trigger optimization if needed
      if (this.shouldTriggerOptimization(result)) {
        await this.optimizer.optimizeRepository(repository.path, { force: false });
      }
    } catch (error) {
      this.emit('analysis:error', {
        repository: repository.name,
        error: error instanceof Error ? error.message : String(error),
        change: combinedChange,
      });
    }
  }

  private async performAnalysis(
    repository: MonitoredRepository,
    change: RepositoryChange,
    _triggerReason: string
  ): Promise<AnalysisResult> {
    const startTime = Date.now();

    // Check cache first
    if (this.config.analysis.cacheResults) {
      const cached = this.getCachedAnalysis(repository.path);
      if (cached && this.config.analysis.incremental) {
        return {
          ...cached,
          triggeredBy: change,
          cacheHit: true,
          duration: Date.now() - startTime,
        };
      }
    }

    // Perform fresh analysis
    const analysis = await this.analyzer.analyze(repository.path);

    const metrics: RepositoryMetrics = {
      chaosScore: analysis.chaosScore || 0,
      complexityScore: analysis.complexityScore || 0,
      fileCount: analysis.files?.length || 0,
      totalLines: analysis.totalLines || 0,
      issuesCount: analysis.issues?.length || 0,
    };

    const result: AnalysisResult = {
      repository: repository.name,
      timestamp: new Date(),
      triggeredBy: change,
      metrics,
      recommendations: this.generateRecommendations(metrics),
      cacheHit: false,
      duration: Date.now() - startTime,
    };

    // Cache result
    if (this.config.analysis.cacheResults) {
      const expires = new Date(Date.now() + this.config.analysis.cacheTtl * 60 * 1000);
      this.analysisCache.set(repository.path, { result, expires });
    }

    repository.lastAnalysis = new Date();

    return result;
  }

  private shouldTriggerOptimization(result: AnalysisResult): boolean {
    // Trigger optimization based on metrics thresholds
    // This would be configurable, but for now use simple heuristics
    return (
      result.metrics.chaosScore > 0.7 ||
      result.metrics.complexityScore > 0.8 ||
      result.metrics.issuesCount > 10
    );
  }

  private generateRecommendations(metrics: RepositoryMetrics): string[] {
    const recommendations: string[] = [];

    if (metrics.chaosScore > 0.8) {
      recommendations.push('High chaos detected - consider refactoring complex functions');
    }

    if (metrics.complexityScore > 0.9) {
      recommendations.push('High complexity - review code structure and dependencies');
    }

    if (metrics.issuesCount > 20) {
      recommendations.push('Multiple code issues found - run linter and fix issues');
    }

    if (metrics.fileCount > 1000) {
      recommendations.push('Large codebase - consider modularization');
    }

    return recommendations;
  }

  private async getCurrentCommit(repoPath: string): Promise<string | undefined> {
    try {
      const { execSync } = await import('child_process');
      const hash = execSync('git rev-parse HEAD', {
        cwd: repoPath,
        encoding: 'utf8',
      }).trim();
      return hash;
    } catch {
      return undefined;
    }
  }

  /**
   * Private method for future use in git-based diff analysis
   * @internal Reserved for future git integration
   */
  // @ts-expect-error Reserved for future git integration feature
  private async _getDiffStats(repoPath: string): Promise<string[]> {
    try {
      const { execSync } = await import('child_process');
      const output = execSync('git diff --name-only HEAD', {
        cwd: repoPath,
        encoding: 'utf8',
      });
      return output.split('\n').filter((line) => line.length > 0);
    } catch {
      return [];
    }
  }

  private startPolling(): void {
    this.pollingTimer = setInterval(async () => {
      for (const repository of this.repositories.values()) {
        try {
          await this.checkForChanges(repository);
        } catch (error) {
          this.emit('polling:error', {
            repository: repository.name,
            error: error instanceof Error ? error.message : String(error),
          });
        }
      }
    }, this.config.polling.interval * 1000);
  }

  private async checkForChanges(repository: MonitoredRepository): Promise<void> {
    // Check for new commits
    const currentCommit = await this.getCurrentCommit(repository.path);
    if (currentCommit && currentCommit !== repository.lastCommit) {
      const change: RepositoryChange = {
        repository: repository.name,
        type: 'commit',
        files: [], // Would need to get changed files from git
        timestamp: new Date(),
        commitHash: currentCommit,
      };

      repository.lastCommit = currentCommit;

      if (this.config.triggers.onCommit) {
        const result = await this.performAnalysis(repository, change, 'commit');
        this.emit('analysis:complete', result);
      }
    }
  }

  private restartFileWatching(): Promise<void> {
    // Stop all current watchers
    for (const repo of this.repositories.values()) {
      for (const watcher of repo.watchers) {
        watcher.close();
      }
      repo.watchers = [];
    }

    // Restart watching for enabled repos
    const promises = Array.from(this.repositories.values())
      .filter((repo) => repo.enabled)
      .map((repo) => this.startFileWatching(repo));

    return Promise.all(promises).then(() => undefined);
  }

  private setupEventHandlers(): void {
    // Set up event handlers for telemetry and coordination with optimizer
    this.on('analysis:complete', (_result: AnalysisResult) => {
      // Could trigger notifications or further processing
    });

    this.on('analysis:error', (_error) => {
      // Log errors and potentially retry
    });
  }
}
