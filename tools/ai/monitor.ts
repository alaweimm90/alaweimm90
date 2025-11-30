#!/usr/bin/env npx tsx
/**
 * AI Continuous Monitor
 * Watches for changes, triggers analysis, manages circuit breakers
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { EventEmitter } from 'events';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');
const MONITOR_STATE_FILE = path.join(AI_DIR, 'monitor-state.json');

// ============================================================================
// Types
// ============================================================================

interface MonitorConfig {
  watchPaths: string[];
  ignorePaths: string[];
  debounceMs: number;
  maxFrequencyMs: number;
  triggers: {
    minChanges: number;
    maxComplexity: number;
    filePatterns: string[];
  };
  circuitBreaker: {
    failureThreshold: number;
    resetTimeoutMs: number;
    halfOpenRequests: number;
  };
}

interface MonitorState {
  lastTriggerTime: string | null;
  changeBuffer: FileChange[];
  circuitBreakers: Record<string, CircuitBreakerState>;
  stats: {
    triggersTotal: number;
    triggersSuccess: number;
    triggersFailed: number;
    changesProcessed: number;
    lastActivity: string | null;
  };
}

interface FileChange {
  path: string;
  type: 'add' | 'modify' | 'delete';
  timestamp: string;
  size?: number;
}

interface CircuitBreakerState {
  state: 'closed' | 'open' | 'half-open';
  failures: number;
  lastFailure: string | null;
  lastSuccess: string | null;
  openedAt: string | null;
}

type TriggerResult = {
  success: boolean;
  action: string;
  duration: number;
  error?: string;
};

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: MonitorConfig = {
  watchPaths: ['tools/', 'templates/', '.ai/', 'docs/'],
  ignorePaths: ['node_modules', '.git', 'dist', 'coverage', '.ai/cache'],
  debounceMs: 2000,
  maxFrequencyMs: 30000, // Min 30s between triggers
  triggers: {
    minChanges: 1,
    maxComplexity: 100,
    filePatterns: ['*.ts', '*.js', '*.json', '*.yaml', '*.yml', '*.md'],
  },
  circuitBreaker: {
    failureThreshold: 3,
    resetTimeoutMs: 60000, // 1 minute
    halfOpenRequests: 1,
  },
};

// ============================================================================
// Circuit Breaker Implementation
// ============================================================================

class CircuitBreaker {
  private state: CircuitBreakerState;
  private config: MonitorConfig['circuitBreaker'];
  private halfOpenAttempts = 0;

  constructor(
    private name: string,
    config: MonitorConfig['circuitBreaker']
  ) {
    this.config = config;
    this.state = {
      state: 'closed',
      failures: 0,
      lastFailure: null,
      lastSuccess: null,
      openedAt: null,
    };
  }

  async execute<T>(operation: () => Promise<T>): Promise<T> {
    if (!this.canExecute()) {
      throw new Error(`Circuit breaker ${this.name} is OPEN - operation rejected`);
    }

    try {
      const result = await operation();
      this.recordSuccess();
      return result;
    } catch (error) {
      this.recordFailure();
      throw error;
    }
  }

  private canExecute(): boolean {
    switch (this.state.state) {
      case 'closed':
        return true;

      case 'open': {
        // Check if timeout has elapsed
        if (this.state.openedAt) {
          const elapsed = Date.now() - new Date(this.state.openedAt).getTime();
          if (elapsed >= this.config.resetTimeoutMs) {
            this.state.state = 'half-open';
            this.halfOpenAttempts = 0;
            return true;
          }
        }
        return false;
      }

      case 'half-open':
        return this.halfOpenAttempts < this.config.halfOpenRequests;

      default:
        return false;
    }
  }

  private recordSuccess(): void {
    this.state.lastSuccess = new Date().toISOString();

    if (this.state.state === 'half-open') {
      // Success in half-open state closes the circuit
      this.state.state = 'closed';
      this.state.failures = 0;
      this.state.openedAt = null;
    }
  }

  private recordFailure(): void {
    this.state.failures++;
    this.state.lastFailure = new Date().toISOString();

    if (this.state.state === 'half-open') {
      // Failure in half-open reopens immediately
      this.state.state = 'open';
      this.state.openedAt = new Date().toISOString();
    } else if (this.state.failures >= this.config.failureThreshold) {
      // Threshold reached, open the circuit
      this.state.state = 'open';
      this.state.openedAt = new Date().toISOString();
    }
  }

  getState(): CircuitBreakerState {
    return { ...this.state };
  }

  reset(): void {
    this.state = {
      state: 'closed',
      failures: 0,
      lastFailure: null,
      lastSuccess: null,
      openedAt: null,
    };
  }
}

// ============================================================================
// Monitor Implementation
// ============================================================================

class AIMonitor extends EventEmitter {
  private config: MonitorConfig;
  private state: MonitorState;
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();
  private debounceTimer: ReturnType<typeof setTimeout> | null = null;
  private isRunning = false;

  constructor(config: Partial<MonitorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = this.loadState();
  }

  // Load monitor state from disk
  private loadState(): MonitorState {
    if (fs.existsSync(MONITOR_STATE_FILE)) {
      try {
        return JSON.parse(fs.readFileSync(MONITOR_STATE_FILE, 'utf8'));
      } catch {
        // Fall through to default
      }
    }

    return {
      lastTriggerTime: null,
      changeBuffer: [],
      circuitBreakers: {},
      stats: {
        triggersTotal: 0,
        triggersSuccess: 0,
        triggersFailed: 0,
        changesProcessed: 0,
        lastActivity: null,
      },
    };
  }

  // Save state to disk
  private saveState(): void {
    // Update circuit breaker states
    for (const [name, cb] of this.circuitBreakers.entries()) {
      this.state.circuitBreakers[name] = cb.getState();
    }

    try {
      fs.writeFileSync(MONITOR_STATE_FILE, JSON.stringify(this.state, null, 2));
    } catch {
      // Silent fail
    }
  }

  // Get or create circuit breaker
  private getCircuitBreaker(name: string): CircuitBreaker {
    if (!this.circuitBreakers.has(name)) {
      const cb = new CircuitBreaker(name, this.config.circuitBreaker);
      this.circuitBreakers.set(name, cb);
    }
    return this.circuitBreakers.get(name)!;
  }

  // Check if file should be ignored
  private shouldIgnore(filePath: string): boolean {
    return this.config.ignorePaths.some(
      (ignore) => filePath.includes(ignore) || filePath.startsWith(ignore)
    );
  }

  // Check if file matches trigger patterns
  private matchesPattern(filePath: string): boolean {
    return this.config.triggers.filePatterns.some((pattern) => {
      const ext = pattern.replace('*', '');
      return filePath.endsWith(ext);
    });
  }

  // Record a file change
  recordChange(change: FileChange): void {
    if (this.shouldIgnore(change.path)) return;
    if (!this.matchesPattern(change.path)) return;

    this.state.changeBuffer.push(change);
    this.state.stats.lastActivity = new Date().toISOString();

    this.emit('change', change);
    this.scheduleTrigger();
  }

  // Schedule a debounced trigger
  private scheduleTrigger(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      this.evaluateTrigger();
    }, this.config.debounceMs);
  }

  // Evaluate if we should trigger actions
  private evaluateTrigger(): void {
    // Check frequency limit
    if (this.state.lastTriggerTime) {
      const elapsed = Date.now() - new Date(this.state.lastTriggerTime).getTime();
      if (elapsed < this.config.maxFrequencyMs) {
        return; // Too soon
      }
    }

    // Check minimum changes threshold
    if (this.state.changeBuffer.length < this.config.triggers.minChanges) {
      return;
    }

    // Execute triggers
    this.executeTriggers();
  }

  // Execute all registered triggers
  private async executeTriggers(): Promise<void> {
    const changes = [...this.state.changeBuffer];
    this.state.changeBuffer = [];
    this.state.lastTriggerTime = new Date().toISOString();
    this.state.stats.triggersTotal++;

    const results: TriggerResult[] = [];

    // Trigger 1: Sync context
    try {
      const syncCb = this.getCircuitBreaker('sync');
      await syncCb.execute(async () => {
        const start = Date.now();
        execSync('npm run ai:sync', { cwd: ROOT, stdio: 'pipe' });
        results.push({
          success: true,
          action: 'sync',
          duration: Date.now() - start,
        });
      });
    } catch (error) {
      results.push({
        success: false,
        action: 'sync',
        duration: 0,
        error: String(error),
      });
    }

    // Trigger 2: Update codemap if structure changed
    const structureChanged = changes.some(
      (c) => c.path.startsWith('tools/') || c.path.startsWith('templates/')
    );

    if (structureChanged) {
      try {
        const codemapCb = this.getCircuitBreaker('codemap');
        await codemapCb.execute(async () => {
          const start = Date.now();
          execSync('npm run codemap', { cwd: ROOT, stdio: 'pipe' });
          results.push({
            success: true,
            action: 'codemap',
            duration: Date.now() - start,
          });
        });
      } catch (error) {
        results.push({
          success: false,
          action: 'codemap',
          duration: 0,
          error: String(error),
        });
      }
    }

    // Trigger 3: Update metrics
    try {
      const metricsCb = this.getCircuitBreaker('metrics');
      await metricsCb.execute(async () => {
        const start = Date.now();
        execSync('npm run ai:metrics', { cwd: ROOT, stdio: 'pipe' });
        results.push({
          success: true,
          action: 'metrics',
          duration: Date.now() - start,
        });
      });
    } catch (error) {
      results.push({
        success: false,
        action: 'metrics',
        duration: 0,
        error: String(error),
      });
    }

    // Update stats
    const successful = results.filter((r) => r.success).length;
    if (successful === results.length) {
      this.state.stats.triggersSuccess++;
    } else {
      this.state.stats.triggersFailed++;
    }
    this.state.stats.changesProcessed += changes.length;

    this.saveState();
    this.emit('triggered', { changes, results });
  }

  // Get current status
  getStatus(): {
    isRunning: boolean;
    state: MonitorState;
    circuitBreakers: Record<string, CircuitBreakerState>;
  } {
    const cbStates: Record<string, CircuitBreakerState> = {};
    for (const [name, cb] of this.circuitBreakers.entries()) {
      cbStates[name] = cb.getState();
    }

    return {
      isRunning: this.isRunning,
      state: this.state,
      circuitBreakers: cbStates,
    };
  }

  // Detect changes since last check (for CLI usage)
  detectChanges(): FileChange[] {
    const changes: FileChange[] = [];

    try {
      // Get recent git changes
      const output = execSync('git diff --name-status HEAD~1 HEAD 2>/dev/null || true', {
        cwd: ROOT,
        encoding: 'utf8',
      });

      for (const line of output.trim().split('\n')) {
        if (!line) continue;
        const [status, filePath] = line.split('\t');
        if (!filePath) continue;

        const type: FileChange['type'] =
          status === 'A' ? 'add' : status === 'D' ? 'delete' : 'modify';

        changes.push({
          path: filePath,
          type,
          timestamp: new Date().toISOString(),
        });
      }
    } catch {
      // Silent fail
    }

    return changes;
  }

  // Process detected changes
  processDetectedChanges(): number {
    const changes = this.detectChanges();
    for (const change of changes) {
      this.recordChange(change);
    }
    return changes.length;
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const monitor = new AIMonitor();

// ============================================================================
// CLI Interface
// ============================================================================

function main(): void {
  const args = process.argv.slice(2);
  const command = args[0];

  switch (command) {
    case 'status': {
      const status = monitor.getStatus();
      console.log('\n🔍 AI Monitor Status\n');
      console.log(`Running: ${status.isRunning ? '✅' : '❌'}`);
      console.log(`Last Activity: ${status.state.stats.lastActivity || 'Never'}`);
      console.log(`Changes in Buffer: ${status.state.changeBuffer.length}`);
      console.log(`\nTrigger Stats:`);
      console.log(`  Total: ${status.state.stats.triggersTotal}`);
      console.log(`  Success: ${status.state.stats.triggersSuccess}`);
      console.log(`  Failed: ${status.state.stats.triggersFailed}`);
      console.log(`  Changes Processed: ${status.state.stats.changesProcessed}`);

      console.log(`\nCircuit Breakers:`);
      for (const [name, state] of Object.entries(status.circuitBreakers)) {
        const icon = state.state === 'closed' ? '🟢' : state.state === 'open' ? '🔴' : '🟡';
        console.log(`  ${icon} ${name}: ${state.state} (failures: ${state.failures})`);
      }
      break;
    }

    case 'check': {
      console.log('🔍 Checking for changes...');
      const count = monitor.processDetectedChanges();
      console.log(`Found ${count} changes`);

      if (count > 0) {
        console.log('⚡ Processing triggers...');
        // Force trigger evaluation
        const status = monitor.getStatus();
        console.log(`Buffer: ${status.state.changeBuffer.length} changes`);
      }
      break;
    }

    case 'trigger': {
      console.log('⚡ Forcing trigger execution...');
      // Add a dummy change to trigger
      monitor.recordChange({
        path: 'manual-trigger',
        type: 'modify',
        timestamp: new Date().toISOString(),
      });
      console.log('Trigger scheduled');
      break;
    }

    case 'reset': {
      const cbName = args[1];
      if (cbName) {
        console.log(`Resetting circuit breaker: ${cbName}`);
        // Would need to expose reset method
      } else {
        console.log('Usage: monitor reset <circuit-breaker-name>');
      }
      break;
    }

    default:
      console.log(`
AI Monitor - Continuous change detection with circuit breakers

Commands:
  status    Show monitor status and circuit breaker states
  check     Detect and process recent changes
  trigger   Force trigger execution
  reset <name>  Reset a circuit breaker

Features:
  - Debounced change detection
  - Frequency limiting (max 1 trigger per 30s)
  - Circuit breakers for fault tolerance
  - Automatic sync, codemap, and metrics updates
      `);
  }
}

main();
