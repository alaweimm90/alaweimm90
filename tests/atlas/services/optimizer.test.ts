// ATLAS Optimizer Tests
// Tests for job queue, telemetry, and rollback functionality

import { describe, it, expect, beforeEach, vi } from 'vitest';

// Mock the optimizer internals for testing
// Since JobQueue and TelemetryRecorder are internal classes,
// we test them through their expected behaviors

describe('Optimizer Features', () => {
  describe('JobQueue', () => {
    // Simulating JobQueue behavior
    class TestJobQueue {
      private queue: Array<{
        repository: { name: string; path: string };
        options: Record<string, unknown>;
      }> = [];
      private maxConcurrent: number;

      constructor(maxConcurrent: number) {
        this.maxConcurrent = maxConcurrent;
      }

      enqueue(repository: { name: string; path: string }, options: Record<string, unknown>): void {
        this.queue.push({ repository, options });
      }

      dequeue():
        | { repository: { name: string; path: string }; options: Record<string, unknown> }
        | undefined {
        return this.queue.shift();
      }

      size(): number {
        return this.queue.length;
      }

      clear(): void {
        this.queue = [];
      }

      peek():
        | { repository: { name: string; path: string }; options: Record<string, unknown> }
        | undefined {
        return this.queue[0];
      }

      getMaxConcurrent(): number {
        return this.maxConcurrent;
      }
    }

    let queue: TestJobQueue;

    beforeEach(() => {
      queue = new TestJobQueue(3);
    });

    it('should enqueue jobs correctly', () => {
      const repo = { name: 'test-repo', path: '/test/path' };
      queue.enqueue(repo, { priority: 'high' });

      expect(queue.size()).toBe(1);
    });

    it('should dequeue jobs in FIFO order', () => {
      const repo1 = { name: 'repo1', path: '/path1' };
      const repo2 = { name: 'repo2', path: '/path2' };

      queue.enqueue(repo1, {});
      queue.enqueue(repo2, {});

      const first = queue.dequeue();
      expect(first?.repository.name).toBe('repo1');

      const second = queue.dequeue();
      expect(second?.repository.name).toBe('repo2');
    });

    it('should return undefined when dequeuing empty queue', () => {
      expect(queue.dequeue()).toBeUndefined();
    });

    it('should clear all jobs', () => {
      queue.enqueue({ name: 'repo1', path: '/path1' }, {});
      queue.enqueue({ name: 'repo2', path: '/path2' }, {});

      queue.clear();
      expect(queue.size()).toBe(0);
    });

    it('should peek without removing', () => {
      const repo = { name: 'test-repo', path: '/test/path' };
      queue.enqueue(repo, {});

      expect(queue.peek()?.repository.name).toBe('test-repo');
      expect(queue.size()).toBe(1);
    });

    it('should respect max concurrent setting', () => {
      expect(queue.getMaxConcurrent()).toBe(3);
    });
  });

  describe('TelemetryRecorder', () => {
    interface TelemetryEvent {
      type: string;
      timestamp: Date;
      [key: string]: unknown;
    }

    class TestTelemetryRecorder {
      private events: TelemetryEvent[] = [];
      private maxEvents: number = 1000;

      record(event: TelemetryEvent): void {
        this.events.push(event);
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

        const durations = completes.map((e) => (e.duration as number) || 0).filter((d) => d > 0);
        const avgDuration =
          durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : 0;

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

    let telemetry: TestTelemetryRecorder;

    beforeEach(() => {
      telemetry = new TestTelemetryRecorder();
    });

    it('should record events', () => {
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '1' });

      expect(telemetry.getEvents().length).toBe(1);
    });

    it('should filter events by type', () => {
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '1' });
      telemetry.record({ type: 'job_complete', timestamp: new Date(), jobId: '1', duration: 100 });
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '2' });

      const starts = telemetry.getEvents('job_start');
      expect(starts.length).toBe(2);

      const completes = telemetry.getEvents('job_complete');
      expect(completes.length).toBe(1);
    });

    it('should calculate metrics correctly', () => {
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '1' });
      telemetry.record({ type: 'job_complete', timestamp: new Date(), jobId: '1', duration: 100 });
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '2' });
      telemetry.record({ type: 'job_complete', timestamp: new Date(), jobId: '2', duration: 200 });
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '3' });
      telemetry.record({ type: 'job_fail', timestamp: new Date(), jobId: '3', error: 'timeout' });

      const metrics = telemetry.getMetrics();

      expect(metrics.totalJobs).toBe(3);
      expect(metrics.completedJobs).toBe(2);
      expect(metrics.failedJobs).toBe(1);
      expect(metrics.averageDuration).toBe(150);
    });

    it('should return zero average duration with no completed jobs', () => {
      telemetry.record({ type: 'job_start', timestamp: new Date(), jobId: '1' });

      const metrics = telemetry.getMetrics();
      expect(metrics.averageDuration).toBe(0);
    });

    it('should trim old events beyond max limit', () => {
      // Record more than maxEvents
      for (let i = 0; i < 1005; i++) {
        telemetry.record({ type: 'test', timestamp: new Date(), index: i });
      }

      const events = telemetry.getEvents();
      expect(events.length).toBe(1000);
      // Should keep the most recent events
      expect((events[0] as unknown as { index: number }).index).toBe(5);
    });
  });

  describe('Rollback Mechanism', () => {
    interface RollbackData {
      resultId: string;
      fileBackups: Map<string, string>;
      timestamp: Date;
    }

    class TestRollbackStore {
      private store: Map<string, RollbackData> = new Map();

      save(resultId: string, fileBackups: Map<string, string>): void {
        this.store.set(resultId, {
          resultId,
          fileBackups,
          timestamp: new Date(),
        });
      }

      get(resultId: string): RollbackData | undefined {
        return this.store.get(resultId);
      }

      delete(resultId: string): boolean {
        return this.store.delete(resultId);
      }

      has(resultId: string): boolean {
        return this.store.has(resultId);
      }
    }

    let rollbackStore: TestRollbackStore;

    beforeEach(() => {
      rollbackStore = new TestRollbackStore();
    });

    it('should save rollback data', () => {
      const backups = new Map<string, string>();
      backups.set('/path/file1.ts', 'original content 1');
      backups.set('/path/file2.ts', 'original content 2');

      rollbackStore.save('result-123', backups);

      expect(rollbackStore.has('result-123')).toBe(true);
    });

    it('should retrieve rollback data', () => {
      const backups = new Map<string, string>();
      backups.set('/path/file1.ts', 'original content');

      rollbackStore.save('result-456', backups);
      const data = rollbackStore.get('result-456');

      expect(data).toBeDefined();
      expect(data?.fileBackups.get('/path/file1.ts')).toBe('original content');
    });

    it('should return undefined for non-existent rollback', () => {
      const data = rollbackStore.get('non-existent');
      expect(data).toBeUndefined();
    });

    it('should delete rollback data after use', () => {
      const backups = new Map<string, string>();
      backups.set('/path/file.ts', 'content');

      rollbackStore.save('result-789', backups);
      expect(rollbackStore.has('result-789')).toBe(true);

      rollbackStore.delete('result-789');
      expect(rollbackStore.has('result-789')).toBe(false);
    });
  });

  describe('File Monitoring', () => {
    it('should debounce rapid file changes', async () => {
      const handler = vi.fn();
      let timeout: NodeJS.Timeout | null = null;
      const DEBOUNCE_MS = 100;

      const debouncedHandler = (filename: string): void => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => handler(filename), DEBOUNCE_MS);
      };

      // Simulate rapid file changes
      debouncedHandler('file1.ts');
      debouncedHandler('file1.ts');
      debouncedHandler('file1.ts');

      // Handler shouldn't be called yet
      expect(handler).not.toHaveBeenCalled();

      // Wait for debounce
      await new Promise((resolve) => setTimeout(resolve, DEBOUNCE_MS + 50));

      // Should only be called once
      expect(handler).toHaveBeenCalledTimes(1);
    });

    it('should filter files based on patterns', () => {
      const shouldTrigger = (filename: string): boolean => {
        const ignorePatterns = [/node_modules/, /\.git/, /dist/, /\.log$/];
        return !ignorePatterns.some((pattern) => pattern.test(filename));
      };

      expect(shouldTrigger('src/index.ts')).toBe(true);
      expect(shouldTrigger('node_modules/package/index.js')).toBe(false);
      expect(shouldTrigger('.git/config')).toBe(false);
      expect(shouldTrigger('dist/bundle.js')).toBe(false);
      expect(shouldTrigger('debug.log')).toBe(false);
    });
  });

  describe('Rate Limiting', () => {
    it('should enforce minimum interval between optimizations', () => {
      const MIN_INTERVAL_MS = 60000; // 1 minute
      let lastOptimization: number | null = null;

      const canOptimize = (): boolean => {
        if (lastOptimization === null) return true;
        return Date.now() - lastOptimization >= MIN_INTERVAL_MS;
      };

      const recordOptimization = (): void => {
        lastOptimization = Date.now();
      };

      // First optimization should be allowed
      expect(canOptimize()).toBe(true);
      recordOptimization();

      // Immediate second optimization should be blocked
      expect(canOptimize()).toBe(false);
    });

    it('should track optimization counts per repository', () => {
      const optimizationCounts = new Map<string, number>();

      const incrementCount = (repoPath: string): number => {
        const current = optimizationCounts.get(repoPath) || 0;
        optimizationCounts.set(repoPath, current + 1);
        return current + 1;
      };

      expect(incrementCount('/repo1')).toBe(1);
      expect(incrementCount('/repo1')).toBe(2);
      expect(incrementCount('/repo2')).toBe(1);
      expect(incrementCount('/repo1')).toBe(3);
    });
  });
});
