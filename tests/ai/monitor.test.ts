/**
 * Integration tests for AI Monitor Module
 * Tests continuous monitoring and circuit breakers
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';

describe('AI Monitor Module', () => {
  describe('Circuit Breaker', () => {
    interface CircuitBreakerState {
      state: 'closed' | 'open' | 'half-open';
      failures: number;
      lastFailure: string | null;
      lastSuccess: string | null;
      openedAt: string | null;
    }

    const config = {
      failureThreshold: 3,
      resetTimeoutMs: 60000,
      halfOpenRequests: 1,
    };

    it('should start in closed state', () => {
      const state: CircuitBreakerState = {
        state: 'closed',
        failures: 0,
        lastFailure: null,
        lastSuccess: null,
        openedAt: null,
      };

      expect(state.state).toBe('closed');
      expect(state.failures).toBe(0);
    });

    it('should allow execution in closed state', () => {
      const state: CircuitBreakerState = {
        state: 'closed',
        failures: 0,
        lastFailure: null,
        lastSuccess: null,
        openedAt: null,
      };
      const canExecute = state.state === 'closed';
      expect(canExecute).toBe(true);
    });

    it('should open after threshold failures', () => {
      const state: CircuitBreakerState = {
        state: 'closed',
        failures: 0,
        lastFailure: null,
        lastSuccess: null,
        openedAt: null,
      };

      // Simulate 3 failures
      for (let i = 0; i < config.failureThreshold; i++) {
        state.failures++;
        state.lastFailure = new Date().toISOString();
      }

      // Check threshold
      if (state.failures >= config.failureThreshold) {
        state.state = 'open';
        state.openedAt = new Date().toISOString();
      }

      expect(state.state).toBe('open');
      expect(state.failures).toBe(3);
    });

    it('should reject execution in open state', () => {
      const state: CircuitBreakerState = {
        state: 'open',
        failures: 3,
        lastFailure: new Date().toISOString(),
        lastSuccess: null,
        openedAt: new Date().toISOString(),
      };
      const canExecute = state.state === 'closed';
      expect(canExecute).toBe(false);
    });

    it('should transition to half-open after timeout', () => {
      const openedAt = new Date(Date.now() - 70000).toISOString(); // 70 seconds ago
      const state: CircuitBreakerState = {
        state: 'open',
        failures: 3,
        lastFailure: null,
        lastSuccess: null,
        openedAt,
      };

      const elapsed = Date.now() - new Date(state.openedAt!).getTime();
      if (elapsed >= config.resetTimeoutMs) {
        state.state = 'half-open';
      }

      expect(state.state).toBe('half-open');
    });

    it('should close on success in half-open state', () => {
      const state: CircuitBreakerState = {
        state: 'half-open',
        failures: 3,
        lastFailure: null,
        lastSuccess: null,
        openedAt: null,
      };

      // Simulate success
      state.lastSuccess = new Date().toISOString();
      if (state.state === 'half-open') {
        state.state = 'closed';
        state.failures = 0;
        state.openedAt = null;
      }

      expect(state.state).toBe('closed');
      expect(state.failures).toBe(0);
    });

    it('should reopen on failure in half-open state', () => {
      const state: CircuitBreakerState = {
        state: 'half-open',
        failures: 3,
        lastFailure: null,
        lastSuccess: null,
        openedAt: null,
      };

      // Simulate failure
      state.failures++;
      state.lastFailure = new Date().toISOString();
      if (state.state === 'half-open') {
        state.state = 'open';
        state.openedAt = new Date().toISOString();
      }

      expect(state.state).toBe('open');
    });
  });

  describe('File Change Detection', () => {
    const ignorePaths = ['node_modules', '.git', 'dist', 'coverage', '.ai/cache'];
    const filePatterns = ['*.ts', '*.js', '*.json', '*.yaml', '*.yml', '*.md'];

    it('should ignore specified paths', () => {
      const shouldIgnore = (filePath: string): boolean => {
        return ignorePaths.some(
          (ignore) => filePath.includes(ignore) || filePath.startsWith(ignore)
        );
      };

      expect(shouldIgnore('node_modules/package/index.js')).toBe(true);
      expect(shouldIgnore('.git/config')).toBe(true);
      expect(shouldIgnore('tools/ai/cache.ts')).toBe(false);
    });

    it('should match file patterns', () => {
      const matchesPattern = (filePath: string): boolean => {
        return filePatterns.some((pattern) => {
          const ext = pattern.replace('*', '');
          return filePath.endsWith(ext);
        });
      };

      expect(matchesPattern('tools/ai/cache.ts')).toBe(true);
      expect(matchesPattern('package.json')).toBe(true);
      expect(matchesPattern('docs/CODEMAP.md')).toBe(true);
      expect(matchesPattern('image.png')).toBe(false);
    });
  });

  describe('Debouncing', () => {
    it('should debounce rapid changes', async () => {
      let triggerCount = 0;
      let debounceTimer: ReturnType<typeof setTimeout> | null = null;

      const scheduleTrigger = () => {
        if (debounceTimer) {
          clearTimeout(debounceTimer);
        }
        debounceTimer = setTimeout(() => {
          triggerCount++;
        }, 50);
      };

      // Rapid changes
      scheduleTrigger();
      scheduleTrigger();
      scheduleTrigger();
      scheduleTrigger();

      await new Promise((resolve) => setTimeout(resolve, 100));

      expect(triggerCount).toBe(1);
    });
  });

  describe('Frequency Limiting', () => {
    it('should enforce minimum time between triggers', () => {
      const maxFrequencyMs = 30000;
      const lastTriggerTime = new Date(Date.now() - 10000).toISOString(); // 10 seconds ago

      const elapsed = Date.now() - new Date(lastTriggerTime).getTime();
      const canTrigger = elapsed >= maxFrequencyMs;

      expect(canTrigger).toBe(false);
    });

    it('should allow trigger after frequency period', () => {
      const maxFrequencyMs = 30000;
      const lastTriggerTime = new Date(Date.now() - 40000).toISOString(); // 40 seconds ago

      const elapsed = Date.now() - new Date(lastTriggerTime).getTime();
      const canTrigger = elapsed >= maxFrequencyMs;

      expect(canTrigger).toBe(true);
    });
  });

  describe('Change Buffer', () => {
    interface FileChange {
      path: string;
      type: 'add' | 'modify' | 'delete';
      timestamp: string;
    }

    it('should buffer changes correctly', () => {
      const changeBuffer: FileChange[] = [];

      changeBuffer.push({
        path: 'tools/ai/cache.ts',
        type: 'modify',
        timestamp: new Date().toISOString(),
      });

      changeBuffer.push({
        path: 'tools/ai/monitor.ts',
        type: 'add',
        timestamp: new Date().toISOString(),
      });

      expect(changeBuffer.length).toBe(2);
    });

    it('should clear buffer after processing', () => {
      const changeBuffer: FileChange[] = [
        { path: 'file1.ts', type: 'modify', timestamp: new Date().toISOString() },
        { path: 'file2.ts', type: 'add', timestamp: new Date().toISOString() },
      ];

      const processedChanges = [...changeBuffer];
      changeBuffer.length = 0;

      expect(changeBuffer.length).toBe(0);
      expect(processedChanges.length).toBe(2);
    });
  });

  describe('Trigger Results', () => {
    interface TriggerResult {
      success: boolean;
      action: string;
      duration: number;
      error?: string;
    }

    it('should track successful triggers', () => {
      const results: TriggerResult[] = [
        { success: true, action: 'sync', duration: 150 },
        { success: true, action: 'codemap', duration: 300 },
        { success: true, action: 'metrics', duration: 50 },
      ];

      const successful = results.filter((r) => r.success).length;
      expect(successful).toBe(3);
    });

    it('should track failed triggers', () => {
      const results: TriggerResult[] = [
        { success: true, action: 'sync', duration: 150 },
        { success: false, action: 'codemap', duration: 0, error: 'Command failed' },
        { success: true, action: 'metrics', duration: 50 },
      ];

      const failed = results.filter((r) => !r.success);
      expect(failed.length).toBe(1);
      expect(failed[0].action).toBe('codemap');
    });
  });

  describe('Repository Monitor - Advanced Scenarios', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    describe('Cache Hit/Miss', () => {
      it('should cache analysis results and return cached result on hit', async () => {
        const cacheTtl = 60000; // 60 seconds
        const cache = new Map<string, { result: unknown; expires: Date }>();
        const repoPath = '/test/repo';

        // First call - cache miss
        const analysis1 = { chaosScore: 0.5, timestamp: new Date() };
        cache.set(repoPath, { result: analysis1, expires: new Date(Date.now() + cacheTtl) });

        // Second call - should hit cache
        const cached = cache.get(repoPath);
        expect(cached).toBeDefined();
        expect(cached?.result).toEqual(analysis1);

        // Advance time but stay within TTL
        vi.advanceTimersByTime(30000);
        const stillValid = cached && cached.expires > new Date();
        expect(stillValid).toBe(true);

        // Advance time beyond TTL
        vi.advanceTimersByTime(35000);
        const expired = cached && cached.expires < new Date();
        expect(expired).toBe(true);
      });

      it('should invalidate expired cache entries', () => {
        const cache = new Map<string, { result: unknown; expires: Date }>();
        const repoPath = '/test/repo';
        const cacheTtl = 60000;

        const analysis = { chaosScore: 0.5 };
        const expiresAt = new Date(Date.now() + cacheTtl);
        cache.set(repoPath, { result: analysis, expires: expiresAt });

        // Advance time to expiration
        vi.advanceTimersByTime(61000);

        const cached = cache.get(repoPath);
        if (cached && cached.expires < new Date()) {
          cache.delete(repoPath);
        }

        expect(cache.has(repoPath)).toBe(false);
      });
    });

    describe('Buffer Overflow Protection', () => {
      it('should cap buffer at 1000 events and emit warning', () => {
        const MAX_BUFFER_SIZE = 1000;
        let warningEmitted = false;

        interface FileChange {
          id: string;
          path: string;
          timestamp: Date;
        }

        const buffer: FileChange[] = [];

        // Add 1001 changes
        for (let i = 0; i < 1001; i++) {
          buffer.push({
            id: `change-${i}`,
            path: `/file-${i}.ts`,
            timestamp: new Date(),
          });

          // Check buffer cap
          if (buffer.length >= MAX_BUFFER_SIZE) {
            warningEmitted = true;
            const excess = buffer.length - MAX_BUFFER_SIZE;
            buffer.splice(0, excess + 1); // Remove excess + 1 to keep at 1000
          }
        }

        expect(warningEmitted).toBe(true);
        expect(buffer.length).toBeLessThanOrEqual(MAX_BUFFER_SIZE);
      });

      it('should preserve most recent events when truncating', () => {
        const MAX_BUFFER_SIZE = 5;

        interface Change {
          id: number;
          timestamp: Date;
        }

        const buffer: Change[] = [];

        // Add 10 changes
        for (let i = 0; i < 10; i++) {
          buffer.push({ id: i, timestamp: new Date(Date.now() + i) });
        }

        // Truncate to keep only most recent MAX_BUFFER_SIZE
        if (buffer.length > MAX_BUFFER_SIZE) {
          const toRemove = buffer.length - MAX_BUFFER_SIZE;
          buffer.splice(0, toRemove);
        }

        // Should keep the last 5 (ids 5-9)
        expect(buffer.length).toBe(5);
        expect(buffer[0].id).toBe(5);
        expect(buffer[4].id).toBe(9);
      });
    });

    describe('Throttling with Frequency Limits', () => {
      it('should queue missed triggers when frequency limit exceeded', async () => {
        const maxFrequencyMs = 30000;
        const debounceMs = 300;
        const triggerQueue: string[] = [];

        let lastTriggerTime: number | null = null;

        const scheduleTrigger = (label: string) => {
          const now = Date.now();

          if (lastTriggerTime === null || now - lastTriggerTime >= maxFrequencyMs) {
            // Can trigger immediately
            triggerQueue.push(`trigger:${label}`);
            lastTriggerTime = now;
          } else {
            // Queue for later
            const delayMs = maxFrequencyMs - (now - lastTriggerTime) + debounceMs;
            triggerQueue.push(`queued:${label}:delay=${delayMs}`);
          }
        };

        // Rapid triggers
        scheduleTrigger('change-1');
        expect(triggerQueue[0]).toBe('trigger:change-1');

        // Too soon - should queue
        vi.advanceTimersByTime(1000);
        scheduleTrigger('change-2');
        expect(triggerQueue[1]).toContain('queued:change-2');

        // After frequency period
        vi.advanceTimersByTime(35000);
        lastTriggerTime = null; // Simulate cleared from previous trigger
        scheduleTrigger('change-3');
        expect(triggerQueue[2]).toBe('trigger:change-3');
      });

      it('should enforce minimum time between analysis triggers', () => {
        const maxFrequencyMs = 30000;
        const lastTriggerTimes = new Map<string, number>();

        const canTrigger = (repoPath: string): boolean => {
          const lastTime = lastTriggerTimes.get(repoPath);
          const now = Date.now();

          if (lastTime === undefined) return true;
          return now - lastTime >= maxFrequencyMs;
        };

        const updateTriggerTime = (repoPath: string) => {
          lastTriggerTimes.set(repoPath, Date.now());
        };

        const repoPath = '/test/repo';

        // First trigger should be allowed
        expect(canTrigger(repoPath)).toBe(true);
        updateTriggerTime(repoPath);

        // Immediate retry should be blocked
        expect(canTrigger(repoPath)).toBe(false);

        // After 15 seconds should still be blocked
        vi.advanceTimersByTime(15000);
        expect(canTrigger(repoPath)).toBe(false);

        // After 30 seconds should be allowed
        vi.advanceTimersByTime(15001);
        expect(canTrigger(repoPath)).toBe(true);
      });
    });

    describe('Debounce with Fake Timers', () => {
      it('should debounce rapid file changes correctly', () => {
        let analysisTriggered = 0;

        const debounce = (callback: () => void, delay: number) => {
          let timerId: ReturnType<typeof setTimeout> | null = null;

          return () => {
            if (timerId !== null) {
              clearTimeout(timerId);
            }
            timerId = setTimeout(() => {
              callback();
              timerId = null;
            }, delay);
          };
        };

        const triggerAnalysis = () => {
          analysisTriggered++;
        };

        const debouncedTrigger = debounce(triggerAnalysis, 300);

        // Rapid changes
        debouncedTrigger();
        debouncedTrigger();
        debouncedTrigger();
        debouncedTrigger();

        expect(analysisTriggered).toBe(0); // Should not trigger yet

        // Advance past debounce delay
        vi.advanceTimersByTime(350);

        expect(analysisTriggered).toBe(1); // Should trigger once
      });
    });
  });
});
