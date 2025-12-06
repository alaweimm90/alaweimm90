import { describe, it, expect } from 'vitest';
import { orchestrate, buildContext, summarizeResults, evaluateCompliance } from '@/agents/turingo/orchestrator';
import type { AgentTask } from '@/agents/turingo/types';

describe('Turingo Orchestrator', () => {
  it('runs core agents and returns results', async () => {
    const tasks: AgentTask[] = [
      { name: 'ringmaster', input: { select: 'qap' } },
      { name: 'puzzle-prodigy', input: { analyze: 'structure' } },
      { name: 'benchmark-bandit', input: { suite: 'qaplib' } },
    ];
    const ctx = buildContext('test-run');
    const results = await orchestrate(tasks, ctx);
    expect(results.length).toBe(3);
    results.forEach(r => {
      expect(r.ok).toBe(true);
      expect(r.data).toBeTruthy();
    });
  });

  it('times out a delayed task when timeout is small', async () => {
    const tasks: AgentTask[] = [
      { name: 'ringmaster', input: { delayMs: 50 } }
    ];
    const ctx = buildContext('timeout-run');
    const results = await orchestrate(tasks, ctx, { timeoutMs: 10 });
    expect(results.length).toBe(1);
    expect(results[0].ok).toBe(false);
    expect(results[0].error).toBe('timeout');
  });

  it('opens circuit after repeated timeouts', async () => {
    const tasks: AgentTask[] = [
      { name: 'ringmaster', input: { delayMs: 50 } },
      { name: 'ringmaster', input: { delayMs: 50 } }
    ];
    const ctx = buildContext('breaker-run');
    const results = await orchestrate(tasks, ctx, { timeoutMs: 10, circuitBreakerThreshold: 1 });
    expect(results.length).toBe(2);
    expect(results[0].error).toBe('timeout');
    expect(results[1].error).toBe('circuit_open');
  });
  it('summarizes results', () => {
    const results = [
      { name: 'ringmaster', ok: true, data: { x: 1 } },
      { name: 'puzzle-prodigy', ok: false, error: 'timeout' },
      { name: 'ringmaster', ok: true, data: { y: 2 } }
    ];
    const summary = summarizeResults(results as any);
    expect(summary.total).toBe(3);
    expect(summary.succeeded).toBe(2);
    expect(summary.failed).toBe(1);
    expect(summary.errorTypes.timeout).toBe(1);
    expect(summary.byAgent['ringmaster'].total).toBe(2);
    expect(summary.byAgent['ringmaster'].ok).toBe(2);
  });
  it('evaluates compliance based on thresholds', () => {
    const results = [
      { name: 'ringmaster', ok: true },
      { name: 'puzzle-prodigy', ok: false, error: 'timeout' },
      { name: 'ringmaster', ok: true }
    ] as any;
    const summary = summarizeResults(results);
    const okEval = evaluateCompliance(summary, { minSuccessRate: 0.5, maxErrorRate: 0.6, maxTimeoutsPerAgent: 2 });
    expect(okEval.ok).toBe(true);
    const badEval = evaluateCompliance(summary, { minSuccessRate: 0.9, maxErrorRate: 0.1, maxTimeoutsPerAgent: 0 });
    expect(badEval.ok).toBe(false);
  });
});