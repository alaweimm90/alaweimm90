import { AgentContext, AgentResult, AgentTask } from './types';
import { AgentRegistry } from './registry';

function timeout(ms: number): Promise<AgentResult> {
  return new Promise((_, reject) => {
    setTimeout(() => reject(new Error('timeout')), ms);
  }) as unknown as Promise<AgentResult>;
}

const cache = new Map<string, { r: AgentResult; e: number }>();

function key(t: AgentTask): string {
  return `${t.name}:${JSON.stringify(t.input)}`;
}

async function execTask(
  t: AgentTask,
  ctx: AgentContext,
  timeoutMs: number,
  retries: number,
  backoffMs: number,
  cacheTTL: number
): Promise<AgentResult> {
  const agent = AgentRegistry.byName(t.name);
  if (!agent) return { name: t.name, ok: false, error: 'agent_not_found' } as AgentResult;
  const k = key(t);
  const c = cache.get(k);
  if (c && c.e > Date.now()) return c.r;
  try {
    let lastErr: unknown;
    for (let i = 0; i <= retries; i++) {
      try {
        const res = await Promise.race([agent.execute(t, ctx), timeout(timeoutMs)]);
        if (res.ok) cache.set(k, { r: res, e: Date.now() + cacheTTL });
        return res;
      } catch (err) {
        lastErr = err;
        if (i < retries) await new Promise(r => setTimeout(r, backoffMs));
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error('error');
  } catch {
    return { name: t.name, ok: false, error: 'timeout' } as AgentResult;
  }
}

export async function orchestrate(
  tasks: AgentTask[],
  ctx: AgentContext,
  opts?: { timeoutMs?: number; retries?: number; backoffMs?: number; cacheTTL?: number; circuitBreakerThreshold?: number }
): Promise<AgentResult[]> {
  const timeoutMs = opts?.timeoutMs ?? 30000;
  const retries = opts?.retries ?? 0;
  const backoffMs = opts?.backoffMs ?? 250;
  const cacheTTL = opts?.cacheTTL ?? 60000;
  const threshold = opts?.circuitBreakerThreshold ?? 1;
  const breaker = new Map<string, number>();
  const wrapped = async (t: AgentTask): Promise<AgentResult> => {
    const count = breaker.get(t.name) || 0;
    if (count >= threshold) return { name: t.name, ok: false, error: 'circuit_open' } as AgentResult;
    const res = await execTask(t, ctx, timeoutMs, retries, backoffMs, cacheTTL);
    if (!res.ok && res.error === 'timeout') breaker.set(t.name, count + 1);
    return res;
  };
  const results: AgentResult[] = [];
  for (const t of tasks) {
    // sequential to honor circuit breaker
    // keep minimal nesting and small function sizes
    // eslint-disabled intentionally not used
    // collect results deterministically
    const r = await wrapped(t);
    results.push(r);
  }
  return results;
}

export function buildContext(runId: string): AgentContext {
  return { now: Date.now(), runId };
}

export interface RunSummary {
  total: number;
  succeeded: number;
  failed: number;
  errorTypes: Record<string, number>;
  byAgent: Record<string, { total: number; ok: number; errors: Record<string, number> }>;
  createdAt: number;
}

export function summarizeResults(results: AgentResult[]): RunSummary {
  const summary: RunSummary = { total: 0, succeeded: 0, failed: 0, errorTypes: {}, byAgent: {}, createdAt: Date.now() };
  for (const r of results) {
    summary.total += 1;
    if (r.ok) summary.succeeded += 1; else summary.failed += 1;
    const a = summary.byAgent[r.name] || { total: 0, ok: 0, errors: {} };
    a.total += 1; if (r.ok) a.ok += 1; else if (r.error) a.errors[r.error] = (a.errors[r.error] || 0) + 1;
    summary.byAgent[r.name] = a;
    if (!r.ok && r.error) summary.errorTypes[r.error] = (summary.errorTypes[r.error] || 0) + 1;
  }
  return summary;
}

export interface GovernanceRules { minSuccessRate?: number; maxErrorRate?: number; maxTimeoutsPerAgent?: number }

export function evaluateCompliance(summary: RunSummary, rules: GovernanceRules) {
  const total = summary.total;
  const successRate = total > 0 ? summary.succeeded / total : 0;
  const errorRate = total > 0 ? summary.failed / total : 0;
  const timeouts = summary.errorTypes['timeout'] || 0;
  const timeoutLimit = rules.maxTimeoutsPerAgent ?? Number.POSITIVE_INFINITY;
  const agentTimeoutBreach = Object.values(summary.byAgent).some(a => (a.errors['timeout'] || 0) > timeoutLimit);
  const ok = (rules.minSuccessRate === undefined || successRate >= rules.minSuccessRate)
    && (rules.maxErrorRate === undefined || errorRate <= rules.maxErrorRate)
    && (rules.maxTimeoutsPerAgent === undefined || !agentTimeoutBreach);
  return { ok, successRate, errorRate, timeouts };
}
