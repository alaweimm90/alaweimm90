import { WorkflowDef, planWorkflow } from './workflows.js';
import { OrchestrationConfig, TaskResult } from '../types/index.js';
import { agentRegistry } from '../agents/registry.js';
import { allowRequest, recordFailure, recordSuccess } from './circuit-breaker.js';
import { executeAction, executeCommand } from './adapters.js';
import * as fs from 'fs';
import * as path from 'path';

interface ExecOptions {
  teamId?: string;
  retries?: number;
  telemetryPath?: string;
  config?: Partial<OrchestrationConfig>;
  concurrency?: number;
}

export async function executeWorkflow(
  def: WorkflowDef,
  opts: ExecOptions = {}
): Promise<TaskResult[]> {
  const cfg: OrchestrationConfig = {
    fallbackChain: ['claude-sonnet', 'claude-opus', 'gpt-4'],
    circuitBreaker: {
      failureThreshold: 5,
      successThreshold: 3,
      timeout: 60000,
      halfOpenRequests: 1,
    },
    routing: { strategy: 'capability' },
    telemetry: {
      enabled: true,
      metricsPath: opts.telemetryPath || path.join('.atlas', 'metrics.json'),
    },
    ...opts.config,
  };
  const plan = planWorkflow(def, cfg, opts.teamId);
  const results: TaskResult[] = [];
  const completed = new Set<string>();
  const pending = new Map(plan.steps.map((ps) => [ps.step.id, ps]));
  const conc = Math.max(1, opts.concurrency ?? 4);

  async function runStep(ps: (typeof plan.steps)[number]): Promise<TaskResult> {
    const res = await doStep(ps);
    const gated = await postGates(ps, res);
    if (cfg.telemetry.enabled) {
      writeMetric(cfg.telemetry.metricsPath, ps.step.id, gated);
      writeEvidence('.atlas/evidence', ps.step.id, gated);
    }
    return gated;
  }

  async function doStep(ps: (typeof plan.steps)[number]): Promise<TaskResult> {
    const agentId = ps.agentId;
    const retries = ps.step.maxRetries ?? opts.retries ?? 1;
    let attempt = 0;
    let last: TaskResult = { success: false, error: 'not_run', agentId };
    while (attempt < retries) {
      attempt++;
      if (!agentId || !allowRequest(agentId, cfg.circuitBreaker))
        return { success: false, error: 'circuit_open', agentId };
      const ar = await executeAction(ps.step);
      agentRegistry.recordRequest(agentId, ar.success, ar.latency || 0, ar.tokensUsed || 0);
      if (ar.success) {
        recordSuccess(agentId, cfg.circuitBreaker);
        last = {
          success: true,
          output: ar.output,
          latency: ar.latency,
          tokensUsed: ar.tokensUsed,
          agentId,
        };
        break;
      }
      recordFailure(agentId, cfg.circuitBreaker);
      last = {
        success: false,
        error: ar.error || 'execution_failed',
        latency: ar.latency,
        tokensUsed: ar.tokensUsed,
        agentId,
      };
    }
    return last;
  }

  async function postGates(ps: (typeof plan.steps)[number], r: TaskResult): Promise<TaskResult> {
    if (!r.success) return r;
    if (ps.step.action === 'imageScan') {
      const gate = await executeCommand('npm run ai:compliance:check');
      if (gate.code !== 0) return { ...r, success: false, error: 'compliance_failed' };
    }
    return r;
  }

  while (pending.size) {
    const ready = Array.from(pending.values()).filter((ps) => {
      const deps = ps.step.dependsOn || [];
      return deps.every((d) => completed.has(d));
    });
    if (ready.length === 0) break;
    const batch = ready.slice(0, conc);
    const promises = batch.map((ps) => runStep(ps));
    const batchResults = await Promise.all(promises);
    for (let i = 0; i < batch.length; i++) {
      const ps = batch[i];
      const r = batchResults[i];
      results.push(r);
      pending.delete(ps.step.id);
      if (r.success) completed.add(ps.step.id);
      else if (ps.step.onError === 'abort') {
        // Abort entire workflow
        pending.clear();
        break;
      }
    }
  }
  return results;
}

function writeEvidence(dir: string, stepId: string, r: TaskResult): void {
  const folder = path.resolve(dir);
  if (!fs.existsSync(folder)) fs.mkdirSync(folder, { recursive: true });
  const file = path.join(folder, `${stepId}.json`);
  const data = {
    stepId,
    timestamp: new Date().toISOString(),
    success: r.success,
    output: r.output,
    error: r.error,
    latency: r.latency,
    agentId: r.agentId,
  };
  fs.writeFileSync(file, JSON.stringify(data, null, 2));
}

function writeMetric(file: string, stepId: string, r: TaskResult): void {
  const dir = path.dirname(file);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  const entry = {
    stepId,
    timestamp: new Date().toISOString(),
    success: r.success,
    latency: r.latency,
    agentId: r.agentId,
  };
  const arr = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : [];
  arr.push(entry);
  fs.writeFileSync(file, JSON.stringify(arr, null, 2));
}
