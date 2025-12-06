import fs from 'node:fs';
import path from 'node:path';
import { orchestrate, buildContext, summarizeResults, evaluateCompliance } from '@/agents/turingo/orchestrator';
import type { AgentTask } from '@/agents/turingo/types';
import { loadWorkflow } from '@/lib/workflows/loader';
import { runMcp } from '@/agents/adapters/mcp';
import { runSupabase } from '@/agents/adapters/supabase';
import { createClient } from '@supabase/supabase-js';

function usage() {
  console.log('usage: agents-cli <scan|apply|apply-all> [--root <path>] [--path <repoPath>] [--workflow <name>] [--out <file>] [--out-dir <dir>] [--include <prefixes>] [--exclude <prefixes>]');
}

function arg(flag: string) {
  const args = process.argv.slice(2);
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

function repoRoot() {
  const r = arg('--root');
  return r ? path.resolve(r) : path.resolve(process.cwd(), '../../../../');
}

async function scan() {
  const root = repoRoot();
  const dirs = fs.readdirSync(root, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => path.join(root, d.name));
  console.log(JSON.stringify({ root, repos: dirs }, null, 2));
}

function tasksFor(workflow: string): AgentTask[] {
  try {
    return loadWorkflow(workflow).tasks as AgentTask[]
  } catch {
    return [{ name: 'ringmaster', input: { select: 'default' } }]
  }
}

async function apply() {
  const workflow = arg('--workflow') || 'turingo';
  const repoPath = arg('--path');
  const out = arg('--out');
  const dry = !!arg('--dry-run');
  if (!repoPath) {
    console.error('missing --path');
    process.exit(1);
  }
  const wf = loadWorkflow(workflow);
  const ctx = buildContext(`apply:${path.basename(repoPath)}`);
  let results: any[] = [];
  if (!dry) {
    if (wf.transport === 'mcp') {
      results = await runMcp(wf.tasks as AgentTask[], { timeoutMs: wf.policy.timeoutMs, retries: wf.policy.maxRetries, backoffMs: wf.policy.backoffMs });
    } else if (wf.transport === 'supabase') {
      const url = process.env.SUPABASE_URL;
      const key = process.env.SUPABASE_ANON_KEY;
      if (url && key) {
        const client = createClient(url, key);
        results = await runSupabase(client as any, wf.tasks as AgentTask[]);
      } else {
        results = await orchestrate(
          wf.tasks as AgentTask[],
          ctx,
          {
            timeoutMs: wf.policy.timeoutMs,
            retries: wf.policy.maxRetries,
            backoffMs: wf.policy.backoffMs,
            cacheTTL: wf.policy.cacheTTL,
            circuitBreakerThreshold: wf.policy.circuitBreakerThreshold,
          }
        );
      }
    } else {
      results = await orchestrate(
        wf.tasks as AgentTask[],
        ctx,
        {
          timeoutMs: wf.policy.timeoutMs,
          retries: wf.policy.maxRetries,
          backoffMs: wf.policy.backoffMs,
          cacheTTL: wf.policy.cacheTTL,
          circuitBreakerThreshold: wf.policy.circuitBreakerThreshold,
        }
      );
    }
  }
  const payload = { repoPath, workflow: wf.name, transport: wf.transport, policy: wf.policy, tasks: wf.tasks, results };
  try {
    const summary = summarizeResults(results as unknown as import('@/agents/turingo/types').AgentResult[]);
    const compliance = evaluateCompliance(summary, {
      minSuccessRate: wf.policy?.governance?.minSuccessRate,
      maxErrorRate: wf.policy?.governance?.maxErrorRate,
      maxTimeoutsPerAgent: wf.policy?.governance?.maxTimeoutsPerAgent
    });
    const globalDir = path.join(repoRoot(), 'outputs', 'agents', 'global');
    fs.mkdirSync(globalDir, { recursive: true });
    const dashboardFile = path.join(globalDir, `${wf.name}-summary.json`);
    fs.writeFileSync(dashboardFile, JSON.stringify({ workflow: wf.name, runId: ctx.runId, summary, compliance }, null, 2));
  } catch {}
  if (out) {
    const p = path.resolve(out);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify(payload, null, 2));
  }
  console.log(JSON.stringify(payload, null, 2));
}

function sanitizeName(p: string) {
  return p.replace(/[:\\/]/g, '-').toLowerCase();
}

async function applyAll() {
  const workflow = arg('--workflow') || 'turingo';
  const outDir = arg('--out-dir') || path.join(repoRoot(), 'outputs', 'agents');
  const include = (arg('--include') || '').split(',').filter(Boolean);
  const exclude = (arg('--exclude') || '').split(',').filter(Boolean);
  const dry = !!arg('--dry-run');
  const root = repoRoot();
  const dirs = fs.readdirSync(root, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => path.join(root, d.name));
  fs.mkdirSync(outDir, { recursive: true });
  const selected = dirs.filter(d => {
    const inc = include.length === 0 || include.some(p => d.includes(p));
    const exc = exclude.some(p => d.includes(p));
    return inc && !exc;
  });
  const results: Array<{ repoPath: string; ok: boolean; output: string }> = [];
  const wf = loadWorkflow(workflow);
  const allResults: import('@/agents/turingo/types').AgentResult[] = [];
  for (const repoPath of selected) {
    const ctx = buildContext(`apply:${path.basename(repoPath)}`);
    let res: any[] = [];
    if (!dry) {
      if (wf.transport === 'mcp') {
        res = await runMcp(wf.tasks as AgentTask[], { timeoutMs: wf.policy.timeoutMs, retries: wf.policy.maxRetries, backoffMs: wf.policy.backoffMs });
      } else if (wf.transport === 'supabase') {
        const url = process.env.SUPABASE_URL;
        const key = process.env.SUPABASE_ANON_KEY;
        if (url && key) {
          const client = createClient(url, key);
          res = await runSupabase(client as any, wf.tasks as AgentTask[]);
        } else {
          res = await orchestrate(
            wf.tasks as AgentTask[],
            ctx,
            {
              timeoutMs: wf.policy.timeoutMs,
              retries: wf.policy.maxRetries,
              backoffMs: wf.policy.backoffMs,
              cacheTTL: wf.policy.cacheTTL,
              circuitBreakerThreshold: wf.policy.circuitBreakerThreshold,
            }
          );
        }
      } else {
        res = await orchestrate(
          wf.tasks as AgentTask[],
          ctx,
          {
            timeoutMs: wf.policy.timeoutMs,
            retries: wf.policy.maxRetries,
            backoffMs: wf.policy.backoffMs,
            cacheTTL: wf.policy.cacheTTL,
            circuitBreakerThreshold: wf.policy.circuitBreakerThreshold,
          }
        );
      }
    }
    const payload = { repoPath, workflow: wf.name, transport: wf.transport, policy: wf.policy, tasks: wf.tasks, results: res };
    allResults.push(...(res as unknown as import('@/agents/turingo/types').AgentResult[]));
    const file = path.join(outDir, `${workflow}-${sanitizeName(repoPath)}.json`);
    fs.writeFileSync(file, JSON.stringify(payload, null, 2));
    results.push({ repoPath, ok: true, output: file });
  }
  try {
    const globalDir = path.join(outDir, 'global');
    fs.mkdirSync(globalDir, { recursive: true });
    const summary = summarizeResults(allResults);
    const compliance = evaluateCompliance(summary, {
      minSuccessRate: wf.policy?.governance?.minSuccessRate,
      maxErrorRate: wf.policy?.governance?.maxErrorRate,
      maxTimeoutsPerAgent: wf.policy?.governance?.maxTimeoutsPerAgent
    });
    const dashboardFile = path.join(globalDir, `${wf.name}-summary.json`);
    fs.writeFileSync(dashboardFile, JSON.stringify({ workflow: wf.name, summary, compliance }, null, 2));
  } catch {}
  console.log(JSON.stringify({ root, workflow, count: results.length, outputs: results }, null, 2));
}

async function main() {
  const cmd = process.argv[2];
  if (!cmd) return usage();
  if (cmd === 'scan') return scan();
  if (cmd === 'apply') return apply();
  if (cmd === 'apply-all') return applyAll();
  usage();
}

main();