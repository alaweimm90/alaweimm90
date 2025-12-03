import * as fs from 'fs';
import * as path from 'path';
import yaml from 'yaml';
import {
  Task,
  TaskPriority,
  TaskType,
  OrchestrationConfig,
  AgentCapability,
} from '@atlas/types/index.js';
import { createRouter } from './router.js';
import { agentRegistry } from '@atlas/agents/registry.js';
import { teamRegistry } from '@atlas/agents/teams.js';

export interface WorkflowStep {
  id: string;
  task: TaskType;
  description: string;
  action?: string;
  agent?: string;
  inputs?: Record<string, string>;
  outputs?: string[];
  maxRetries?: number;
  onError?: 'continue' | 'rollback' | 'abort';
  dependsOn?: string[];
  parallelGroup?: string;
  condition?: {
    envEquals?: string;
  };
  matrix?: Record<string, string[]>;
}

export interface WorkflowDef {
  id: string;
  name: string;
  env?: string;
  steps: WorkflowStep[];
  agents?: Record<TaskType, string>;
}

export interface PlannedStep {
  step: WorkflowStep;
  agentId: string;
}

export interface WorkflowPlan {
  id: string;
  name: string;
  env?: string;
  steps: PlannedStep[];
}

export function loadWorkflow(file: string): WorkflowDef {
  const p = path.resolve(file);
  const txt = fs.readFileSync(p, 'utf8');
  const data = yaml.parse(txt) as WorkflowDef;
  const expanded = expandMatrix(data);
  validateWorkflow(expanded);
  return expanded;
}

export function planWorkflow(
  def: WorkflowDef,
  config?: Partial<OrchestrationConfig>,
  teamId?: string
): WorkflowPlan {
  const router = createRouter(config);
  const steps: PlannedStep[] = [];
  for (const s of def.steps) {
    if (s.condition?.envEquals && def.env && s.condition.envEquals !== def.env) continue;
    const preferred = def.agents?.[s.task];
    let agentId = preferred || '';

    if (!agentId && teamId) {
      const team = teamRegistry.get(teamId);
      if (team) {
        const required = mapTaskToCapability(s.task);
        const candidates = team.members
          .map((id) => agentRegistry.get(id))
          .filter(
            (a): a is NonNullable<typeof a> =>
              !!a && a.status === 'available' && a.capabilities.includes(required)
          );
        if (candidates.length) {
          candidates.sort((a, b) => {
            const srA = a.metrics.totalRequests
              ? a.metrics.successfulRequests / a.metrics.totalRequests
              : 0;
            const srB = b.metrics.totalRequests
              ? b.metrics.successfulRequests / b.metrics.totalRequests
              : 0;
            if (Math.abs(srB - srA) > 0.1) return srB - srA;
            return a.metrics.avgLatency - b.metrics.avgLatency;
          });
          agentId = candidates[0].id;
        }
      }
    }

    if (!agentId) {
      const decision = router.routeWithFallback(makeTask(def, s));
      agentId = decision ? decision.agentId : '';
    }
    steps.push({ step: s, agentId });
  }
  return { id: def.id, name: def.name, env: def.env, steps };
}

function makeTask(def: WorkflowDef, s: WorkflowStep): Task {
  return {
    id: `${def.id}:${s.id}`,
    type: s.task,
    description: s.description,
    context: { files: [], additionalContext: def.env },
    priority: 'high' as TaskPriority,
    status: 'routing',
    createdAt: new Date().toISOString(),
  };
}

function mapTaskToCapability(task: TaskType): AgentCapability {
  const map: Record<TaskType, AgentCapability> = {
    code_generation: 'code_generation',
    code_review: 'code_review',
    refactoring: 'refactoring',
    documentation: 'documentation',
    testing: 'testing',
    debugging: 'debugging',
    analysis: 'analysis',
    explanation: 'explanation',
    chat: 'chat',
  };
  return map[task];
}

function expandMatrix(def: WorkflowDef): WorkflowDef {
  const steps: WorkflowStep[] = [];
  for (const s of def.steps) {
    if (!s.matrix || Object.keys(s.matrix).length === 0) {
      steps.push(s);
      continue;
    }
    const keys = Object.keys(s.matrix);
    const combos: Record<string, string>[] = [];
    function build(idx: number, acc: Record<string, string>): void {
      if (idx >= keys.length) {
        combos.push(acc);
        return;
      }
      const k = keys[idx];
      for (const v of s.matrix![k]) build(idx + 1, { ...acc, [k]: v });
    }
    build(0, {});
    for (const combo of combos) {
      const label = keys.map((k) => `${k}=${combo[k]}`).join(',');
      const id = `${s.id}[${label}]`;
      const inputs = { ...(s.inputs || {}) };
      for (const k of keys) inputs[k] = combo[k];
      steps.push({ ...s, id, inputs });
    }
  }
  return { ...def, steps };
}

function validateWorkflow(def: WorkflowDef): void {
  if (!def.id || !def.name) throw new Error('workflow_missing_id_or_name');
  if (!Array.isArray(def.steps) || def.steps.length === 0) throw new Error('workflow_no_steps');
  const ids = new Set<string>();
  for (const s of def.steps) {
    if (!s.id || !s.task || !s.description) throw new Error('step_missing_fields');
    if (ids.has(s.id)) throw new Error(`duplicate_step_id:${s.id}`);
    ids.add(s.id);
  }
  const graph: Record<string, string[]> = {};
  for (const s of def.steps) graph[s.id] = (s.dependsOn || []).slice();
  const temp = new Set<string>();
  const perm = new Set<string>();
  function visit(n: string): void {
    if (perm.has(n)) return;
    if (temp.has(n)) throw new Error('workflow_cycle');
    temp.add(n);
    for (const d of graph[n] || []) {
      if (!ids.has(d)) throw new Error(`unknown_dependency:${d}`);
      visit(d);
    }
    temp.delete(n);
    perm.add(n);
  }
  for (const k of Object.keys(graph)) visit(k);
}
