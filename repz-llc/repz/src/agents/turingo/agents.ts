import { Agent, AgentContext, AgentResult, AgentTask, AgentName } from './types';

function ok(name: AgentName, data: unknown): AgentResult {
  return { name, ok: true, data };
}

function fail(name: AgentName, error: string): AgentResult {
  return { name, ok: false, error };
}

class BaseAgent implements Agent {
  name: AgentName;
  constructor(name: AgentName) {
    this.name = name;
  }
  async execute(task: AgentTask, ctx: AgentContext): Promise<AgentResult> {
    const delayMs = typeof (task.input as Record<string, unknown>)?.['delayMs'] === 'number' ? Number((task.input as Record<string, unknown>)['delayMs']) : 0;
    if (delayMs > 0) await new Promise(r => setTimeout(r, delayMs));
    const payload = { input: task.input, runId: ctx.runId, ts: ctx.now };
    return ok(this.name, payload);
  }
}

export class Ringmaster extends BaseAgent {
  constructor() { super('ringmaster'); }
}

export class BlueprintBoss extends BaseAgent {
  constructor() { super('blueprint-boss'); }
}

export class DealMaker extends BaseAgent {
  constructor() { super('deal-maker'); }
}

export class EthicsEnforcer extends BaseAgent {
  constructor() { super('ethics-enforcer'); }
}

export class PuzzleProdigy extends BaseAgent {
  constructor() { super('puzzle-prodigy'); }
}

export class QuantumQuokka extends BaseAgent {
  constructor() { super('quantum-quokka'); }
}

export class MLMagician extends BaseAgent {
  constructor() { super('ml-magician'); }
}

export class AnalogyAlchemist extends BaseAgent {
  constructor() { super('analogy-alchemist'); }
}

export class ProofPirate extends BaseAgent {
  constructor() { super('proof-pirate'); }
}

export class VerificationVigilante extends BaseAgent {
  constructor() { super('verification-vigilante'); }
}

export class BenchmarkBandit extends BaseAgent {
  constructor() { super('benchmark-bandit'); }
}

export class CodeCowboy extends BaseAgent {
  constructor() { super('code-cowboy'); }
}

export class NoveltyNinja extends BaseAgent {
  constructor() { super('novelty-ninja'); }
}

export class SkepticSorcerer extends BaseAgent {
  constructor() { super('skeptic-sorcerer'); }
}