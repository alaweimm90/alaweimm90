export type AgentName =
  | 'ringmaster'
  | 'blueprint-boss'
  | 'deal-maker'
  | 'ethics-enforcer'
  | 'puzzle-prodigy'
  | 'quantum-quokka'
  | 'ml-magician'
  | 'analogy-alchemist'
  | 'proof-pirate'
  | 'verification-vigilante'
  | 'benchmark-bandit'
  | 'code-cowboy'
  | 'novelty-ninja'
  | 'skeptic-sorcerer';

export interface AgentTask {
  name: AgentName;
  input: Record<string, unknown>;
}

export interface AgentResult {
  name: AgentName;
  ok: boolean;
  data?: unknown;
  error?: string;
}

export interface AgentContext {
  now: number;
  runId: string;
}

export interface Agent {
  name: AgentName;
  execute(task: AgentTask, ctx: AgentContext): Promise<AgentResult>;
}