import * as path from 'path';
import { CircuitBreakerConfig, CircuitBreakerState } from '../types/index.js';
import { ensureDir, loadJson, saveJson } from '@ai/utils/file-persistence.js';

const ROOT = process.cwd();
const ATLAS_DIR = path.join(ROOT, '.atlas');
const FILE = path.join(ATLAS_DIR, 'circuit.json');

type Store = Record<string, CircuitBreakerState>;

class CircuitStore {
  private store: Store;
  constructor() {
    ensureDir(ATLAS_DIR);
    this.store = loadJson<Store>(FILE, {}) || {};
  }
  get(id: string): CircuitBreakerState {
    const s = this.store[id];
    if (s) return s;
    const init: CircuitBreakerState = { state: 'closed', failures: 0, successes: 0 };
    this.store[id] = init;
    saveJson(FILE, this.store);
    return init;
  }
  set(id: string, s: CircuitBreakerState): void {
    this.store[id] = s;
    saveJson(FILE, this.store);
  }
}

const store = new CircuitStore();

export function allowRequest(agentId: string, cfg: CircuitBreakerConfig): boolean {
  const s = store.get(agentId);
  if (s.state === 'open') {
    const opened = s.openedAt ? new Date(s.openedAt).getTime() : 0;
    if (Date.now() - opened >= cfg.timeout) {
      s.state = 'half_open';
      s.successes = 0;
      s.failures = 0;
      store.set(agentId, s);
      return true;
    }
    return false;
  }
  if (s.state === 'half_open') {
    const total = s.successes + s.failures;
    return total < cfg.halfOpenRequests;
  }
  return true;
}

export function recordSuccess(agentId: string, cfg: CircuitBreakerConfig): void {
  const s = store.get(agentId);
  s.successes++;
  s.lastSuccess = new Date().toISOString();
  if (s.state === 'half_open' && s.successes >= cfg.successThreshold) {
    s.state = 'closed';
    s.failures = 0;
    s.successes = 0;
  }
  store.set(agentId, s);
}

export function recordFailure(agentId: string, cfg: CircuitBreakerConfig): void {
  const s = store.get(agentId);
  s.failures++;
  s.lastFailure = new Date().toISOString();
  if (s.state === 'closed' && s.failures >= cfg.failureThreshold) {
    s.state = 'open';
    s.openedAt = new Date().toISOString();
  }
  if (s.state === 'half_open') s.state = 'open';
  store.set(agentId, s);
}
