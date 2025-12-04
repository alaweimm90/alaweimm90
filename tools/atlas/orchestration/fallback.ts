/**
 * ATLAS Fallback Manager
 * Resilience and fault tolerance with circuit breaker pattern
 */

import * as path from 'path';
import type { CircuitState } from '@atlas/types/index.js';
import { Task, CircuitBreakerConfig, CircuitBreakerState } from '@atlas/types/index.js';
import { agentRegistry, Agent } from '@atlas/agents/registry.js';
import { loadJson, saveJson, ensureDir } from '@ai/utils/file-persistence.js';

const ROOT = process.cwd();
const ATLAS_DIR = path.join(ROOT, '.atlas');
const CIRCUIT_STATE_FILE = path.join(ATLAS_DIR, 'circuit-states.json');

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: CircuitBreakerConfig = {
  failureThreshold: 5,
  successThreshold: 3,
  timeout: 60000, // 1 minute
  halfOpenRequests: 1,
};

// ============================================================================
// Circuit Breaker Implementation
// ============================================================================

export class CircuitBreaker {
  private agentId: string;
  private config: CircuitBreakerConfig;
  private state: CircuitBreakerState;
  private halfOpenCount: number = 0;

  constructor(agentId: string, config: CircuitBreakerConfig = DEFAULT_CONFIG) {
    this.agentId = agentId;
    this.config = config;
    this.state = {
      state: 'closed',
      failures: 0,
      successes: 0,
    };
  }

  /**
   * Check if request is allowed through the circuit
   */
  canRequest(): boolean {
    this.checkTimeout();

    switch (this.state.state) {
      case 'closed':
        return true;
      case 'open':
        return false;
      case 'half_open':
        // Only allow limited requests in half-open state
        return this.halfOpenCount < this.config.halfOpenRequests;
    }
  }

  /**
   * Record a successful request
   */
  recordSuccess(): void {
    this.state.lastSuccess = new Date().toISOString();
    this.state.successes++;

    switch (this.state.state) {
      case 'closed':
        // Reset failure count on success
        this.state.failures = 0;
        break;
      case 'half_open':
        // If we've had enough successes, close the circuit
        if (this.state.successes >= this.config.successThreshold) {
          this.close();
        }
        break;
      case 'open':
        // Shouldn't happen, but handle it
        break;
    }

    // Update agent status
    agentRegistry.updateStatus(this.agentId, 'available');
  }

  /**
   * Record a failed request
   */
  recordFailure(): void {
    this.state.lastFailure = new Date().toISOString();
    this.state.failures++;

    switch (this.state.state) {
      case 'closed':
        // If we've exceeded threshold, open the circuit
        if (this.state.failures >= this.config.failureThreshold) {
          this.open();
        }
        break;
      case 'half_open':
        // Any failure in half-open state reopens the circuit
        this.open();
        break;
      case 'open':
        // Already open, nothing to do
        break;
    }
  }

  /**
   * Open the circuit (block requests)
   */
  private open(): void {
    this.state.state = 'open';
    this.state.openedAt = new Date().toISOString();
    this.state.successes = 0;
    this.halfOpenCount = 0;
    agentRegistry.updateStatus(this.agentId, 'circuit_open');
  }

  /**
   * Close the circuit (allow requests)
   */
  private close(): void {
    this.state.state = 'closed';
    this.state.failures = 0;
    this.state.successes = 0;
    this.halfOpenCount = 0;
    delete this.state.openedAt;
    agentRegistry.updateStatus(this.agentId, 'available');
  }

  /**
   * Transition to half-open state
   */
  private halfOpen(): void {
    this.state.state = 'half_open';
    this.state.successes = 0;
    this.halfOpenCount = 0;
  }

  /**
   * Check if timeout has elapsed and transition state
   */
  private checkTimeout(): void {
    if (this.state.state !== 'open' || !this.state.openedAt) return;

    const openedAt = new Date(this.state.openedAt).getTime();
    const elapsed = Date.now() - openedAt;

    if (elapsed >= this.config.timeout) {
      this.halfOpen();
    }
  }

  /**
   * Get current state
   */
  getState(): CircuitBreakerState {
    this.checkTimeout();
    return { ...this.state };
  }

  /**
   * Force reset the circuit
   */
  reset(): void {
    this.close();
  }

  /**
   * Increment half-open request count
   */
  incrementHalfOpenCount(): void {
    if (this.state.state === 'half_open') {
      this.halfOpenCount++;
    }
  }
}

// ============================================================================
// Fallback Manager Implementation
// ============================================================================

interface FallbackState {
  circuits: Record<string, CircuitBreakerState>;
  lastUpdated: string;
}

export class FallbackManager {
  private circuits: Map<string, CircuitBreaker> = new Map();
  private fallbackChain: string[];
  private config: CircuitBreakerConfig;

  constructor(
    fallbackChain: string[] = ['claude-sonnet', 'claude-opus', 'gpt-4'],
    config: CircuitBreakerConfig = DEFAULT_CONFIG
  ) {
    this.fallbackChain = fallbackChain;
    this.config = config;
    ensureDir(ATLAS_DIR);
    this.loadState();
  }

  private loadState(): void {
    const state = loadJson<FallbackState>(CIRCUIT_STATE_FILE);
    if (state?.circuits) {
      for (const [agentId, circuitState] of Object.entries(state.circuits)) {
        const breaker = this.getOrCreateBreaker(agentId);
        // Restore state (simplified - in production would restore full state)
        if (circuitState.state === 'open') {
          breaker.recordFailure();
          for (let i = 1; i < this.config.failureThreshold; i++) {
            breaker.recordFailure();
          }
        }
      }
    }
  }

  private saveState(): void {
    const circuits: Record<string, CircuitBreakerState> = {};
    this.circuits.forEach((breaker, agentId) => {
      circuits[agentId] = breaker.getState();
    });
    saveJson(CIRCUIT_STATE_FILE, {
      circuits,
      lastUpdated: new Date().toISOString(),
    });
  }

  /**
   * Get or create a circuit breaker for an agent
   */
  private getOrCreateBreaker(agentId: string): CircuitBreaker {
    if (!this.circuits.has(agentId)) {
      this.circuits.set(agentId, new CircuitBreaker(agentId, this.config));
    }
    return this.circuits.get(agentId)!;
  }

  /**
   * Get the next available agent for a task
   */
  getNextAgent(preferredAgentId?: string): Agent | null {
    // Try preferred agent first
    if (preferredAgentId) {
      const breaker = this.getOrCreateBreaker(preferredAgentId);
      if (breaker.canRequest()) {
        const agent = agentRegistry.get(preferredAgentId);
        if (agent && agent.status !== 'unavailable') {
          breaker.incrementHalfOpenCount();
          return agent;
        }
      }
    }

    // Try fallback chain
    for (const agentId of this.fallbackChain) {
      if (agentId === preferredAgentId) continue; // Already tried

      const breaker = this.getOrCreateBreaker(agentId);
      if (breaker.canRequest()) {
        const agent = agentRegistry.get(agentId);
        if (agent && agent.status !== 'unavailable') {
          breaker.incrementHalfOpenCount();
          return agent;
        }
      }
    }

    // No agent available
    return null;
  }

  /**
   * Execute a task with fallback support
   */
  async executeWithFallback<T>(
    _task: Task,
    executor: (agent: Agent) => Promise<T>,
    preferredAgentId?: string
  ): Promise<{ result: T; agentUsed: string } | null> {
    const tried = new Set<string>();

    // Try preferred agent first
    if (preferredAgentId && !tried.has(preferredAgentId)) {
      tried.add(preferredAgentId);
      const result = await this.tryAgent(preferredAgentId, executor);
      if (result !== null) {
        return { result, agentUsed: preferredAgentId };
      }
    }

    // Try fallback chain
    for (const agentId of this.fallbackChain) {
      if (tried.has(agentId)) continue;
      tried.add(agentId);

      const result = await this.tryAgent(agentId, executor);
      if (result !== null) {
        return { result, agentUsed: agentId };
      }
    }

    return null;
  }

  /**
   * Try to execute with a specific agent
   */
  private async tryAgent<T>(
    agentId: string,
    executor: (agent: Agent) => Promise<T>
  ): Promise<T | null> {
    const breaker = this.getOrCreateBreaker(agentId);

    if (!breaker.canRequest()) {
      return null;
    }

    const agent = agentRegistry.get(agentId);
    if (!agent || agent.status === 'unavailable') {
      return null;
    }

    breaker.incrementHalfOpenCount();
    const startTime = Date.now();

    try {
      const result = await executor(agent);
      const latency = Date.now() - startTime;

      breaker.recordSuccess();
      agentRegistry.recordRequest(agentId, true, latency, 0);
      this.saveState();

      return result;
    } catch {
      breaker.recordFailure();
      agentRegistry.recordRequest(agentId, false, Date.now() - startTime, 0);
      this.saveState();

      return null;
    }
  }

  /**
   * Record success for an agent
   */
  recordSuccess(agentId: string, latency: number, tokens: number): void {
    const breaker = this.getOrCreateBreaker(agentId);
    breaker.recordSuccess();
    agentRegistry.recordRequest(agentId, true, latency, tokens);
    this.saveState();
  }

  /**
   * Record failure for an agent
   */
  recordFailure(agentId: string): void {
    const breaker = this.getOrCreateBreaker(agentId);
    breaker.recordFailure();
    agentRegistry.recordRequest(agentId, false, 0, 0);
    this.saveState();
  }

  /**
   * Get circuit breaker status for all agents
   */
  getStatus(): Record<string, CircuitBreakerState> {
    const status: Record<string, CircuitBreakerState> = {};
    for (const agentId of this.fallbackChain) {
      const breaker = this.getOrCreateBreaker(agentId);
      status[agentId] = breaker.getState();
    }
    return status;
  }

  /**
   * Reset circuit breaker for an agent
   */
  resetCircuit(agentId: string): boolean {
    const breaker = this.circuits.get(agentId);
    if (!breaker) return false;

    breaker.reset();
    this.saveState();
    return true;
  }

  /**
   * Reset all circuit breakers
   */
  resetAll(): void {
    this.circuits.forEach((breaker) => {
      breaker.reset();
    });
    this.saveState();
  }

  /**
   * Check if any agent is available
   */
  hasAvailableAgent(): boolean {
    for (const agentId of this.fallbackChain) {
      const breaker = this.getOrCreateBreaker(agentId);
      if (breaker.canRequest()) {
        const agent = agentRegistry.get(agentId);
        if (agent && agent.status !== 'unavailable') {
          return true;
        }
      }
    }
    return false;
  }

  /**
   * Get health summary
   */
  getHealth(): {
    healthy: number;
    degraded: number;
    unhealthy: number;
    total: number;
  } {
    let healthy = 0;
    let degraded = 0;
    let unhealthy = 0;

    for (const agentId of this.fallbackChain) {
      const breaker = this.getOrCreateBreaker(agentId);
      const state = breaker.getState();

      switch (state.state) {
        case 'closed':
          healthy++;
          break;
        case 'half_open':
          degraded++;
          break;
        case 'open':
          unhealthy++;
          break;
      }
    }

    return {
      healthy,
      degraded,
      unhealthy,
      total: this.fallbackChain.length,
    };
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const fallbackManager = new FallbackManager();

// Re-export types
export type { CircuitState, CircuitBreakerConfig, CircuitBreakerState };
