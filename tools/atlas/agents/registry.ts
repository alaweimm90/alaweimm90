/**
 * ATLAS Agent Registry
 * Central registry for AI agents with capability mapping and status tracking
 */

import * as path from 'path';
import {
  Agent,
  AgentProvider,
  AgentCapability,
  AgentStatus,
  AgentConfig,
  AgentMetrics,
} from '../types/index.js';
import { loadJson, saveJson, ensureDir } from '../../ai/utils/file-persistence.js';

const ROOT = process.cwd();
const ATLAS_DIR = path.join(ROOT, '.atlas');
const REGISTRY_FILE = path.join(ATLAS_DIR, 'agent-registry.json');

// ============================================================================
// Default Agent Configurations
// ============================================================================

const DEFAULT_AGENTS: Omit<Agent, 'registeredAt' | 'metrics'>[] = [
  {
    id: 'claude-sonnet',
    name: 'Claude Sonnet 4',
    provider: 'anthropic',
    model: 'claude-sonnet-4-20250514',
    capabilities: [
      'code_generation',
      'code_review',
      'refactoring',
      'documentation',
      'explanation',
      'chat',
    ],
    status: 'available',
    config: {
      apiKeyEnv: 'ANTHROPIC_API_KEY',
      maxTokens: 8192,
      temperature: 0.7,
      timeout: 120000,
      retries: 3,
      costPerToken: 0.000003,
    },
  },
  {
    id: 'claude-opus',
    name: 'Claude Opus 4',
    provider: 'anthropic',
    model: 'claude-opus-4-20250514',
    capabilities: [
      'code_generation',
      'code_review',
      'refactoring',
      'documentation',
      'analysis',
      'explanation',
      'chat',
    ],
    status: 'available',
    config: {
      apiKeyEnv: 'ANTHROPIC_API_KEY',
      maxTokens: 8192,
      temperature: 0.5,
      timeout: 180000,
      retries: 2,
      costPerToken: 0.000015,
    },
  },
  {
    id: 'gpt-4',
    name: 'GPT-4 Turbo',
    provider: 'openai',
    model: 'gpt-4-turbo-preview',
    capabilities: ['code_generation', 'code_review', 'documentation', 'explanation', 'chat'],
    status: 'unavailable',
    config: {
      apiKeyEnv: 'OPENAI_API_KEY',
      maxTokens: 4096,
      temperature: 0.7,
      timeout: 120000,
      retries: 3,
      costPerToken: 0.00001,
    },
  },
  {
    id: 'gemini-pro',
    name: 'Gemini Pro',
    provider: 'google',
    model: 'gemini-pro',
    capabilities: ['code_generation', 'documentation', 'explanation', 'chat'],
    status: 'unavailable',
    config: {
      apiKeyEnv: 'GOOGLE_API_KEY',
      maxTokens: 8192,
      temperature: 0.7,
      timeout: 120000,
      retries: 3,
      costPerToken: 0.000001,
    },
  },
];

// ============================================================================
// Registry State
// ============================================================================

interface RegistryState {
  agents: Agent[];
  lastUpdated: string;
  version: string;
}

// ============================================================================
// Agent Registry Implementation
// ============================================================================

class AgentRegistry {
  private state: RegistryState;

  constructor() {
    ensureDir(ATLAS_DIR);
    this.state = this.loadState();
  }

  private loadState(): RegistryState {
    const defaultState: RegistryState = {
      agents: [],
      lastUpdated: new Date().toISOString(),
      version: '1.0.0',
    };

    const loaded = loadJson<RegistryState>(REGISTRY_FILE, defaultState);

    // Initialize with default agents if empty
    if (!loaded || loaded.agents.length === 0) {
      const initializedAgents = DEFAULT_AGENTS.map((agent) => ({
        ...agent,
        registeredAt: new Date().toISOString(),
        metrics: this.createEmptyMetrics(),
      }));

      const state: RegistryState = {
        agents: initializedAgents,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0',
      };

      saveJson(REGISTRY_FILE, state);
      return state;
    }

    return loaded;
  }

  private saveState(): void {
    this.state.lastUpdated = new Date().toISOString();
    saveJson(REGISTRY_FILE, this.state);
  }

  private createEmptyMetrics(): AgentMetrics {
    return {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      avgLatency: 0,
      totalTokens: 0,
      errorRate: 0,
    };
  }

  // ============================================================================
  // Public API
  // ============================================================================

  /**
   * Register a new agent or update existing
   */
  register(agent: Omit<Agent, 'registeredAt' | 'metrics'>): Agent {
    const existing = this.state.agents.findIndex((a) => a.id === agent.id);

    const fullAgent: Agent = {
      ...agent,
      registeredAt:
        existing >= 0 ? this.state.agents[existing].registeredAt : new Date().toISOString(),
      metrics: existing >= 0 ? this.state.agents[existing].metrics : this.createEmptyMetrics(),
    };

    if (existing >= 0) {
      this.state.agents[existing] = fullAgent;
    } else {
      this.state.agents.push(fullAgent);
    }

    this.saveState();
    return fullAgent;
  }

  /**
   * Unregister an agent
   */
  unregister(agentId: string): boolean {
    const index = this.state.agents.findIndex((a) => a.id === agentId);
    if (index < 0) return false;

    this.state.agents.splice(index, 1);
    this.saveState();
    return true;
  }

  /**
   * Get agent by ID
   */
  get(agentId: string): Agent | undefined {
    return this.state.agents.find((a) => a.id === agentId);
  }

  /**
   * Get all registered agents
   */
  getAll(): Agent[] {
    return [...this.state.agents];
  }

  /**
   * Get agents by provider
   */
  getByProvider(provider: AgentProvider): Agent[] {
    return this.state.agents.filter((a) => a.provider === provider);
  }

  /**
   * Get agents by capability
   */
  getByCapability(capability: AgentCapability): Agent[] {
    return this.state.agents.filter((a) => a.capabilities.includes(capability));
  }

  /**
   * Get available agents (not busy, not circuit_open)
   */
  getAvailable(): Agent[] {
    return this.state.agents.filter((a) => a.status === 'available');
  }

  /**
   * Get agents that can handle a specific capability and are available
   */
  getCapableAndAvailable(capability: AgentCapability): Agent[] {
    return this.state.agents.filter(
      (a) => a.status === 'available' && a.capabilities.includes(capability)
    );
  }

  /**
   * Update agent status
   */
  updateStatus(agentId: string, status: AgentStatus): boolean {
    const agent = this.state.agents.find((a) => a.id === agentId);
    if (!agent) return false;

    agent.status = status;
    this.saveState();
    return true;
  }

  /**
   * Update agent metrics after a request
   */
  recordRequest(agentId: string, success: boolean, latency: number, tokens: number): void {
    const agent = this.state.agents.find((a) => a.id === agentId);
    if (!agent) return;

    agent.metrics.totalRequests++;
    if (success) {
      agent.metrics.successfulRequests++;
    } else {
      agent.metrics.failedRequests++;
    }

    // Update running average latency
    agent.metrics.avgLatency =
      (agent.metrics.avgLatency * (agent.metrics.totalRequests - 1) + latency) /
      agent.metrics.totalRequests;

    agent.metrics.totalTokens += tokens;
    agent.metrics.errorRate = agent.metrics.failedRequests / agent.metrics.totalRequests;
    agent.lastUsed = new Date().toISOString();

    this.saveState();
  }

  /**
   * Check if agent API key is configured
   */
  isConfigured(agentId: string): boolean {
    const agent = this.state.agents.find((a) => a.id === agentId);
    if (!agent || !agent.config.apiKeyEnv) return false;

    return !!process.env[agent.config.apiKeyEnv];
  }

  /**
   * Get agents sorted by performance (success rate, latency)
   */
  getByPerformance(capability?: AgentCapability): Agent[] {
    const agents = capability ? this.getByCapability(capability) : this.getAll();

    return agents.sort((a, b) => {
      // Prefer lower error rate
      const errorDiff = a.metrics.errorRate - b.metrics.errorRate;
      if (Math.abs(errorDiff) > 0.1) return errorDiff;

      // Then prefer lower latency
      return a.metrics.avgLatency - b.metrics.avgLatency;
    });
  }

  /**
   * Get agents sorted by cost
   */
  getByCost(capability?: AgentCapability): Agent[] {
    const agents = capability ? this.getByCapability(capability) : this.getAll();

    return agents.sort((a, b) => {
      const costA = a.config.costPerToken ?? Infinity;
      const costB = b.config.costPerToken ?? Infinity;
      return costA - costB;
    });
  }

  /**
   * Get registry statistics
   */
  getStats(): {
    total: number;
    available: number;
    configured: number;
    byProvider: Record<AgentProvider, number>;
    byCapability: Record<AgentCapability, number>;
  } {
    const stats = {
      total: this.state.agents.length,
      available: this.getAvailable().length,
      configured: this.state.agents.filter((a) => this.isConfigured(a.id)).length,
      byProvider: {} as Record<AgentProvider, number>,
      byCapability: {} as Record<AgentCapability, number>,
    };

    for (const agent of this.state.agents) {
      stats.byProvider[agent.provider] = (stats.byProvider[agent.provider] || 0) + 1;
      for (const cap of agent.capabilities) {
        stats.byCapability[cap] = (stats.byCapability[cap] || 0) + 1;
      }
    }

    return stats;
  }

  /**
   * Reset agent registry to defaults
   */
  reset(): void {
    this.state = {
      agents: DEFAULT_AGENTS.map((agent) => ({
        ...agent,
        registeredAt: new Date().toISOString(),
        metrics: this.createEmptyMetrics(),
      })),
      lastUpdated: new Date().toISOString(),
      version: '1.0.0',
    };
    this.saveState();
  }
}

// ============================================================================
// Singleton Export
// ============================================================================

export const agentRegistry = new AgentRegistry();

// Re-export types for convenience
export type { Agent, AgentProvider, AgentCapability, AgentStatus, AgentConfig, AgentMetrics };
