/**
 * ATLAS Integration Configuration
 *
 * Configuration and setup for seamless integration between
 * AI Tools Suite and ATLAS orchestration platform.
 */

// TODO: Import when atlas-integration module is available
// import { ATLASIntegration } from '../src/core/atlas-integration.js';

interface AgentInfo {
  agent_id: string;
  name: string;
  provider: string;
  capabilities: string[];
  health?: { status: string };
}

// Placeholder class until module is available
class ATLASIntegration {
  constructor(public config: ATLASIntegrationConfig) {}
  async getHealth(): Promise<{ status: string }> {
    return { status: 'healthy' };
  }
  async getAgents(): Promise<AgentInfo[]> {
    return [];
  }
}

export interface ATLASIntegrationConfig {
  url?: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  preferredProviders?: string[];
  costLimits?: {
    maxPerTask?: number;
    maxPerHour?: number;
  };
}

/**
 * Create ATLAS integration instance with default configuration
 */
export function createATLASIntegration(config: ATLASIntegrationConfig = {}): ATLASIntegration {
  const defaultConfig = {
    url: process.env.ATLAS_URL || 'http://localhost:8000',
    apiKey: process.env.ATLAS_API_KEY,
    timeout: 30000,
    retryAttempts: 3,
    preferredProviders: ['anthropic', 'openai', 'google'],
    costLimits: {
      maxPerTask: 0.5,
      maxPerHour: 10.0,
    },
    ...config,
  };

  return new ATLASIntegration(defaultConfig);
}

/**
 * Validate ATLAS connection and capabilities
 */
export async function validateATLASConnection(atlas: ATLASIntegration): Promise<boolean> {
  try {
    const health = await atlas.getHealth();
    return health.status === 'healthy';
  } catch (error) {
    console.error('ATLAS connection validation failed:', error);
    return false;
  }
}

interface SimplifiedAgent {
  id: string;
  name: string;
  provider: string;
  capabilities: string[];
  health: string;
}

/**
 * Get available agents and their capabilities
 */
export async function getAvailableAgents(atlas: ATLASIntegration): Promise<SimplifiedAgent[]> {
  try {
    const agents = await atlas.getAgents();
    return agents.map((agent) => ({
      id: agent.agent_id,
      name: agent.name,
      provider: agent.provider,
      capabilities: agent.capabilities,
      health: agent.health?.status || 'unknown',
    }));
  } catch (error) {
    console.error('Failed to get available agents:', error);
    return [];
  }
}

/**
 * Setup ATLAS integration for AI Tools
 */
export async function setupATLASIntegration(
  config: ATLASIntegrationConfig = {}
): Promise<{ atlas: ATLASIntegration; agents: SimplifiedAgent[] }> {
  const atlas = createATLASIntegration(config);

  // Validate connection
  const isConnected = await validateATLASConnection(atlas);
  if (!isConnected) {
    throw new Error('Failed to connect to ATLAS. Please check your configuration.');
  }

  // Get available agents
  const agents = await getAvailableAgents(atlas);

  console.log(`✅ Connected to ATLAS at ${config.url || 'http://localhost:8000'}`);
  console.log(`📊 Available agents: ${agents.length}`);
  console.log(`🤖 Agents: ${agents.map((a) => `${a.name} (${a.provider})`).join(', ')}`);
  return { atlas, agents };
}
