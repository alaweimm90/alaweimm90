/**
 * LaunchOrchestrator Integration Tests
 *
 * Tests for the central orchestrator that coordinates all agents
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { LaunchOrchestrator } from './LaunchOrchestrator';
import { AgentFactory } from '../agents';
import {
  Product,
  ProductStage,
  AgentType,
  LaunchStatus,
} from '../types';

describe('LaunchOrchestrator Integration Tests', () => {
  let orchestrator: LaunchOrchestrator;
  const registeredAgents:unknown[] = [];

  beforeEach(() => {
    orchestrator = new LaunchOrchestrator();
  });

  afterEach(() => {
    // Cleanup registered agents
    registeredAgents.forEach(agent => agent.shutdown());
    registeredAgents.length = 0;
  });

  describe('Agent Registration', () => {
    it('should register agents successfully', () => {
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);
      if (agent) {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);

        const agents = orchestrator.getAgents();
        expect(agents).toHaveLength(1);
        expect(agents[0].getType()).toBe(AgentType.COMPETITOR_ANALYST);
      }
    });

    it('should register multiple agents', () => {
      const agents = AgentFactory.createAgentsForPhase('analysis');
      agents.forEach(agent => {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);
      });

      const registeredList = orchestrator.getAgents();
      expect(registeredList).toHaveLength(5); // Market Intelligence agents
    });

    it('should retrieve agent by ID', () => {
      const agent = AgentFactory.createAgent(AgentType.BRAND_ARCHITECT, 'brand-001');
      if (agent) {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);

        const retrieved = orchestrator.getAgent('brand-001');
        expect(retrieved).toBeDefined();
        expect(retrieved?.getType()).toBe(AgentType.BRAND_ARCHITECT);
      }
    });

    it('should retrieve agents by type', () => {
      const analysisAgents = AgentFactory.createAgentsForPhase('analysis');
      analysisAgents.forEach(agent => {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);
      });

      const competitorAgents = orchestrator.getAgentsByType(AgentType.COMPETITOR_ANALYST);
      expect(competitorAgents).toHaveLength(1);
      expect(competitorAgents[0].getType()).toBe(AgentType.COMPETITOR_ANALYST);
    });
  });

  describe('Launch Planning', () => {
    beforeEach(() => {
      // Register all necessary agents for launch planning
      const allAgents = AgentFactory.createAllAgents();
      allAgents.forEach(agent => {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);
      });
    });

    it('should create launch plan for product', async () => {
      const product: Product = {
        id: 'prod-001',
        name: 'Test Product',
        description: 'A test product for launch',
        category: 'luxury_automotive',
        targetMarket: {
          segments: [],
          primaryPersona: {
            name: 'Premium Buyer',
            age: [35, 55],
            income: [100000, 500000],
            occupation: ['Executive', 'Entrepreneur'],
            psychographics: ['Luxury seekers'],
            behaviors: ['Early adopter'],
            painPoints: ['Finding authentic luxury'],
            goals: ['Status expression'],
            channels: ['Instagram', 'LinkedIn'],
          },
          secondaryPersonas: [],
          geography: ['US', 'EU'],
          tam: 10000000000,
          sam: 1000000000,
          som: 100000000,
        },
        pricing: {
          model: 'tiered',
          basePrice: 199,
          currency: 'USD',
        },
        features: [],
        differentiators: ['Premium quality', 'Exclusive design'],
        stage: ProductStage.READY_TO_LAUNCH,
      };

      const plan = await orchestrator.planLaunch(product);

      expect(plan).toBeDefined();
      expect(plan.id).toBeDefined();
      expect(plan.product).toEqual(product);
      expect(plan.strategy).toBeDefined();
      expect(plan.resources).toBeDefined();
      expect(plan.timeline).toBeDefined();
      expect(plan.metrics).toBeDefined();
      expect(plan.status).toBeDefined();
    });

    it('should include all necessary components in launch plan', async () => {
      const product: Product = {
        id: 'prod-002',
        name: 'Premium Product',
        description: 'Premium test product',
        category: 'merchandise',
        targetMarket: {
          segments: [],
          primaryPersona: {
            name: 'Luxury Consumer',
            age: [30, 60],
            income: [150000, 1000000],
            occupation: ['Business Owner'],
            psychographics: ['Luxury lifestyle'],
            behaviors: ['Brand loyal'],
            painPoints: ['Quality concerns'],
            goals: ['Express success'],
            channels: ['Instagram'],
          },
          secondaryPersonas: [],
          geography: ['Global'],
          tam: 5000000000,
          sam: 500000000,
          som: 50000000,
        },
        pricing: {
          model: 'one-time',
          basePrice: 299,
          currency: 'USD',
        },
        features: [],
        differentiators: ['Exclusivity'],
        stage: ProductStage.READY_TO_LAUNCH,
      };

      const plan = await orchestrator.planLaunch(product);

      // Verify plan components
      expect(plan.strategy).toHaveProperty('type');
      expect(plan.strategy).toHaveProperty('channels');
      expect(plan.strategy).toHaveProperty('messaging');
      expect(plan.strategy).toHaveProperty('budget');

      expect(plan.resources).toHaveProperty('agents');
      expect(plan.resources).toHaveProperty('budget');

      expect(plan.timeline).toHaveProperty('startDate');
      expect(plan.timeline).toHaveProperty('launchDate');
      expect(plan.timeline).toHaveProperty('phases');

      expect(plan.metrics).toHaveProperty('targets');

      expect(Array.isArray(plan.risks)).toBe(true);
    });
  });

  describe('Multi-Phase Orchestration', () => {
    beforeEach(() => {
      // Register phase-specific agents
      const phases = ['analysis', 'creative', 'execution', 'optimization', 'supporting'] as const;
      phases.forEach(phase => {
        const agents = AgentFactory.createAgentsForPhase(phase);
        agents.forEach(agent => {
          registeredAgents.push(agent);
          orchestrator.registerAgent(agent);
        });
      });
    });

    it('should have agents for all launch phases', () => {
      const agents = orchestrator.getAgents();
      expect(agents.length).toBe(26);

      // Verify we have agents from each category
      expect(orchestrator.getAgentsByType(AgentType.COMPETITOR_ANALYST).length).toBeGreaterThan(0);
      expect(orchestrator.getAgentsByType(AgentType.BRAND_ARCHITECT).length).toBeGreaterThan(0);
      expect(orchestrator.getAgentsByType(AgentType.CAMPAIGN_MANAGER).length).toBeGreaterThan(0);
      expect(orchestrator.getAgentsByType(AgentType.ANALYTICS_INTERPRETER).length).toBeGreaterThan(0);
      expect(orchestrator.getAgentsByType(AgentType.DATA_COLLECTOR).length).toBeGreaterThan(0);
    });

    it('should coordinate agents across phases', () => {
      const marketAgents = orchestrator.getAgentsByType(AgentType.COMPETITOR_ANALYST);
      const creativeAgents = orchestrator.getAgentsByType(AgentType.BRAND_ARCHITECT);
      const executionAgents = orchestrator.getAgentsByType(AgentType.CAMPAIGN_MANAGER);

      expect(marketAgents.length).toBeGreaterThan(0);
      expect(creativeAgents.length).toBeGreaterThan(0);
      expect(executionAgents.length).toBeGreaterThan(0);
    });
  });

  describe('Agent Communication', () => {
    it('should enable inter-agent communication', () => {
      const agent1 = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);
      const agent2 = AgentFactory.createAgent(AgentType.BRAND_ARCHITECT);

      if (agent1 && agent2) {
        registeredAgents.push(agent1, agent2);
        orchestrator.registerAgent(agent1);
        orchestrator.registerAgent(agent2);

        // Agents should be able to communicate through the event bus
        expect(agent1.getId()).toBeDefined();
        expect(agent2.getId()).toBeDefined();
      }
    });
  });

  describe('Error Handling', () => {
    it('should handle missing agents gracefully', () => {
      const agent = orchestrator.getAgent('non-existent-agent');
      expect(agent).toBeUndefined();
    });

    it('should handle empty agent list', () => {
      const agents = orchestrator.getAgents();
      expect(agents).toEqual([]);
    });

    it('should handle invalid agent type query', () => {
      const agents = orchestrator.getAgentsByType('invalid_type' as AgentType);
      expect(agents).toEqual([]);
    });
  });

  describe('Launch Orchestration Flow', () => {
    it('should initialize with default constraints', () => {
      const orch = new LaunchOrchestrator();
      expect(orch).toBeDefined();
    });

    it('should accept custom constraints', () => {
      const constraints = {
        truthfulness: {
          noFalseAdvertising: true,
          accurateClaimsOnly: true,
          transparentPricing: true,
        },
        fairness: {
          noBiasAmplification: true,
          equalOpportunity: true,
          inclusiveMessaging: true,
        },
        privacy: {
          gdprCompliant: true,
          ccpaCompliant: true,
          userConsentRequired: true,
        },
        safety: {
          noBrandDamage: true,
          noReputationalRisk: true,
          noLegalViolations: true,
        },
      };

      const orch = new LaunchOrchestrator(constraints);
      expect(orch).toBeDefined();
    });
  });

  describe('Performance', () => {
    it('should handle large number of agents efficiently', () => {
      const startTime = Date.now();

      const allAgents = AgentFactory.createAllAgents();
      allAgents.forEach(agent => {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);
      });

      const duration = Date.now() - startTime;

      expect(orchestrator.getAgents().length).toBe(26);
      expect(duration).toBeLessThan(1000); // Should register quickly
    });

    it('should retrieve agents efficiently', () => {
      const allAgents = AgentFactory.createAllAgents();
      allAgents.forEach(agent => {
        registeredAgents.push(agent);
        orchestrator.registerAgent(agent);
      });

      const startTime = Date.now();

      for (let i = 0; i < 100; i++) {
        orchestrator.getAgents();
        orchestrator.getAgentsByType(AgentType.COMPETITOR_ANALYST);
      }

      const duration = Date.now() - startTime;
      expect(duration).toBeLessThan(100); // Should be fast
    });
  });
});
