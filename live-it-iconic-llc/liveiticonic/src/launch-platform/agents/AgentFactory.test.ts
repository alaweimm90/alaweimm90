/**
 * AgentFactory Tests
 *
 * Tests for the agent factory that creates and manages all agent instances
 */

import { describe, it, expect, afterEach } from 'vitest';
import { AgentFactory } from './index';
import { AgentType } from '../types';
import { BaseAgent } from '../core/BaseAgent';

describe('AgentFactory', () => {
  const createdAgents: BaseAgent[] = [];

  afterEach(() => {
    // Cleanup all created agents
    createdAgents.forEach(agent => agent.shutdown());
    createdAgents.length = 0;
  });

  describe('Agent Creation', () => {
    it('should create CompetitorAnalyst agent', () => {
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.COMPETITOR_ANALYST);
      if (agent) createdAgents.push(agent);
    });

    it('should create TrendDetector agent', () => {
      const agent = AgentFactory.createAgent(AgentType.TREND_DETECTOR);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.TREND_DETECTOR);
      if (agent) createdAgents.push(agent);
    });

    it('should create BrandArchitect agent', () => {
      const agent = AgentFactory.createAgent(AgentType.BRAND_ARCHITECT);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.BRAND_ARCHITECT);
      if (agent) createdAgents.push(agent);
    });

    it('should create CampaignManager agent', () => {
      const agent = AgentFactory.createAgent(AgentType.CAMPAIGN_MANAGER);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.CAMPAIGN_MANAGER);
      if (agent) createdAgents.push(agent);
    });

    it('should create AnalyticsInterpreter agent', () => {
      const agent = AgentFactory.createAgent(AgentType.ANALYTICS_INTERPRETER);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.ANALYTICS_INTERPRETER);
      if (agent) createdAgents.push(agent);
    });

    it('should create DataCollector agent', () => {
      const agent = AgentFactory.createAgent(AgentType.DATA_COLLECTOR);
      expect(agent).toBeDefined();
      expect(agent?.getType()).toBe(AgentType.DATA_COLLECTOR);
      if (agent) createdAgents.push(agent);
    });

    it('should create agent with custom ID', () => {
      const customId = 'custom-agent-12345';
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST, customId);
      expect(agent).toBeDefined();
      expect(agent?.getId()).toBe(customId);
      if (agent) createdAgents.push(agent);
    });
  });

  describe('All Agents Creation', () => {
    it('should create all 26 agents', () => {
      const agents = AgentFactory.createAllAgents();
      createdAgents.push(...agents);

      expect(agents).toHaveLength(26);
    });

    it('should create unique agent instances', () => {
      const agents = AgentFactory.createAllAgents();
      createdAgents.push(...agents);

      const ids = agents.map(a => a.getId());
      const uniqueIds = new Set(ids);

      expect(uniqueIds.size).toBe(agents.length);
    });

    it('should create agents with correct types', () => {
      const agents = AgentFactory.createAllAgents();
      createdAgents.push(...agents);

      const types = agents.map(a => a.getType());

      // Check for each category
      expect(types).toContain(AgentType.COMPETITOR_ANALYST);
      expect(types).toContain(AgentType.BRAND_ARCHITECT);
      expect(types).toContain(AgentType.CAMPAIGN_MANAGER);
      expect(types).toContain(AgentType.ANALYTICS_INTERPRETER);
      expect(types).toContain(AgentType.DATA_COLLECTOR);
    });

    it('should have all agents with capabilities', () => {
      const agents = AgentFactory.createAllAgents();
      createdAgents.push(...agents);

      agents.forEach(agent => {
        const capabilities = agent.getCapabilities();
        expect(capabilities.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Phase-Based Creation', () => {
    it('should create analysis phase agents', () => {
      const agents = AgentFactory.createAgentsForPhase('analysis');
      createdAgents.push(...agents);

      expect(agents.length).toBe(5); // Market Intelligence agents

      const types = agents.map(a => a.getType());
      expect(types).toContain(AgentType.COMPETITOR_ANALYST);
      expect(types).toContain(AgentType.TREND_DETECTOR);
      expect(types).toContain(AgentType.CUSTOMER_RESEARCHER);
      expect(types).toContain(AgentType.PRICING_STRATEGIST);
      expect(types).toContain(AgentType.MARKET_SIZER);
    });

    it('should create creative phase agents', () => {
      const agents = AgentFactory.createAgentsForPhase('creative');
      createdAgents.push(...agents);

      expect(agents.length).toBe(5); // Creative & Branding agents

      const types = agents.map(a => a.getType());
      expect(types).toContain(AgentType.BRAND_ARCHITECT);
      expect(types).toContain(AgentType.COPYWRITER);
      expect(types).toContain(AgentType.VISUAL_DESIGNER);
      expect(types).toContain(AgentType.VIDEO_PRODUCER);
      expect(types).toContain(AgentType.STORYTELLER);
    });

    it('should create execution phase agents', () => {
      const agents = AgentFactory.createAgentsForPhase('execution');
      createdAgents.push(...agents);

      expect(agents.length).toBe(6); // Launch Execution agents

      const types = agents.map(a => a.getType());
      expect(types).toContain(AgentType.CAMPAIGN_MANAGER);
      expect(types).toContain(AgentType.SOCIAL_MEDIA_STRATEGIST);
      expect(types).toContain(AgentType.INFLUENCER_OUTREACH);
      expect(types).toContain(AgentType.PR_MANAGER);
      expect(types).toContain(AgentType.EMAIL_MARKETER);
      expect(types).toContain(AgentType.CONTENT_DISTRIBUTOR);
    });

    it('should create optimization phase agents', () => {
      const agents = AgentFactory.createAgentsForPhase('optimization');
      createdAgents.push(...agents);

      expect(agents.length).toBe(5); // Optimization agents

      const types = agents.map(a => a.getType());
      expect(types).toContain(AgentType.ANALYTICS_INTERPRETER);
      expect(types).toContain(AgentType.CONVERSION_OPTIMIZER);
      expect(types).toContain(AgentType.SEO_SPECIALIST);
      expect(types).toContain(AgentType.PAID_ADS_MANAGER);
      expect(types).toContain(AgentType.FEEDBACK_ANALYZER);
    });

    it('should create supporting phase agents', () => {
      const agents = AgentFactory.createAgentsForPhase('supporting');
      createdAgents.push(...agents);

      expect(agents.length).toBe(5); // Supporting agents

      const types = agents.map(a => a.getType());
      expect(types).toContain(AgentType.DATA_COLLECTOR);
      expect(types).toContain(AgentType.QUALITY_CONTROLLER);
      expect(types).toContain(AgentType.COMPLIANCE_CHECKER);
      expect(types).toContain(AgentType.BUDGET_MANAGER);
      expect(types).toContain(AgentType.RISK_ASSESSOR);
    });
  });

  describe('Agent Metadata', () => {
    it('should return correct agent count', () => {
      const stats = AgentFactory.getImplementedAgentsCount();

      expect(stats.total).toBe(26);
      expect(stats.byCategory['Market Intelligence']).toBe(5);
      expect(stats.byCategory['Creative & Branding']).toBe(5);
      expect(stats.byCategory['Launch Execution']).toBe(6);
      expect(stats.byCategory['Optimization']).toBe(5);
      expect(stats.byCategory['Supporting']).toBe(5);
    });

    it('should return all implemented types', () => {
      const types = AgentFactory.getImplementedTypes();

      expect(types).toHaveLength(26);
      expect(types).toContain(AgentType.COMPETITOR_ANALYST);
      expect(types).toContain(AgentType.BRAND_ARCHITECT);
      expect(types).toContain(AgentType.CAMPAIGN_MANAGER);
      expect(types).toContain(AgentType.ANALYTICS_INTERPRETER);
      expect(types).toContain(AgentType.DATA_COLLECTOR);
    });

    it('should check if agent type is implemented', () => {
      expect(AgentFactory.isImplemented(AgentType.COMPETITOR_ANALYST)).toBe(true);
      expect(AgentFactory.isImplemented(AgentType.BRAND_ARCHITECT)).toBe(true);
      expect(AgentFactory.isImplemented(AgentType.CAMPAIGN_MANAGER)).toBe(true);
    });
  });

  describe('Agent Lifecycle', () => {
    it('should create agents that can be shutdown', () => {
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);
      expect(agent).toBeDefined();

      if (agent) {
        expect(() => agent.shutdown()).not.toThrow();
      }
    });

    it('should create agents that can execute tasks', async () => {
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST);
      if (agent) {
        createdAgents.push(agent);

        const capabilities = agent.getCapabilities();
        expect(capabilities.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Error Handling', () => {
    it('should return null for invalid agent type', () => {
      // Force an invalid type by casting
      const agent = AgentFactory.createAgent('invalid_type' as AgentType);
      expect(agent).toBeNull();
    });

    it('should handle empty custom ID', () => {
      const agent = AgentFactory.createAgent(AgentType.COMPETITOR_ANALYST, '');
      // Should use default ID
      expect(agent).toBeDefined();
      expect(agent?.getId()).toBeTruthy();
      if (agent) createdAgents.push(agent);
    });
  });
});
