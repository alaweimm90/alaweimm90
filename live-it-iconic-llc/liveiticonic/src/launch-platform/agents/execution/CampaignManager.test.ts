/**
 * CampaignManager Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CampaignManagerAgent } from './CampaignManager';
import { AgentStatus, Task, TaskStatus, AgentPriority } from '../../types';

describe('CampaignManagerAgent', () => {
  let agent: CampaignManagerAgent;

  beforeEach(() => {
    agent = new CampaignManagerAgent('campaign-manager-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize correctly', () => {
      expect(agent.getName()).toBe('Campaign Manager');
      expect(agent.getType()).toBe('campaign_manager');
    });

    it('should have campaign management capabilities', () => {
      const capabilities = agent.getCapabilities();
      const capNames = capabilities.map(c => c.name);

      expect(capNames).toContain('plan_campaign');
      expect(capNames).toContain('coordinate_execution');
      expect(capNames).toContain('track_performance');
      expect(capNames).toContain('optimize_realtime');
    });
  });

  describe('Campaign Planning', () => {
    it('should create campaign plan', async () => {
      const task: Task = {
        id: 'campaign-task-001',
        name: 'Plan Campaign',
        description: 'Create launch campaign',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.CRITICAL,
        dependencies: [],
        estimatedDuration: 7000,
        deliverables: ['campaign_plan'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'plan_campaign',
        product: {},
        budget: 100000,
        timeline: 90,
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.plan).toBeDefined();
      expect(result.output.plan.phases).toBeDefined();
      expect(result.output.plan.channels).toBeDefined();
      expect(result.output.plan.timeline).toBeDefined();
    });

    it('should coordinate campaign execution', async () => {
      const task: Task = {
        id: 'campaign-task-002',
        name: 'Coordinate Execution',
        description: 'Coordinate launch',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['execution_plan'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'coordinate_execution',
        campaignPlan: {},
        agents: [],
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.taskAssignments).toBeDefined();
      expect(result.output.schedule).toBeDefined();
      expect(result.output.dependencies).toBeDefined();
    });

    it('should track campaign performance', async () => {
      const task: Task = {
        id: 'campaign-task-003',
        name: 'Track Performance',
        description: 'Monitor campaign metrics',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 4500,
        deliverables: ['performance_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'track_performance',
        campaignId: 'camp-001',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.metrics).toBeDefined();
      expect(result.output.status).toBeDefined();
      expect(result.output.insights).toBeDefined();
    });
  });
});
