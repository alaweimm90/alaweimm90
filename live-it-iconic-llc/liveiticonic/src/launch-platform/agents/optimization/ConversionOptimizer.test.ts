/**
 * ConversionOptimizer Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { ConversionOptimizerAgent } from './ConversionOptimizer';
import { Task, TaskStatus, AgentPriority } from '../../types';

describe('ConversionOptimizerAgent', () => {
  let agent: ConversionOptimizerAgent;

  beforeEach(() => {
    agent = new ConversionOptimizerAgent('conversion-optimizer-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Conversion Optimization', () => {
    it('should design A/B tests', async () => {
      const task: Task = {
        id: 'cro-task-001',
        name: 'Design A/B Test',
        description: 'Create test design',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['test_design'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'design_ab_test',
        element: 'landing_page',
        hypothesis: 'New headline will increase conversions',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.testDesign).toBeDefined();
      expect(result.output.variants).toBeDefined();
      expect(result.output.methodology).toBeDefined();
    });

    it('should optimize conversion funnels', async () => {
      const task: Task = {
        id: 'cro-task-002',
        name: 'Optimize Funnel',
        description: 'Improve conversion funnel',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['funnel_optimization'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'optimize_funnel',
        currentFunnel: {},
        metrics: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.optimizations).toBeDefined();
      expect(result.output.expectedImpact).toBeDefined();
    });

    it('should develop CRO strategy', async () => {
      const task: Task = {
        id: 'cro-task-003',
        name: 'CRO Strategy',
        description: 'Create optimization strategy',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 7000,
        deliverables: ['cro_strategy'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'develop_cro_strategy',
        data: {},
        goals: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.strategy).toBeDefined();
      expect(result.output.prioritizedTests).toBeDefined();
    });
  });
});
