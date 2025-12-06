/**
 * BudgetManager Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { BudgetManagerAgent } from './BudgetManager';
import { Task, TaskStatus, AgentPriority } from '../../types';

describe('BudgetManagerAgent', () => {
  let agent: BudgetManagerAgent;

  beforeEach(() => {
    agent = new BudgetManagerAgent('budget-manager-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Budget Management', () => {
    it('should track spending', async () => {
      const task: Task = {
        id: 'budget-task-001',
        name: 'Track Spending',
        description: 'Monitor campaign spending',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 4000,
        deliverables: ['spending_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'track_spending',
        expenses: [],
        budget: 100000,
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.spending).toBeDefined();
      expect(result.output.remaining).toBeDefined();
    });

    it('should forecast costs', async () => {
      const task: Task = {
        id: 'budget-task-002',
        name: 'Forecast Costs',
        description: 'Predict future costs',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 5500,
        deliverables: ['cost_forecast'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'forecast_costs',
        historical: [],
        plans: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.forecast).toBeDefined();
      expect(result.output.confidence).toBeGreaterThan(0);
    });

    it('should optimize budget allocation', async () => {
      const task: Task = {
        id: 'budget-task-003',
        name: 'Optimize Allocation',
        description: 'Optimize budget distribution',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['allocation_plan'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'optimize_allocation',
        budget: 100000,
        channels: [],
        performance: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.allocation).toBeDefined();
      expect(result.output.expectedROI).toBeGreaterThan(0);
    });

    it('should report ROI', async () => {
      const task: Task = {
        id: 'budget-task-004',
        name: 'Report ROI',
        description: 'Calculate return on investment',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 4500,
        deliverables: ['roi_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'report_roi',
        spending: {},
        revenue: {},
        timeframe: 'monthly',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.roi).toBeGreaterThan(0);
      expect(result.output.breakdown).toBeDefined();
    });
  });
});
