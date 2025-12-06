/**
 * DataCollector Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DataCollectorAgent } from './DataCollector';
import { Task, TaskStatus, AgentPriority } from '../../types';

describe('DataCollectorAgent', () => {
  let agent: DataCollectorAgent;

  beforeEach(() => {
    agent = new DataCollectorAgent('data-collector-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Data Collection', () => {
    it('should aggregate data from multiple sources', async () => {
      const task: Task = {
        id: 'data-task-001',
        name: 'Aggregate Data',
        description: 'Collect data from sources',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 5500,
        deliverables: ['aggregated_data'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'aggregate_data',
        sources: ['analytics', 'crm', 'social'],
        timeframe: '30_days',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.datasets).toBeDefined();
      expect(Array.isArray(result.output.datasets)).toBe(true);
      expect(result.output.metadata).toBeDefined();
      expect(result.output.quality).toBeGreaterThan(0);
    });

    it('should validate data quality', async () => {
      const task: Task = {
        id: 'data-task-002',
        name: 'Validate Quality',
        description: 'Check data quality',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 4500,
        deliverables: ['validation_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'validate_quality',
        data: [],
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.validated).toBeDefined();
      expect(result.output.score).toBeGreaterThan(0);
      expect(result.output.score).toBeLessThanOrEqual(1);
    });

    it('should transform data', async () => {
      const task: Task = {
        id: 'data-task-003',
        name: 'Transform Data',
        description: 'Clean and transform data',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['transformed_data'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'transform_data',
        rawData: [],
        schema: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.transformed).toBeDefined();
      expect(result.output.report).toBeDefined();
    });
  });
});
