/**
 * AnalyticsInterpreter Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { AnalyticsInterpreterAgent } from './AnalyticsInterpreter';
import { Task, TaskStatus, AgentPriority } from '../../types';

describe('AnalyticsInterpreterAgent', () => {
  let agent: AnalyticsInterpreterAgent;

  beforeEach(() => {
    agent = new AnalyticsInterpreterAgent('analytics-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Analytics Analysis', () => {
    it('should analyze campaign metrics', async () => {
      const task: Task = {
        id: 'analytics-task-001',
        name: 'Analyze Metrics',
        description: 'Interpret campaign data',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['analysis'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'analyze_metrics',
        data: { impressions: 100000, clicks: 5000 },
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.insights).toBeDefined();
      expect(result.output.recommendations).toBeDefined();
    });

    it('should identify patterns in data', async () => {
      const task: Task = {
        id: 'analytics-task-002',
        name: 'Identify Patterns',
        description: 'Find data patterns',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['patterns'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'identify_patterns',
        dataset: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.patterns).toBeDefined();
      expect(Array.isArray(result.output.patterns)).toBe(true);
    });

    it('should generate insights from metrics', async () => {
      const task: Task = {
        id: 'analytics-task-003',
        name: 'Generate Insights',
        description: 'Extract actionable insights',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 5500,
        deliverables: ['insights'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'generate_insights',
        metrics: {},
        context: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.insights).toBeDefined();
      expect(result.output.actionable).toBe(true);
    });
  });
});
