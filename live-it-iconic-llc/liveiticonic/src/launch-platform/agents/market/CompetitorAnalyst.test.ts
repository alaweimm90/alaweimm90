/**
 * CompetitorAnalyst Agent Tests
 *
 * Tests for the CompetitorAnalyst agent
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CompetitorAnalystAgent } from './CompetitorAnalyst';
import { EventBus } from '../../core/EventBus';
import { AgentStatus, Task, TaskStatus, AgentPriority } from '../../types';

describe('CompetitorAnalystAgent', () => {
  let agent: CompetitorAnalystAgent;
  let eventBus: EventBus;

  beforeEach(() => {
    eventBus = new EventBus();
    agent = new CompetitorAnalystAgent('competitor-analyst-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize with correct type', () => {
      expect(agent.getType()).toBe('competitor_analyst');
    });

    it('should have required capabilities', () => {
      const capabilities = agent.getCapabilities();
      const capabilityNames = capabilities.map(c => c.name);

      expect(capabilityNames).toContain('analyze_competitors');
      expect(capabilityNames).toContain('identify_gaps');
      expect(capabilityNames).toContain('track_activities');
      expect(capabilityNames).toContain('predict_moves');
    });

    it('should start with IDLE status', () => {
      expect(agent.getState().status).toBe(AgentStatus.IDLE);
    });
  });

  describe('Competitor Analysis', () => {
    it('should analyze competitors successfully', async () => {
      const task: Task = {
        id: 'task-001',
        name: 'Analyze Competitors',
        description: 'Analyze luxury automotive competitors',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['competitor_analysis'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'analyze_competitors',
        market: 'luxury_automotive',
        focus: 'pricing_strategies',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output).toBeDefined();
      expect(result.output.competitors).toBeDefined();
      expect(Array.isArray(result.output.competitors)).toBe(true);
      expect(result.output.competitors.length).toBeGreaterThan(0);
    });

    it('should identify competitive gaps', async () => {
      const task: Task = {
        id: 'task-002',
        name: 'Identify Gaps',
        description: 'Find market gaps',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 4000,
        deliverables: ['gap_analysis'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'identify_gaps',
        competitorData: {
          competitors: [
            { name: 'Competitor A', strengths: ['brand'] },
            { name: 'Competitor B', strengths: ['distribution'] },
          ],
        },
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.gaps).toBeDefined();
      expect(result.output.opportunities).toBeDefined();
      expect(Array.isArray(result.output.gaps)).toBe(true);
    });

    it('should track competitor activities', async () => {
      const task: Task = {
        id: 'task-003',
        name: 'Track Activities',
        description: 'Monitor competitor movements',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 3500,
        deliverables: ['activity_report'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'track_activities',
        competitors: ['comp-001', 'comp-002'],
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.activities).toBeDefined();
      expect(result.output.alerts).toBeDefined();
    });

    it('should predict competitor moves', async () => {
      const task: Task = {
        id: 'task-004',
        name: 'Predict Moves',
        description: 'Forecast competitor strategies',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['predictions'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'predict_moves',
        competitor: 'comp-001',
        historical: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.predictions).toBeDefined();
      expect(result.output.confidence).toBeDefined();
      expect(result.output.confidence).toBeGreaterThan(0);
      expect(result.output.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Performance Metrics', () => {
    it('should track task completion metrics', async () => {
      const task: Task = {
        id: 'task-metrics',
        name: 'Test Metrics',
        description: 'Test metric tracking',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 3000,
        deliverables: [],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'analyze_competitors',
      };

      await agent.executeTask(task);

      const state = agent.getState();
      expect(state.metrics.tasksCompleted).toBe(1);
      expect(state.metrics.successRate).toBe(1.0);
      expect(state.metrics.averageDuration).toBeGreaterThan(0);
    });

    it('should measure execution duration', async () => {
      const task: Task = {
        id: 'task-duration',
        name: 'Duration Test',
        description: 'Test duration measurement',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 3000,
        deliverables: [],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'analyze_competitors',
      };

      const result = await agent.executeTask(task);

      expect(result.duration).toBeGreaterThan(0);
      expect(result.duration).toBeLessThan(10000); // Should complete quickly in tests
    });
  });

  describe('Capability Validation', () => {
    it('should have correct capability structure', () => {
      const capabilities = agent.getCapabilities();

      capabilities.forEach(cap => {
        expect(cap).toHaveProperty('name');
        expect(cap).toHaveProperty('description');
        expect(cap).toHaveProperty('inputs');
        expect(cap).toHaveProperty('outputs');
        expect(cap).toHaveProperty('estimatedDuration');
        expect(cap).toHaveProperty('successMetrics');
        expect(Array.isArray(cap.successMetrics)).toBe(true);
      });
    });

    it('should have realistic duration estimates', () => {
      const capabilities = agent.getCapabilities();

      capabilities.forEach(cap => {
        expect(cap.estimatedDuration).toBeGreaterThan(0);
        expect(cap.estimatedDuration).toBeLessThan(60000); // < 1 minute
      });
    });

    it('should define success metrics for each capability', () => {
      const capabilities = agent.getCapabilities();

      capabilities.forEach(cap => {
        expect(cap.successMetrics.length).toBeGreaterThan(0);
        cap.successMetrics.forEach(metric => {
          expect(metric).toHaveProperty('name');
          expect(metric).toHaveProperty('target');
          expect(metric).toHaveProperty('unit');
        });
      });
    });
  });
});
