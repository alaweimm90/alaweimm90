/**
 * CopyWriter Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { CopyWriterAgent } from './CopyWriter';
import { AgentStatus, Task, TaskStatus, AgentPriority } from '../../types';

describe('CopyWriterAgent', () => {
  let agent: CopyWriterAgent;

  beforeEach(() => {
    agent = new CopyWriterAgent('copywriter-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Content Creation', () => {
    it('should write headlines', async () => {
      const task: Task = {
        id: 'copy-task-001',
        name: 'Write Headlines',
        description: 'Create compelling headlines',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 4000,
        deliverables: ['headlines'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'write_headlines',
        productInfo: { name: 'Test Product' },
        count: 10,
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.headlines).toBeDefined();
      expect(Array.isArray(result.output.headlines)).toBe(true);
      expect(result.output.headlines.length).toBeGreaterThan(0);
    });

    it('should craft product descriptions', async () => {
      const task: Task = {
        id: 'copy-task-002',
        name: 'Product Description',
        description: 'Write product copy',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 5000,
        deliverables: ['description'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'craft_product_copy',
        product: { name: 'Test', features: [] },
        tone: 'professional',
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.copy).toBeDefined();
      expect(result.output.wordCount).toBeGreaterThan(0);
    });

    it('should optimize existing copy', async () => {
      const task: Task = {
        id: 'copy-task-003',
        name: 'Optimize Copy',
        description: 'Improve existing copy',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 4500,
        deliverables: ['optimized_copy'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'optimize_copy',
        original: 'Test copy to optimize',
        metrics: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.optimized).toBeDefined();
      expect(result.output.improvements).toBeDefined();
    });
  });
});
