/**
 * SocialMediaStrategist Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { SocialMediaStrategistAgent } from './SocialMediaStrategist';
import { Task, TaskStatus, AgentPriority } from '../../types';

describe('SocialMediaStrategistAgent', () => {
  let agent: SocialMediaStrategistAgent;

  beforeEach(() => {
    agent = new SocialMediaStrategistAgent('social-strategist-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Platform Strategy', () => {
    it('should develop platform-specific strategies', async () => {
      const task: Task = {
        id: 'social-task-001',
        name: 'Platform Strategy',
        description: 'Create social media strategy',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['platform_strategy'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'develop_platform_strategy',
        platforms: ['instagram', 'twitter', 'linkedin'],
        audience: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.strategy).toBeDefined();
      expect(result.output.strategy.instagram).toBeDefined();
    });

    it('should create content calendar', async () => {
      const task: Task = {
        id: 'social-task-002',
        name: 'Content Calendar',
        description: 'Plan content schedule',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 5500,
        deliverables: ['content_calendar'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'create_content_calendar',
        strategy: {},
        duration: 30,
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.calendar).toBeDefined();
      expect(Array.isArray(result.output.calendar)).toBe(true);
    });
  });
});
