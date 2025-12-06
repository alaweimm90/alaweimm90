/**
 * BrandArchitect Agent Tests
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { BrandArchitectAgent } from './BrandArchitect';
import { AgentStatus, Task, TaskStatus, AgentPriority } from '../../types';

describe('BrandArchitectAgent', () => {
  let agent: BrandArchitectAgent;

  beforeEach(() => {
    agent = new BrandArchitectAgent('brand-architect-test-001');
  });

  afterEach(() => {
    agent.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize correctly', () => {
      expect(agent.getName()).toBe('Brand Architect');
      expect(agent.getType()).toBe('brand_architect');
    });

    it('should have brand strategy capabilities', () => {
      const capabilities = agent.getCapabilities();
      const capNames = capabilities.map(c => c.name);

      expect(capNames).toContain('define_identity');
      expect(capNames).toContain('create_positioning');
      expect(capNames).toContain('develop_voice');
      expect(capNames).toContain('design_guidelines');
    });
  });

  describe('Brand Identity', () => {
    it('should define brand identity', async () => {
      const task: Task = {
        id: 'brand-task-001',
        name: 'Define Identity',
        description: 'Create brand identity',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.CRITICAL,
        dependencies: [],
        estimatedDuration: 8000,
        deliverables: ['brand_identity'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'define_identity',
        productInfo: { name: 'Test Product' },
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.identity).toBeDefined();
      expect(result.output.identity.mission).toBeDefined();
      expect(result.output.identity.values).toBeDefined();
      expect(result.output.identity.personality).toBeDefined();
    });

    it('should create brand positioning', async () => {
      const task: Task = {
        id: 'brand-task-002',
        name: 'Create Positioning',
        description: 'Develop positioning strategy',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 7000,
        deliverables: ['positioning'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'create_positioning',
        target: { segment: 'luxury' },
        competitors: [],
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.positioning).toBeDefined();
      expect(result.output.positioning.category).toBeDefined();
      expect(result.output.positioning.differentiation).toBeDefined();
    });

    it('should develop brand voice', async () => {
      const task: Task = {
        id: 'brand-task-003',
        name: 'Develop Voice',
        description: 'Create brand voice guidelines',
        assignedTo: agent.getId(),
        status: TaskStatus.PENDING,
        priority: AgentPriority.HIGH,
        dependencies: [],
        estimatedDuration: 6000,
        deliverables: ['voice_guidelines'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        action: 'develop_voice',
        brandIdentity: {},
      };

      const result = await agent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output.voice).toBeDefined();
      expect(result.output.voice.attributes).toBeDefined();
      expect(result.output.voice.examples).toBeDefined();
    });
  });
});
