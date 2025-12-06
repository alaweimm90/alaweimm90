/**
 * BaseAgent Tests
 *
 * Tests for the abstract BaseAgent class that all specialized agents extend
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { BaseAgent } from './BaseAgent';
import { EventBus } from './EventBus';
import { stateManager } from './StateManager';
import {
  AgentConfig,
  AgentType,
  AgentStatus,
  AgentPriority,
  MessageType,
  Task,
  TaskStatus,
} from '../types';

// Concrete implementation for testing
class TestAgent extends BaseAgent {
  public executeCallCount = 0;
  public lastParams: Record<string, unknown> | null = null;

  protected async execute(params: Record<string, unknown>): Promise<unknown> {
    this.executeCallCount++;
    this.lastParams = params;

    if (params.shouldFail) {
      throw new Error('Test execution failure');
    }

    return {
      success: true,
      data: 'test result',
      processedParams: params,
    };
  }
}

describe('BaseAgent', () => {
  let testAgent: TestAgent;
  let eventBus: EventBus;
  let config: AgentConfig;

  beforeEach(() => {
    // Create a fresh EventBus for each test
    eventBus = new EventBus();

    config = {
      id: 'test-agent-001',
      name: 'Test Agent',
      type: AgentType.COMPETITOR_ANALYST,
      capabilities: [
        {
          name: 'test_capability',
          description: 'Test capability',
          inputs: { param1: 'string' },
          outputs: { result: 'string' },
          constraints: [],
          dependencies: [],
          estimatedDuration: 1000,
          successMetrics: [
            { name: 'test_metric', target: 1.0, unit: 'score' },
          ],
        },
      ],
      maxConcurrentTasks: 3,
      timeout: 5000,
      retryAttempts: 3,
      learningEnabled: true,
    };

    testAgent = new TestAgent(config, eventBus);
  });

  afterEach(() => {
    testAgent.shutdown();
  });

  describe('Initialization', () => {
    it('should initialize with correct config', () => {
      expect(testAgent.getId()).toBe('test-agent-001');
      expect(testAgent.getName()).toBe('Test Agent');
      expect(testAgent.getType()).toBe(AgentType.COMPETITOR_ANALYST);
    });

    it('should register with state manager', () => {
      const state = stateManager.getAgent('test-agent-001');
      expect(state).toBeDefined();
      expect(state?.id).toBe('test-agent-001');
      expect(state?.type).toBe(AgentType.COMPETITOR_ANALYST);
      expect(state?.status).toBe(AgentStatus.IDLE);
    });

    it('should subscribe to event bus', () => {
      // Agent should be subscribed to its ID and type
      expect(testAgent['subscriptionIds'].length).toBeGreaterThanOrEqual(2);
    });

    it('should start with IDLE status', () => {
      const state = testAgent.getState();
      expect(state.status).toBe(AgentStatus.IDLE);
    });

    it('should initialize with empty task history', () => {
      const history = testAgent.getTaskHistory();
      expect(history).toEqual([]);
    });
  });

  describe('Capabilities', () => {
    it('should return agent capabilities', () => {
      const capabilities = testAgent.getCapabilities();
      expect(capabilities).toHaveLength(1);
      expect(capabilities[0].name).toBe('test_capability');
    });

    it('should have correct capability structure', () => {
      const capabilities = testAgent.getCapabilities();
      const cap = capabilities[0];

      expect(cap).toHaveProperty('name');
      expect(cap).toHaveProperty('description');
      expect(cap).toHaveProperty('inputs');
      expect(cap).toHaveProperty('outputs');
      expect(cap).toHaveProperty('successMetrics');
      expect(cap).toHaveProperty('estimatedDuration');
    });
  });

  describe('Task Execution', () => {
    it('should execute task successfully', async () => {
      const task: Task = {
        id: 'task-001',
        name: 'Test Task',
        description: 'A test task',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: ['result'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
      };

      const result = await testAgent.executeTask(task);

      expect(result.success).toBe(true);
      expect(result.output).toBeDefined();
      expect(result.duration).toBeGreaterThan(0);
      expect(testAgent.executeCallCount).toBe(1);
    });

    it('should handle task execution failure', async () => {
      const task: Task = {
        id: 'task-002',
        name: 'Failing Task',
        description: 'A task that will fail',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: ['result'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        shouldFail: true,
      };

      await expect(testAgent.executeTask(task)).rejects.toThrow('Test execution failure');

      const state = testAgent.getState();
      expect(state.status).toBe(AgentStatus.FAILED);
    });

    it('should update status during task execution', async () => {
      const task: Task = {
        id: 'task-003',
        name: 'Status Test Task',
        description: 'Test status changes',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: ['result'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
      };

      // Start with IDLE status
      expect(testAgent.getState().status).toBe(AgentStatus.IDLE);

      const resultPromise = testAgent.executeTask(task);

      // Should change to EXECUTING during execution
      // (May be brief, so we check after)

      await resultPromise;

      // Should be COMPLETED after success
      expect(testAgent.getState().status).toBe(AgentStatus.COMPLETED);
    });

    it('should record task in history', async () => {
      const task: Task = {
        id: 'task-004',
        name: 'History Test',
        description: 'Test history recording',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: ['result'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
      };

      await testAgent.executeTask(task);

      const history = testAgent.getTaskHistory();
      expect(history).toHaveLength(1);
      expect(history[0].success).toBe(true);
      expect(history[0].duration).toBeGreaterThan(0);
    });

    it('should track metrics from task execution', async () => {
      const task: Task = {
        id: 'task-005',
        name: 'Metrics Test',
        description: 'Test metrics tracking',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: ['result'],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
      };

      await testAgent.executeTask(task);

      const state = testAgent.getState();
      expect(state.metrics.tasksCompleted).toBe(1);
      expect(state.metrics.tasksFailed).toBe(0);
      expect(state.metrics.successRate).toBe(1.0);
      expect(state.metrics.averageDuration).toBeGreaterThan(0);
    });
  });

  describe('Message Handling', () => {
    it('should handle REQUEST messages', async () => {
      const message = {
        id: 'msg-001',
        from: 'orchestrator-001',
        to: 'test-agent-001',
        type: MessageType.REQUEST,
        priority: AgentPriority.NORMAL,
        payload: { action: 'test' },
        timestamp: new Date(),
        requiresAck: true,
      };

      // Spy on event bus publish to capture response
      const publishSpy = vi.spyOn(eventBus, 'publish');

      await testAgent['handleMessage'](message);

      // Should publish response
      expect(publishSpy).toHaveBeenCalled();
      expect(testAgent.executeCallCount).toBe(1);
    });

    it('should send error response on execution failure', async () => {
      const message = {
        id: 'msg-002',
        from: 'orchestrator-001',
        to: 'test-agent-001',
        type: MessageType.REQUEST,
        priority: AgentPriority.NORMAL,
        payload: { shouldFail: true },
        timestamp: new Date(),
        requiresAck: true,
      };

      const publishSpy = vi.spyOn(eventBus, 'publish');

      await testAgent['handleMessage'](message);

      // Should publish error message
      expect(publishSpy).toHaveBeenCalled();
      const errorCall = publishSpy.mock.calls.find(
        call => call[0].type === MessageType.ERROR
      );
      expect(errorCall).toBeDefined();
    });

    it('should handle BROADCAST messages', async () => {
      const message = {
        id: 'msg-003',
        from: 'orchestrator-001',
        to: AgentType.COMPETITOR_ANALYST,
        type: MessageType.BROADCAST,
        priority: AgentPriority.NORMAL,
        payload: { announcement: 'test broadcast' },
        timestamp: new Date(),
        requiresAck: false,
      };

      // Should not throw
      await expect(testAgent['handleMessage'](message)).resolves.not.toThrow();
    });

    it('should send messages to other agents', async () => {
      const publishSpy = vi.spyOn(eventBus, 'publish');

      await testAgent['sendMessage'](
        'other-agent-001',
        { data: 'test' },
        MessageType.REQUEST
      );

      expect(publishSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          from: 'test-agent-001',
          to: 'other-agent-001',
          type: MessageType.REQUEST,
        })
      );
    });
  });

  describe('State Management', () => {
    it('should return current state', () => {
      const state = testAgent.getState();

      expect(state).toHaveProperty('id');
      expect(state).toHaveProperty('type');
      expect(state).toHaveProperty('status');
      expect(state).toHaveProperty('metrics');
      expect(state).toHaveProperty('lastActivity');
    });

    it('should update state on status change', () => {
      testAgent['setStatus'](AgentStatus.EXECUTING);

      const state = stateManager.getAgent('test-agent-001');
      expect(state?.status).toBe(AgentStatus.EXECUTING);
    });

    it('should track metrics correctly', async () => {
      // Execute successful task
      const task1: Task = {
        id: 'task-metrics-1',
        name: 'Success Task',
        description: 'Should succeed',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: [],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
      };

      await testAgent.executeTask(task1);

      // Execute failing task
      const task2: Task = {
        id: 'task-metrics-2',
        name: 'Failing Task',
        description: 'Should fail',
        assignedTo: 'test-agent-001',
        status: TaskStatus.PENDING,
        priority: AgentPriority.NORMAL,
        dependencies: [],
        estimatedDuration: 1000,
        deliverables: [],
        createdAt: new Date(),
        dueDate: new Date(Date.now() + 86400000),
        shouldFail: true,
      };

      try {
        await testAgent.executeTask(task2);
      } catch {
        // Expected to fail
      }

      const state = testAgent.getState();
      expect(state.metrics.tasksCompleted).toBe(1);
      expect(state.metrics.tasksFailed).toBe(1);
      expect(state.metrics.successRate).toBe(0.5);
    });
  });

  describe('Lifecycle Management', () => {
    it('should pause agent', () => {
      testAgent.pause();
      expect(testAgent.getState().status).toBe(AgentStatus.PAUSED);
    });

    it('should resume agent', () => {
      testAgent.pause();
      testAgent.resume();
      expect(testAgent.getState().status).toBe(AgentStatus.IDLE);
    });

    it('should shutdown cleanly', () => {
      const initialSubCount = testAgent['subscriptionIds'].length;
      expect(initialSubCount).toBeGreaterThan(0);

      testAgent.shutdown();

      expect(testAgent['subscriptionIds']).toHaveLength(0);
      expect(testAgent.getState().status).toBe(AgentStatus.IDLE);
    });

    it('should not receive messages after shutdown', async () => {
      testAgent.shutdown();

      const initialExecuteCount = testAgent.executeCallCount;

      const message = {
        id: 'msg-after-shutdown',
        from: 'orchestrator-001',
        to: 'test-agent-001',
        type: MessageType.REQUEST,
        priority: AgentPriority.NORMAL,
        payload: { action: 'test' },
        timestamp: new Date(),
        requiresAck: false,
      };

      // Publish directly through event bus
      await eventBus.publish(message);

      // Agent should not have processed it
      expect(testAgent.executeCallCount).toBe(initialExecuteCount);
    });
  });

  describe('Learning', () => {
    it('should store learning data when enabled', async () => {
      await testAgent['learn']({ experience: 'test' });

      expect(testAgent['learningData'].size).toBe(1);
    });

    it('should not store learning data when disabled', async () => {
      const nonLearningConfig = { ...config, learningEnabled: false };
      const nonLearningAgent = new TestAgent(nonLearningConfig, eventBus);

      await nonLearningAgent['learn']({ experience: 'test' });

      expect(nonLearningAgent['learningData'].size).toBe(0);

      nonLearningAgent.shutdown();
    });

    it('should limit learning data size', async () => {
      // Add more than 1000 items
      for (let i = 0; i < 1050; i++) {
        await testAgent['learn']({ iteration: i });
      }

      // Should be capped at 1000
      expect(testAgent['learningData'].size).toBeLessThanOrEqual(1000);
    });
  });

  describe('Inter-Agent Communication', () => {
    it('should broadcast to agent type', async () => {
      const publishSpy = vi.spyOn(eventBus, 'broadcast');

      await testAgent['broadcastToType'](
        AgentType.TREND_DETECTOR,
        { data: 'broadcast test' }
      );

      expect(publishSpy).toHaveBeenCalledWith(
        AgentType.TREND_DETECTOR,
        { data: 'broadcast test' },
        'test-agent-001'
      );
    });

    it('should request data from another agent', async () => {
      const requestSpy = vi.spyOn(eventBus, 'request').mockResolvedValue({
        success: true,
        data: 'response data',
      });

      const result = await testAgent['requestFromAgent'](
        'other-agent-001',
        { query: 'test' }
      );

      expect(requestSpy).toHaveBeenCalledWith(
        'other-agent-001',
        { query: 'test' },
        'test-agent-001',
        undefined
      );
      expect(result).toEqual({
        success: true,
        data: 'response data',
      });
    });
  });
});
