/**
 * LiveItIconic Launch Platform - Base Agent Class
 *
 * Abstract base class that all specialized agents extend
 */

import {
  AgentConfig,
  AgentCapability,
  AgentMessage,
  AgentStatus,
  AgentType,
  AgentPriority,
  MessageType,
  Task,
  ActionResult,
} from '../types';
import { EventBus, eventBus } from './EventBus';
import { stateManager, AgentState } from './StateManager';

export abstract class BaseAgent {
  protected config: AgentConfig;
  protected status: AgentStatus = AgentStatus.IDLE;
  protected currentTask?: Task;
  protected eventBus: EventBus;
  protected subscriptionIds: string[] = [];
  protected taskHistory: ActionResult[] = [];
  protected learningData: Map<string, unknown> = new Map();

  constructor(config: AgentConfig, customEventBus?: EventBus) {
    this.config = config;
    this.eventBus = customEventBus || eventBus;
    this.initialize();
  }

  /**
   * Initialize agent
   */
  protected initialize(): void {
    // Subscribe to messages addressed to this agent
    const subId = this.eventBus.subscribe(
      this.config.id,
      this.handleMessage.bind(this),
      AgentPriority.NORMAL
    );
    this.subscriptionIds.push(subId);

    // Subscribe to broadcast messages for this agent type
    const typeSub = this.eventBus.subscribe(
      this.config.type,
      this.handleMessage.bind(this),
      AgentPriority.NORMAL
    );
    this.subscriptionIds.push(typeSub);

    // Register with state manager
    stateManager.registerAgent(this.getState());

    console.log(`[${this.config.name}] Agent initialized (${this.config.id})`);
  }

  /**
   * Handle incoming messages
   */
  protected async handleMessage(message: AgentMessage): Promise<void> {
    console.log(
      `[${this.config.name}] Received message: ${message.type} from ${message.from}`
    );

    try {
      switch (message.type) {
        case MessageType.REQUEST:
          await this.handleRequest(message);
          break;
        case MessageType.BROADCAST:
          await this.handleBroadcast(message);
          break;
        case MessageType.STATUS_UPDATE:
          await this.handleStatusUpdate(message);
          break;
        default:
          console.log(`[${this.config.name}] Unhandled message type: ${message.type}`);
      }
    } catch (error) {
      console.error(`[${this.config.name}] Error handling message:`, error);

      if (message.requiresAck) {
        await this.sendErrorResponse(message, error);
      }
    }
  }

  /**
   * Handle request messages
   */
  protected async handleRequest(message: AgentMessage): Promise<void> {
    this.setStatus(AgentStatus.THINKING);

    try {
      // Execute the capability
      const result = await this.execute(message.payload);

      // Send response
      const response: AgentMessage = {
        id: `resp_${message.id}`,
        from: this.config.id,
        to: message.from,
        type: MessageType.RESPONSE,
        priority: message.priority,
        payload: {
          success: true,
          result,
          originalMessageId: message.id,
        },
        timestamp: new Date(),
        requiresAck: false,
      };

      await this.eventBus.publish(response);
      this.setStatus(AgentStatus.IDLE);
    } catch (error) {
      this.setStatus(AgentStatus.FAILED);
      throw error;
    }
  }

  /**
   * Handle broadcast messages
   */
  protected async handleBroadcast(message: AgentMessage): Promise<void> {
    // Override in subclasses to handle broadcasts
    console.log(`[${this.config.name}] Received broadcast:`, message.payload);
  }

  /**
   * Handle status update messages
   */
  protected async handleStatusUpdate(message: AgentMessage): Promise<void> {
    // Override in subclasses to handle status updates
    console.log(`[${this.config.name}] Received status update:`, message.payload);
  }

  /**
   * Execute agent capability - must be implemented by subclasses
   */
  protected abstract execute(params: Record<string, unknown>): Promise<unknown>;

  /**
   * Send error response
   */
  protected async sendErrorResponse(originalMessage: AgentMessage, error: Record<string, unknown>): Promise<void> {
    const errorMessage: AgentMessage = {
      id: `error_${originalMessage.id}`,
      from: this.config.id,
      to: originalMessage.from,
      type: MessageType.ERROR,
      priority: AgentPriority.HIGH,
      payload: {
        error: error instanceof Error ? error.message : String(error),
        originalMessageId: originalMessage.id,
      },
      timestamp: new Date(),
      requiresAck: false,
    };

    await this.eventBus.publish(errorMessage);
  }

  /**
   * Send message to another agent
   */
  protected async sendMessage(
    to: string | string[],
    payload: Record<string, unknown>,
    type: MessageType = MessageType.REQUEST,
    requiresAck: boolean = true
  ): Promise<void> {
    const message: AgentMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      from: this.config.id,
      to,
      type,
      priority: AgentPriority.NORMAL,
      payload,
      timestamp: new Date(),
      requiresAck,
    };

    await this.eventBus.publish(message);
  }

  /**
   * Request data from another agent
   */
  protected async requestFromAgent(agentId: string, params: Record<string, unknown>, timeout?: number): Promise<unknown> {
    return await this.eventBus.request(agentId, params, this.config.id, timeout);
  }

  /**
   * Broadcast message to all agents of a type
   */
  protected async broadcastToType(agentType: AgentType, payload: Record<string, unknown>): Promise<void> {
    await this.eventBus.broadcast(agentType, payload, this.config.id);
  }

  /**
   * Execute a task
   */
  async executeTask(task: Task): Promise<ActionResult> {
    const startTime = Date.now();
    this.currentTask = task;
    this.setStatus(AgentStatus.EXECUTING);

    console.log(`[${this.config.name}] Starting task: ${task.name}`);

    try {
      // Update task status
      stateManager.updateTaskStatus(task.id, task.id, {
        status: 'in_progress',
      });

      // Execute the task
      const output = await this.execute(task);

      const duration = Date.now() - startTime;

      // Create action result
      const result: ActionResult = {
        success: true,
        output,
        metrics: {
          duration,
        },
        duration,
      };

      // Record in history
      this.taskHistory.push(result);

      // Update task status
      stateManager.updateTaskStatus(task.id, task.id, {
        status: 'completed',
        completedAt: new Date(),
        actualDuration: duration,
      });

      // Update agent metrics
      this.updateMetrics(result);

      this.setStatus(AgentStatus.COMPLETED);
      console.log(`[${this.config.name}] Completed task: ${task.name} in ${duration}ms`);

      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      const errorMessage = error instanceof Error ? error.message : String(error);

      const result: ActionResult = {
        success: false,
        output: null,
        metrics: {
          duration,
        },
        duration,
        error: errorMessage,
      };

      this.taskHistory.push(result);

      // Update task status
      stateManager.updateTaskStatus(task.id, task.id, {
        status: 'failed',
      });

      this.setStatus(AgentStatus.FAILED);
      console.error(`[${this.config.name}] Task failed: ${task.name}`, error);

      throw error;
    } finally {
      this.currentTask = undefined;
    }
  }

  /**
   * Update agent status
   */
  protected setStatus(status: AgentStatus): void {
    this.status = status;
    stateManager.updateAgentStatus(this.config.id, status, this.currentTask);
  }

  /**
   * Update agent metrics
   */
  protected updateMetrics(result: ActionResult): void {
    const state = stateManager.getAgent(this.config.id);
    if (!state) return;

    const metrics = state.metrics;
    const totalTasks = metrics.tasksCompleted + metrics.tasksFailed;

    if (result.success) {
      metrics.tasksCompleted++;
    } else {
      metrics.tasksFailed++;
    }

    // Update average duration
    metrics.averageDuration =
      (metrics.averageDuration * totalTasks + result.duration) / (totalTasks + 1);

    // Update success rate
    metrics.successRate = metrics.tasksCompleted / (totalTasks + 1);

    stateManager.updateAgent(this.config.id, { metrics });
  }

  /**
   * Learn from experience
   */
  protected async learn(experience: Record<string, unknown>): Promise<void> {
    if (!this.config.learningEnabled) return;

    // Store in learning data
    const key = `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.learningData.set(key, experience);

    // Limit learning data size
    if (this.learningData.size > 1000) {
      const firstKey = this.learningData.keys().next().value;
      this.learningData.delete(firstKey);
    }

    console.log(`[${this.config.name}] Learned from experience`);
  }

  /**
   * Get agent capabilities
   */
  getCapabilities(): AgentCapability[] {
    return this.config.capabilities;
  }

  /**
   * Get agent state
   */
  getState(): AgentState {
    return {
      id: this.config.id,
      type: this.config.type,
      status: this.status,
      currentTask: this.currentTask,
      metrics: {
        tasksCompleted: this.taskHistory.filter(t => t.success).length,
        tasksInProgress: this.currentTask ? 1 : 0,
        tasksFailed: this.taskHistory.filter(t => !t.success).length,
        averageDuration:
          this.taskHistory.reduce((sum, t) => sum + t.duration, 0) / this.taskHistory.length || 0,
        successRate:
          this.taskHistory.filter(t => t.success).length / this.taskHistory.length || 0,
      },
      lastActivity: new Date(),
    };
  }

  /**
   * Get task history
   */
  getTaskHistory(): ActionResult[] {
    return [...this.taskHistory];
  }

  /**
   * Pause agent
   */
  pause(): void {
    this.setStatus(AgentStatus.PAUSED);
    console.log(`[${this.config.name}] Agent paused`);
  }

  /**
   * Resume agent
   */
  resume(): void {
    this.setStatus(AgentStatus.IDLE);
    console.log(`[${this.config.name}] Agent resumed`);
  }

  /**
   * Shutdown agent
   */
  shutdown(): void {
    // Unsubscribe from all topics
    this.subscriptionIds.forEach(id => this.eventBus.unsubscribe(id));
    this.subscriptionIds = [];

    this.setStatus(AgentStatus.IDLE);
    console.log(`[${this.config.name}] Agent shutdown`);
  }

  /**
   * Get agent ID
   */
  getId(): string {
    return this.config.id;
  }

  /**
   * Get agent name
   */
  getName(): string {
    return this.config.name;
  }

  /**
   * Get agent type
   */
  getType(): AgentType {
    return this.config.type;
  }
}
