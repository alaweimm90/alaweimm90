/**
 * Base Agent class
 */

import { AgentConfig, AgentContext, AgentTask, TaskStatus, AgentResult } from './types';

export abstract class BaseAgent {
  protected config: AgentConfig;

  protected context?: AgentContext;

  protected tasks: Map<string, AgentTask> = new Map();

  constructor(config: AgentConfig) {
    this.config = config;
  }

  /**
   * Initialize the agent with context
   * @param context
   */
  public async initialize(context: AgentContext): Promise<void> {
    this.context = context;
    await this.onInitialize();
  }

  /**
   * Execute a task - to be implemented by subclasses
   */
  public abstract execute(task: AgentTask): Promise<AgentResult>;

  /**
   * Hook for subclasses to perform custom initialization
   */
  protected async onInitialize(): Promise<void> {
    // Override in subclass
  }

  /**
   * Get agent configuration
   */
  public getConfig(): AgentConfig {
    return this.config;
  }

  /**
   * Get agent context
   */
  public getContext(): AgentContext | undefined {
    return this.context;
  }

  /**
   * Get a task by ID
   * @param id
   */
  public getTask(id: string): AgentTask | undefined {
    return this.tasks.get(id);
  }

  /**
   * Register a new task
   * @param task
   */
  public registerTask(task: AgentTask): void {
    this.tasks.set(task.id, task);
  }

  /**
   * Update task status
   * @param taskId
   * @param status
   */
  public updateTaskStatus(taskId: string, status: TaskStatus): void {
    const task = this.tasks.get(taskId);
    if (task) {
      task.status = status;
    }
  }

  /**
   * Get all tasks
   */
  public getTasks(): AgentTask[] {
    return Array.from(this.tasks.values());
  }

  /**
   * Check if agent can execute task
   * @param task
   */
  public canExecute(task: AgentTask): boolean {
    // Check if agent has required capabilities
    return this.config.capabilities.includes(task.type);
  }
}

/**
 * Code Agent - Specialized for code manipulation
 */
export class CodeAgent extends BaseAgent {
  public async execute(task: AgentTask): Promise<AgentResult> {
    const startTime = Date.now();

    try {
      if (!this.canExecute(task)) {
        throw new Error(`CodeAgent cannot execute task type: ${task.type}`);
      }

      this.updateTaskStatus(task.id, TaskStatus.RUNNING);

      // Implement code-specific logic
      const result = await this.executeCodeTask(task);

      this.updateTaskStatus(task.id, TaskStatus.COMPLETED);

      return {
        success: true,
        data: result,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      this.updateTaskStatus(task.id, TaskStatus.FAILED);
      return {
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        duration: Date.now() - startTime,
      };
    }
  }

  private async executeCodeTask(task: AgentTask): Promise<unknown> {
    // Implementation for code-specific tasks
    return {
      message: 'Code task executed',
      taskId: task.id,
    };
  }
}

/**
 * Analysis Agent - Specialized for analysis and reporting
 */
export class AnalysisAgent extends BaseAgent {
  public async execute(task: AgentTask): Promise<AgentResult> {
    const startTime = Date.now();

    try {
      if (!this.canExecute(task)) {
        throw new Error(`AnalysisAgent cannot execute task type: ${task.type}`);
      }

      this.updateTaskStatus(task.id, TaskStatus.RUNNING);

      const result = await this.performAnalysis(task);

      this.updateTaskStatus(task.id, TaskStatus.COMPLETED);

      return {
        success: true,
        data: result,
        duration: Date.now() - startTime,
      };
    } catch (error) {
      this.updateTaskStatus(task.id, TaskStatus.FAILED);
      return {
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
        duration: Date.now() - startTime,
      };
    }
  }

  private async performAnalysis(task: AgentTask): Promise<unknown> {
    // Implementation for analysis tasks
    return {
      message: 'Analysis completed',
      taskId: task.id,
    };
  }
}
