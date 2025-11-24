/**
 * Agent Orchestrator - Manages agent execution and workflows
 */

import { BaseAgent } from './agent';
import { AgentContext, Workflow, WorkflowStep, OrchestrationRule, AgentTask, TaskStatus } from './types';

export class AgentOrchestrator {
  private agents: Map<string, BaseAgent> = new Map();

  private context?: AgentContext;

  private workflows: Map<string, Workflow> = new Map();

  private rules: Map<string, OrchestrationRule> = new Map();

  /**
   * Initialize orchestrator with context
   * @param context
   */
  public async initialize(context: AgentContext): Promise<void> {
    this.context = context;

    // Initialize all registered agents
    for (const agent of this.agents.values()) {
      await agent.initialize(context);
    }
  }

  /**
   * Register an agent
   * @param agent
   */
  public registerAgent(agent: BaseAgent): void {
    const config = agent.getConfig();
    this.agents.set(config.id, agent);
  }

  /**
   * Get an agent by ID
   * @param id
   */
  public getAgent(id: string): BaseAgent | undefined {
    return this.agents.get(id);
  }

  /**
   * Get all agents
   */
  public getAgents(): BaseAgent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Register a workflow
   * @param workflow
   */
  public registerWorkflow(workflow: Workflow): void {
    this.workflows.set(workflow.id, workflow);
  }

  /**
   * Get a workflow by ID
   * @param id
   */
  public getWorkflow(id: string): Workflow | undefined {
    return this.workflows.get(id);
  }

  /**
   * Register an orchestration rule
   * @param rule
   */
  public registerRule(rule: OrchestrationRule): void {
    this.rules.set(rule.id, rule);
  }

  /**
   * Get a rule by ID
   * @param id
   */
  public getRule(id: string): OrchestrationRule | undefined {
    return this.rules.get(id);
  }

  /**
   * Execute a task with the appropriate agent
   * @param task
   */
  public async executeTask(task: AgentTask): Promise<unknown> {
    if (!task.agentId) {
      throw new Error('Task must specify an agentId');
    }

    const agent = this.getAgent(task.agentId);
    if (!agent) {
      throw new Error(`Agent not found: ${task.agentId}`);
    }

    if (!agent.canExecute(task)) {
      throw new Error(`Agent ${task.agentId} cannot execute task type: ${task.type}`);
    }

    const result = await agent.execute(task);
    return result;
  }

  /**
   * Execute a workflow
   * @param workflowId
   */
  public async executeWorkflow(workflowId: string): Promise<Map<string, unknown>> {
    const workflow = this.getWorkflow(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    if (!workflow.enabled) {
      throw new Error(`Workflow is disabled: ${workflowId}`);
    }

    const results = new Map<string, unknown>();

    // Execute steps in order (simplified - doesn't handle parallel or conditional logic yet)
    for (const step of workflow.steps) {
      const result = await this.executeWorkflowStep(step, results);
      results.set(step.id, result);
    }

    return results;
  }

  /**
   * Execute a single workflow step
   * @param step
   * @param previousResults
   */
  private async executeWorkflowStep(step: WorkflowStep, previousResults: Map<string, unknown>): Promise<unknown> {
    switch (step.type) {
      case 'task':
        return this.executeStepTask(step);
      case 'conditional':
        return this.executeConditionalStep(step, previousResults);
      case 'parallel':
      case 'sequential':
        // Placeholder for future implementation
        return { message: 'Step executed', stepId: step.id };
      default:
        throw new Error(`Unknown step type: ${step.type}`);
    }
  }

  /**
   * Execute a task step
   * @param step
   */
  private async executeStepTask(step: WorkflowStep): Promise<unknown> {
    if (!step.agentId) {
      throw new Error('Task step must specify an agentId');
    }

    const task: AgentTask = {
      id: step.id,
      name: step.name,
      description: '',
      type: step.action || 'unknown',
      input: step.inputs,
      status: TaskStatus.PENDING,
      agentId: step.agentId,
    };

    return this.executeTask(task);
  }

  /**
   * Execute a conditional step
   * @param step
   * @param previousResults
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private async executeConditionalStep(step: WorkflowStep, previousResults: Map<string, unknown>): Promise<unknown> {
    // Placeholder - evaluate condition and execute based on result
    if (step.condition) {
      // Simple condition evaluation (can be enhanced)
      return { message: 'Conditional evaluated', stepId: step.id };
    }
    return { message: 'Conditional skipped', stepId: step.id };
  }

  /**
   * Apply an orchestration rule
   * @param ruleName
   */
  public async applyRule(ruleName: string): Promise<unknown> {
    const rule = Array.from(this.rules.values()).find(r => r.name === ruleName);
    if (!rule) {
      throw new Error(`Rule not found: ${ruleName}`);
    }

    if (!rule.enabled) {
      throw new Error(`Rule is disabled: ${ruleName}`);
    }

    const results = [];
    for (const action of rule.actions) {
      const result = await this.executeAction(action);
      results.push(result);
    }

    return results;
  }

  /**
   * Execute an action from a rule
   * @param action
   */
  private async executeAction(action: OrchestrationAction): Promise<unknown> {
    switch (action.type) {
      case 'execute':
        return { message: 'Action executed', target: action.target };
      case 'schedule':
        return { message: 'Action scheduled', target: action.target };
      case 'webhook':
        return { message: 'Webhook triggered', target: action.target };
      default:
        throw new Error(`Unknown action type: ${action.type}`);
    }
  }
}


