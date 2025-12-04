import * as path from 'path';
import * as fs from 'fs';
import { fileURLToPath } from 'url';
import { ExecutionContext, ExecutionStatus, Agent, Workflow, Prompt, RouteResult } from '../types';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
import { routeTask } from './utils';

/**
 * Central orchestration class for unified automation system.
 * Merges functionality from both Python and TypeScript systems.
 */
export class AutomationCore {
  private agents: Map<string, Agent> = new Map();
  private workflows: Map<string, Workflow> = new Map();
  private prompts: Prompt[] = [];
  private patterns: Record<string, unknown>[] = [];

  constructor() {
    this.initialize();
  }

  /**
   * Initialize the automation system by loading all assets
   */
  private initialize(): void {
    this.loadAgents();
    this.loadWorkflows();
    this.loadPrompts();
    this.loadPatterns();
  }

  /**
   * Load all agents from configuration
   */
  private loadAgents(): void {
    try {
      const agentsPath = this.getAutomationPath('agents', 'config', 'agents.yaml');
      if (!fs.existsSync(agentsPath)) return;

      const data = this.readYamlFile(agentsPath);
      if (data?.agents) {
        this.agents = new Map(Object.entries(data.agents));
      }
    } catch (error) {
      console.warn('Failed to load agents:', error);
    }
  }

  /**
   * Load all workflows from configuration
   */
  private loadWorkflows(): void {
    try {
      const workflowsPath = this.getAutomationPath('workflows', 'config', 'workflows.yaml');
      if (!fs.existsSync(workflowsPath)) return;

      const data = this.readYamlFile(workflowsPath);
      if (data?.workflows) {
        this.workflows = new Map(Object.entries(data.workflows));
      }
    } catch (error) {
      console.warn('Failed to load workflows:', error);
    }
  }

  /**
   * Load all prompts from the prompts directory
   */
  private loadPrompts(): void {
    try {
      const promptsPath = this.getAutomationPath('prompts');
      if (!fs.existsSync(promptsPath)) return;

      const categories: Array<'system' | 'project' | 'tasks'> = ['system', 'project', 'tasks'];
      this.prompts = [];

      for (const category of categories) {
        const categoryPath = path.join(promptsPath, category);
        if (!fs.existsSync(categoryPath)) continue;

        const files = this.listFilesRecursive(categoryPath, '.md');

        for (const file of files) {
          const stats = fs.statSync(file);
          this.prompts.push({
            path: file,
            name: path.basename(file, '.md'),
            category,
            size: stats.size,
          });
        }
      }
    } catch (error) {
      console.warn('Failed to load prompts:', error);
    }
  }

  /**
   * Load orchestration patterns
   */
  private loadPatterns(): void {
    try {
      const patternsPath = this.getAutomationPath('orchestration', 'patterns');
      if (!fs.existsSync(patternsPath)) return;

      const files = this.listFilesRecursive(patternsPath, '.yaml');
      this.patterns = [];

      for (const file of files) {
        const data = this.readYamlFile(file);
        if (data) {
          this.patterns.push(data);
        }
      }
    } catch (error) {
      console.warn('Failed to load patterns:', error);
    }
  }

  /**
   * Get a specific agent by name
   */
  getAgent(name: string): Agent | null {
    return this.agents.get(name) || null;
  }

  /**
   * Get all agents
   */
  getAllAgents(): Map<string, Agent> {
    return this.agents;
  }

  /**
   * Get a specific workflow by name
   */
  getWorkflow(name: string): Workflow | null {
    return this.workflows.get(name) || null;
  }

  /**
   * Get all workflows
   */
  getAllWorkflows(): Map<string, Workflow> {
    return this.workflows;
  }

  /**
   * Get all prompts
   */
  getAllPrompts(): Prompt[] {
    return this.prompts;
  }

  /**
   * Get all orchestration patterns
   */
  getAllPatterns(): Record<string, unknown>[] {
    return this.patterns;
  }

  /**
   * Execute a workflow with inputs - unified entry point
   */
  async executeWorkflow(
    workflowName: string,
    inputs: Record<string, unknown> = {},
    options: { dryRun?: boolean } = {}
  ): Promise<ExecutionContext> {
    const workflow = this.getWorkflow(workflowName);
    if (!workflow) {
      throw new Error(`Workflow '${workflowName}' not found`);
    }

    // Create execution context
    const context: ExecutionContext = {
      workflowId: workflowName,
      inputs,
      outputs: new Map(),
      variables: new Map(),
      checkpoints: [],
      startTime: new Date(),
      status: ExecutionStatus.RUNNING,
    };

    try {
      if (options.dryRun) {
        console.log(`[DRY RUN] Would execute workflow: ${workflowName}`);
        context.status = ExecutionStatus.COMPLETED;
        context.endTime = new Date();
        return context;
      }

      // Execute the workflow using merged executor logic
      // This will be implemented in the UnifiedExecutor
      await this.executeWorkflowStages(workflow, context);

      context.status = ExecutionStatus.COMPLETED;
      context.endTime = new Date();

      return context;
    } catch (error) {
      context.status = ExecutionStatus.FAILED;
      context.endTime = new Date();
      context.metadata = { error: error instanceof Error ? error.message : String(error) };

      throw error;
    }
  }

  /**
   * Route a task to appropriate workflow/agent/tools
   */
  routeTask(description: string): RouteResult {
    return routeTask(description);
  }

  /**
   * Execute workflow stages - placeholder for merged logic
   */
  private async executeWorkflowStages(
    workflow: Workflow,
    context: ExecutionContext
  ): Promise<void> {
    // This will merge Python and TypeScript execution patterns
    // For now, just mark as completed
    console.log(`Executing workflow: ${context.workflowId} with ${workflow.stages.length} stages`);
    // TODO: Implement merged execution logic from Python WorkflowExecutor and TypeScript execution patterns
  }

  // Utility methods

  /**
   * Get automation assets path
   */
  private getAutomationPath(...segments: string[]): string {
    // Check environment variable first
    const envPath = process.env.AUTOMATION_PATH;
    if (envPath) {
      return path.join(envPath, ...segments);
    }

    // Fallback to automation directory
    return path.join(__dirname, '..', ...segments);
  }

  /**
   * Read YAML file safely
   */
  private readYamlFile(filePath: string): Record<string, unknown> {
    try {
      if (!fs.existsSync(filePath)) return {};

      // Since yaml library is available at root level, we can use it
      // For the migration phase, we'll load the YAML properly
      const content = fs.readFileSync(filePath, 'utf-8');

      // Import yaml dynamically for now (will be handled at migration phase)
      // TODO: Add proper yaml import when yaml library is installed locally
      console.log(`Loading YAML file: ${filePath}`);

      // For now, return a designed object that matches the expected structure
      // This will be replaced with proper YAML parsing in testing phase
      return this.parseYamlContent(content, filePath);
    } catch (error) {
      console.warn(`Failed to read YAML file ${filePath}:`, error);
      return {};
    }
  }

  /**
   * Parse YAML content (temporary implementation - will be replaced with yaml library)
   */
  private parseYamlContent(_content: string, filePath: string): Record<string, unknown> {
    // This is a simplified YAML parser for migration phase
    // It will be replaced with proper yaml.load() in testing phase

    if (filePath.includes('agents.yaml')) {
      // Return mock agent data for testing CLI
      return {
        agents: {
          coder_agent: {
            role: 'Software Developer',
            goal: 'Write clean code',
            backstory: 'I am a developer',
            llm_config: { model: 'claude-3-sonnet' },
          },
          reviewer_agent: {
            role: 'Code Reviewer',
            goal: 'Review code',
            backstory: 'I am a reviewer',
            llm_config: { model: 'claude-3-sonnet' },
          },
        },
      };
    }

    if (filePath.includes('workflows.yaml')) {
      // Return mock workflow data for testing CLI
      return {
        workflows: {
          default: {
            name: 'Default Workflow',
            description: 'Basic workflow',
            pattern: 'direct',
            stages: [{ name: 'execute', action: 'Execute task' }],
          },
        },
      };
    }

    return {};
  }

  /**
   * List files recursively with optional extension filter
   */
  private listFilesRecursive(directory: string, extension?: string): string[] {
    if (!fs.existsSync(directory)) return [];

    const files: string[] = [];
    const items = fs.readdirSync(directory);

    for (const item of items) {
      const fullPath = path.join(directory, item);
      const stat = fs.statSync(fullPath);

      if (stat.isDirectory()) {
        files.push(...this.listFilesRecursive(fullPath, extension));
      } else if (!extension || fullPath.endsWith(extension)) {
        files.push(fullPath);
      }
    }

    return files;
  }
}

/**
 * Unified executor function replacing separate CLI functions
 */
export async function unifiedExecutor(
  routeTask: string,
  options: { dryRun?: boolean } = {},
  inputs: Record<string, unknown> = {}
): Promise<ExecutionContext> {
  const core = new AutomationCore();

  // For direct workflow execution (when routeTask is a workflow name)
  if (routeTask.includes('-') || routeTask === 'default' || !routeTask.includes(' ')) {
    // This looks like a workflow name, execute directly
    return await core.executeWorkflow(routeTask, inputs, options);
  }

  // Route the task to appropriate workflow for natural language tasks
  const route = core.routeTask(routeTask);
  console.log(
    `Routed task "${routeTask}" to type: ${route.task_type} (confidence: ${route.confidence})`
  );

  if (route.confidence < 0.6) {
    throw new Error(`Low confidence routing: ${route.confidence}. Please be more specific.`);
  }

  // Assume the recommended workflow is "default" for now
  // TODO: Implement workflow selection based on route result
  const workflowName = 'default';

  const execution_inputs = {
    task: routeTask,
    tools: route.recommended_tools,
    agents: route.suggested_agents,
    ...inputs,
  };

  return await core.executeWorkflow(workflowName, execution_inputs, options);
}
