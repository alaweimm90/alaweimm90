import type { ExecutionContext, ExecutionResult, WorkflowStage } from '../types';
import { loadAgents, loadWorkflow } from '../index';

/**
 * Execute a workflow by name
 */
export async function executeWorkflow(
  workflowName: string,
  inputs: Record<string, unknown> = {}
): Promise<ExecutionResult> {
  const startTime = Date.now();
  const stagesCompleted: string[] = [];
  const outputs = new Map<string, unknown>();

  try {
    // Load workflow
    const workflow = loadWorkflow(workflowName);
    if (!workflow) {
      return {
        success: false,
        outputs,
        duration_ms: Date.now() - startTime,
        stages_completed: stagesCompleted,
        error: `Workflow not found: ${workflowName}`
      };
    }

    // Load agents
    const agents = loadAgents();

    // Create execution context
    const context: ExecutionContext = {
      workflow,
      agents,
      variables: new Map(Object.entries(inputs)),
      checkpoints: []
    };

    // Execute based on pattern
    switch (workflow.pattern) {
      case 'prompt_chaining':
        await executeChaining(context, outputs, stagesCompleted);
        break;
      case 'parallelization':
        await executeParallel(context, outputs, stagesCompleted);
        break;
      case 'routing':
        await executeRouting(context, outputs, stagesCompleted);
        break;
      case 'orchestrator_workers':
        await executeOrchestratorWorkers(context, outputs, stagesCompleted);
        break;
      case 'evaluator_optimizer':
        await executeEvaluatorOptimizer(context, outputs, stagesCompleted);
        break;
      default:
        await executeSequential(context, outputs, stagesCompleted);
    }

    return {
      success: true,
      outputs,
      duration_ms: Date.now() - startTime,
      stages_completed: stagesCompleted
    };
  } catch (error) {
    return {
      success: false,
      outputs,
      duration_ms: Date.now() - startTime,
      stages_completed: stagesCompleted,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Execute stages sequentially (default)
 */
async function executeSequential(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  for (const stage of context.workflow.stages) {
    await executeStage(stage, context, outputs);
    stagesCompleted.push(stage.name);
    context.checkpoints.push(`completed:${stage.name}`);
  }
}

/**
 * Execute prompt chaining pattern
 */
async function executeChaining(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  let previousOutput: unknown = null;

  for (const stage of context.workflow.stages) {
    // Pass previous output as input
    if (previousOutput !== null) {
      context.variables.set('previous_output', previousOutput);
    }

    const result = await executeStage(stage, context, outputs);
    previousOutput = result;
    stagesCompleted.push(stage.name);
  }
}

/**
 * Execute parallel pattern
 */
async function executeParallel(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  // Group stages by dependencies
  const parallelStages = context.workflow.stages.filter(s => s.parallel);
  const sequentialStages = context.workflow.stages.filter(s => !s.parallel);

  // Execute parallel stages concurrently
  if (parallelStages.length > 0) {
    const promises = parallelStages.map(stage => executeStage(stage, context, outputs));
    await Promise.all(promises);
    stagesCompleted.push(...parallelStages.map(s => s.name));
  }

  // Execute sequential stages
  for (const stage of sequentialStages) {
    await executeStage(stage, context, outputs);
    stagesCompleted.push(stage.name);
  }
}

/**
 * Execute routing pattern
 */
async function executeRouting(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  for (const stage of context.workflow.stages) {
    // Check condition if present
    if (stage.condition) {
      const conditionMet = evaluateCondition(stage.condition, context);
      if (!conditionMet) {
        continue;
      }
    }

    await executeStage(stage, context, outputs);
    stagesCompleted.push(stage.name);
  }
}

/**
 * Execute orchestrator-workers pattern
 */
async function executeOrchestratorWorkers(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  const orchestrator = context.workflow.orchestrator;
  const workers = context.workflow.workers || [];

  // Orchestrator plans the work
  console.log(`Orchestrator (${orchestrator}) planning work...`);

  // Workers execute in parallel
  const workerPromises = workers.map(async (worker) => {
    console.log(`Worker (${worker}) executing...`);
    return { worker, result: 'completed' };
  });

  await Promise.all(workerPromises);

  // Execute stages
  for (const stage of context.workflow.stages) {
    await executeStage(stage, context, outputs);
    stagesCompleted.push(stage.name);
  }
}

/**
 * Execute evaluator-optimizer pattern
 */
async function executeEvaluatorOptimizer(
  context: ExecutionContext,
  outputs: Map<string, unknown>,
  stagesCompleted: string[]
): Promise<void> {
  const maxIterations = 3;
  let iteration = 0;
  let quality = 0;

  while (iteration < maxIterations && quality < 0.9) {
    for (const stage of context.workflow.stages) {
      await executeStage(stage, context, outputs);
      stagesCompleted.push(`${stage.name}_iter${iteration}`);
    }

    // Simulate quality evaluation
    quality = 0.7 + (iteration * 0.15);
    iteration++;
  }
}

/**
 * Execute a single stage
 */
async function executeStage(
  stage: WorkflowStage,
  context: ExecutionContext,
  outputs: Map<string, unknown>
): Promise<unknown> {
  console.log(`Executing stage: ${stage.name}`);

  // Get agent if specified
  const agent = stage.agent ? context.agents.get(stage.agent) : null;

  // Gather inputs
  const inputs: Record<string, unknown> = {};
  if (stage.inputs) {
    for (const inputName of stage.inputs) {
      inputs[inputName] = outputs.get(inputName) || context.variables.get(inputName);
    }
  }

  // Simulate execution
  const result = {
    stage: stage.name,
    agent: agent?.role || 'default',
    action: stage.action,
    inputs,
    timestamp: new Date().toISOString()
  };

  // Store outputs
  if (stage.outputs) {
    for (const outputName of stage.outputs) {
      outputs.set(outputName, result);
    }
  }

  return result;
}

/**
 * Evaluate a condition string
 */
function evaluateCondition(condition: string, context: ExecutionContext): boolean {
  // Simple condition evaluation
  const value = context.variables.get(condition);
  return Boolean(value);
}

/**
 * Create a checkpoint for recovery
 */
export function createCheckpoint(context: ExecutionContext): string {
  const checkpoint = {
    timestamp: new Date().toISOString(),
    workflow: context.workflow.name,
    variables: Object.fromEntries(context.variables),
    checkpoints: context.checkpoints
  };

  return JSON.stringify(checkpoint);
}

/**
 * Restore from a checkpoint
 */
export function restoreCheckpoint(checkpointData: string): Partial<ExecutionContext> {
  const data = JSON.parse(checkpointData);
  return {
    variables: new Map(Object.entries(data.variables)),
    checkpoints: data.checkpoints
  };
}
