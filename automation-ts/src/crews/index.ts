import * as path from 'path';
import { getAutomationPath, readYamlFile, listFilesRecursive } from '../utils/file';
import type { Crew, CrewMember, CrewTask } from '../types';

/**
 * Load all crews from the automation folder
 */
export function loadCrews(): Map<string, Crew> {
  const automationPath = getAutomationPath();
  const crewsPath = path.join(automationPath, 'orchestration', 'crews');

  const crews = new Map<string, Crew>();
  const files = listFilesRecursive(crewsPath, '.yaml');

  for (const file of files) {
    const crew = readYamlFile<Crew>(file);
    if (crew?.name) {
      crews.set(crew.name, crew);
    }
  }

  return crews;
}

/**
 * Load a specific crew by name
 */
export function loadCrew(name: string): Crew | null {
  const crews = loadCrews();
  return crews.get(name) || null;
}

/**
 * List all available crews
 */
export function listCrews(): string[] {
  const crews = loadCrews();
  return Array.from(crews.keys());
}

/**
 * Get crew members with their roles
 */
export function getCrewMembers(crewName: string): CrewMember[] {
  const crew = loadCrew(crewName);
  return crew?.agents || [];
}

/**
 * Get crew tasks in execution order
 */
export function getCrewTasks(crewName: string): CrewTask[] {
  const crew = loadCrew(crewName);
  if (!crew?.tasks) return [];

  // Sort by priority
  return [...crew.tasks].sort((a, b) => a.priority - b.priority);
}

/**
 * Validate crew configuration
 */
export function validateCrew(crew: Crew): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!crew.name) {
    errors.push('Crew missing name');
  }

  if (!crew.agents || crew.agents.length === 0) {
    errors.push('Crew has no agents');
  }

  if (!crew.tasks || crew.tasks.length === 0) {
    errors.push('Crew has no tasks');
  }

  // Check task assignments
  if (crew.agents && crew.tasks) {
    const agentNames = crew.agents.map(a => a.name);
    for (const task of crew.tasks) {
      if (!agentNames.includes(task.assigned_to)) {
        errors.push(`Task "${task.name}" assigned to unknown agent: ${task.assigned_to}`);
      }
    }
  }

  // Check task dependencies
  if (crew.tasks) {
    const taskNames = crew.tasks.map(t => t.name);
    for (const task of crew.tasks) {
      if (task.depends_on) {
        for (const dep of task.depends_on) {
          if (!taskNames.includes(dep)) {
            errors.push(`Task "${task.name}" depends on unknown task: ${dep}`);
          }
        }
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Execute a crew workflow
 */
export async function executeCrew(
  crewName: string,
  _inputs: Record<string, unknown> = {}
): Promise<{
  success: boolean;
  results: Map<string, unknown>;
  duration_ms: number;
  error?: string;
}> {
  const startTime = Date.now();
  const results = new Map<string, unknown>();

  try {
    const crew = loadCrew(crewName);
    if (!crew) {
      return {
        success: false,
        results,
        duration_ms: Date.now() - startTime,
        error: `Crew not found: ${crewName}`
      };
    }

    // Validate crew
    const validation = validateCrew(crew);
    if (!validation.valid) {
      return {
        success: false,
        results,
        duration_ms: Date.now() - startTime,
        error: `Invalid crew: ${validation.errors.join(', ')}`
      };
    }

    // Get tasks in order
    const tasks = getCrewTasks(crewName);

    // Execute tasks based on process type
    if (crew.process.type === 'hierarchical') {
      await executeHierarchical(crew, tasks, results);
    } else {
      await executeSequential(crew, tasks, results);
    }

    return {
      success: true,
      results,
      duration_ms: Date.now() - startTime
    };
  } catch (error) {
    return {
      success: false,
      results,
      duration_ms: Date.now() - startTime,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

/**
 * Execute tasks hierarchically (with manager oversight)
 */
async function executeHierarchical(
  crew: Crew,
  tasks: CrewTask[],
  results: Map<string, unknown>
): Promise<void> {
  const manager = crew.process.manager;
  console.log(`Manager (${manager}) coordinating crew...`);

  for (const phase of crew.process.workflow) {
    console.log(`Phase: ${phase.phase}`);

    const phaseTasks = tasks.filter(t => phase.tasks.includes(t.name));

    if (phase.parallel) {
      // Execute phase tasks in parallel
      await Promise.all(phaseTasks.map(task => executeTask(task, results)));
    } else {
      // Execute phase tasks sequentially
      for (const task of phaseTasks) {
        await executeTask(task, results);
      }
    }
  }
}

/**
 * Execute tasks sequentially
 */
async function executeSequential(
  crew: Crew,
  tasks: CrewTask[],
  results: Map<string, unknown>
): Promise<void> {
  for (const task of tasks) {
    await executeTask(task, results);
  }
}

/**
 * Execute a single task
 */
async function executeTask(
  task: CrewTask,
  results: Map<string, unknown>
): Promise<void> {
  console.log(`Executing task: ${task.name} (assigned to: ${task.assigned_to})`);

  // Simulate task execution
  const result = {
    task: task.name,
    agent: task.assigned_to,
    output: task.expected_output,
    timestamp: new Date().toISOString()
  };

  results.set(task.name, result);
}
