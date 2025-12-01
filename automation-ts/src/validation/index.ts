import * as path from 'path';
import { getAutomationPath, readYamlFile, listFilesRecursive, readMarkdownFile } from '../utils/file';
import type { ValidationResult, ValidationError, ValidationWarning, Agent, Workflow } from '../types';

/**
 * Validate all automation assets
 */
export function validateAll(): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  // Validate agents
  const agentResult = validateAgents();
  errors.push(...agentResult.errors);
  warnings.push(...agentResult.warnings);

  // Validate workflows
  const workflowResult = validateWorkflows();
  errors.push(...workflowResult.errors);
  warnings.push(...workflowResult.warnings);

  // Validate prompts
  const promptResult = validatePrompts();
  errors.push(...promptResult.errors);
  warnings.push(...promptResult.warnings);

  // Cross-reference validation
  const crossRefResult = validateCrossReferences();
  errors.push(...crossRefResult.errors);
  warnings.push(...crossRefResult.warnings);

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

/**
 * Validate agent configurations
 */
export function validateAgents(): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const automationPath = getAutomationPath();
  const agentsPath = path.join(automationPath, 'agents', 'config', 'agents.yaml');

  const data = readYamlFile<{ agents: Record<string, Agent> }>(agentsPath);

  if (!data) {
    errors.push({
      type: 'file_not_found',
      message: 'agents.yaml not found',
      path: agentsPath
    });
    return { valid: false, errors, warnings };
  }

  if (!data.agents) {
    errors.push({
      type: 'invalid_structure',
      message: 'agents.yaml missing "agents" key',
      path: agentsPath
    });
    return { valid: false, errors, warnings };
  }

  for (const [name, agent] of Object.entries(data.agents)) {
    // Required fields
    if (!agent.role) {
      errors.push({
        type: 'missing_field',
        message: `Agent "${name}" missing required field: role`,
        path: `agents.${name}`
      });
    }

    if (!agent.goal) {
      errors.push({
        type: 'missing_field',
        message: `Agent "${name}" missing required field: goal`,
        path: `agents.${name}`
      });
    }

    // Warnings for optional but recommended fields
    if (!agent.backstory) {
      warnings.push({
        type: 'missing_recommended',
        message: `Agent "${name}" missing backstory`,
        path: `agents.${name}`
      });
    }

    if (!agent.tools || agent.tools.length === 0) {
      warnings.push({
        type: 'no_tools',
        message: `Agent "${name}" has no tools assigned`,
        path: `agents.${name}`
      });
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate workflow configurations
 */
export function validateWorkflows(): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const automationPath = getAutomationPath();
  const workflowsPath = path.join(automationPath, 'workflows', 'config', 'workflows.yaml');

  const data = readYamlFile<{ workflows: Record<string, Workflow> }>(workflowsPath);

  if (!data) {
    errors.push({
      type: 'file_not_found',
      message: 'workflows.yaml not found',
      path: workflowsPath
    });
    return { valid: false, errors, warnings };
  }

  if (!data.workflows) {
    errors.push({
      type: 'invalid_structure',
      message: 'workflows.yaml missing "workflows" key',
      path: workflowsPath
    });
    return { valid: false, errors, warnings };
  }

  for (const [name, workflow] of Object.entries(data.workflows)) {
    // Required fields
    if (!workflow.name) {
      errors.push({
        type: 'missing_field',
        message: `Workflow "${name}" missing required field: name`,
        path: `workflows.${name}`
      });
    }

    if (!workflow.pattern) {
      errors.push({
        type: 'missing_field',
        message: `Workflow "${name}" missing required field: pattern`,
        path: `workflows.${name}`
      });
    }

    if (!workflow.stages || workflow.stages.length === 0) {
      errors.push({
        type: 'missing_field',
        message: `Workflow "${name}" has no stages defined`,
        path: `workflows.${name}`
      });
    }

    // Validate stages
    if (workflow.stages) {
      for (let i = 0; i < workflow.stages.length; i++) {
        const stage = workflow.stages[i];
        if (!stage.name) {
          errors.push({
            type: 'missing_field',
            message: `Workflow "${name}" stage ${i} missing name`,
            path: `workflows.${name}.stages[${i}]`
          });
        }
        if (!stage.action) {
          errors.push({
            type: 'missing_field',
            message: `Workflow "${name}" stage "${stage.name || i}" missing action`,
            path: `workflows.${name}.stages[${i}]`
          });
        }
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate prompt files
 */
export function validatePrompts(): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const automationPath = getAutomationPath();
  const promptsPath = path.join(automationPath, 'prompts');

  const categories = ['system', 'project', 'tasks'];

  for (const category of categories) {
    const categoryPath = path.join(promptsPath, category);
    const files = listFilesRecursive(categoryPath, '.md');

    if (files.length === 0) {
      warnings.push({
        type: 'empty_category',
        message: `No prompts found in category: ${category}`,
        path: categoryPath
      });
      continue;
    }

    for (const file of files) {
      const content = readMarkdownFile(file);

      if (!content) {
        errors.push({
          type: 'unreadable_file',
          message: `Cannot read prompt file`,
          path: file
        });
        continue;
      }

      // Check for basic structure
      if (!content.includes('#')) {
        warnings.push({
          type: 'no_heading',
          message: `Prompt file has no markdown headings`,
          path: file
        });
      }

      if (content.length < 100) {
        warnings.push({
          type: 'short_content',
          message: `Prompt file is very short (${content.length} chars)`,
          path: file
        });
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validate cross-references between agents and workflows
 */
export function validateCrossReferences(): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  const automationPath = getAutomationPath();

  // Load agents
  const agentsPath = path.join(automationPath, 'agents', 'config', 'agents.yaml');
  const agentsData = readYamlFile<{ agents: Record<string, Agent> }>(agentsPath);
  const agentNames = agentsData?.agents ? Object.keys(agentsData.agents) : [];

  // Load workflows
  const workflowsPath = path.join(automationPath, 'workflows', 'config', 'workflows.yaml');
  const workflowsData = readYamlFile<{ workflows: Record<string, Workflow> }>(workflowsPath);

  if (workflowsData?.workflows) {
    for (const [name, workflow] of Object.entries(workflowsData.workflows)) {
      // Check orchestrator reference
      if (workflow.orchestrator && !agentNames.includes(workflow.orchestrator)) {
        errors.push({
          type: 'invalid_reference',
          message: `Workflow "${name}" references unknown orchestrator: ${workflow.orchestrator}`,
          path: `workflows.${name}.orchestrator`
        });
      }

      // Check worker references
      if (workflow.workers) {
        for (const worker of workflow.workers) {
          if (!agentNames.includes(worker)) {
            errors.push({
              type: 'invalid_reference',
              message: `Workflow "${name}" references unknown worker: ${worker}`,
              path: `workflows.${name}.workers`
            });
          }
        }
      }

      // Check stage agent references
      if (workflow.stages) {
        for (const stage of workflow.stages) {
          if (stage.agent && !agentNames.includes(stage.agent)) {
            errors.push({
              type: 'invalid_reference',
              message: `Workflow "${name}" stage "${stage.name}" references unknown agent: ${stage.agent}`,
              path: `workflows.${name}.stages.${stage.name}`
            });
          }
        }
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}
