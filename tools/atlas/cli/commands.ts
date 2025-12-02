// ATLAS CLI Commands - Command definitions and handlers

import { Command } from 'commander';
import { registerAnalyzeCommands } from './commands/analyze.js';
import { registerTemplateCommands } from './commands/template.js';
import { registerDashboardCommands } from './commands/dashboard.js';
import { registerAiCommands } from '../integrations/ai.js';
import { registerWorkflowCommands } from './commands/workflow.js';
import { registerTeamCommands } from './commands/team.js';

/**
 * Register all ATLAS CLI commands
 */
export function registerCommands(program: Command): void {
  // Register existing analyze commands
  registerAnalyzeCommands(program);

  // Register new KILO-integrated commands
  registerTemplateCommands(program);
  registerDashboardCommands(program);

  // Register AI tools integration
  registerAiCommands(program);

  // Agentic workflows and teams
  registerWorkflowCommands(program);
  registerTeamCommands(program);
}

/**
 * Create and configure the main CLI program
 */
export function createCLI(): Command {
  const program = new Command();

  program
    .name('atlas')
    .description(
      'ATLAS CLI - Repository analysis and DevOps tools with KILO governance and AI integration'
    )
    .version('1.0.0');

  // Register all commands
  registerCommands(program);

  return program;
}
