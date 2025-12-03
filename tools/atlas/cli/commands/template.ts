/**
 * Template commands for ATLAS CLI
 * Stub implementation - to be completed
 */

import { Command } from 'commander';

export function registerTemplateCommands(program: Command): void {
  const cmd = program.command('template').description('Template management commands');

  cmd
    .command('list')
    .description('List available templates')
    .action(() => {
      console.log('Templates feature coming soon...');
    });

  cmd
    .command('apply <name>')
    .description('Apply a template')
    .action((name) => {
      console.log(`Applying template: ${name} (coming soon...)`);
    });
}
