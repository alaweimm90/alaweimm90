import { Command } from 'commander';
import { teamRegistry } from '../../agents/teams.js';

export function registerTeamCommands(program: Command): void {
  const cmd = program.command('team').description('Agent team management');

  cmd.command('list').action(() => {
    const teams = teamRegistry.list();
    for (const t of teams) {
      const line = `${t.id} ${t.name} members=${t.members.length} strategy=${t.routingStrategy}`;

      console.log(line);
    }
  });

  cmd
    .command('register')
    .requiredOption('--id <id>', 'Team id')
    .requiredOption('--name <name>', 'Team name')
    .requiredOption('--members <list>', 'Comma separated agent ids')
    .option('--strategy <s>', 'Routing strategy', 'capability')
    .action(
      (opts: {
        id: string;
        name: string;
        members: string;
        strategy: 'capability' | 'load_balance' | 'cost' | 'latency';
      }) => {
        const m = opts.members
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean);
        const team = {
          id: opts.id,
          name: opts.name,
          members: m,
          capabilities: [],
          routingStrategy: opts.strategy,
        };
        const saved = teamRegistry.register(team);

        console.log(`${saved.id} registered`);
      }
    );
}
