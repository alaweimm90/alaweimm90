import { Command } from 'commander';
import { loadWorkflow, planWorkflow } from '@atlas/orchestration/workflows.js';
import { executeWorkflow } from '@atlas/orchestration/executor.js';
import chalk from 'chalk';
import { output, getDefaultStyle } from '@atlas/cli/utils.js';

const NO_COLOR = Boolean(process.env.NO_COLOR);
const cyan = (s: string) => (NO_COLOR ? s : chalk.cyan(s));
const gray = (s: string) => (NO_COLOR ? s : chalk.gray(s));
const green = (s: string) => (NO_COLOR ? s : chalk.green(s));
const yellow = (s: string) => (NO_COLOR ? s : chalk.yellow(s));

export function registerWorkflowCommands(program: Command): void {
  const cmd = program.command('workflow').description('Workflow planning and execution');

  cmd
    .command('plan')
    .requiredOption('--file <path>', 'Workflow YAML file')
    .option('--team <id>', 'Team id for planning')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action((opts: { file: string }) => {
      const def = loadWorkflow(opts.file);
      const plan = planWorkflow(def, undefined, (opts as any).team);
      const style = (opts as any).style || getDefaultStyle();
      if (style === 'json') {
        console.log(
          JSON.stringify(
            plan.steps.map((s) => ({
              id: s.step.id,
              task: s.step.task,
              agent: s.agentId || 'unassigned',
            })),
            null,
            2
          )
        );
        return;
      }
      for (const p of plan.steps) {
        const id = cyan(p.step.id);
        const task = gray(p.step.task);
        const agent = p.agentId ? green(p.agentId) : yellow('unassigned');
        output.info(`${id} ${task} → ${agent}`);
      }
    });

  cmd
    .command('run')
    .requiredOption('--file <path>', 'Workflow YAML file')
    .option('--retries <n>', 'Retries per step', '1')
    .option('--team <id>', 'Team id for planning')
    .option('--concurrency <n>', 'Parallelism', '4')
    .option('--style <mode>', 'Output style: compact|json|verbose', getDefaultStyle())
    .action(async (opts: { file: string; retries: string }) => {
      const def = loadWorkflow(opts.file);
      const results = await executeWorkflow(def, {
        retries: parseInt(opts.retries, 10),
        teamId: (opts as any).team,
        concurrency: parseInt((opts as any).concurrency, 10),
      });
      const style = (opts as any).style || getDefaultStyle();
      if (style === 'json') {
        console.log(JSON.stringify(results, null, 2));
        return;
      }
      let ok = 0,
        fail = 0;
      for (const r of results) {
        const agent = r.agentId || 'n/a';
        const msg = r.output || r.error || '';
        if (r.success) {
          ok++;
          output.success(`${green(agent)} ${gray('ok')} ${msg}`);
        } else {
          fail++;
          output.error(`${NO_COLOR ? agent : chalk.red(agent)} ${gray('fail')} ${msg}`);
        }
      }
      output.info(
        `${gray('summary')} ${green(String(ok))}/${results.length} ok, ${NO_COLOR ? String(fail) : chalk.red(String(fail))} fail`
      );
    });
}
