import { loadWorkflow, planWorkflow } from '../orchestration/workflows.js';
import chalk from 'chalk';

function getArg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0) return process.argv[i + 1];
  return undefined;
}

const NO_COLOR = Boolean(process.env.NO_COLOR);
const cyan = (s: string) => (NO_COLOR ? s : chalk.cyan(s));
const gray = (s: string) => (NO_COLOR ? s : chalk.gray(s));
const green = (s: string) => (NO_COLOR ? s : chalk.green(s));
const yellow = (s: string) => (NO_COLOR ? s : chalk.yellow(s));

const file = getArg('file');
if (!file) {
  // eslint-disable-next-line no-console
  console.error('Usage: tsx tools/atlas/cli/workflow-plan.ts --file <path> [--style compact|json]');
  process.exit(1);
}

const style = getArg('style') || process.env.ATLAS_OUTPUT_STYLE || 'compact';
const def = loadWorkflow(file);
const plan = planWorkflow(def);
if (style === 'json') {
  // eslint-disable-next-line no-console
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
  process.exit(0);
}
for (const p of plan.steps) {
  // eslint-disable-next-line no-console
  console.log(
    `${cyan(p.step.id)} ${gray(p.step.task)} → ${p.agentId ? green(p.agentId) : yellow('unassigned')}`
  );
}
