import { loadWorkflow } from '../orchestration/workflows.js';
import { executeWorkflow } from '../orchestration/executor.js';
import chalk from 'chalk';

function getArg(name: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  if (i >= 0) return process.argv[i + 1];
  return undefined;
}

const NO_COLOR = Boolean(process.env.NO_COLOR);
const gray = (s: string) => (NO_COLOR ? s : chalk.gray(s));
const green = (s: string) => (NO_COLOR ? s : chalk.green(s));
const red = (s: string) => (NO_COLOR ? s : chalk.red(s));

const file = getArg('file');
const retriesStr = getArg('retries') || '1';
const style = getArg('style') || process.env.ATLAS_OUTPUT_STYLE || 'compact';
if (!file) {
  console.error(
    'Usage: tsx tools/atlas/cli/workflow-run.ts --file <path> [--retries <n>] [--style compact|json]'
  );
  process.exit(1);
}

const def = loadWorkflow(file);
executeWorkflow(def, { retries: parseInt(retriesStr, 10) }).then((res) => {
  if (style === 'json') {
    console.log(JSON.stringify(res, null, 2));
    return;
  }
  let ok = 0,
    fail = 0;
  for (const r of res) {
    const agent = r.agentId || 'n/a';
    const msg = r.output || r.error || '';
    if (r.success) {
      ok++;

      console.log(`✅ ${green(agent)} ${gray('ok')} ${msg}`);
    } else {
      fail++;

      console.log(`❌ ${red(agent)} ${gray('fail')} ${msg}`);
    }
  }

  console.log(
    `${gray('summary')} ${green(String(ok))}/${res.length} ok, ${red(String(fail))} fail`
  );
});
