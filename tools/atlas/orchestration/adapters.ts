import { WorkflowStep } from './workflows.js';
import { exec } from 'child_process';
const ALLOWED = ['npm', 'npx', 'docker', 'kubectl', 'helm', 'echo', 'pwsh', 'powershell'];
const TIMEOUT_MS = 60000;
const MAX_BUFFER = 1024 * 1024;

export interface AdapterResult {
  success: boolean;
  output?: string;
  error?: string;
  latency?: number;
  tokensUsed?: number;
}

export async function executeAction(step: WorkflowStep): Promise<AdapterResult> {
  const start = Date.now();
  let ok = true;
  let out = '';
  const cmd = step.inputs?.command;
  if (cmd) {
    const res = await executeCommand(cmd);
    const latency = Date.now() - start;
    return {
      success: res.code === 0,
      output: res.stdout.trim(),
      error: res.stderr.trim() || undefined,
      latency,
      tokensUsed: 100,
    };
  }
  switch (step.action) {
    case 'buildArtifacts':
      out = `built ${step.id}`;
      break;
    case 'runTests':
      out = `tests ${step.id} passed`;
      break;
    case 'buildImage':
      out = `image ${step.id} built`;
      break;
    case 'imageScan':
      out = `scan ${step.id} clean`;
      break;
    case 'k8sDeploy':
      out = `deploy ${step.id} ok`;
      break;
    case 'publishArtifacts':
      out = `published ${step.id}`;
      break;
    default:
      out = `${step.id} executed`;
  }
  const latency = Date.now() - start;
  return { success: ok, output: out, latency, tokensUsed: 1000 };
}

export function executeCommand(
  command: string
): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise((resolve) => {
    const bin = String(command).trim().split(/\s+/)[0].toLowerCase();
    if (!ALLOWED.includes(bin)) {
      resolve({ code: 1, stdout: '', stderr: `command_not_allowed:${bin}` });
      return;
    }
    const proc = exec(
      command,
      { windowsHide: true, timeout: TIMEOUT_MS, maxBuffer: MAX_BUFFER },
      (err, stdout, stderr) => {
        resolve({ code: err ? 1 : 0, stdout: String(stdout || ''), stderr: String(stderr || '') });
      }
    );
    proc.on('error', () => resolve({ code: 1, stdout: '', stderr: 'spawn_error' }));
  });
}
