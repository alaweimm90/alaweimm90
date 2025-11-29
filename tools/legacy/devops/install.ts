import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export type InstallType = 'npm' | 'pip';

/**
 * Run npm install in target directory
 */
function runNpmInstall(targetDir: string): boolean {
  const packageJsonPath = path.join(targetDir, 'package.json');
  if (!fs.existsSync(packageJsonPath)) {
    console.log('No package.json found, skipping npm install');
    return true;
  }

  try {
    console.log(`Running npm install in ${targetDir}...`);
    execSync('npm install', { cwd: targetDir, stdio: 'inherit' });
    return true;
  } catch (e) {
    console.error('npm install failed:', (e as Error).message);
    return false;
  }
}

/**
 * Run pip install in target directory
 */
function runPipInstall(targetDir: string): boolean {
  const requirementsPath = path.join(targetDir, 'requirements.txt');
  if (!fs.existsSync(requirementsPath)) {
    console.log('No requirements.txt found, skipping pip install');
    return true;
  }

  try {
    console.log(`Running pip install in ${targetDir}...`);
    execSync(`pip install -r ${requirementsPath}`, { cwd: targetDir, stdio: 'inherit' });
    return true;
  } catch (e) {
    console.error('pip install failed:', (e as Error).message);
    return false;
  }
}

/**
 * Run dependency installation
 */
export function install(targetDir: string, type: InstallType): boolean {
  switch (type) {
    case 'npm':
      return runNpmInstall(targetDir);
    case 'pip':
      return runPipInstall(targetDir);
    default:
      console.error(`Unknown install type: ${type}`);
      return false;
  }
}
