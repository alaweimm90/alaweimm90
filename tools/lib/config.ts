/**
 * Unified Configuration Management
 * Consolidated from tools/devops/config.ts and tools/lib/config.ts
 */
import * as path from 'node:path';
import * as fs from 'node:fs';

/**
 * Target resolution: arg → env (DEVOPS_TARGET_DIR) → local .metaHub
 */
export function resolveTargetDir(args: string[]): string {
  // Check for --target argument first (both formats)
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--target' && args[i + 1]) {
      return path.resolve(args[i + 1]);
    }
    if (args[i].startsWith('--target=')) {
      return path.resolve(args[i].split('=')[1]);
    }
  }

  // Check environment variable
  if (process.env.DEVOPS_TARGET_DIR) {
    return path.resolve(process.env.DEVOPS_TARGET_DIR);
  }

  // Default to .metaHub in current directory
  return path.resolve(process.cwd(), '.metaHub');
}

/**
 * Suggested external folder for generated content
 */
export const SUGGESTED_EXTERNAL_FOLDER = 'C:\\Users\\mesha\\Desktop\\GitHub\\.metaHub\\tools';

/**
 * Parse placeholder arguments (KEY=VALUE format)
 */
export function parsePlaceholders(args: string[]): Record<string, string> {
  const placeholders: Record<string, string> = {};
  for (const arg of args) {
    if (arg.includes('=') && !arg.startsWith('--')) {
      const [key, ...valueParts] = arg.split('=');
      placeholders[key] = valueParts.join('=');
    }
  }
  return placeholders;
}

/**
 * Parse boolean flag from args
 */
export function parseFlag(args: string[], flag: string, defaultValue = false): boolean {
  const arg = args.find((a) => a.startsWith(`--${flag}=`));
  if (arg) {
    return arg.split('=')[1] === 'true';
  }
  return defaultValue;
}

/**
 * Parse string option from args
 */
export function parseOption(args: string[], option: string): string | undefined {
  const arg = args.find((a) => a.startsWith(`--${option}=`));
  if (arg) {
    return arg.split('=')[1];
  }
  return undefined;
}

export interface DevOpsConfigOptions {
  targetDir?: string;
  templates?: Record<string, boolean>;
  placeholders?: Record<string, string>;
}

/**
 * Load configuration from file or return defaults
 */
export function loadConfig(configPath: string): DevOpsConfigOptions {
  const defaults: DevOpsConfigOptions = {
    targetDir: '.metaHub',
    templates: {
      cicd: true,
      k8s: true,
      monitoring: true,
      logging: true,
    },
    placeholders: {
      '{{PROJECT_NAME}}': path.basename(process.cwd()),
      '{{TIMESTAMP}}': new Date().toISOString(),
    },
  };

  if (fs.existsSync(configPath)) {
    try {
      const fileConfig = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      return { ...defaults, ...fileConfig };
    } catch {
      console.warn(`Warning: Could not parse config at ${configPath}`);
    }
  }

  return defaults;
}

/**
 * DevOps configuration class
 */
export class DevOpsConfig {
  targetDir: string;
  templates: Record<string, boolean>;
  placeholders: Record<string, string>;

  constructor(options: DevOpsConfigOptions = {}) {
    this.targetDir = options.targetDir || '.metaHub';
    this.templates = options.templates || {};
    this.placeholders = options.placeholders || {};
  }

  getTargetPath(subpath: string): string {
    return path.join(this.targetDir, subpath);
  }
}
