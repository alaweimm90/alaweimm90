// ATLAS DevOps Configuration Management
import * as path from 'path';
import * as fs from 'fs';

/**
 * Resolve target directory from CLI arguments and environment
 */
export function resolveTargetDir(args: string[]): string {
  // Check for --target argument first
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
