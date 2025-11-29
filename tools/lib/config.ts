import path from 'node:path';

/**
 * Target resolution: arg → env (DEVOPS_TARGET_DIR) → local .metaHub
 */
export function resolveTargetDir(args: string[]): string {
  const targetArg = args.find((a) => a.startsWith('--target='));
  if (targetArg) {
    return targetArg.split('=')[1];
  }
  if (process.env.DEVOPS_TARGET_DIR) {
    return process.env.DEVOPS_TARGET_DIR;
  }
  return path.join(process.cwd(), '.metaHub');
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
