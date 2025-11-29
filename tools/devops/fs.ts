// ATLAS DevOps File System Utilities
import * as fs from 'fs';
import * as path from 'path';

export interface TemplateManifest {
  name?: string;
  version?: string;
  files?: string[];
  [key: string]: unknown;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Find template manifest files in a directory
 */
export function findManifests(dirPath: string): string[] {
  if (!fs.existsSync(dirPath)) {
    return [];
  }

  return fs
    .readdirSync(dirPath)
    .filter((file) => file.endsWith('.manifest.json') || file === 'manifest.json')
    .map((file) => path.join(dirPath, file));
}

/**
 * Validate a template against its manifest
 */
export function validateTemplate(manifest: TemplateManifest, dirPath: string): ValidationResult {
  const errors: string[] = [];

  if (!manifest.name) {
    errors.push('Missing required field: name');
  }

  if (!manifest.version) {
    errors.push('Missing required field: version');
  }

  if (manifest.files && Array.isArray(manifest.files)) {
    for (const file of manifest.files) {
      const filePath = path.join(dirPath, file);
      if (!fs.existsSync(filePath)) {
        errors.push(`Missing file: ${file}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Copy directory recursively with placeholder replacement
 */
export function copyDirWithReplacements(
  src: string,
  dest: string,
  placeholders: Record<string, string> = {}
): void {
  if (!fs.existsSync(dest)) {
    fs.mkdirSync(dest, { recursive: true });
  }

  const entries = fs.readdirSync(src, { withFileTypes: true });

  for (const entry of entries) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);

    if (entry.isDirectory()) {
      copyDirWithReplacements(srcPath, destPath, placeholders);
    } else {
      let content = fs.readFileSync(srcPath, 'utf8');

      for (const [placeholder, value] of Object.entries(placeholders)) {
        content = content.replace(new RegExp(placeholder, 'g'), value);
      }

      fs.writeFileSync(destPath, content);
    }
  }
}

/**
 * DevOps file system utilities class
 */
export class DevOpsFS {
  basePath: string;

  constructor(basePath: string = process.cwd()) {
    this.basePath = basePath;
  }

  resolve(...paths: string[]): string {
    return path.resolve(this.basePath, ...paths);
  }

  exists(filePath: string): boolean {
    return fs.existsSync(this.resolve(filePath));
  }

  readJSON<T = unknown>(filePath: string): T {
    const content = fs.readFileSync(this.resolve(filePath), 'utf8');
    return JSON.parse(content) as T;
  }

  writeJSON(filePath: string, data: unknown): void {
    const fullPath = this.resolve(filePath);
    fs.mkdirSync(path.dirname(fullPath), { recursive: true });
    fs.writeFileSync(fullPath, JSON.stringify(data, null, 2));
  }
}
