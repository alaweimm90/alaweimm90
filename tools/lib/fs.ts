/**
 * Unified File System Utilities
 * Consolidated from tools/devops/fs.ts and tools/lib/fs.ts
 */
import * as fs from 'node:fs';
import * as path from 'node:path';

export interface TemplateManifest {
  category?: string;
  name?: string;
  version?: string;
  description?: string;
  tags?: string[];
  files?: string[];
  requiredFiles?: string[];
  placeholders?: string[];
  dependencies?: string[];
  [key: string]: unknown;
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * Recursively walk directory and return all file paths
 */
export function walk(dir: string, files: string[] = []): string[] {
  if (!fs.existsSync(dir)) {
    return files;
  }
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath, files);
    } else {
      files.push(fullPath);
    }
  }
  return files;
}

/**
 * Find template manifest files in a directory
 */
export function findManifests(dirPath: string): string[] {
  if (!fs.existsSync(dirPath)) {
    return [];
  }

  // Support multiple manifest patterns
  return fs
    .readdirSync(dirPath)
    .filter(
      (file) =>
        file.endsWith('.manifest.json') ||
        file === 'manifest.json' ||
        file.endsWith('template.json')
    )
    .map((file) => path.join(dirPath, file));
}

/**
 * Safely read and parse JSON file
 */
export function readJson<T>(filePath: string): T | null {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return JSON.parse(content) as T;
  } catch {
    return null;
  }
}

/**
 * Replace {{PLACEHOLDER}} patterns in content
 */
export function replaceVars(content: string, vars: Record<string, string>): string {
  let result = content;
  for (const [key, value] of Object.entries(vars)) {
    result = result.replace(new RegExp(`\\{\\{${key}\\}\\}`, 'g'), value);
  }
  return result;
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

  // Check files array (legacy format)
  if (manifest.files && Array.isArray(manifest.files)) {
    for (const file of manifest.files) {
      const filePath = path.join(dirPath, file);
      if (!fs.existsSync(filePath)) {
        errors.push(`Missing file: ${file}`);
      }
    }
  }

  // Check requiredFiles array (new format)
  if (manifest.requiredFiles && Array.isArray(manifest.requiredFiles)) {
    for (const file of manifest.requiredFiles) {
      const filePath = path.join(dirPath, file);
      if (!fs.existsSync(filePath)) {
        errors.push(`Required file not found: ${file}`);
      } else {
        const stat = fs.statSync(filePath);
        if (stat.size === 0) {
          errors.push(`Required file is empty: ${file}`);
        }
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate content (YAML/JSON basic checks)
 */
export function validateContent(filePath: string): ValidationResult {
  const errors: string[] = [];

  if (!fs.existsSync(filePath)) {
    return { valid: false, errors: ['File not found'] };
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const ext = path.extname(filePath).toLowerCase();

  if (ext === '.json') {
    try {
      JSON.parse(content);
    } catch (e) {
      errors.push(`Invalid JSON: ${(e as Error).message}`);
    }
  } else if (ext === '.yaml' || ext === '.yml') {
    // Basic YAML check: must have key-value or list structure
    const lines = content.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#'));
    if (lines.length === 0) {
      errors.push('YAML file is empty');
    } else {
      const hasStructure = lines.some((l) => l.includes(':') || l.trim().startsWith('-'));
      if (!hasStructure) {
        errors.push('YAML file has no valid structure');
      }
    }
  }

  return { valid: errors.length === 0, errors };
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
 * Copy template files to target with placeholder substitution
 */
export function copyTemplateFiles(
  manifest: TemplateManifest,
  templateDir: string,
  targetDir: string,
  vars: Record<string, string>
): void {
  const files = manifest.requiredFiles || manifest.files || [];
  for (const file of files) {
    const srcPath = path.join(templateDir, file);
    const destPath = path.join(targetDir, file);

    if (!fs.existsSync(srcPath)) {
      console.warn(`Warning: Required file not found: ${srcPath}`);
      continue;
    }

    const destDir = path.dirname(destPath);
    fs.mkdirSync(destDir, { recursive: true });

    const content = fs.readFileSync(srcPath, 'utf-8');
    const replaced = replaceVars(content, vars);
    fs.writeFileSync(destPath, replaced, 'utf-8');
    console.log(`  Copied: ${file}`);
  }
}

/**
 * Write template metadata to target
 */
export function writeTemplateMeta(
  targetDir: string,
  manifest: Pick<TemplateManifest, 'name' | 'version'>
): void {
  const metaPath = path.join(targetDir, '.template-meta.json');
  const meta = {
    name: manifest.name,
    version: manifest.version,
  };
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2), 'utf-8');
}

/**
 * Ensure directory exists
 */
export function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
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
