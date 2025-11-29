import fs from 'node:fs';
import path from 'node:path';

export interface TemplateManifest {
  category: string;
  name: string;
  version: string;
  description: string;
  tags: string[];
  requiredFiles: string[];
  placeholders: string[];
  dependencies: string[];
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
 * Find all template.json manifests in a directory
 */
export function findManifests(baseDir: string): string[] {
  return walk(baseDir).filter((f) => f.endsWith('template.json'));
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
    result = result.replaceAll(`{{${key}}}`, value);
  }
  return result;
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
  for (const file of manifest.requiredFiles) {
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
 * Validate template manifest
 */
export function validateTemplate(
  manifest: TemplateManifest,
  templateDir: string
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check required fields
  if (!manifest.name) errors.push('Missing name');
  if (!manifest.version) errors.push('Missing version');
  if (!manifest.category) errors.push('Missing category');

  // Check required files exist and are non-empty
  for (const file of manifest.requiredFiles) {
    const filePath = path.join(templateDir, file);
    if (!fs.existsSync(filePath)) {
      errors.push(`Required file not found: ${file}`);
    } else {
      const stat = fs.statSync(filePath);
      if (stat.size === 0) {
        errors.push(`Required file is empty: ${file}`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Validate content (YAML/JSON basic checks)
 */
export function validateContent(filePath: string): { valid: boolean; errors: string[] } {
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
 * Ensure directory exists
 */
export function ensureDir(dir: string): void {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}
