import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

/**
 * Get the automation folder path
 */
export function getAutomationPath(): string {
  // Check environment variable first
  if (process.env.AUTOMATION_PATH) {
    return process.env.AUTOMATION_PATH;
  }

  // Default to sibling automation folder
  const defaultPath = path.join(__dirname, '..', '..', '..', 'automation');
  if (fs.existsSync(defaultPath)) {
    return defaultPath;
  }

  // Fallback to current directory
  return process.cwd();
}

/**
 * Read and parse a YAML file
 */
export function readYamlFile<T>(filePath: string): T | null {
  try {
    if (!fs.existsSync(filePath)) {
      return null;
    }
    const content = fs.readFileSync(filePath, 'utf-8');
    return yaml.load(content) as T;
  } catch (error) {
    console.error(`Error reading YAML file ${filePath}:`, error);
    return null;
  }
}

/**
 * Write data to a YAML file
 */
export function writeYamlFile(filePath: string, data: unknown): boolean {
  try {
    const content = yaml.dump(data, { indent: 2, lineWidth: 120 });
    fs.writeFileSync(filePath, content, 'utf-8');
    return true;
  } catch (error) {
    console.error(`Error writing YAML file ${filePath}:`, error);
    return false;
  }
}

/**
 * Read a markdown file
 */
export function readMarkdownFile(filePath: string): string | null {
  try {
    if (!fs.existsSync(filePath)) {
      return null;
    }
    return fs.readFileSync(filePath, 'utf-8');
  } catch (error) {
    console.error(`Error reading markdown file ${filePath}:`, error);
    return null;
  }
}

/**
 * List files in a directory with optional extension filter
 */
export function listFiles(dirPath: string, extension?: string): string[] {
  try {
    if (!fs.existsSync(dirPath)) {
      return [];
    }

    let files = fs.readdirSync(dirPath);

    if (extension) {
      files = files.filter(f => f.endsWith(extension));
    }

    return files.map(f => path.join(dirPath, f));
  } catch (error) {
    console.error(`Error listing files in ${dirPath}:`, error);
    return [];
  }
}

/**
 * Get file stats
 */
export function getFileStats(filePath: string): fs.Stats | null {
  try {
    return fs.statSync(filePath);
  } catch {
    return null;
  }
}

/**
 * Recursively list all files in a directory
 */
export function listFilesRecursive(dirPath: string, extension?: string): string[] {
  const results: string[] = [];

  function walk(dir: string) {
    try {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          walk(fullPath);
        } else if (entry.isFile()) {
          if (!extension || entry.name.endsWith(extension)) {
            results.push(fullPath);
          }
        }
      }
    } catch (error) {
      console.error(`Error walking directory ${dir}:`, error);
    }
  }

  walk(dirPath);
  return results;
}

/**
 * Ensure a directory exists
 */
export function ensureDir(dirPath: string): boolean {
  try {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
    return true;
  } catch (error) {
    console.error(`Error creating directory ${dirPath}:`, error);
    return false;
  }
}
