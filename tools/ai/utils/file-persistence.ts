/**
 * Shared File Persistence Utility
 *
 * Centralizes JSON file I/O operations used across AI tools.
 * Eliminates duplicate code in: cache.ts, compliance.ts, monitor.ts,
 * telemetry.ts, errors.ts, issues.ts, orchestrator.ts, sync.ts, security.ts
 */

import * as fs from 'fs';
import * as path from 'path';

export interface PersistenceOptions {
  createDir?: boolean;
  pretty?: boolean;
  encoding?: BufferEncoding;
}

const DEFAULT_OPTIONS: PersistenceOptions = {
  createDir: true,
  pretty: true,
  encoding: 'utf8'
};

/**
 * Save data to a JSON file
 */
export function saveJson<T>(filePath: string, data: T, options: PersistenceOptions = {}): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  if (opts.createDir) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  const content = opts.pretty
    ? JSON.stringify(data, null, 2)
    : JSON.stringify(data);

  fs.writeFileSync(filePath, content, { encoding: opts.encoding });
}

/**
 * Load data from a JSON file
 */
export function loadJson<T>(filePath: string, defaultValue?: T): T | null {
  if (!fs.existsSync(filePath)) {
    return defaultValue ?? null;
  }

  try {
    const content = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(content) as T;
  } catch {
    return defaultValue ?? null;
  }
}

/**
 * Append an item to a JSON array file
 */
export function appendToJsonArray<T>(filePath: string, item: T, maxItems?: number): void {
  const existing = loadJson<T[]>(filePath, []) ?? [];
  existing.push(item);

  const toSave = maxItems ? existing.slice(-maxItems) : existing;
  saveJson(filePath, toSave);
}

/**
 * Check if a file exists
 */
export function fileExists(filePath: string): boolean {
  return fs.existsSync(filePath);
}

/**
 * Ensure a directory exists
 */
export function ensureDir(dirPath: string): void {
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }
}

/**
 * Read a text file
 */
export function readText(filePath: string, defaultValue = ''): string {
  if (!fs.existsSync(filePath)) {
    return defaultValue;
  }

  try {
    return fs.readFileSync(filePath, 'utf8');
  } catch {
    return defaultValue;
  }
}

/**
 * Write a text file
 */
export function writeText(filePath: string, content: string, options: PersistenceOptions = {}): void {
  const opts = { ...DEFAULT_OPTIONS, ...options };

  if (opts.createDir) {
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  fs.writeFileSync(filePath, content, { encoding: opts.encoding });
}

/**
 * State manager for services that need persistent state
 */
export class StateManager<T> {
  private state: T;

  constructor(
    private readonly filePath: string,
    private readonly defaultState: T
  ) {
    this.state = this.load();
  }

  private load(): T {
    return loadJson<T>(this.filePath, this.defaultState) ?? this.defaultState;
  }

  get(): T {
    return this.state;
  }

  set(newState: T): void {
    this.state = newState;
    this.save();
  }

  update(updater: (state: T) => T): void {
    this.state = updater(this.state);
    this.save();
  }

  save(): void {
    saveJson(this.filePath, this.state);
  }

  reload(): T {
    this.state = this.load();
    return this.state;
  }
}
