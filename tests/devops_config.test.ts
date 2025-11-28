import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import path from 'node:path';
import {
  resolveTargetDir,
  parsePlaceholders,
  parseFlag,
  parseOption,
} from '../tools/devops/config.js';

describe('DevOps CLI - Configuration', () => {
  const originalEnv = process.env.DEVOPS_TARGET_DIR;

  beforeEach(() => {
    delete process.env.DEVOPS_TARGET_DIR;
  });

  afterEach(() => {
    if (originalEnv) {
      process.env.DEVOPS_TARGET_DIR = originalEnv;
    } else {
      delete process.env.DEVOPS_TARGET_DIR;
    }
  });

  describe('resolveTargetDir', () => {
    it('should use --target argument when provided', () => {
      const result = resolveTargetDir(['--target=/custom/path']);
      expect(result).toBe('/custom/path');
    });

    it('should use DEVOPS_TARGET_DIR env when set', () => {
      process.env.DEVOPS_TARGET_DIR = '/env/path';
      const result = resolveTargetDir([]);
      expect(result).toBe('/env/path');
    });

    it('should default to .metaHub in cwd', () => {
      const result = resolveTargetDir([]);
      expect(result).toBe(path.join(process.cwd(), '.metaHub'));
    });

    it('should prefer argument over env', () => {
      process.env.DEVOPS_TARGET_DIR = '/env/path';
      const result = resolveTargetDir(['--target=/arg/path']);
      expect(result).toBe('/arg/path');
    });
  });

  describe('parsePlaceholders', () => {
    it('should parse KEY=VALUE pairs', () => {
      const result = parsePlaceholders(['PROJECT_NAME=myapp', 'REGISTRY=ghcr.io']);
      expect(result).toEqual({
        PROJECT_NAME: 'myapp',
        REGISTRY: 'ghcr.io',
      });
    });

    it('should ignore flags', () => {
      const result = parsePlaceholders(['--target=/path', 'NAME=value']);
      expect(result).toEqual({ NAME: 'value' });
    });

    it('should handle values with equals signs', () => {
      const result = parsePlaceholders(['URL=http://host:8080?key=value']);
      expect(result).toEqual({ URL: 'http://host:8080?key=value' });
    });
  });

  describe('parseFlag', () => {
    it('should parse true flag', () => {
      expect(parseFlag(['--apply=true'], 'apply')).toBe(true);
    });

    it('should parse false flag', () => {
      expect(parseFlag(['--apply=false'], 'apply')).toBe(false);
    });

    it('should return default when not found', () => {
      expect(parseFlag([], 'apply', true)).toBe(true);
      expect(parseFlag([], 'apply', false)).toBe(false);
    });
  });

  describe('parseOption', () => {
    it('should parse option value', () => {
      expect(parseOption(['--search=*'], 'search')).toBe('*');
    });

    it('should return undefined when not found', () => {
      expect(parseOption([], 'search')).toBeUndefined();
    });
  });
});
