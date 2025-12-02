import { describe, it, expect } from 'vitest';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { validateTemplate, validateContent, type TemplateManifest } from '../tools/lib/fs.js';

describe('DevOps CLI - Validation', () => {
  describe('validateTemplate', () => {
    const createTempDir = (): string => {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'devops-test-'));
      return dir;
    };

    it('should validate complete manifest', () => {
      const tempDir = createTempDir();
      fs.writeFileSync(path.join(tempDir, 'file1.txt'), 'content');
      fs.writeFileSync(path.join(tempDir, 'file2.txt'), 'content');

      const manifest: TemplateManifest = {
        category: 'test',
        name: 'test-template',
        version: '1.0.0',
        description: 'Test template',
        tags: ['test'],
        requiredFiles: ['file1.txt', 'file2.txt'],
        placeholders: [],
        dependencies: [],
      };

      const result = validateTemplate(manifest, tempDir);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);

      fs.rmSync(tempDir, { recursive: true });
    });

    it('should detect missing required files', () => {
      const tempDir = createTempDir();

      const manifest: TemplateManifest = {
        category: 'test',
        name: 'test-template',
        version: '1.0.0',
        description: 'Test template',
        tags: ['test'],
        requiredFiles: ['missing.txt'],
        placeholders: [],
        dependencies: [],
      };

      const result = validateTemplate(manifest, tempDir);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Required file not found: missing.txt');

      fs.rmSync(tempDir, { recursive: true });
    });

    it('should detect empty required files', () => {
      const tempDir = createTempDir();
      fs.writeFileSync(path.join(tempDir, 'empty.txt'), '');

      const manifest: TemplateManifest = {
        category: 'test',
        name: 'test-template',
        version: '1.0.0',
        description: 'Test template',
        tags: ['test'],
        requiredFiles: ['empty.txt'],
        placeholders: [],
        dependencies: [],
      };

      const result = validateTemplate(manifest, tempDir);
      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Required file is empty: empty.txt');

      fs.rmSync(tempDir, { recursive: true });
    });
  });

  describe('validateContent', () => {
    const createTempFile = (name: string, content: string): string => {
      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'devops-test-'));
      const filePath = path.join(tempDir, name);
      fs.writeFileSync(filePath, content);
      return filePath;
    };

    it('should validate valid JSON', () => {
      const filePath = createTempFile('test.json', '{"key": "value"}');
      const result = validateContent(filePath);
      expect(result.valid).toBe(true);
      fs.rmSync(path.dirname(filePath), { recursive: true });
    });

    it('should detect invalid JSON', () => {
      const filePath = createTempFile('test.json', '{invalid json}');
      const result = validateContent(filePath);
      expect(result.valid).toBe(false);
      expect(result.errors[0]).toContain('Invalid JSON');
      fs.rmSync(path.dirname(filePath), { recursive: true });
    });

    it('should validate valid YAML', () => {
      const filePath = createTempFile('test.yaml', 'key: value\nlist:\n  - item1');
      const result = validateContent(filePath);
      expect(result.valid).toBe(true);
      fs.rmSync(path.dirname(filePath), { recursive: true });
    });

    it('should detect empty YAML', () => {
      const filePath = createTempFile('test.yaml', '# just comments\n');
      const result = validateContent(filePath);
      expect(result.valid).toBe(false);
      fs.rmSync(path.dirname(filePath), { recursive: true });
    });
  });
});
