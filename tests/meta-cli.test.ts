/**
 * Tests for the unified meta-cli
 */

import { describe, it, expect } from 'vitest';
import { spawn } from 'child_process';
import * as path from 'path';

/**
 * Helper function to run CLI commands
 */
function runCLI(args: string[]): Promise<{ stdout: string; stderr: string; code: number }> {
  const CLI_PATH = path.join(process.cwd(), 'tools', 'cli', 'meta-cli.ts');
  return new Promise((resolve) => {
    const child = spawn('npx', ['tsx', CLI_PATH, ...args], {
      cwd: process.cwd(),
      env: process.env,
      shell: true, // Required for Windows compatibility
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      resolve({ stdout, stderr, code: code || 0 });
    });
  });
}

describe('meta-cli', () => {
  describe('Main Command', () => {
    it('should display help when no arguments provided', async () => {
      const result = await runCLI([]);
      expect(result.stdout).toContain('Meta-governance repository CLI');
      expect(result.stdout).toContain('Commands:');
      // Commander.js may return 0 or 1 when showing help without explicit command
      expect([0, 1]).toContain(result.code);
    });

    it('should display version with --version flag', async () => {
      const result = await runCLI(['--version']);
      expect(result.stdout).toContain('2.0.0');
      expect(result.code).toBe(0);
    });

    it('should display help with --help flag', async () => {
      const result = await runCLI(['--help']);
      expect(result.stdout).toContain('Meta-governance repository CLI');
      expect(result.stdout).toContain('ai');
      expect(result.stdout).toContain('ORCHEX');
      expect(result.stdout).toContain('devops');
      expect(result.stdout).toContain('automation');
      expect(result.code).toBe(0);
    });
  });

  describe('AI Command', () => {
    it('should display AI help', async () => {
      const result = await runCLI(['ai', '--help']);
      expect(result.stdout).toContain('AI orchestration');
      expect(result.stdout).toContain('Commands:');
      expect(result.stdout).toContain('cache');
      expect(result.stdout).toContain('monitor');
      expect(result.stdout).toContain('compliance');
      expect(result.code).toBe(0);
    });

    it('should display cache subcommand help', async () => {
      const result = await runCLI(['ai', 'cache', '--help']);
      expect(result.stdout).toContain('AI cache management');
      expect(result.stdout).toContain('stats');
      expect(result.stdout).toContain('clear');
      expect(result.code).toBe(0);
    });
  });

  describe('Dev Command', () => {
    it('should display dev help', async () => {
      const result = await runCLI(['dev', '--help']);
      expect(result.stdout).toContain('Development tools');
      expect(result.stdout).toContain('lint');
      expect(result.stdout).toContain('format');
      expect(result.stdout).toContain('test');
      expect(result.stdout).toContain('type-check');
      expect(result.code).toBe(0);
    });
  });

  describe('ORCHEX Command', () => {
    it('should display ORCHEX help', async () => {
      const result = await runCLI(['ORCHEX', '--help']);
      expect(result.stdout).toContain('ORCHEX research orchestration');
      expect(result.code).toBe(0);
    });
  });

  describe('DevOps Command', () => {
    it('should display devops help', async () => {
      const result = await runCLI(['devops', '--help']);
      expect(result.stdout).toContain('DevOps template');
      expect(result.code).toBe(0);
    });

    it('should display template subcommand help', async () => {
      const result = await runCLI(['devops', 'template', '--help']);
      expect(result.stdout).toContain('Template management');
      expect(result.stdout).toContain('list');
      expect(result.stdout).toContain('apply');
      expect(result.code).toBe(0);
    });
  });

  describe('Automation Command', () => {
    it('should display automation help', async () => {
      const result = await runCLI(['automation', '--help']);
      expect(result.stdout).toContain('Workflow automation');
      expect(result.stdout).toContain('list');
      expect(result.stdout).toContain('execute');
      expect(result.code).toBe(0);
    });

    it('should accept auto alias', async () => {
      const result = await runCLI(['auto', '--help']);
      expect(result.stdout).toContain('Workflow automation');
      expect(result.code).toBe(0);
    });
  });

  describe('Error Handling', () => {
    it('should handle unknown commands gracefully', async () => {
      const result = await runCLI(['unknown-command']);
      expect(result.stderr + result.stdout).toContain('unknown command');
      expect(result.code).toBe(1);
    }, 10000); // Increase timeout for error cases
  });
});

describe('CLI Command Structure', () => {
  it('should have consistent help format', async () => {
    const mainHelp = await runCLI(['--help']);
    const aiHelp = await runCLI(['ai', '--help']);
    const devHelp = await runCLI(['dev', '--help']);

    // All should have usage line
    expect(mainHelp.stdout).toMatch(/Usage:/i);
    expect(aiHelp.stdout).toMatch(/Usage:/i);
    expect(devHelp.stdout).toMatch(/Usage:/i);

    // All should have options section
    expect(mainHelp.stdout).toMatch(/Options:/i);
    expect(aiHelp.stdout).toMatch(/Options:/i);
    expect(devHelp.stdout).toMatch(/Options:/i);

    // All should have commands section (except leaf commands)
    expect(mainHelp.stdout).toMatch(/Commands:/i);
    expect(aiHelp.stdout).toMatch(/Commands:/i);
    expect(devHelp.stdout).toMatch(/Commands:/i);
  });

  it('should have proper command hierarchy', async () => {
    // Test 3-level deep command
    const cacheHelp = await runCLI(['ai', 'cache', '--help']);
    expect(cacheHelp.stdout).toContain('stats');
    expect(cacheHelp.stdout).toContain('clear');
    expect(cacheHelp.code).toBe(0);
  });
});
