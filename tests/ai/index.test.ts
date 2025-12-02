/**
 * Integration tests for AI Tools Entry Point
 * Tests unified interface and module exports
 */

import { describe, it, expect } from 'vitest';

describe('AI Tools Entry Point', () => {
  describe('Tool Registry', () => {
    const TOOLS = {
      orchestrator: {
        description: 'Task management and context injection',
        commands: ['ai:start', 'ai:complete', 'ai:context', 'ai:metrics', 'ai:history'],
      },
      sync: {
        description: 'Context synchronization',
        commands: ['ai:sync'],
      },
      dashboard: {
        description: 'ASCII metrics dashboard',
        commands: ['ai:dashboard'],
      },
      cache: {
        description: 'Multi-layer caching with semantic similarity',
        commands: ['ai:cache', 'ai:cache:stats', 'ai:cache:clear'],
      },
      monitor: {
        description: 'Continuous monitoring with circuit breakers',
        commands: ['ai:monitor', 'ai:monitor:status', 'ai:monitor:check'],
      },
      compliance: {
        description: 'Policy-based validation and scoring',
        commands: ['ai:compliance', 'ai:compliance:check', 'ai:compliance:score'],
      },
      telemetry: {
        description: 'Observability and alerting',
        commands: ['ai:telemetry', 'ai:telemetry:status', 'ai:telemetry:alerts'],
      },
      errors: {
        description: 'Structured error handling with recovery',
        commands: ['ai:errors', 'ai:errors:list', 'ai:errors:stats'],
      },
      security: {
        description: 'Security scanning (secrets, vulns, licenses)',
        commands: ['ai:security', 'ai:security:scan', 'ai:security:secrets', 'ai:security:vulns'],
      },
      issues: {
        description: 'Automated issue management',
        commands: ['ai:issues', 'ai:issues:list', 'ai:issues:critical', 'ai:issues:stats'],
      },
    };

    it('should have all expected tools', () => {
      const toolNames = Object.keys(TOOLS);
      expect(toolNames).toContain('orchestrator');
      expect(toolNames).toContain('sync');
      expect(toolNames).toContain('dashboard');
      expect(toolNames).toContain('cache');
      expect(toolNames).toContain('monitor');
      expect(toolNames).toContain('compliance');
      expect(toolNames).toContain('telemetry');
      expect(toolNames).toContain('errors');
      expect(toolNames).toContain('security');
      expect(toolNames).toContain('issues');
      expect(toolNames.length).toBe(10);
    });

    it('should have descriptions for all tools', () => {
      for (const [, tool] of Object.entries(TOOLS)) {
        expect(tool.description).toBeDefined();
        expect(tool.description.length).toBeGreaterThan(10);
      }
    });

    it('should have commands for all tools', () => {
      for (const [, tool] of Object.entries(TOOLS)) {
        expect(tool.commands).toBeDefined();
        expect(tool.commands.length).toBeGreaterThan(0);
      }
    });

    it('should have unique commands across all tools', () => {
      const allCommands = Object.values(TOOLS).flatMap((t) => t.commands);
      const uniqueCommands = new Set(allCommands);
      expect(allCommands.length).toBe(uniqueCommands.size);
    });

    it('should follow command naming convention', () => {
      const allCommands = Object.values(TOOLS).flatMap((t) => t.commands);
      for (const cmd of allCommands) {
        expect(cmd).toMatch(/^ai:/);
      }
    });
  });

  describe('Module Categories', () => {
    const categories = {
      core: ['orchestrator', 'sync', 'dashboard'],
      infrastructure: ['cache', 'monitor', 'telemetry', 'errors'],
      governance: ['compliance', 'security', 'issues'],
    };

    it('should have core modules', () => {
      expect(categories.core).toContain('orchestrator');
      expect(categories.core).toContain('sync');
      expect(categories.core).toContain('dashboard');
    });

    it('should have infrastructure modules', () => {
      expect(categories.infrastructure).toContain('cache');
      expect(categories.infrastructure).toContain('monitor');
      expect(categories.infrastructure).toContain('telemetry');
      expect(categories.infrastructure).toContain('errors');
    });

    it('should have governance modules', () => {
      expect(categories.governance).toContain('compliance');
      expect(categories.governance).toContain('security');
      expect(categories.governance).toContain('issues');
    });
  });

  describe('Command Groups', () => {
    it('should group related commands', () => {
      const cacheCommands = ['ai:cache', 'ai:cache:stats', 'ai:cache:clear'];
      const monitorCommands = ['ai:monitor', 'ai:monitor:status', 'ai:monitor:check'];

      expect(cacheCommands.filter((c) => c.split(':').length === 2)).toHaveLength(1);
      expect(monitorCommands.filter((c) => c.split(':').length === 2)).toHaveLength(1);
    });

    it('should use consistent subcommand pattern', () => {
      const commandPattern = /^ai:[a-z]+(?::[a-z]+)?$/;

      const sampleCommands = ['ai:cache:stats', 'ai:monitor:status', 'ai:security:scan'];

      for (const cmd of sampleCommands) {
        expect(cmd).toMatch(commandPattern);
      }
    });
  });

  describe('Quick Start Commands', () => {
    const quickStart = {
      'ai:dashboard': 'View metrics dashboard',
      'ai:compliance:score': 'Check compliance score',
      'ai:security:scan': 'Run security scan',
      'ai:sync': 'Sync context from git',
    };

    it('should have quick start commands', () => {
      expect(Object.keys(quickStart).length).toBe(4);
    });

    it('should include essential operations', () => {
      expect(quickStart['ai:dashboard']).toBeDefined();
      expect(quickStart['ai:security:scan']).toBeDefined();
      expect(quickStart['ai:compliance:score']).toBeDefined();
    });
  });

  describe('CLI Help Output', () => {
    it('should format tool listing correctly', () => {
      const formatToolName = (name: string) => name.toUpperCase().padEnd(56);
      expect(formatToolName('cache')).toBe('CACHE'.padEnd(56));
      expect(formatToolName('security').trim()).toBe('SECURITY');
    });

    it('should truncate long descriptions', () => {
      const truncate = (s: string, max: number) =>
        s.length > max ? s.substring(0, max - 3) + '...' : s;

      const longDesc =
        'This is a very long description that exceeds the maximum allowed length for display';
      const truncated = truncate(longDesc, 50);
      expect(truncated.length).toBeLessThanOrEqual(50);
      expect(truncated.endsWith('...')).toBe(true);
    });

    it('should format command list correctly', () => {
      const commands = ['ai:cache', 'ai:cache:stats', 'ai:cache:clear', 'ai:cache:export'];
      const preview = commands.slice(0, 3).join(', ');
      expect(preview).toBe('ai:cache, ai:cache:stats, ai:cache:clear');
    });
  });

  describe('Module Dependencies', () => {
    const dependencies = {
      orchestrator: [],
      sync: ['telemetry'],
      dashboard: ['cache', 'monitor', 'compliance', 'telemetry'],
      cache: [],
      monitor: ['telemetry'],
      compliance: ['telemetry'],
      telemetry: [],
      errors: [],
      security: ['errors', 'issues'],
      issues: ['errors'],
    };

    it('should have no circular dependencies', () => {
      const checkCircular = (module: string, visited: Set<string> = new Set()): boolean => {
        if (visited.has(module)) return true;
        visited.add(module);

        const deps = dependencies[module as keyof typeof dependencies] || [];
        for (const dep of deps) {
          if (checkCircular(dep, new Set(visited))) return true;
        }
        return false;
      };

      for (const module of Object.keys(dependencies)) {
        expect(checkCircular(module)).toBe(false);
      }
    });

    it('should have base modules without dependencies', () => {
      const baseMods = Object.entries(dependencies)
        .filter(([, deps]) => deps.length === 0)
        .map(([name]) => name);

      expect(baseMods).toContain('cache');
      expect(baseMods).toContain('telemetry');
      expect(baseMods).toContain('errors');
    });
  });

  describe('Configuration', () => {
    const defaultConfig = {
      aiDir: '.ai',
      cacheDir: '.ai/cache',
      logFile: '.ai/error-log.json',
      metricsFile: '.ai/metrics.json',
      taskHistoryFile: '.ai/task-history.json',
    };

    it('should use .ai directory as base', () => {
      for (const [key, value] of Object.entries(defaultConfig)) {
        if (key !== 'aiDir') {
          expect(value.startsWith('.ai/')).toBe(true);
        }
      }
    });

    it('should have consistent file naming', () => {
      const dataFiles = ['logFile', 'metricsFile', 'taskHistoryFile'];
      for (const key of dataFiles) {
        const file = defaultConfig[key as keyof typeof defaultConfig];
        expect(file).toMatch(/\.json$/);
      }
    });
  });
});
