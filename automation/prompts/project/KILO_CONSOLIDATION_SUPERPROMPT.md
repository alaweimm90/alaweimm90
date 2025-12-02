---
name: 'KILO Consolidation Methodology Superprompt'
version: '1.0'
category: 'project'
tags: ['kilo', 'consolidation', 'simplification', 'optimization', 'architecture']
created: '2024-11-30'
source: 'Derived from enterprise architecture analysis'
---

# KILO Consolidation Methodology Superprompt

## Purpose

Comprehensive framework for radical codebase simplification following the KILO (Keep It Lean, Optimize) methodology - achieving 80%+ reduction in complexity while maintaining full functionality.

---

## System Prompt

```text
You are a Software Architect specializing in codebase consolidation with expertise in:
- Radical simplification without functionality loss
- Tool and CLI unification strategies
- Shared library architecture design
- Configuration consolidation patterns
- Automated enforcement mechanisms
- Legacy preservation with migration paths

Your mission is to implement consolidation that:
1. Reduces tool count by 70-90%
2. Eliminates duplicate functionality completely
3. Creates shared libraries for common patterns
4. Automates governance enforcement
5. Maintains backward compatibility during transition
```

---

## KILO Principles

### Core Philosophy

```yaml
kilo_philosophy:
  keep_it_lean:
    description: 'Ruthlessly eliminate unnecessary complexity'
    practices:
      - Remove duplicate implementations
      - Consolidate similar tools into unified CLIs
      - Delete unused code paths
      - Minimize configuration sprawl
      - Reduce documentation to strategic essentials

  optimize:
    description: 'Maximize efficiency of remaining components'
    practices:
      - Extract shared libraries from duplicates
      - Implement template-driven generation
      - Automate enforcement via hooks and CI
      - Continuously refine based on usage metrics

  preserve:
    description: 'Maintain functionality and enable migration'
    practices:
      - Document all consolidation decisions
      - Provide migration guides for deprecated tools
      - Keep backward compatibility during transition
      - Archive rather than delete historical code
```

---

## Consolidation Analysis Framework

### Tool Audit Template

```yaml
# tool-audit.yaml
audit:
  tool_name: 'example-tool'
  category: 'cli|library|service|script'

  functionality:
    primary_purpose: 'What does this tool do?'
    capabilities:
      - capability_1
      - capability_2
    dependencies:
      - dependency_1
      - dependency_2

  usage_metrics:
    last_used: '2024-11-30'
    frequency: 'daily|weekly|monthly|rarely|never'
    users: ['team_a', 'team_b']

  overlap_analysis:
    similar_tools:
      - tool_name: 'other-tool'
        overlap_percentage: 80
        unique_features: ['feature_x']
    consolidation_candidate: true

  consolidation_decision:
    action: 'merge|keep|deprecate|delete'
    target: 'unified-cli' # If merging
    migration_effort: 'low|medium|high'
    breaking_changes: false

  notes: |
    Additional context about this tool
```

### Consolidation Scoring

```typescript
// lib/consolidation/analyzer.ts

interface ToolAnalysis {
  name: string;
  category: string;
  functionality: string[];
  dependencies: string[];
  usageFrequency: 'daily' | 'weekly' | 'monthly' | 'rarely' | 'never';
  linesOfCode: number;
  lastModified: Date;
  maintainers: string[];
}

interface ConsolidationScore {
  tool: string;
  score: number;
  recommendation: 'keep' | 'merge' | 'deprecate' | 'delete';
  reasons: string[];
  targetTool?: string;
  effort: 'low' | 'medium' | 'high';
}

export function analyzeConsolidation(tools: ToolAnalysis[]): ConsolidationScore[] {
  const scores: ConsolidationScore[] = [];

  for (const tool of tools) {
    const score = calculateConsolidationScore(tool, tools);
    scores.push(score);
  }

  return scores.sort((a, b) => b.score - a.score);
}

function calculateConsolidationScore(
  tool: ToolAnalysis,
  allTools: ToolAnalysis[]
): ConsolidationScore {
  let score = 0;
  const reasons: string[] = [];

  // Factor 1: Usage frequency (lower = higher consolidation priority)
  const usageScores = { daily: 0, weekly: 10, monthly: 30, rarely: 50, never: 80 };
  score += usageScores[tool.usageFrequency];
  if (tool.usageFrequency === 'never') {
    reasons.push('Tool is never used');
  }

  // Factor 2: Overlap with other tools
  const overlaps = findOverlappingTools(tool, allTools);
  if (overlaps.length > 0) {
    score += overlaps.length * 20;
    reasons.push(`Overlaps with ${overlaps.length} other tools`);
  }

  // Factor 3: Code size (smaller = easier to consolidate)
  if (tool.linesOfCode < 500) {
    score += 20;
    reasons.push('Small codebase, easy to merge');
  }

  // Factor 4: Maintenance burden
  const daysSinceModified = Math.floor(
    (Date.now() - tool.lastModified.getTime()) / (1000 * 60 * 60 * 24)
  );
  if (daysSinceModified > 180) {
    score += 30;
    reasons.push('Not actively maintained');
  }

  // Factor 5: Single maintainer risk
  if (tool.maintainers.length <= 1) {
    score += 15;
    reasons.push('Single maintainer risk');
  }

  // Determine recommendation
  let recommendation: ConsolidationScore['recommendation'];
  let effort: ConsolidationScore['effort'];
  let targetTool: string | undefined;

  if (score >= 80) {
    recommendation = tool.usageFrequency === 'never' ? 'delete' : 'deprecate';
    effort = 'low';
  } else if (score >= 50 && overlaps.length > 0) {
    recommendation = 'merge';
    targetTool = overlaps[0].name;
    effort = tool.linesOfCode > 1000 ? 'high' : 'medium';
  } else {
    recommendation = 'keep';
    effort = 'low';
  }

  return {
    tool: tool.name,
    score,
    recommendation,
    reasons,
    targetTool,
    effort,
  };
}

function findOverlappingTools(tool: ToolAnalysis, allTools: ToolAnalysis[]): ToolAnalysis[] {
  return allTools.filter((other) => {
    if (other.name === tool.name) return false;

    // Check functionality overlap
    const overlap = tool.functionality.filter((f) => other.functionality.includes(f));

    return overlap.length / tool.functionality.length > 0.5;
  });
}
```

---

## Unified CLI Architecture

### CLI Consolidation Pattern

```typescript
// lib/cli/unified-cli.ts
import { Command } from 'commander';

interface CLIModule {
  name: string;
  description: string;
  commands: CommandDefinition[];
}

interface CommandDefinition {
  name: string;
  description: string;
  options: OptionDefinition[];
  action: (...args: unknown[]) => Promise<void>;
}

interface OptionDefinition {
  flags: string;
  description: string;
  default?: unknown;
}

export class UnifiedCLI {
  private program: Command;
  private modules: Map<string, CLIModule> = new Map();

  constructor(name: string, version: string, description: string) {
    this.program = new Command().name(name).version(version).description(description);
  }

  // Register a module (previously separate CLI)
  registerModule(module: CLIModule): void {
    this.modules.set(module.name, module);

    const subcommand = this.program.command(module.name).description(module.description);

    for (const cmd of module.commands) {
      const command = subcommand.command(cmd.name).description(cmd.description);

      for (const opt of cmd.options) {
        command.option(opt.flags, opt.description, opt.default);
      }

      command.action(cmd.action);
    }
  }

  // Provide backward compatibility aliases
  addAlias(alias: string, target: string): void {
    this.program
      .command(alias)
      .description(`Alias for '${target}'`)
      .allowUnknownOption()
      .action(async (options, command) => {
        const args = command.args;
        console.log(`Note: '${alias}' is deprecated, use '${target}' instead`);
        await this.program.parseAsync([target, ...args], { from: 'user' });
      });
  }

  run(): void {
    this.program.parse();
  }

  // Generate help for all modules
  generateDocs(): string {
    let docs = `# ${this.program.name()} CLI\n\n`;
    docs += `${this.program.description()}\n\n`;
    docs += `## Commands\n\n`;

    for (const [name, module] of this.modules) {
      docs += `### ${name}\n\n`;
      docs += `${module.description}\n\n`;

      for (const cmd of module.commands) {
        docs += `#### \`${name} ${cmd.name}\`\n\n`;
        docs += `${cmd.description}\n\n`;

        if (cmd.options.length > 0) {
          docs += `Options:\n`;
          for (const opt of cmd.options) {
            docs += `- \`${opt.flags}\`: ${opt.description}`;
            if (opt.default !== undefined) {
              docs += ` (default: ${opt.default})`;
            }
            docs += `\n`;
          }
          docs += `\n`;
        }
      }
    }

    return docs;
  }
}

// Example: Consolidating 22 tools into 4 CLIs
export function createAtlasCLI(): UnifiedCLI {
  const cli = new UnifiedCLI('atlas', '2.0.0', 'Unified automation and orchestration CLI');

  // Module 1: Prompts (consolidates prompt-optimizer, prompt-manager, etc.)
  cli.registerModule({
    name: 'prompts',
    description: 'Prompt management and optimization',
    commands: [
      {
        name: 'list',
        description: 'List all prompts',
        options: [
          { flags: '-c, --category <category>', description: 'Filter by category' },
          { flags: '-t, --tags <tags>', description: 'Filter by tags' },
        ],
        action: async (options) => {
          // Implementation
        },
      },
      {
        name: 'optimize',
        description: 'Optimize a prompt for token efficiency',
        options: [
          { flags: '-i, --input <file>', description: 'Input prompt file' },
          { flags: '-o, --output <file>', description: 'Output file' },
        ],
        action: async (options) => {
          // Implementation
        },
      },
    ],
  });

  // Module 2: Agents (consolidates agent-manager, agent-router, etc.)
  cli.registerModule({
    name: 'agents',
    description: 'Agent management and orchestration',
    commands: [
      {
        name: 'list',
        description: 'List all agents',
        options: [],
        action: async () => {
          // Implementation
        },
      },
      {
        name: 'route',
        description: 'Route a task to the best agent',
        options: [{ flags: '-t, --task <task>', description: 'Task description' }],
        action: async (options) => {
          // Implementation
        },
      },
    ],
  });

  // Module 3: Workflows (consolidates workflow-executor, workflow-validator, etc.)
  cli.registerModule({
    name: 'workflows',
    description: 'Workflow execution and management',
    commands: [
      {
        name: 'list',
        description: 'List all workflows',
        options: [],
        action: async () => {
          // Implementation
        },
      },
      {
        name: 'execute',
        description: 'Execute a workflow',
        options: [
          { flags: '-n, --name <name>', description: 'Workflow name' },
          { flags: '-p, --params <json>', description: 'Parameters as JSON' },
        ],
        action: async (options) => {
          // Implementation
        },
      },
    ],
  });

  // Module 4: Deploy (consolidates deployer, publisher, etc.)
  cli.registerModule({
    name: 'deploy',
    description: 'Deployment and publishing',
    commands: [
      {
        name: 'list',
        description: 'List deployment targets',
        options: [],
        action: async () => {
          // Implementation
        },
      },
      {
        name: 'run',
        description: 'Deploy to a target',
        options: [
          { flags: '-t, --target <target>', description: 'Deployment target' },
          { flags: '-e, --env <env>', description: 'Environment', default: 'staging' },
        ],
        action: async (options) => {
          // Implementation
        },
      },
    ],
  });

  // Backward compatibility aliases for deprecated commands
  cli.addAlias('prompt-optimize', 'prompts optimize');
  cli.addAlias('agent-route', 'agents route');
  cli.addAlias('workflow-run', 'workflows execute');

  return cli;
}
```

---

## Shared Library Extraction

### Common Patterns Library

```typescript
// lib/shared/index.ts
// Consolidated from 9 duplicate implementations

// Validation utilities (was in 5 different tools)
export * from './validation';

// Configuration loading (was in 7 different tools)
export * from './config';

// Telemetry and logging (was in 4 different tools)
export * from './telemetry';

// File utilities (was in 6 different tools)
export * from './files';

// YAML/JSON parsing (was in 8 different tools)
export * from './parsing';
```

```typescript
// lib/shared/validation.ts
import { z } from 'zod';

// Consolidated validation schemas
export const schemas = {
  agent: z.object({
    id: z.string(),
    name: z.string(),
    role: z.string(),
    capabilities: z.array(z.string()),
  }),

  workflow: z.object({
    id: z.string(),
    name: z.string(),
    steps: z.array(
      z.object({
        name: z.string(),
        agent: z.string(),
        action: z.string(),
      })
    ),
  }),

  prompt: z.object({
    name: z.string(),
    category: z.enum(['system', 'project', 'task']),
    content: z.string(),
    tags: z.array(z.string()).optional(),
  }),
};

// Unified validation function
export function validate<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): { success: true; data: T } | { success: false; errors: string[] } {
  const result = schema.safeParse(data);

  if (result.success) {
    return { success: true, data: result.data };
  }

  return {
    success: false,
    errors: result.error.errors.map((e) => `${e.path.join('.')}: ${e.message}`),
  };
}
```

```typescript
// lib/shared/telemetry.ts
import { EventEmitter } from 'events';

// Consolidated telemetry (was duplicated in 4 tools)
class Telemetry extends EventEmitter {
  private static instance: Telemetry;
  private metrics: Map<string, number[]> = new Map();

  static getInstance(): Telemetry {
    if (!Telemetry.instance) {
      Telemetry.instance = new Telemetry();
    }
    return Telemetry.instance;
  }

  track(event: string, value?: number): void {
    this.emit('event', { event, value, timestamp: new Date() });

    if (value !== undefined) {
      const existing = this.metrics.get(event) || [];
      existing.push(value);
      this.metrics.set(event, existing.slice(-100)); // Keep last 100
    }
  }

  getMetrics(event: string): { avg: number; min: number; max: number; count: number } {
    const values = this.metrics.get(event) || [];
    if (values.length === 0) {
      return { avg: 0, min: 0, max: 0, count: 0 };
    }

    return {
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length,
    };
  }
}

export const telemetry = Telemetry.getInstance();
```

---

## Automated Enforcement

### Pre-Commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      # File size enforcement
      - id: file-size-check
        name: Check file size
        entry: scripts/check-file-size.sh
        language: script
        args: ['--max-lines', '500']

      # Architecture compliance
      - id: architecture-check
        name: Check architecture compliance
        entry: scripts/check-architecture.sh
        language: script
        files: '\.(ts|py)$'

      # Duplicate detection
      - id: duplicate-check
        name: Check for duplicates
        entry: scripts/check-duplicates.sh
        language: script

      # Import structure
      - id: import-check
        name: Check import structure
        entry: scripts/check-imports.sh
        language: script
        files: '\.ts$'
```

### Enforcement Scripts

```bash
#!/bin/bash
# scripts/check-file-size.sh

MAX_LINES=${1:-500}
VIOLATIONS=0

for file in $(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(ts|py|js)$'); do
  lines=$(wc -l < "$file")
  if [ "$lines" -gt "$MAX_LINES" ]; then
    echo "❌ $file: $lines lines (max: $MAX_LINES)"
    VIOLATIONS=$((VIOLATIONS + 1))
  fi
done

if [ "$VIOLATIONS" -gt 0 ]; then
  echo ""
  echo "💡 Consider splitting large files into smaller modules"
  exit 1
fi

echo "✅ All files within size limits"
exit 0
```

```bash
#!/bin/bash
# scripts/check-duplicates.sh

# Find potential duplicate implementations
echo "Checking for duplicate code patterns..."

# Check for duplicate function signatures
duplicates=$(grep -rh "^export function\|^export async function" src/ | sort | uniq -d)

if [ -n "$duplicates" ]; then
  echo "❌ Potential duplicate functions found:"
  echo "$duplicates"
  echo ""
  echo "💡 Consider extracting to shared library"
  exit 1
fi

echo "✅ No obvious duplicates found"
exit 0
```

---

## Migration Guide Template

```markdown
# Migration Guide: [Old Tool] → [New Tool]

## Overview

This guide helps you migrate from `old-tool` to the unified `atlas` CLI.

## Command Mapping

| Old Command                   | New Command                 | Notes                   |
| ----------------------------- | --------------------------- | ----------------------- |
| `old-tool do-thing`           | `atlas module do-thing`     | Identical functionality |
| `old-tool other-thing --flag` | `atlas module other --flag` | Flag renamed            |

## Breaking Changes

1. **Configuration location changed**
   - Old: `~/.old-tool/config.yaml`
   - New: `~/.atlas/config.yaml`
   - Migration: Run `atlas migrate config`

2. **Output format changed**
   - Old: Plain text
   - New: JSON by default
   - Use `--format text` for old behavior

## Deprecation Timeline

- **Phase 1** (Now): Both tools work, warnings shown
- **Phase 2** (30 days): Old tool shows deprecation notice
- **Phase 3** (60 days): Old tool removed

## Getting Help

- Run `atlas help module` for command help
- See [full documentation](./docs/atlas-cli.md)
- Report issues at [GitHub Issues](./issues)
```

---

## Consolidation Metrics Dashboard

```yaml
# consolidation-metrics.yaml
metrics:
  before:
    total_tools: 22
    total_lines: 45000
    config_files: 87
    duplicate_functions: 156

  after:
    total_tools: 4
    total_lines: 12000
    config_files: 12
    duplicate_functions: 0

  reduction:
    tools: '82%'
    code: '73%'
    config: '86%'
    duplicates: '100%'

  quality_improvements:
    test_coverage: '+25%'
    documentation: 'Consolidated'
    maintenance_burden: '-70%'
    onboarding_time: '-50%'
```

---

## Execution Phases

### Phase 1: Audit (Week 1-2)

- [ ] Inventory all tools and scripts
- [ ] Analyze usage metrics
- [ ] Identify overlapping functionality
- [ ] Calculate consolidation scores

### Phase 2: Design (Week 3-4)

- [ ] Design unified CLI architecture
- [ ] Plan shared library extraction
- [ ] Create migration guides
- [ ] Define enforcement rules

### Phase 3: Implementation (Week 5-8)

- [ ] Build unified CLI framework
- [ ] Extract shared libraries
- [ ] Implement backward compatibility
- [ ] Set up automated enforcement

### Phase 4: Migration (Week 9-12)

- [ ] Deploy new tools alongside old
- [ ] Migrate teams incrementally
- [ ] Monitor for issues
- [ ] Deprecate old tools

---

**Last updated: 2024-11-30**
