/**
 * Tool Metadata Definitions
 * Extracted from generator.ts for maintainability
 */

export interface ToolDoc {
  name: string;
  description: string;
  category: 'core' | 'infrastructure' | 'governance';
  commands: CommandDoc[];
  exports?: string[];
  configuration?: ConfigDoc[];
  examples?: ExampleDoc[];
}

export interface CommandDoc {
  name: string;
  script: string;
  description: string;
  args?: ArgDoc[];
}

export interface ArgDoc {
  name: string;
  type: string;
  description: string;
  required: boolean;
  default?: string;
}

export interface ConfigDoc {
  name: string;
  type: string;
  description: string;
  default: string;
}

export interface ExampleDoc {
  title: string;
  code: string;
  description?: string;
}

export const TOOLS: ToolDoc[] = [
  {
    name: 'Orchestrator',
    description: 'Task management and context injection for AI-assisted development',
    category: 'core',
    commands: [
      {
        name: 'start',
        script: 'npm run ai:start',
        description: 'Start tracking a new task',
        args: [
          {
            name: 'type',
            type: 'string',
            description: 'Task type (feature, bugfix, refactor, docs, test, devops)',
            required: true,
          },
          {
            name: 'scope',
            type: 'string',
            description: 'Comma-separated scope areas',
            required: false,
          },
          { name: 'description', type: 'string', description: 'Task description', required: true },
        ],
      },
      {
        name: 'complete',
        script: 'npm run ai:complete',
        description: 'Mark current task as complete',
        args: [
          {
            name: 'success',
            type: 'boolean',
            description: 'Whether task was successful',
            required: true,
          },
          {
            name: 'filesChanged',
            type: 'string',
            description: 'Comma-separated list of changed files',
            required: false,
          },
          {
            name: 'linesAdded',
            type: 'number',
            description: 'Lines of code added',
            required: false,
            default: '0',
          },
          {
            name: 'linesRemoved',
            type: 'number',
            description: 'Lines of code removed',
            required: false,
            default: '0',
          },
          {
            name: 'testsAdded',
            type: 'number',
            description: 'Number of tests added',
            required: false,
            default: '0',
          },
          { name: 'notes', type: 'string', description: 'Completion notes', required: false },
        ],
      },
      {
        name: 'context',
        script: 'npm run ai:context',
        description: 'Get AI context for a task type',
      },
      {
        name: 'metrics',
        script: 'npm run ai:metrics',
        description: 'View or update AI effectiveness metrics',
      },
      { name: 'history', script: 'npm run ai:history', description: 'View task history' },
    ],
    examples: [
      {
        title: 'Start a feature task',
        code: 'npm run ai:start feature auth,api "Add OAuth authentication"',
      },
      {
        title: 'Complete a task',
        code: 'npm run ai:complete true "src/auth.ts,src/api.ts" 150 20 5 "Added OAuth flow"',
      },
    ],
  },
  {
    name: 'Sync',
    description: 'Context synchronization from git and other sources',
    category: 'core',
    commands: [
      { name: 'sync', script: 'npm run ai:sync', description: 'Synchronize all context sources' },
    ],
    examples: [{ title: 'Run sync', code: 'npm run ai:sync' }],
  },
  {
    name: 'Dashboard',
    description: 'ASCII metrics dashboard for AI effectiveness visualization',
    category: 'core',
    commands: [
      {
        name: 'dashboard',
        script: 'npm run ai:dashboard',
        description: 'Display ASCII metrics dashboard',
      },
    ],
    examples: [{ title: 'View dashboard', code: 'npm run ai:dashboard' }],
  },
  {
    name: 'Cache',
    description: 'Multi-layer caching with semantic similarity for AI operations',
    category: 'infrastructure',
    commands: [
      { name: 'cache', script: 'npm run ai:cache', description: 'Cache management CLI' },
      { name: 'stats', script: 'npm run ai:cache:stats', description: 'Show cache statistics' },
      {
        name: 'clear',
        script: 'npm run ai:cache:clear',
        description: 'Clear cache entries',
        args: [
          {
            name: 'layer',
            type: 'string',
            description: 'Cache layer to clear (semantic, template, result, analysis)',
            required: false,
          },
        ],
      },
    ],
    exports: ['cache'],
    configuration: [
      { name: 'maxEntries', type: 'number', description: 'Maximum cache entries', default: '1000' },
      {
        name: 'maxSizeBytes',
        type: 'number',
        description: 'Maximum cache size in bytes',
        default: '52428800',
      },
      {
        name: 'defaultTtlMs',
        type: 'number',
        description: 'Default TTL in milliseconds',
        default: '3600000',
      },
      {
        name: 'enableSemanticSimilarity',
        type: 'boolean',
        description: 'Enable semantic matching',
        default: 'true',
      },
      {
        name: 'similarityThreshold',
        type: 'number',
        description: 'Similarity threshold (0-1)',
        default: '0.85',
      },
    ],
    examples: [
      { title: 'View cache stats', code: 'npm run ai:cache:stats' },
      { title: 'Clear semantic layer', code: 'npm run ai:cache:clear semantic' },
    ],
  },
  {
    name: 'Monitor',
    description: 'Continuous monitoring with circuit breakers for fault tolerance',
    category: 'infrastructure',
    commands: [
      { name: 'monitor', script: 'npm run ai:monitor', description: 'Monitor CLI' },
      {
        name: 'status',
        script: 'npm run ai:monitor:status',
        description: 'Show monitor status and circuit breaker states',
      },
      {
        name: 'check',
        script: 'npm run ai:monitor:check',
        description: 'Check for changes and trigger actions',
      },
    ],
    exports: ['monitor', 'circuitBreaker'],
    configuration: [
      {
        name: 'debounceMs',
        type: 'number',
        description: 'Debounce time for changes',
        default: '2000',
      },
      {
        name: 'maxFrequencyMs',
        type: 'number',
        description: 'Minimum time between triggers',
        default: '30000',
      },
      {
        name: 'failureThreshold',
        type: 'number',
        description: 'Circuit breaker failure threshold',
        default: '3',
      },
      {
        name: 'resetTimeoutMs',
        type: 'number',
        description: 'Circuit breaker reset timeout',
        default: '60000',
      },
    ],
    examples: [{ title: 'Check monitor status', code: 'npm run ai:monitor:status' }],
  },
  {
    name: 'Compliance',
    description: 'Policy-based validation with quantitative scoring and recommendations',
    category: 'governance',
    commands: [
      { name: 'compliance', script: 'npm run ai:compliance', description: 'Compliance CLI' },
      {
        name: 'check',
        script: 'npm run ai:compliance:check',
        description: 'Run compliance check on files',
        args: [{ name: 'files', type: 'string[]', description: 'Files to check', required: false }],
      },
      {
        name: 'score',
        script: 'npm run ai:compliance:score',
        description: 'Quick compliance score check',
      },
    ],
    exports: ['compliance'],
    examples: [
      { title: 'Run compliance check', code: 'npm run ai:compliance:check src/api.ts' },
      { title: 'Quick score', code: 'npm run ai:compliance:score' },
    ],
  },
  {
    name: 'Telemetry',
    description: 'Observability and alerting for AI operations',
    category: 'infrastructure',
    commands: [
      { name: 'telemetry', script: 'npm run ai:telemetry', description: 'Telemetry CLI' },
      {
        name: 'status',
        script: 'npm run ai:telemetry:status',
        description: 'Show telemetry status',
      },
      { name: 'alerts', script: 'npm run ai:telemetry:alerts', description: 'Show active alerts' },
    ],
    exports: ['telemetry'],
    examples: [{ title: 'Check telemetry status', code: 'npm run ai:telemetry:status' }],
  },
  {
    name: 'Errors',
    description: 'Structured error handling with automatic recovery strategies',
    category: 'infrastructure',
    commands: [
      { name: 'errors', script: 'npm run ai:errors', description: 'Error handler CLI' },
      {
        name: 'list',
        script: 'npm run ai:errors:list',
        description: 'List recent errors',
        args: [
          {
            name: 'severity',
            type: 'string',
            description: 'Filter by severity (low, medium, high, critical)',
            required: false,
          },
        ],
      },
      { name: 'stats', script: 'npm run ai:errors:stats', description: 'Show error statistics' },
    ],
    exports: ['errorHandler', 'ErrorCodes', 'AIOperationError'],
    examples: [
      { title: 'List critical errors', code: 'npm run ai:errors:list critical' },
      { title: 'View error stats', code: 'npm run ai:errors:stats' },
    ],
  },
  {
    name: 'Security',
    description: 'Security scanning for secrets, vulnerabilities, and license compliance',
    category: 'governance',
    commands: [
      { name: 'security', script: 'npm run ai:security', description: 'Security scanner CLI' },
      { name: 'scan', script: 'npm run ai:security:scan', description: 'Run full security scan' },
      {
        name: 'secrets',
        script: 'npm run ai:security:secrets',
        description: 'Scan for exposed secrets',
      },
      {
        name: 'vulns',
        script: 'npm run ai:security:vulns',
        description: 'Scan for vulnerabilities',
      },
    ],
    exports: ['securityScanner'],
    examples: [
      { title: 'Run full scan', code: 'npm run ai:security:scan' },
      { title: 'Check for secrets', code: 'npm run ai:security:secrets src/' },
    ],
  },
  {
    name: 'Issues',
    description: 'Automated issue management and tracking',
    category: 'governance',
    commands: [
      { name: 'issues', script: 'npm run ai:issues', description: 'Issue manager CLI' },
      { name: 'list', script: 'npm run ai:issues:list', description: 'List tracked issues' },
      {
        name: 'critical',
        script: 'npm run ai:issues:critical',
        description: 'List critical issues',
      },
      { name: 'stats', script: 'npm run ai:issues:stats', description: 'Show issue statistics' },
    ],
    exports: ['issueManager'],
    examples: [
      { title: 'List critical issues', code: 'npm run ai:issues:critical' },
      { title: 'View issue stats', code: 'npm run ai:issues:stats' },
    ],
  },
];
