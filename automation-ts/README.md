# Automation TypeScript CLI

TypeScript CLI for AI automation asset management. Provides full parity with the Python `automation/` CLI plus additional deployment features.

## Installation

```bash
cd automation-ts
npm install
npm run build
```

## Usage

### System Information

```bash
npx automation info
```

### Prompts

```bash
# List all prompts
npx automation prompts list

# Filter by category
npx automation prompts list --category system
```

### Agents

```bash
npx automation agents list
```

### Workflows

```bash
npx automation workflows list
```

### Task Routing

```bash
# Route a task to appropriate tools
npx automation route "debug the authentication error"
npx automation route "implement user management feature"
```

### Orchestration Patterns

```bash
npx automation patterns
```

### Validation

```bash
npx automation validate
```

### Workflow Execution

```bash
npx automation execute literature_review
npx automation execute code_review --input '{"pr_number": 123}'
```

### Deployment

```bash
# List projects
npx automation deploy list
npx automation deploy list --org alaweimm90-science
npx automation deploy list --type scientific

# Statistics
npx automation deploy stats

# Templates
npx automation deploy templates
```

### Crews

```bash
npx automation crews list
```

## Development

```bash
# Watch mode
npm run dev

# Run tests
npm test

# Type check
npm run type-check

# Lint
npm run lint
```

## Project Structure

```
automation-ts/
├── src/
│   ├── index.ts            # Library entry point
│   ├── types/index.ts      # TypeScript interfaces
│   ├── utils/file.ts       # File utilities
│   ├── cli/index.ts        # CLI commands (Commander.js)
│   ├── executor/index.ts   # Workflow execution engine
│   ├── validation/index.ts # Asset validation
│   ├── deployment/index.ts # Deployment registry
│   ├── crews/index.ts      # Crew management
│   └── __tests__/          # Jest tests
├── dist/                    # Compiled JavaScript
├── package.json
├── tsconfig.json
└── README.md
```

## Integration

This CLI shares configurations with the Python `automation/` folder:
- `automation/agents/config/agents.yaml`
- `automation/workflows/config/workflows.yaml`
- `automation/prompts/`
- `automation/orchestration/`

## License

MIT
