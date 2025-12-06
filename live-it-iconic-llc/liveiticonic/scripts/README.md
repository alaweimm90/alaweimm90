# Scripts Directory

This directory contains build scripts, automation tools, and utility scripts for the Live It Iconic project.

## Directory Structure

```
scripts/
├── brand/                    # Brand asset management
│   ├── asset-validator.ts
│   └── image-processor.ts
├── setup/                    # Development environment setup
│   └── dev-environment.ts
├── workflow/                 # Workflow automation
│   └── workflow-engine.ts
├── analysis/                 # Code analysis tools
│   ├── memory-analysis.ts
│   ├── performance-analysis.ts
│   └── documentation-audit.ts
├── build/                    # Build and deployment scripts
└── db/                       # Database migration and seed scripts
```

## Analysis Scripts

### Memory Analysis
Analyzes code for potential memory issues.

```bash
npm run analyze:memory
```

**Output**: `reports/memory-analysis.json`

**What it checks**:
- Large arrays and objects
- Event listener leaks
- Closure memory retention
- Component state management

### Performance Analysis
Identifies performance bottlenecks and code complexity.

```bash
npm run analyze:performance
```

**Output**: `reports/performance-analysis.json`

**Metrics**:
- Cyclomatic complexity (target: < 10)
- Function length (target: < 50 lines)
- Nested loops (target: < 3 levels)
- Large component re-renders

### Documentation Audit
Measures documentation coverage across the codebase.

```bash
npm run analyze:docs
```

**Output**: `reports/documentation-audit.json`

**Coverage targets**:
- Overall: 80%+
- Services: 90%+
- Components: 75%+
- Public APIs: 100%

## Brand Scripts

### Asset Validator
Validates brand assets meet quality standards.

```bash
npm run brand:validate
```

**Checks**:
- Image dimensions (logos, icons)
- File formats (WebP, PNG, SVG)
- Color palette consistency
- File size limits

### Image Processor
Optimizes and converts images for web use.

```bash
npm run brand:process
```

**Actions**:
- Converts images to WebP
- Generates multiple sizes
- Compresses images
- Creates responsive variants

## Setup Scripts

### Development Environment
Sets up local development environment.

```bash
npm run setup:dev
```

**What it does**:
1. Checks Node.js version (>=18)
2. Installs dependencies
3. Creates `.env.local` from `.env.example`
4. Sets up Supabase locally (if Docker available)
5. Runs database migrations
6. Seeds test data
7. Verifies environment

## Workflow Scripts

### Workflow Engine
Automates common development workflows.

```bash
npm run workflow:run -- <workflow-name>
```

**Available workflows**:
- `new-feature` - Scaffolds new feature files
- `new-component` - Creates component with test and story
- `new-service` - Creates service with singleton pattern
- `pre-commit` - Runs linting, formatting, type checking
- `pre-push` - Runs tests and builds

**Example**:
```bash
npm run workflow:run -- new-service PaymentService
```

Creates:
- `src/services/paymentService.ts`
- `tests/unit/services/paymentService.test.ts`
- Type definitions in `src/types/`

## Build Scripts

### Production Build
```bash
npm run build
```

Runs:
1. TypeScript compilation
2. Vite build with optimizations
3. Asset optimization
4. Service worker generation
5. Source map generation

### Build with Analysis
```bash
npm run build:analyze
```

Additional steps:
- Bundle size analysis
- Dependency tree visualization
- Build time profiling

### Preview Build
```bash
npm run build:preview
```

Builds and serves locally to test production build.

## Database Scripts

### Run Migrations
```bash
npm run db:migrate
```

Applies pending database migrations to Supabase.

### Seed Database
```bash
npm run db:seed
```

Seeds database with:
- Sample users
- Test products (hoodies, caps, t-shirts)
- Sample orders and payments
- Product reviews

### Reset Database
```bash
npm run db:reset
```

⚠️ **Warning**: Drops all data and resets to clean state.

Only for development environments.

### Generate Types from Database
```bash
npm run db:types
```

Generates TypeScript types from Supabase schema.

Output: `src/types/supabase.ts`

## Utility Scripts

### Format Code
```bash
npm run format
```

Formats all code with Prettier.

### Lint Code
```bash
npm run lint
```

Runs ESLint and reports issues.

### Fix Linting Issues
```bash
npm run lint:fix
```

Auto-fixes ESLint issues where possible.

### Type Check
```bash
npm run typecheck
```

Runs TypeScript compiler in check mode (no output).

### Update Dependencies
```bash
npm run deps:update
```

Updates dependencies to latest compatible versions.

Shows:
- Outdated packages
- Breaking changes
- Security vulnerabilities

### Check Bundle Size
```bash
npm run size
```

Analyzes bundle size and shows breakdown by chunk.

## Creating New Scripts

### TypeScript Scripts

Create scripts in TypeScript for type safety:

```typescript
// scripts/my-script.ts
import { resolve } from 'path';

async function main() {
  console.log('Running my script...');

  // Your script logic here
}

main().catch(console.error);
```

Run with:
```bash
tsx scripts/my-script.ts
```

### Add to package.json

```json
{
  "scripts": {
    "my-script": "tsx scripts/my-script.ts"
  }
}
```

## Script Best Practices

### 1. Use TypeScript
TypeScript scripts are easier to maintain and refactor.

### 2. Handle Errors Gracefully
```typescript
async function main() {
  try {
    await riskyOperation();
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}
```

### 3. Provide Progress Feedback
```typescript
import ora from 'ora';

const spinner = ora('Processing files...').start();

await processFiles();

spinner.succeed('Files processed successfully!');
```

### 4. Accept CLI Arguments
```typescript
import { program } from 'commander';

program
  .option('-e, --env <environment>', 'Environment', 'development')
  .parse(process.argv);

const options = program.opts();
console.log('Environment:', options.env);
```

### 5. Document Script Usage
Add help text:
```typescript
program
  .description('Processes brand assets')
  .option('-i, --input <path>', 'Input directory')
  .option('-o, --output <path>', 'Output directory')
  .helpOption('-h, --help', 'Display help');
```

## Environment Variables

Scripts can access environment variables:

```typescript
import { config } from 'dotenv';

config(); // Load .env file

const supabaseUrl = process.env.VITE_SUPABASE_URL;
const supabaseKey = process.env.VITE_SUPABASE_ANON_KEY;
```

## Performance Considerations

- Use streams for large file processing
- Parallelize independent operations
- Cache expensive computations
- Show progress for long-running scripts

## Security

- Never commit secrets in scripts
- Use environment variables for sensitive data
- Validate user input in CLI scripts
- Use `--dry-run` flags for destructive operations

## CI/CD Integration

Scripts are used in GitHub Actions workflows:

```yaml
# .github/workflows/test.yml
- name: Run analysis scripts
  run: |
    npm run analyze:memory
    npm run analyze:performance
    npm run analyze:docs
```

## Troubleshooting

### Script Permissions Error
```bash
chmod +x scripts/my-script.sh
```

### TypeScript Import Errors
Add to `tsconfig.json`:
```json
{
  "compilerOptions": {
    "moduleResolution": "node",
    "esModuleInterop": true
  }
}
```

### Module Not Found
Install dependencies:
```bash
npm install --save-dev @types/node tsx
```

## Resources

- [Node.js Script Best Practices](https://nodejs.org/en/docs/guides/)
- [Commander.js (CLI)](https://github.com/tj/commander.js)
- [Ora (Spinners)](https://github.com/sindresorhus/ora)
- [Chalk (Colors)](https://github.com/chalk/chalk)

---

For more details, see:
- [Contributing Guide](../CONTRIBUTING.md)
- [Development Setup](../QUICK_START.md)
