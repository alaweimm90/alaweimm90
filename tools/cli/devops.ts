#!/usr/bin/env node
/**
 * Unified DevOps CLI - Template management and code generation
 *
 * Consolidates:
 * - builder.ts → template list/apply
 * - coder.ts → generate
 * - bootstrap.ts → init
 * - install.ts → setup
 */

import { Command } from 'commander';
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { resolveTargetDir, parsePlaceholders } from '../lib/config.js';
import {
  ensureDir,
  findManifests,
  readJson,
  copyTemplateFiles,
  writeTemplateMeta,
  validateTemplate,
  type TemplateManifest,
} from '../lib/fs.js';

const TEMPLATES_DIR = path.join(process.cwd(), 'templates', 'devops');

interface DiscoveredTemplate {
  manifest: TemplateManifest;
  manifestPath: string;
  templateDir: string;
}

interface PlannedFile {
  path: string;
  content: string;
  action: 'ADD' | 'UPDATE';
}

// ============================================================================
// Template Operations (from builder.ts)
// ============================================================================

/**
 * Discover all templates from manifests
 */
function discoverTemplates(): DiscoveredTemplate[] {
  const manifestPaths = findManifests(TEMPLATES_DIR);
  const templates: DiscoveredTemplate[] = [];

  for (const manifestPath of manifestPaths) {
    const manifest = readJson<TemplateManifest>(manifestPath);
    if (manifest) {
      templates.push({
        manifest,
        manifestPath,
        templateDir: path.dirname(manifestPath),
      });
    }
  }

  return templates;
}

/**
 * Filter templates by search pattern
 */
function filterTemplates(templates: DiscoveredTemplate[], search?: string): DiscoveredTemplate[] {
  if (!search || search === '*') {
    return templates;
  }

  const pattern = search.toLowerCase();
  return templates.filter((t) => {
    return (
      t.manifest.name.toLowerCase().includes(pattern) ||
      t.manifest.description.toLowerCase().includes(pattern) ||
      t.manifest.tags.some((tag) => tag.toLowerCase().includes(pattern))
    );
  });
}

async function cmdTemplateList(options: { search?: string }): Promise<void> {
  const allTemplates = discoverTemplates();
  const templates = filterTemplates(allTemplates, options.search);

  console.log('\nAvailable Templates:\n');
  for (const t of templates) {
    console.log(`  ${t.manifest.name} (${t.manifest.category})`);
    console.log(`    ${t.manifest.description}`);
    console.log(`    Tags: ${t.manifest.tags.join(', ')}`);
    console.log(`    Version: ${t.manifest.version}\n`);
  }
  console.log(`Total: ${templates.length} templates`);
}

async function cmdTemplateApply(
  name: string,
  options: { target?: string; dryRun?: boolean }
): Promise<void> {
  const allTemplates = discoverTemplates();
  const template = allTemplates.find((t) => t.manifest.name === name);

  if (!template) {
    console.error(`Template not found: ${name}`);
    console.error('Use "devops template list" to see available templates');
    process.exit(1);
  }

  const targetDir = options.target || resolveTargetDir([]);
  const vars = parsePlaceholders(process.argv.slice(2));

  if (options.dryRun) {
    console.log('\nDry run - would apply:');
    console.log(`  Template: ${template.manifest.name}`);
    console.log(`  Target: ${targetDir}`);
    console.log(`  Files: ${template.manifest.requiredFiles.join(', ')}`);
    console.log('\nRemove --dry-run to apply changes');
    return;
  }

  console.log(`\nApplying template: ${template.manifest.name}`);
  console.log(`  Target: ${targetDir}`);

  // Validate first
  const validation = validateTemplate(template.manifest, template.templateDir);
  if (!validation.valid) {
    console.error('Validation errors:');
    for (const error of validation.errors) {
      console.error(`  - ${error}`);
    }
    process.exit(1);
  }

  // Copy files
  copyTemplateFiles(template.manifest, template.templateDir, targetDir, vars);

  // Write metadata
  writeTemplateMeta(targetDir, template.manifest);

  console.log('Template applied successfully!');
}

// ============================================================================
// Code Generation (from coder.ts)
// ============================================================================

/**
 * Plan Node.js service with CI, K8s, Helm, and Prometheus
 */
function planNodeService(vars: Record<string, string>): PlannedFile[] {
  const projectName = vars.PROJECT_NAME || 'my-service';
  const registry = vars.REGISTRY || 'ghcr.io';
  const imageTag = vars.IMAGE_TAG || 'latest';

  const files: PlannedFile[] = [];

  // Service source
  files.push({
    path: 'service/src/index.ts',
    content: `import http from 'node:http';

const port = Number(process.env.PORT ?? 8080);

function handler(req: http.IncomingMessage, res: http.ServerResponse): void {
  const method = req.method ?? 'GET';
  const url = req.url ?? '/';

  if (method === 'GET' && url === '/health') {
    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/plain');
    res.end('ok');
    return;
  }

  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify({ service: '${projectName}', status: 'running' }));
}

const server = http.createServer(handler);
server.listen(port, () => {
  console.log(\`${projectName} listening on port \${port}\`);
});
`,
    action: 'ADD',
  });

  // Package.json
  files.push({
    path: 'service/package.json',
    content: JSON.stringify(
      {
        name: projectName,
        version: '1.0.0',
        type: 'module',
        scripts: {
          start: 'node dist/index.js',
          build: 'tsc',
          dev: 'tsx src/index.ts',
        },
        devDependencies: {
          '@types/node': '^22.9.0',
          typescript: '^5.6.3',
          tsx: '^4.19.2',
        },
      },
      null,
      2
    ),
    action: 'ADD',
  });

  // tsconfig.json
  files.push({
    path: 'service/tsconfig.json',
    content: JSON.stringify(
      {
        compilerOptions: {
          target: 'ES2022',
          module: 'ESNext',
          moduleResolution: 'bundler',
          strict: true,
          outDir: 'dist',
          esModuleInterop: true,
        },
        include: ['src/**/*.ts'],
      },
      null,
      2
    ),
    action: 'ADD',
  });

  // Dockerfile
  files.push({
    path: 'service/Dockerfile',
    content: `FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
RUN npm ci --production
EXPOSE 8080
CMD ["node", "dist/index.js"]
`,
    action: 'ADD',
  });

  // CI workflow
  files.push({
    path: 'ci/ci.yml',
    content: `name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: cd service && npm ci
      - run: cd service && npm run build
      - run: cd service && npm test || true
`,
    action: 'ADD',
  });

  // K8s deployment
  files.push({
    path: 'k8s/deployment.yaml',
    content: `apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${projectName}
  labels:
    app: ${projectName}
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ${projectName}
  template:
    metadata:
      labels:
        app: ${projectName}
    spec:
      containers:
        - name: ${projectName}
          image: ${registry}/${projectName}:${imageTag}
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "200m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
`,
    action: 'ADD',
  });

  // K8s service
  files.push({
    path: 'k8s/service.yaml',
    content: `apiVersion: v1
kind: Service
metadata:
  name: ${projectName}
spec:
  selector:
    app: ${projectName}
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
`,
    action: 'ADD',
  });

  // Helm Chart.yaml
  files.push({
    path: 'helm/Chart.yaml',
    content: `apiVersion: v2
name: ${projectName}
description: Helm chart for ${projectName}
type: application
version: 0.1.0
appVersion: "1.0.0"
`,
    action: 'ADD',
  });

  // Helm values.yaml
  files.push({
    path: 'helm/values.yaml',
    content: `replicaCount: 2

image:
  repository: ${registry}/${projectName}
  tag: ${imageTag}
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80

resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "200m"
`,
    action: 'ADD',
  });

  // Prometheus config
  files.push({
    path: 'monitoring/prometheus.yml',
    content: `global:
  scrape_interval: 15s

scrape_configs:
  - job_name: '${projectName}'
    static_configs:
      - targets: ['${projectName}:8080']
    metrics_path: /metrics
`,
    action: 'ADD',
  });

  return files;
}

/**
 * Apply planned changes
 */
function applyChanges(files: PlannedFile[], targetDir: string): void {
  console.log('\nApplying Changes:\n');

  for (const file of files) {
    const fullPath = path.join(targetDir, file.path);
    const dir = path.dirname(fullPath);

    ensureDir(dir);
    fs.writeFileSync(fullPath, file.content, 'utf-8');

    const action = fs.existsSync(fullPath) ? 'UPDATE' : 'ADD';
    console.log(`  [${action}] ${file.path}`);
  }

  console.log(`\nApplied ${files.length} files to ${targetDir}`);
}

/**
 * Preview planned changes
 */
function previewChanges(files: PlannedFile[], targetDir: string): void {
  console.log('\nPlanned Changes:\n');

  for (const file of files) {
    const fullPath = path.join(targetDir, file.path);
    const exists = fs.existsSync(fullPath);
    const action = exists ? 'UPDATE' : 'ADD';

    console.log(`  [${action}] ${file.path}`);
  }

  console.log(`\nTotal: ${files.length} files`);
  console.log(`Target: ${targetDir}`);
  console.log('\nRemove --dry-run to apply changes');
}

async function cmdGenerate(
  type: string,
  options: { target?: string; dryRun?: boolean }
): Promise<void> {
  const targetDir = options.target || resolveTargetDir([]);
  const vars = parsePlaceholders(process.argv.slice(2));

  let files: PlannedFile[] = [];

  switch (type) {
    case 'node-service':
      files = planNodeService(vars);
      break;
    default:
      console.error(`Unknown generation type: ${type}`);
      console.error('\nAvailable types:');
      console.error('  node-service    Generate Node.js service with CI, K8s, Helm, Prometheus');
      process.exit(1);
  }

  if (options.dryRun) {
    previewChanges(files, targetDir);
  } else {
    applyChanges(files, targetDir);
  }
}

// ============================================================================
// Workspace Initialization (from bootstrap.ts)
// ============================================================================

async function cmdInit(options: { target?: string }): Promise<void> {
  const targetDir = options.target || resolveTargetDir([]);

  console.log(`Initializing workspace at: ${targetDir}`);

  // Create main directory
  ensureDir(targetDir);

  // Create subdirectories
  const subdirs = ['tools', 'templates', 'ci', 'k8s', 'helm', 'monitoring', 'service', 'docs'];

  for (const subdir of subdirs) {
    const subdirPath = path.join(targetDir, subdir);
    ensureDir(subdirPath);
    console.log(`  Created: ${subdir}/`);
  }

  // Create README if not exists
  const readmePath = path.join(targetDir, 'README.md');
  if (!fs.existsSync(readmePath)) {
    const readme = `# .metaHub Workspace

Generated by DevOps CLI.

## Structure

- \`tools/\` - CLI tools and scripts
- \`templates/\` - Template files
- \`ci/\` - CI/CD configurations
- \`k8s/\` - Kubernetes manifests
- \`helm/\` - Helm charts
- \`monitoring/\` - Monitoring configurations
- \`service/\` - Service source code
- \`docs/\` - Documentation

## Usage

See the main repository documentation for usage instructions.
`;
    fs.writeFileSync(readmePath, readme, 'utf-8');
    console.log('  Created: README.md');
  }

  console.log('\nWorkspace initialized successfully!');
}

// ============================================================================
// Dependency Setup (from install.ts)
// ============================================================================

type InstallType = 'npm' | 'pip';

/**
 * Run npm install in target directory
 */
function runNpmInstall(targetDir: string): boolean {
  const packageJsonPath = path.join(targetDir, 'package.json');
  if (!fs.existsSync(packageJsonPath)) {
    console.log('No package.json found, skipping npm install');
    return true;
  }

  try {
    console.log(`Running npm install in ${targetDir}...`);
    execSync('npm install', { cwd: targetDir, stdio: 'inherit' });
    return true;
  } catch (e) {
    console.error('npm install failed:', (e as Error).message);
    return false;
  }
}

/**
 * Run pip install in target directory
 */
function runPipInstall(targetDir: string): boolean {
  const requirementsPath = path.join(targetDir, 'requirements.txt');
  if (!fs.existsSync(requirementsPath)) {
    console.log('No requirements.txt found, skipping pip install');
    return true;
  }

  try {
    console.log(`Running pip install in ${targetDir}...`);
    execSync(`pip install -r ${requirementsPath}`, { cwd: targetDir, stdio: 'inherit' });
    return true;
  } catch (e) {
    console.error('pip install failed:', (e as Error).message);
    return false;
  }
}

/**
 * Check dependencies without installing
 */
function checkDependencies(targetDir: string): void {
  console.log(`Checking dependencies in: ${targetDir}\n`);

  const packageJsonPath = path.join(targetDir, 'package.json');
  const requirementsPath = path.join(targetDir, 'requirements.txt');

  if (fs.existsSync(packageJsonPath)) {
    console.log('  ✓ Found package.json (npm)');
  }

  if (fs.existsSync(requirementsPath)) {
    console.log('  ✓ Found requirements.txt (pip)');
  }

  if (!fs.existsSync(packageJsonPath) && !fs.existsSync(requirementsPath)) {
    console.log('  No dependency files found');
  }
}

async function cmdSetup(options: {
  target?: string;
  checkOnly?: boolean;
  type?: string;
}): Promise<void> {
  const targetDir = options.target || resolveTargetDir([]);

  if (options.checkOnly) {
    checkDependencies(targetDir);
    return;
  }

  const installType = (options.type as InstallType) || 'npm';

  let success = false;
  switch (installType) {
    case 'npm':
      success = runNpmInstall(targetDir);
      break;
    case 'pip':
      success = runPipInstall(targetDir);
      break;
    default:
      console.error(`Unknown install type: ${installType}`);
      console.error('Available types: npm, pip');
      process.exit(1);
  }

  if (!success) {
    process.exit(1);
  }
}

// ============================================================================
// CLI Setup
// ============================================================================

const program = new Command();

program
  .name('devops')
  .description('DevOps template operations and code generation')
  .version('1.0.0');

// Template commands
const templateCmd = program.command('template').description('Template operations');

templateCmd
  .command('list')
  .option('--search <pattern>', 'Search pattern for filtering templates')
  .description('List available templates')
  .action(cmdTemplateList);

templateCmd
  .command('apply <name>')
  .option('--target <dir>', 'Target directory (default: .metaHub)')
  .option('--dry-run', 'Preview changes without applying')
  .description('Apply a template to target directory')
  .action(cmdTemplateApply);

// Generate command
program
  .command('generate <type>')
  .option('--target <dir>', 'Target directory (default: .metaHub)')
  .option('--dry-run', 'Preview changes without applying')
  .description('Generate code from template (types: node-service)')
  .action(cmdGenerate);

// Init command
program
  .command('init')
  .option('--target <dir>', 'Target directory (default: .metaHub)')
  .description('Initialize new workspace with directory structure')
  .action(cmdInit);

// Setup command
program
  .command('setup')
  .option('--target <dir>', 'Target directory (default: .metaHub)')
  .option('--check-only', 'Check dependencies without installing')
  .option('--type <type>', 'Install type: npm or pip (default: npm)')
  .description('Setup dependencies (npm or pip)')
  .action(cmdSetup);

program.parse();
