import fs from 'node:fs';
import path from 'node:path';
import { resolveTargetDir, parsePlaceholders, parseFlag, parseOption } from './config.js';
import { ensureDir } from './fs.js';

interface PlannedFile {
  path: string;
  content: string;
  action: 'ADD' | 'UPDATE';
}

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
  console.log('\nUse --dry-run=false to apply changes');
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

    const exists = fs.existsSync(fullPath);
    const action = exists ? 'UPDATE' : 'ADD';
    console.log(`  [${action}] ${file.path}`);
  }

  console.log(`\nApplied ${files.length} files to ${targetDir}`);
}

/**
 * Main CLI entry point
 */
function main(): void {
  const args = process.argv.slice(2);

  const targetDir = resolveTargetDir(args);
  const action = parseOption(args, 'action');
  const dryRun = parseFlag(args, 'dry-run', true);
  const vars = parsePlaceholders(args);

  if (!action) {
    console.log('Usage: coder.ts --action=<action> [--dry-run=true|false] [VARS...]');
    console.log('\nActions:');
    console.log('  node-service    Generate Node.js service with CI, K8s, Helm, Prometheus');
    console.log('\nVariables:');
    console.log('  PROJECT_NAME=   Service name');
    console.log('  REGISTRY=       Container registry');
    console.log('  IMAGE_TAG=      Image tag');
    return;
  }

  let files: PlannedFile[] = [];

  switch (action) {
    case 'node-service':
      files = planNodeService(vars);
      break;
    default:
      console.error(`Unknown action: ${action}`);
      process.exit(1);
  }

  if (dryRun) {
    previewChanges(files, targetDir);
  } else {
    applyChanges(files, targetDir);
  }
}

main();
