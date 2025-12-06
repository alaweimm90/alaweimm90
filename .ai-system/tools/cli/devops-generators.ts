/**
 * DevOps CLI Code Generators
 *
 * Contains template generators for:
 * - Node.js services with CI, K8s, Helm, Prometheus
 */

export interface PlannedFile {
  path: string;
  content: string;
  action: 'ADD' | 'UPDATE';
}

/**
 * Plan Node.js service with CI, K8s, Helm, and Prometheus
 */
export function planNodeService(vars: Record<string, string>): PlannedFile[] {
  const projectName = vars.PROJECT_NAME || 'my-service';
  const registry = vars.REGISTRY || 'ghcr.io';
  const imageTag = vars.IMAGE_TAG || 'latest';

  const files: PlannedFile[] = [];

  // Service source
  files.push({
    path: 'service/src/index.ts',
    content: generateServiceSource(projectName),
    action: 'ADD',
  });

  // Package.json
  files.push({
    path: 'service/package.json',
    content: generatePackageJson(projectName),
    action: 'ADD',
  });

  // tsconfig.json
  files.push({
    path: 'service/tsconfig.json',
    content: generateTsConfig(),
    action: 'ADD',
  });

  // Dockerfile
  files.push({
    path: 'service/Dockerfile',
    content: generateDockerfile(),
    action: 'ADD',
  });

  // CI workflow
  files.push({
    path: 'ci/ci.yml',
    content: generateCiWorkflow(),
    action: 'ADD',
  });

  // K8s deployment
  files.push({
    path: 'k8s/deployment.yaml',
    content: generateK8sDeployment(projectName, registry, imageTag),
    action: 'ADD',
  });

  // K8s service
  files.push({
    path: 'k8s/service.yaml',
    content: generateK8sService(projectName),
    action: 'ADD',
  });

  // Helm Chart.yaml
  files.push({
    path: 'helm/Chart.yaml',
    content: generateHelmChart(projectName),
    action: 'ADD',
  });

  // Helm values.yaml
  files.push({
    path: 'helm/values.yaml',
    content: generateHelmValues(projectName, registry, imageTag),
    action: 'ADD',
  });

  // Prometheus config
  files.push({
    path: 'monitoring/prometheus.yml',
    content: generatePrometheusConfig(projectName),
    action: 'ADD',
  });

  return files;
}

// Individual template generators
function generateServiceSource(projectName: string): string {
  return `import http from 'node:http';

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
`;
}

function generatePackageJson(projectName: string): string {
  return JSON.stringify(
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
  );
}

function generateTsConfig(): string {
  return JSON.stringify(
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
  );
}

function generateDockerfile(): string {
  return `FROM node:20-alpine AS builder
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
`;
}

function generateCiWorkflow(): string {
  return `name: CI

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
`;
}

function generateK8sDeployment(projectName: string, registry: string, imageTag: string): string {
  return `apiVersion: apps/v1
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
`;
}

function generateK8sService(projectName: string): string {
  return `apiVersion: v1
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
`;
}

function generateHelmChart(projectName: string): string {
  return `apiVersion: v2
name: ${projectName}
description: Helm chart for ${projectName}
type: application
version: 0.1.0
appVersion: "1.0.0"
`;
}

function generateHelmValues(projectName: string, registry: string, imageTag: string): string {
  return `replicaCount: 2

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
`;
}

function generatePrometheusConfig(projectName: string): string {
  return `global:
  scrape_interval: 15s

scrape_configs:
  - job_name: '${projectName}'
    static_configs:
      - targets: ['${projectName}:8080']
    metrics_path: /metrics
`;
}
