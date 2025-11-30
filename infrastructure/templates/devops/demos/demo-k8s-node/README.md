# {{PROJECT_NAME}}

A production-ready Node.js service with CI/CD, Kubernetes, Helm, and monitoring.

## Structure

```
{{PROJECT_NAME}}/
├── service/           # Node.js application
│   ├── src/          # TypeScript source
│   ├── Dockerfile    # Container image
│   └── package.json  # Dependencies
├── ci/               # CI/CD workflows
├── k8s/              # Kubernetes manifests
├── helm/             # Helm chart
└── monitoring/       # Prometheus configuration
```

## Quick Start

### Local Development

```bash
cd service
npm install
npm run dev
```

### Build Docker Image

```bash
cd service
docker build -t {{REGISTRY}}/{{PROJECT_NAME}}:{{IMAGE_TAG}} .
```

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/
```

### Deploy with Helm

```bash
helm install {{PROJECT_NAME}} ./helm
```

## Endpoints

- `GET /` - Service info
- `GET /health` - Health check

## Configuration

Environment variables:
- `PORT` - Server port (default: 8080)
- `NODE_ENV` - Environment (production/development)

## Monitoring

Prometheus scrape configuration is included in `monitoring/prometheus.yml`.
