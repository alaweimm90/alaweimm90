# ATLAS Kubernetes Deployment

Kubernetes manifests for deploying the ATLAS Multi-Agent Orchestration Platform.

## Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- (Optional) Kustomize
- (Optional) Ingress controller (nginx-ingress)

## Quick Start

### 1. Create Namespace

```bash
kubectl create namespace atlas
```

### 2. Configure Secrets

Edit `secret.yaml` with your API keys (base64 encoded):

```bash
# Encode your API keys
echo -n "your-atlas-api-key" | base64
echo -n "sk-ant-xxx" | base64
echo -n "sk-xxx" | base64
```

### 3. Deploy

Using kubectl:

```bash
kubectl apply -f k8s/atlas/ -n atlas
```

Using kustomize:

```bash
kubectl apply -k k8s/atlas/
```

### 4. Verify

```bash
kubectl get pods -n atlas
kubectl get svc -n atlas
```

## Components

| File                 | Description                         |
| -------------------- | ----------------------------------- |
| `deployment.yaml`    | Main API deployment with 2 replicas |
| `service.yaml`       | ClusterIP and LoadBalancer services |
| `configmap.yaml`     | Environment configuration           |
| `secret.yaml`        | API keys and secrets                |
| `pvc.yaml`           | Persistent storage for SQLite       |
| `hpa.yaml`           | Horizontal Pod Autoscaler           |
| `ingress.yaml`       | Ingress configuration               |
| `kustomization.yaml` | Kustomize configuration             |

## Configuration

### Environment Variables

| Variable                | Description            | Default      |
| ----------------------- | ---------------------- | ------------ |
| `NODE_ENV`              | Environment            | `production` |
| `ATLAS_PORT`            | API port               | `3000`       |
| `ATLAS_STORAGE_BACKEND` | Storage backend        | `sqlite`     |
| `ATLAS_API_KEY`         | API authentication key | Required     |
| `ANTHROPIC_API_KEY`     | Anthropic API key      | Optional     |
| `OPENAI_API_KEY`        | OpenAI API key         | Optional     |
| `GOOGLE_AI_API_KEY`     | Google AI API key      | Optional     |

### Scaling

Default configuration:

- Min replicas: 2
- Max replicas: 10
- CPU target: 70%
- Memory target: 80%

## Accessing the API

### ClusterIP (internal)

```bash
kubectl port-forward svc/atlas-api 3000:80 -n atlas
curl http://localhost:3000/health
```

### LoadBalancer (external)

```bash
kubectl get svc atlas-api-lb -n atlas
# Use the EXTERNAL-IP
curl http://<EXTERNAL-IP>/health
```

### Ingress

Configure `ingress.yaml` with your domain and TLS certificate.

## Monitoring

Check health:

```bash
kubectl exec -it deployment/atlas-api -n atlas -- wget -qO- http://localhost:3000/health
```

Check logs:

```bash
kubectl logs -f deployment/atlas-api -n atlas
```

## Cleanup

```bash
kubectl delete -k k8s/atlas/
# or
kubectl delete namespace atlas
```
