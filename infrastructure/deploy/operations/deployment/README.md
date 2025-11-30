# ATLAS Production Deployment Procedures

This document outlines the procedures for deploying ATLAS to production environments.

## Prerequisites

### Infrastructure Requirements
- Kubernetes cluster (v1.24+) with sufficient resources
- Ingress controller (NGINX recommended)
- cert-manager for SSL certificates
- Storage classes for persistent volumes
- Load balancer for external access

### Required Tools
- `kubectl` configured for target cluster
- `helm` v3.0+
- `terraform` v1.0+ (for infrastructure provisioning)
- Docker registry access
- Git access to deployment repository

### Environment Preparation
1. **Configure DNS**: Set up DNS records for ATLAS endpoints
2. **SSL Certificates**: Obtain or configure certificates for HTTPS
3. **Secrets Management**: Prepare API keys and credentials
4. **Network Security**: Configure firewalls and security groups

## Deployment Methods

### Method 1: Helm Chart Deployment (Recommended)

#### Initial Setup
```bash
# Add required Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add elastic https://helm.elastic.co
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Create namespace
kubectl create namespace atlas-system

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

#### Deploy ATLAS
```bash
# Clone deployment repository
git clone <deployment-repo>
cd deploy/helm

# Update values.yaml with your configuration
cp values.yaml.example values.yaml
# Edit values.yaml with your settings

# Install ATLAS
helm install atlas ./atlas -f values.yaml
```

#### Verify Deployment
```bash
# Check pod status
kubectl get pods -n atlas-system

# Check services
kubectl get svc -n atlas-system

# Check ingress
kubectl get ingress -n atlas-system

# Verify health endpoints
curl https://api.atlas.your-domain.com/health
```

### Method 2: Kubernetes Manifests

#### Deploy Core Infrastructure
```bash
# Apply namespace and RBAC
kubectl apply -f deploy/kubernetes/namespace.yaml

# Apply secrets and configmaps
kubectl apply -f deploy/kubernetes/secrets.yaml
kubectl apply -f deploy/kubernetes/configmap.yaml

# Deploy storage services
kubectl apply -f deploy/kubernetes/storage/
```

#### Deploy ATLAS Services
```bash
# Deploy in order: storage -> orchestration -> execution -> optimization -> api-gateway
kubectl apply -f deploy/kubernetes/storage/
kubectl apply -f deploy/kubernetes/orchestration/
kubectl apply -f deploy/kubernetes/execution/
kubectl apply -f deploy/kubernetes/optimization/
kubectl apply -f deploy/kubernetes/api-gateway/
```

#### Deploy Ingress and Monitoring
```bash
# Deploy ingress
kubectl apply -f deploy/kubernetes/ingress.yaml

# Deploy monitoring stack
kubectl apply -f deploy/monitoring/
```

## Configuration

### Environment Variables
Set the following environment variables in your deployment:

```yaml
# API Gateway
JWT_SECRET: "your-jwt-secret"
API_RATE_LIMIT_REQUESTS: "1000"

# AI Providers
ANTHROPIC_API_KEY: "your-anthropic-key"
OPENAI_API_KEY: "your-openai-key"
GOOGLE_API_KEY: "your-google-key"

# Database
POSTGRES_PASSWORD: "your-db-password"
REDIS_PASSWORD: "your-redis-password"
```

### Secrets Management
Use Kubernetes secrets or external secret management:

```bash
# Create secrets from files
kubectl create secret generic atlas-secrets \
  --from-literal=jwt-secret=your-jwt-secret \
  --from-literal=anthropic-api-key=your-anthropic-key \
  --from-file=ssl-cert=cert.pem \
  --from-file=ssl-key=key.pem
```

## Scaling Configuration

### Horizontal Pod Autoscaling
ATLAS services are configured with HPA for automatic scaling:

```bash
# Check HPA status
kubectl get hpa -n atlas-system

# Manually scale if needed
kubectl scale deployment atlas-api-gateway --replicas=5 -n atlas-system
```

### Resource Limits
Adjust resource requests/limits based on load:

```yaml
resources:
  requests:
    memory: "512Mi"
    cpu: "500m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Rollback Procedures

### Helm Rollback
```bash
# Check release history
helm history atlas -n atlas-system

# Rollback to previous version
helm rollback atlas 1 -n atlas-system

# Rollback to specific version
helm rollback atlas <revision> -n atlas-system
```

### Manual Rollback
```bash
# Scale down new deployment
kubectl scale deployment atlas-api-gateway-v2 --replicas=0 -n atlas-system

# Scale up previous deployment
kubectl scale deployment atlas-api-gateway-v1 --replicas=3 -n atlas-system

# Update service selector
kubectl patch svc atlas-api-gateway -p '{"spec":{"selector":{"version":"v1"}}}'
```

## Post-Deployment Verification

### Health Checks
```bash
# Check all services are running
kubectl get pods -n atlas-system

# Verify API endpoints
curl -k https://api.atlas.your-domain.com/health
curl -k https://api.atlas.your-domain.com/v1/tasks

# Check database connectivity
kubectl exec -it atlas-storage-0 -n atlas-system -- psql -U atlas -d atlas -c "SELECT 1"
```

### Monitoring Setup
```bash
# Access Grafana
kubectl port-forward svc/grafana 3000:80 -n monitoring
# Open http://localhost:3000

# Access Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Open http://localhost:9090

# Check alerts
kubectl get alertmanager -n monitoring
```

### Load Testing
```bash
# Run load tests
ab -n 1000 -c 10 https://api.atlas.your-domain.com/health

# Monitor performance during load
kubectl logs -f deployment/atlas-api-gateway -n atlas-system
```

## Troubleshooting

### Common Issues

#### Pods Not Starting
```bash
# Check pod status
kubectl describe pod <pod-name> -n atlas-system

# Check logs
kubectl logs <pod-name> -n atlas-system

# Check events
kubectl get events -n atlas-system --sort-by=.metadata.creationTimestamp
```

#### Service Unavailable
```bash
# Check service endpoints
kubectl get endpoints -n atlas-system

# Check service configuration
kubectl describe svc <service-name> -n atlas-system
```

#### Ingress Issues
```bash
# Check ingress status
kubectl describe ingress atlas-api-gateway-ingress -n atlas-system

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller
```

## Maintenance Procedures

### Certificate Renewal
```bash
# Check certificate status
kubectl get certificate -n atlas-system

# Renew certificates
kubectl delete certificate atlas-tls-cert -n atlas-system
kubectl apply -f deploy/kubernetes/ingress.yaml
```

### Log Rotation
```bash
# Check log sizes
kubectl exec -it <pod-name> -n atlas-system -- du -h /app/logs

# Rotate logs (if using sidecar)
kubectl delete pod <pod-name> -n atlas-system
```

### Database Maintenance
```bash
# Create backup
kubectl exec -it atlas-postgres-0 -n atlas-system -- pg_dump -U atlas atlas > backup.sql

# Run maintenance
kubectl exec -it atlas-postgres-0 -n atlas-system -- vacuumdb -U atlas --analyze atlas
```

## Security Considerations

### Network Policies
```bash
# Apply network policies
kubectl apply -f deploy/security/network-policies.yaml
```

### RBAC Configuration
```bash
# Create service accounts
kubectl apply -f deploy/security/rbac.yaml
```

### Secret Rotation
```bash
# Rotate API keys
kubectl create secret generic atlas-secrets-new --from-literal=anthropic-api-key=new-key
kubectl patch deployment atlas-execution -p '{"spec":{"template":{"spec":{"containers":[{"name":"execution","env":[{"name":"ANTHROPIC_API_KEY","valueFrom":{"secretKeyRef":{"name":"atlas-secrets-new","key":"anthropic-api-key"}}}]}}]}}}}
```

## Performance Optimization

### Resource Tuning
- Monitor resource usage with Grafana dashboards
- Adjust HPA thresholds based on observed patterns
- Configure appropriate resource requests/limits

### Database Optimization
- Monitor query performance
- Adjust connection pool settings
- Implement proper indexing

### Caching Strategy
- Configure Redis for session and API response caching
- Implement application-level caching for frequently accessed data

## Backup and Recovery

See `backup-recovery/README.md` for detailed procedures.

## Support

For deployment issues:
1. Check monitoring dashboards for alerts
2. Review pod logs for error messages
3. Consult troubleshooting guides
4. Contact platform team for assistance

## Change Management

### Deployment Approval
- All production deployments require approval
- Use CI/CD pipelines for automated deployments
- Maintain deployment logs and change records

### Version Control
- Tag releases in Git
- Maintain deployment manifests in version control
- Document all configuration changes