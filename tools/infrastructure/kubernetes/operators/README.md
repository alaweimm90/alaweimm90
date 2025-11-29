# SuperTool Kubernetes Operator

A Kubernetes operator for managing SuperTool deployments using the SuperTool Custom Resource.

## Overview

The SuperTool Operator automates the deployment and management of SuperTool instances in Kubernetes. It watches for `SuperTool` Custom Resources and automatically creates and manages all necessary Kubernetes resources.

## Architecture

```
SuperTool CR → Operator → [Deployment, Service, HPA, Ingress, NetworkPolicy, etc.]
```

### Managed Resources

When you create a `SuperTool` CR, the operator automatically creates and manages:

1. **Deployment**: Main application deployment
2. **Service**: ClusterIP and/or LoadBalancer services
3. **HPA**: Horizontal Pod Autoscaler (if enabled)
4. **PDB**: Pod Disruption Budget
5. **Ingress**: Ingress resource (if enabled)
6. **NetworkPolicy**: Network policies (if enabled)
7. **ServiceAccount**: Dedicated service account
8. **ConfigMap**: Application configuration
9. **Secret**: Application secrets (from templates)
10. **ServiceMonitor**: Prometheus monitoring (if enabled)
11. **PVC**: Persistent storage (if enabled)

## Installation

### Prerequisites

- Kubernetes cluster 1.24+
- kubectl configured
- Helm 3+ (optional)

### Install CRD

```bash
kubectl apply -f devops/kubernetes/crds/supertool-crd.yaml
```

### Install Operator

```bash
# Create operator namespace
kubectl create namespace supertool-system

# Deploy operator
kubectl apply -f devops/kubernetes/operators/supertool-operator.yaml
```

### Verify Installation

```bash
# Check operator is running
kubectl get pods -n supertool-system

# Check CRD is installed
kubectl get crd supertools.devops.supertool.io
```

## Usage

### Create a SuperTool Instance

```bash
kubectl apply -f devops/kubernetes/crds/supertool-example.yaml
```

### Check Status

```bash
# List all SuperTool instances
kubectl get supertools -A

# Get detailed status
kubectl get supertool supertool-production -n production -o yaml

# Check operator logs
kubectl logs -n supertool-system -l app.kubernetes.io/name=supertool-operator
```

### Update Configuration

Edit the SuperTool CR:

```bash
kubectl edit supertool supertool-production -n production
```

The operator will automatically reconcile the changes.

### Scale Instances

```bash
# Using kubectl scale
kubectl scale supertool supertool-production -n production --replicas=5

# Or edit the CR
kubectl patch supertool supertool-production -n production \
  --type='json' -p='[{"op": "replace", "path": "/spec/replicas", "value": 5}]'
```

### Delete Instance

```bash
kubectl delete supertool supertool-production -n production
```

The operator will automatically clean up all managed resources.

## Operator Features

### Reconciliation Loop

The operator continuously watches for:

- SuperTool CR creation/updates/deletion
- Changes to managed resources
- Drift detection and auto-remediation

### Status Tracking

The operator maintains status in the SuperTool CR:

```yaml
status:
  phase: Running
  replicas: 3
  readyReplicas: 3
  availableReplicas: 3
  conditions:
    - type: Ready
      status: 'True'
      lastTransitionTime: '2025-11-28T10:00:00Z'
      reason: DeploymentReady
      message: 'All pods are ready'
```

### Finalizers

The operator uses finalizers to ensure proper cleanup:

- Removes all managed resources
- Cleans up PVCs (based on retention policy)
- Removes finalizers after cleanup

### Leader Election

Multiple operator replicas can run with leader election enabled for high availability.

## Monitoring

### Metrics

The operator exposes Prometheus metrics on port 8080:

- `supertool_reconcile_total`: Total reconciliations
- `supertool_reconcile_errors_total`: Total reconciliation errors
- `supertool_reconcile_duration_seconds`: Reconciliation duration

Access metrics:

```bash
kubectl port-forward -n supertool-system \
  svc/supertool-operator-metrics 8080:8080

curl http://localhost:8080/metrics
```

### Health Checks

Health endpoints on port 8081:

- `/healthz`: Liveness probe
- `/readyz`: Readiness probe

## Development

### Building the Operator

The operator is built using Kubebuilder/Operator SDK:

```bash
# Initialize project
operator-sdk init --domain supertool.io --repo github.com/username/supertool-operator

# Create API
operator-sdk create api --group devops --version v1alpha1 --kind SuperTool --resource --controller

# Generate manifests
make manifests

# Build and push image
make docker-build docker-push IMG=ghcr.io/username/supertool-operator:1.0.0

# Deploy
make deploy IMG=ghcr.io/username/supertool-operator:1.0.0
```

### Controller Logic (Pseudocode)

```go
func (r *SuperToolReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    // 1. Fetch SuperTool CR
    superTool := &devopsv1alpha1.SuperTool{}
    if err := r.Get(ctx, req.NamespacedName, superTool); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }

    // 2. Handle deletion with finalizers
    if !superTool.DeletionTimestamp.IsZero() {
        return r.handleDeletion(ctx, superTool)
    }

    // 3. Ensure finalizer exists
    if !containsString(superTool.Finalizers, finalizerName) {
        superTool.Finalizers = append(superTool.Finalizers, finalizerName)
        if err := r.Update(ctx, superTool); err != nil {
            return ctrl.Result{}, err
        }
    }

    // 4. Reconcile resources
    if err := r.reconcileDeployment(ctx, superTool); err != nil {
        return ctrl.Result{}, err
    }
    if err := r.reconcileService(ctx, superTool); err != nil {
        return ctrl.Result{}, err
    }
    if superTool.Spec.Autoscaling.Enabled {
        if err := r.reconcileHPA(ctx, superTool); err != nil {
            return ctrl.Result{}, err
        }
    }
    if superTool.Spec.Ingress.Enabled {
        if err := r.reconcileIngress(ctx, superTool); err != nil {
            return ctrl.Result{}, err
        }
    }
    // ... reconcile other resources

    // 5. Update status
    if err := r.updateStatus(ctx, superTool); err != nil {
        return ctrl.Result{}, err
    }

    return ctrl.Result{RequeueAfter: 5 * time.Minute}, nil
}
```

## Troubleshooting

### Operator Not Starting

```bash
# Check operator logs
kubectl logs -n supertool-system -l app.kubernetes.io/name=supertool-operator

# Check RBAC permissions
kubectl auth can-i --as=system:serviceaccount:supertool-system:supertool-operator \
  create deployments
```

### Resources Not Created

```bash
# Check operator events
kubectl get events -n supertool-system

# Check SuperTool status
kubectl describe supertool <name> -n <namespace>

# Check operator logs for errors
kubectl logs -n supertool-system -l app.kubernetes.io/name=supertool-operator --tail=100
```

### Status Not Updating

The operator reconciles every 5 minutes or when resources change. Force reconciliation:

```bash
# Annotate the CR to trigger reconciliation
kubectl annotate supertool <name> -n <namespace> \
  force-reconcile="$(date +%s)" --overwrite
```

## Best Practices

1. **Use Namespaces**: Deploy instances in dedicated namespaces
2. **Resource Limits**: Always set resource requests/limits
3. **Monitoring**: Enable Prometheus monitoring
4. **Persistence**: Use appropriate storage classes
5. **Security**: Enable network policies and pod security standards
6. **High Availability**: Use multiple replicas in production
7. **Backups**: Regularly backup PVCs and configuration

## Examples

### Minimal Configuration

```yaml
apiVersion: devops.supertool.io/v1alpha1
kind: SuperTool
metadata:
  name: supertool-minimal
  namespace: default
spec:
  version: '1.0.0'
  replicas: 1
```

### Production Configuration

```yaml
apiVersion: devops.supertool.io/v1alpha1
kind: SuperTool
metadata:
  name: supertool-production
  namespace: production
spec:
  version: '1.0.0'
  replicas: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
  resources:
    limits:
      cpu: '2000m'
      memory: '4Gi'
    requests:
      cpu: '500m'
      memory: '1Gi'
  ingress:
    enabled: true
    host: supertool.example.com
    tls: true
  monitoring:
    enabled: true
  security:
    podSecurityStandard: restricted
    networkPolicy:
      enabled: true
  persistence:
    enabled: true
    storageClass: fast-ssd
    size: 50Gi
```

## Contributing

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](../../../LICENSE).
