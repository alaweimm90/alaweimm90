# Phase 1: Infrastructure Consolidation

## Objective

Merge scattered infrastructure folders (deploy/, templates/) into .metaHub/ for unified DevOps management.

## Pre-Flight Checks

```bash
# Create backup branch
git checkout -b consolidation-phase-1
git branch backup-pre-phase-1

# Verify clean state
git status

# Document current structure
tree deploy/ templates/ .metaHub/infra/ -L 2 > phase1-before.txt
```

## Execution Steps

### Step 1: Docker Infrastructure

```bash
# Move Docker configurations
git mv deploy/docker/ .metaHub/docker/

# Verify move
ls -la .metaHub/docker/
```

### Step 2: Kubernetes Manifests

```bash
# Move K8s configurations
git mv deploy/kubernetes/ .metaHub/k8s/

# Verify move
ls -la .metaHub/k8s/
```

### Step 3: Terraform Infrastructure

```bash
# Move Terraform configs
git mv deploy/terraform/ .metaHub/infra/terraform/

# Verify move
ls -la .metaHub/infra/terraform/
```

### Step 4: Monitoring Stack

```bash
# Move monitoring configs
git mv deploy/monitoring/ .metaHub/monitoring/

# Verify move
ls -la .metaHub/monitoring/
```

### Step 5: DevOps Templates

```bash
# Move DevOps templates
git mv templates/devops/ .metaHub/templates/devops/

# Verify move
ls -la .metaHub/templates/devops/
```

### Step 6: GitHub Templates

```bash
# Move GitHub org templates
git mv templates/org-github/ .metaHub/templates/org-github/

# Verify move
ls -la .metaHub/templates/org-github/
```

### Step 7: Cleanup Empty Folders

```bash
# Remove empty deploy/ folder
rmdir deploy/

# Remove empty templates/ folder
rmdir templates/
```

## Path Updates Required

### Update package.json

```json
{
  "scripts": {
    "docker:build": "docker-compose -f .metaHub/docker/docker-compose.yml build",
    "k8s:apply": "kubectl apply -f .metaHub/k8s/",
    "terraform:plan": "cd .metaHub/infra/terraform && terraform plan"
  }
}
```

### Update CI/CD Workflows

Search and replace in `.github/workflows/*.yml`:

- `deploy/docker/` → `.metaHub/docker/`
- `deploy/kubernetes/` → `.metaHub/k8s/`
- `deploy/terraform/` → `.metaHub/infra/terraform/`
- `templates/devops/` → `.metaHub/templates/devops/`

### Update Documentation

Search and replace in `docs/**/*.md`:

- `deploy/` → `.metaHub/`
- `templates/` → `.metaHub/templates/`

## Validation Tests

### Test 1: Docker Build

```bash
cd .metaHub/docker/
docker-compose config
```

### Test 2: Kubernetes Validation

```bash
kubectl apply --dry-run=client -f .metaHub/k8s/
```

### Test 3: Terraform Validation

```bash
cd .metaHub/infra/terraform/
terraform init
terraform validate
```

### Test 4: CI/CD Pipeline

```bash
# Trigger test workflow
git add .
git commit -m "Phase 1: Infrastructure consolidation"
git push origin consolidation-phase-1
```

## Rollback Procedure

```bash
git checkout backup-pre-phase-1
git branch -D consolidation-phase-1
```

## Success Criteria

- [ ] All files moved successfully
- [ ] No broken imports or references
- [ ] CI/CD pipelines pass
- [ ] Docker builds work
- [ ] Kubernetes manifests valid
- [ ] Terraform validates
- [ ] Documentation updated
- [ ] Git history preserved

## Commit Message

```
feat: consolidate infrastructure into .metaHub

- Move deploy/docker/ → .metaHub/docker/
- Move deploy/kubernetes/ → .metaHub/k8s/
- Move deploy/terraform/ → .metaHub/infra/terraform/
- Move deploy/monitoring/ → .metaHub/monitoring/
- Move templates/devops/ → .metaHub/templates/devops/
- Move templates/org-github/ → .metaHub/templates/org-github/
- Update all path references in CI/CD and docs
- Remove empty deploy/ and templates/ folders

BREAKING CHANGE: Infrastructure paths updated
```
