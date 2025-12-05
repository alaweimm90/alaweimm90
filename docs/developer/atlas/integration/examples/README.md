# Workflow Examples

This section provides practical examples of common workflows that leverage the ORCHEX-KILO integration. Each example includes step-by-step instructions, expected outputs, and integration benefits.

## Code Quality Assurance Workflow

### Scenario

A development team wants to ensure all code changes meet quality standards and organizational policies before merging.

### Workflow Steps

1. **Initial Code Analysis**

   ```bash
   ORCHEX analyze repo . --governance-check --format table
   ```

   **Expected Output:**

   ```
   Repository Analysis Results
   ┌─────────────────┬─────────────┬─────────┐
   │ Metric          │ Value       │ Status  │
   ├─────────────────┼─────────────┼─────────┤
   │ Files Analyzed  │ 245         │ info    │
   │ Total Lines     │ 15432       │ info    │
   │ Complexity Score│ 3.2         │ warning │
   │ Chaos Level     │ 2.1         │ good    │
   │ Maintainability │ 7.8         │ good    │
   └─────────────────┴─────────────┴─────────┘

   Governance Check: PASSED
   Compliance Score: 8.5/10
   ```

2. **Compliance Verification**

   ```bash
   ORCHEX compliance check . --policies security,code_quality --format summary
   ```

   **Expected Output:**

   ```
   Compliance Summary
   ==================
   Total Files: 245
   Compliant Files: 238
   Violations: 7

   Policy Breakdown:
   - Security: 2 violations
   - Code Quality: 5 violations

   Recommendations:
   - Address 2 high-severity security issues
   - Fix 3 code quality violations in production code
   ```

3. **Automated Refactoring (if enabled)**
   ```bash
   ORCHEX analyze repo . --auto-refactor --compliance-level standard
   ```

### Integration Benefits

- **Unified Analysis**: Single command combines ORCHEX analysis with KILO governance
- **Policy Enforcement**: Automatic validation against organizational standards
- **Consistent Quality**: Standardized quality checks across all repositories

## CI/CD Pipeline Integration

### Scenario

Integrate quality checks and automated deployment preparation into CI/CD pipelines.

### GitHub Actions Example

```yaml
# .github/workflows/ORCHEX-kilo-pipeline.yml
name: ORCHEX-KILO Quality Gate
on: [push, pull_request]

jobs:
  quality-gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'

      - name: Install ORCHEX-KILO
        run: npm install -g @ORCHEX/cli @kilo/cli

      - name: Configure Integration
        run: |
          ORCHEX config set kilo.endpoint ${{ secrets.KILO_ENDPOINT }}
          ORCHEX config set kilo.apiKey ${{ secrets.KILO_API_KEY }}

      - name: Code Analysis & Governance Check
        run: ORCHEX analyze repo . --governance-check --format json > analysis.json

      - name: Compliance Verification
        run: ORCHEX compliance check . --strict --format json > compliance.json

      - name: Generate Reports
        run: |
          ORCHEX compliance report --output compliance-report.html --format html
          ORCHEX analyze repo . --format summary > analysis-summary.txt

      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: quality-reports
          path: |
            analysis.json
            compliance.json
            compliance-report.html
            analysis-summary.txt

      - name: Quality Gate Check
        run: |
          if [ $(jq '.complianceScore' compliance.json) -lt 7 ]; then
            echo "Quality gate failed: Compliance score too low"
            exit 1
          fi

  deploy-prep:
    needs: quality-gate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Generate Deployment Templates
        run: |
          ORCHEX template get cicd/github-actions --param.appName=my-app --apply
          ORCHEX template get k8s/deployment --param.namespace=production --apply

      - name: Validate Deployment Configuration
        run: ORCHEX compliance check ./k8s/ --policies security --format summary
```

### Jenkins Pipeline Example

```groovy
// Jenkinsfile
pipeline {
    agent any

    stages {
        stage('Quality Analysis') {
            steps {
                script {
                    // Install ORCHEX-KILO CLI
                    sh 'npm install -g @ORCHEX/cli @kilo/cli'

                    // Configure integration
                    sh 'ORCHEX config set kilo.endpoint $KILO_ENDPOINT'
                    sh 'ORCHEX config set kilo.apiKey $KILO_API_KEY'

                    // Run integrated analysis
                    sh 'ORCHEX analyze repo . --governance-check --format json > analysis.json'

                    // Archive results
                    archiveArtifacts artifacts: 'analysis.json', fingerprint: true
                }
            }
        }

        stage('Compliance Check') {
            steps {
                script {
                    def complianceResult = sh(
                        script: 'ORCHEX compliance check . --strict --format json',
                        returnStdout: true
                    ).trim()

                    def compliance = readJSON text: complianceResult
                    if (compliance.complianceScore < 7.0) {
                        error "Compliance score ${compliance.complianceScore} is below threshold 7.0"
                    }
                }
            }
        }

        stage('Generate Deployment') {
            when {
                branch 'main'
            }
            steps {
                sh 'ORCHEX template get k8s/deployment --param.namespace=prod --apply'
                sh 'ORCHEX compliance check ./k8s/ --policies security'
            }
        }
    }

    post {
        always {
            sh 'ORCHEX compliance report --output compliance-report.html --format html'
            publishHTML([
                allowMissing: false,
                alwaysLinkToLastBuild: true,
                keepAll: true,
                reportDir: '.',
                reportFiles: 'compliance-report.html',
                reportName: 'Compliance Report'
            ])
        }
    }
}
```

## Development Workflow Automation

### Scenario

Automate common development tasks with integrated quality checks.

### Pre-commit Hook Setup

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running ORCHEX-KILO pre-commit checks..."

# Run quick analysis
ORCHEX analyze scan . --health-check > /dev/null
if [ $? -ne 0 ]; then
    echo "Repository health check failed"
    exit 1
fi

# Check compliance for staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.js\|\.ts\|\.py' | tr '\n' ',')
if [ ! -z "$STAGED_FILES" ]; then
    ORCHEX compliance check $STAGED_FILES --policies security,code_quality --format summary
    if [ $? -ne 0 ]; then
        echo "Compliance check failed. Please fix violations before committing."
        exit 1
    fi
fi

echo "Pre-commit checks passed!"
```

### IDE Integration Script

```bash
#!/bin/bash
# scripts/dev-workflow.sh

PROJECT_PATH=${1:-"."}
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)

echo "=== ORCHEX-KILO Development Workflow ==="
echo "Project: $PROJECT_PATH"
echo "Branch: $BRANCH_NAME"
echo

# Step 1: Repository analysis
echo "Step 1: Analyzing repository..."
ORCHEX analyze repo "$PROJECT_PATH" --format summary

# Step 2: Compliance check
echo -e "\nStep 2: Checking compliance..."
ORCHEX compliance check "$PROJECT_PATH" --format summary

# Step 3: Generate missing templates (if any)
echo -e "\nStep 3: Checking for required templates..."
if [ ! -f ".github/workflows/ci.yml" ]; then
    echo "Generating CI/CD workflow..."
    ORCHEX template get cicd/github-actions --apply
fi

if [ ! -f "Dockerfile" ]; then
    echo "Generating Dockerfile..."
    ORCHEX template get cicd/dockerfile --param.language=typescript --apply
fi

# Step 4: Final validation
echo -e "\nStep 4: Final validation..."
ORCHEX bridge status --health-check

echo -e "\n=== Workflow Complete ==="
echo "Ready for development or deployment!"
```

## Infrastructure as Code Workflow

### Scenario

Generate and validate infrastructure templates with compliance checking.

### Kubernetes Deployment Setup

```bash
# 1. Analyze current infrastructure needs
ORCHEX analyze repo . --include-patterns "*.yml,*.yaml,Dockerfile" --format json

# 2. Generate Kubernetes manifests
ORCHEX template get k8s/deployment --param.appName=my-app --param.replicas=3 --apply
ORCHEX template get k8s/service --param.appName=my-app --param.port=8080 --apply
ORCHEX template get k8s/configmap --param.appName=my-app --apply

# 3. Validate infrastructure compliance
ORCHEX compliance check ./k8s/ --policies security --format detailed

# 4. Generate monitoring setup
ORCHEX template get monitoring/prometheus --param.namespace=monitoring --apply
ORCHEX template get monitoring/grafana --param.namespace=monitoring --apply

# 5. Final infrastructure validation
ORCHEX compliance check ./k8s/ ./monitoring/ --policies security,performance --strict
```

### AWS Infrastructure Setup

```bash
# Generate CloudFormation templates
ORCHEX template get iac/cloudformation/vpc --param.environment=production --apply
ORCHEX template get iac/cloudformation/ecs-cluster --param.clusterName=my-cluster --apply

# Validate infrastructure code
ORCHEX compliance check ./iac/ --policies security,performance --format json

# Generate deployment pipeline
ORCHEX template get cicd/github-actions --param.deployTarget=aws --param.region=us-west-2 --apply

# Final compliance check
ORCHEX compliance report --path ./iac/ --format html --output infra-compliance.html
```

## Security-First Development

### Scenario

Implement security-focused development practices with automated validation.

### Security Analysis Workflow

```bash
#!/bin/bash
# scripts/security-audit.sh

echo "=== ORCHEX-KILO Security Audit ==="

# 1. Security-focused repository analysis
echo "Running security analysis..."
ORCHEX analyze repo . --depth deep --include-patterns "*.js,*.ts,*.py,*.java" --format table

# 2. Comprehensive security compliance check
echo -e "\nChecking security compliance..."
ORCHEX compliance check . --policies security --strict --format detailed

# 3. Dependency vulnerability scan
echo -e "\nScanning dependencies..."
ORCHEX analyze repo . --governance-check --format json | jq '.vulnerabilities[]?'

# 4. Generate security recommendations
echo -e "\nGenerating security recommendations..."
ORCHEX compliance report --policies security --format markdown --output SECURITY-RECOMMENDATIONS.md

# 5. Check for security templates
echo -e "\nChecking security infrastructure..."
if [ ! -f "security/scan.yml" ]; then
    ORCHEX template get security/trivy-scan --apply
fi

if [ ! -f "security/audit-log.sh" ]; then
    ORCHEX template get security/audit-logging --apply
fi

echo "=== Security Audit Complete ==="
```

### Automated Security Fixes

```bash
# Find and fix common security issues
ORCHEX compliance check . --policies security --fix --format summary

# Apply security hardening templates
ORCHEX template get security/hardening --param.level=high --apply

# Validate security improvements
ORCHEX compliance check . --policies security --format json > security-before.json
# ... apply fixes ...
ORCHEX compliance check . --policies security --format json > security-after.json

# Compare results
echo "Security improvements:"
jq -r '.complianceScore' security-before.json security-after.json
```

## Multi-Repository Governance

### Scenario

Apply consistent governance across multiple repositories.

### Organization-Wide Analysis

```bash
#!/bin/bash
# scripts/org-analysis.sh

REPOS=(
    "repo1"
    "repo2"
    "repo3"
)

echo "=== Organization Analysis ==="

for repo in "${REPOS[@]}"; do
    echo "Analyzing $repo..."
    cd "$repo"

    # Run integrated analysis
    ORCHEX analyze repo . --governance-check --format json > "../reports/$repo-analysis.json"

    # Generate compliance report
    ORCHEX compliance report --output "../reports/$repo-compliance.html" --format html

    cd ..
done

# Aggregate results
echo "Generating organization summary..."
ORCHEX config set output.format json
ORCHEX analyze repo ./repos --format summary > org-summary.json

echo "=== Organization Analysis Complete ==="
```

### Governance Dashboard Setup

```bash
# Generate governance dashboard
ORCHEX template get dashboard/governance --param.repos=repo1,repo2,repo3 --apply

# Set up monitoring
ORCHEX template get monitoring/grafana --param.dashboard=governance --apply

# Configure alerts
ORCHEX config set monitoring.alerts.governance.enabled true
ORCHEX config set monitoring.alerts.governance.threshold 7.0

# Validate dashboard setup
ORCHEX compliance check ./dashboard/ --policies security
```

## Performance Optimization Workflow

### Scenario

Identify and fix performance issues with automated refactoring.

### Performance Analysis and Optimization

```bash
# 1. Performance analysis
ORCHEX analyze complexity . --threshold 10 --format detailed
ORCHEX analyze chaos . --detailed --format json

# 2. Identify performance bottlenecks
ORCHEX analyze repo . --depth deep --include-patterns "*.js,*.ts" --format json | \
    jq '.performanceIssues[] | select(.severity == "high")'

# 3. Apply performance optimizations
ORCHEX analyze repo . --auto-refactor --focus performance

# 4. Validate improvements
ORCHEX analyze repo . --format json > performance-after.json

# 5. Generate performance report
ORCHEX compliance report --policies performance --format html --output performance-report.html
```

### Database Optimization

```bash
# Analyze database queries and schemas
ORCHEX template get db/postgres --param.optimization=performance --apply

# Check database compliance
ORCHEX compliance check ./db/ --policies performance --format detailed

# Generate optimization recommendations
ORCHEX analyze repo . --include-patterns "migrations/*.sql" --governance-check
```

## Custom Workflow Creation

### Scenario

Create reusable workflows for specific project needs.

### Workflow Definition

```json
// workflows/custom-quality-gate.json
{
  "name": "custom-quality-gate",
  "description": "Custom quality gate for Node.js projects",
  "steps": [
    {
      "name": "dependency-check",
      "command": "ORCHEX compliance check package.json --policies security",
      "required": true
    },
    {
      "name": "code-analysis",
      "command": "ORCHEX analyze repo . --depth medium --governance-check",
      "required": true
    },
    {
      "name": "test-coverage",
      "command": "ORCHEX analyze repo . --include-patterns 'test/**/*.js' --format json",
      "required": false
    },
    {
      "name": "generate-report",
      "command": "ORCHEX compliance report --output quality-report.html --format html",
      "required": false
    }
  ],
  "triggers": {
    "pre-commit": true,
    "pre-push": true,
    "ci": true
  }
}
```

### Workflow Execution

```bash
# Run custom workflow
ORCHEX workflow run custom-quality-gate

# Run with parameters
ORCHEX workflow run custom-quality-gate --param.minCoverage=80

# List available workflows
ORCHEX workflow list

# Create new workflow
ORCHEX workflow create ./workflows/my-workflow.json
```

## Integration Testing Workflow

### Scenario

Ensure integration components work correctly together.

### Bridge Testing Workflow

```bash
#!/bin/bash
# scripts/test-integration.sh

echo "=== Integration Testing ==="

# 1. Test bridge connectivity
echo "Testing bridge status..."
ORCHEX bridge status --health-check --format json

# 2. Test KILO connectivity
echo "Testing KILO integration..."
ORCHEX config test --component kilo

# 3. Test ORCHEX functionality
echo "Testing ORCHEX analysis..."
ORCHEX analyze scan . --format json > /dev/null

# 4. Test integrated workflow
echo "Testing integrated analysis..."
ORCHEX analyze repo . --governance-check --format json > integration-test.json

# 5. Validate results
echo "Validating integration results..."
if [ $(jq '.governanceCheck.passed' integration-test.json) = "true" ]; then
    echo "✓ Integration test passed"
else
    echo "✗ Integration test failed"
    exit 1
fi

echo "=== Integration Testing Complete ==="
```

### End-to-End Testing

```bash
# Full integration test
ORCHEX workflow run integration-test

# Performance testing
ORCHEX bridge test --performance --report integration-performance.json

# Load testing
ORCHEX bridge test --load-test --duration 300 --concurrency 10
```

These examples demonstrate the power and flexibility of the ORCHEX-KILO integration for various development scenarios. Each workflow can be customized and extended based on specific project requirements.
