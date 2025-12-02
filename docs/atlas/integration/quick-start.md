# Quick Start Guide

Get started with the ATLAS-KILO integration in under 15 minutes. This guide covers the essential steps to set up and use the integrated system.

## Prerequisites

- Node.js 16 or higher
- ATLAS CLI installed (`npm install -g @atlas/cli`)
- KILO CLI installed (`npm install -g @kilo/cli`)
- Access to KILO API and valid API key

## Step 1: Installation

Install the integration packages:

```bash
npm install -g @atlas/integrations @kilo/bridge
```

Verify installation:

```bash
atlas --version
kilo --version
```

## Step 2: Initialize Integration

Initialize the integration in your project:

```bash
cd your-project
atlas init --integration kilo
```

This creates the basic configuration file structure.

## Step 3: Configure Connection

Set up the connection to your KILO instance:

```bash
# Configure KILO endpoint
atlas config set kilo.endpoint "https://kilo-api.yourcompany.com"

# Set API key (use environment variable for security)
export KILO_API_KEY="your-api-key-here"
atlas config set kilo.apiKey "${KILO_API_KEY}"

# Configure organization (if required)
atlas config set kilo.organization "your-org"
```

## Step 4: Test Connection

Verify the integration is working:

```bash
# Test bridge connectivity
atlas bridge test

# Check bridge status
atlas bridge status
```

You should see both bridges reporting as "active".

## Step 5: First Integrated Analysis

Run your first integrated analysis:

```bash
# Analyze repository with governance checks
atlas analyze repo . --governance-check --format table
```

This command will:

1. Analyze your code with ATLAS
2. Validate results against KILO policies
3. Display combined results

## Step 6: Try Template Access

Access KILO DevOps templates:

```bash
# List available templates
atlas template list cicd

# Get a GitHub Actions template
atlas template get cicd/github-actions --apply
```

## Step 7: Compliance Checking

Check code compliance:

```bash
# Quick compliance check
atlas compliance check . --format summary

# Detailed security check
atlas compliance check . --policies security --format detailed
```

## Next Steps

### For Development Teams

1. **Set up pre-commit hooks:**

   ```bash
   # Add to .git/hooks/pre-commit
   atlas analyze scan . --governance-check
   ```

2. **Configure CI/CD integration:**

   ```yaml
   # Add to your CI pipeline
   - run: atlas analyze repo . --governance-check --format json
   - run: atlas compliance check . --strict
   ```

3. **Create custom workflows:**
   ```bash
   atlas workflow create ./workflows/dev-checks.json
   ```

### For DevOps Teams

1. **Set up monitoring:**

   ```bash
   atlas config set monitoring.enabled true
   atlas template get monitoring/grafana --apply
   ```

2. **Configure automated remediation:**

   ```bash
   atlas config set integration.autoRemediate true
   ```

3. **Set up governance dashboards:**
   ```bash
   atlas template get dashboard/governance --apply
   ```

## Common Issues

### Connection Problems

```bash
# Check API key
echo $KILO_API_KEY

# Test endpoint connectivity
curl -H "Authorization: Bearer $KILO_API_KEY" https://kilo-api.yourcompany.com/health

# Reset bridge configuration
atlas bridge configure a2k --reset
```

### Permission Issues

```bash
# Check API key permissions
atlas bridge test --auth-only

# Verify organization access
atlas config show kilo.organization

# Update API key
atlas config set kilo.apiKey "new-api-key"
```

### Performance Issues

```bash
# Enable caching
atlas config set bridges.a2k.templates.cacheEnabled true

# Adjust timeouts
atlas config set bridges.a2k.validation.timeoutMs 60000

# Check system resources
atlas doctor --performance
```

## Getting Help

- **Documentation:** See the full documentation in this directory
- **CLI Help:** Run `atlas --help` or `atlas <command> --help`
- **Troubleshooting:** Check `troubleshooting.md` for common issues
- **Community:** Join the ATLAS-KILO community forums

## Example Project

Here's a complete example for a Node.js project:

```bash
# Initialize project
mkdir my-integrated-app
cd my-integrated-app
npm init -y

# Set up integration
atlas init --integration kilo
atlas config set kilo.endpoint "https://kilo-api.example.com"
export KILO_API_KEY="your-key"
atlas config set kilo.apiKey "${KILO_API_KEY}"

# Create basic app structure
mkdir src
echo "console.log('Hello, ATLAS-KILO!');" > src/index.js

# Run integrated analysis
atlas analyze repo . --governance-check

# Add CI/CD
atlas template get cicd/github-actions --param.nodeVersion=18 --apply

# Check compliance
atlas compliance check . --format summary

echo "🎉 ATLAS-KILO integration complete!"
```

This quick start guide should get you productive with the ATLAS-KILO integration in minutes. For more advanced features, refer to the detailed documentation sections.
