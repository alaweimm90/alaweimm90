# Quick Start Guide

Get started with the ORCHEX-KILO integration in under 15 minutes. This guide covers the essential steps to set up and use the integrated system.

## Prerequisites

- Node.js 16 or higher
- ORCHEX CLI installed (`npm install -g @ORCHEX/cli`)
- KILO CLI installed (`npm install -g @kilo/cli`)
- Access to KILO API and valid API key

## Step 1: Installation

Install the integration packages:

```bash
npm install -g @ORCHEX/integrations @kilo/bridge
```

Verify installation:

```bash
ORCHEX --version
kilo --version
```

## Step 2: Initialize Integration

Initialize the integration in your project:

```bash
cd your-project
ORCHEX init --integration kilo
```

This creates the basic configuration file structure.

## Step 3: Configure Connection

Set up the connection to your KILO instance:

```bash
# Configure KILO endpoint
ORCHEX config set kilo.endpoint "https://kilo-api.yourcompany.com"

# Set API key (use environment variable for security)
export KILO_API_KEY="your-api-key-here"
ORCHEX config set kilo.apiKey "${KILO_API_KEY}"

# Configure organization (if required)
ORCHEX config set kilo.organization "your-org"
```

## Step 4: Test Connection

Verify the integration is working:

```bash
# Test bridge connectivity
ORCHEX bridge test

# Check bridge status
ORCHEX bridge status
```

You should see both bridges reporting as "active".

## Step 5: First Integrated Analysis

Run your first integrated analysis:

```bash
# Analyze repository with governance checks
ORCHEX analyze repo . --governance-check --format table
```

This command will:

1. Analyze your code with ORCHEX
2. Validate results against KILO policies
3. Display combined results

## Step 6: Try Template Access

Access KILO DevOps templates:

```bash
# List available templates
ORCHEX template list cicd

# Get a GitHub Actions template
ORCHEX template get cicd/github-actions --apply
```

## Step 7: Compliance Checking

Check code compliance:

```bash
# Quick compliance check
ORCHEX compliance check . --format summary

# Detailed security check
ORCHEX compliance check . --policies security --format detailed
```

## Next Steps

### For Development Teams

1. **Set up pre-commit hooks:**

   ```bash
   # Add to .git/hooks/pre-commit
   ORCHEX analyze scan . --governance-check
   ```

2. **Configure CI/CD integration:**

   ```yaml
   # Add to your CI pipeline
   - run: ORCHEX analyze repo . --governance-check --format json
   - run: ORCHEX compliance check . --strict
   ```

3. **Create custom workflows:**
   ```bash
   ORCHEX workflow create ./workflows/dev-checks.json
   ```

### For DevOps Teams

1. **Set up monitoring:**

   ```bash
   ORCHEX config set monitoring.enabled true
   ORCHEX template get monitoring/grafana --apply
   ```

2. **Configure automated remediation:**

   ```bash
   ORCHEX config set integration.autoRemediate true
   ```

3. **Set up governance dashboards:**
   ```bash
   ORCHEX template get dashboard/governance --apply
   ```

## Common Issues

### Connection Problems

```bash
# Check API key
echo $KILO_API_KEY

# Test endpoint connectivity
curl -H "Authorization: Bearer $KILO_API_KEY" https://kilo-api.yourcompany.com/health

# Reset bridge configuration
ORCHEX bridge configure a2k --reset
```

### Permission Issues

```bash
# Check API key permissions
ORCHEX bridge test --auth-only

# Verify organization access
ORCHEX config show kilo.organization

# Update API key
ORCHEX config set kilo.apiKey "new-api-key"
```

### Performance Issues

```bash
# Enable caching
ORCHEX config set bridges.a2k.templates.cacheEnabled true

# Adjust timeouts
ORCHEX config set bridges.a2k.validation.timeoutMs 60000

# Check system resources
ORCHEX doctor --performance
```

## Getting Help

- **Documentation:** See the full documentation in this directory
- **CLI Help:** Run `ORCHEX --help` or `ORCHEX <command> --help`
- **Troubleshooting:** Check `troubleshooting.md` for common issues
- **Community:** Join the ORCHEX-KILO community forums

## Example Project

Here's a complete example for a Node.js project:

```bash
# Initialize project
mkdir my-integrated-app
cd my-integrated-app
npm init -y

# Set up integration
ORCHEX init --integration kilo
ORCHEX config set kilo.endpoint "https://kilo-api.example.com"
export KILO_API_KEY="your-key"
ORCHEX config set kilo.apiKey "${KILO_API_KEY}"

# Create basic app structure
mkdir src
echo "console.log('Hello, ORCHEX-KILO!');" > src/index.js

# Run integrated analysis
ORCHEX analyze repo . --governance-check

# Add CI/CD
ORCHEX template get cicd/github-actions --param.nodeVersion=18 --apply

# Check compliance
ORCHEX compliance check . --format summary

echo "🎉 ORCHEX-KILO integration complete!"
```

This quick start guide should get you productive with the ORCHEX-KILO integration in minutes. For more advanced features, refer to the detailed documentation sections.
