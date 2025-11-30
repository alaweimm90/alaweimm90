# Common Issues and Solutions

Comprehensive troubleshooting guide for the most frequently encountered issues with ATLAS, including symptoms, root causes, and step-by-step solutions.

---

## Installation and Setup Issues

### "atlas: command not found"

**Symptoms:**
- `atlas --version` returns "command not found"
- ATLAS commands fail to execute
- Auto-completion doesn't work

**Possible Causes:**
1. ATLAS CLI not installed
2. npm global bin path not in PATH
3. Using npx without proper setup
4. Permission issues with global packages

**Solutions:**

**Solution 1: Check Installation**
```bash
# Verify npm global packages location
npm config get prefix

# Check if ATLAS is installed
npm list -g @atlas/cli

# Reinstall if missing
npm install -g @atlas/cli
```

**Solution 2: Fix PATH**
```bash
# Add npm global bin to PATH (Linux/macOS)
echo 'export PATH="$(npm config get prefix)/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Add npm global bin to PATH (Windows)
npm config get prefix
# Add the returned path to your system PATH environment variable
```

**Solution 3: Use npx**
```bash
# Use npx for one-off commands
npx @atlas/cli --version

# Create alias for convenience
alias atlas="npx @atlas/cli"
```

**Solution 4: Fix Permissions (Linux/macOS)**
```bash
# Fix npm permissions
sudo chown -R $(whoami) $(npm config get prefix)/{lib/node_modules,bin,share}
```

### "Node.js version too old"

**Symptoms:**
- Installation fails with version errors
- Runtime errors about unsupported Node.js version
- Features not working as expected

**Possible Causes:**
1. Node.js version below minimum requirement (16.0.0)
2. Using system Node.js instead of managed version
3. PATH pointing to wrong Node.js installation

**Solutions:**

**Solution 1: Update Node.js**
```bash
# Using nvm (recommended)
nvm install 18
nvm use 18

# Using fnm
fnm install 18
fnm use 18

# Direct download from nodejs.org
# Download and install latest LTS version
```

**Solution 2: Check Version**
```bash
# Verify Node.js version
node --version  # Should be 16.0.0 or higher
npm --version   # Should be 7.0.0 or higher

# Check which node is being used
which node
```

**Solution 3: Update PATH**
```bash
# Ensure correct Node.js is first in PATH
export PATH="/usr/local/bin:$PATH"  # Adjust path as needed
```

---

## Agent Registration Issues

### "Agent registration failed: Invalid API key"

**Symptoms:**
- Agent registration returns authentication error
- API key validation fails
- Agent shows as "offline" after registration

**Possible Causes:**
1. Invalid or expired API key
2. API key doesn't have required permissions
3. Environment variable not set correctly
4. Network connectivity issues

**Solutions:**

**Solution 1: Verify API Key**
```bash
# Check if environment variable is set
echo $ANTHROPIC_API_KEY  # For Claude
echo $OPENAI_API_KEY     # For GPT-4
echo $GOOGLE_API_KEY     # For Gemini

# Test API key validity (example for Anthropic)
curl -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "content-type: application/json" \
     https://api.anthropic.com/v1/messages \
     -d '{"model": "claude-3-haiku-20240307", "max_tokens": 1, "messages": [{"role": "user", "content": "test"}]}'
```

**Solution 2: Set Environment Variables**
```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="your-key-here"
export OPENAI_API_KEY="your-key-here"
export GOOGLE_API_KEY="your-key-here"

# Reload shell
source ~/.bashrc
```

**Solution 3: Check API Key Permissions**
```bash
# For Anthropic: Ensure key has access to Claude models
# For OpenAI: Ensure key has access to GPT-4 models
# For Google: Ensure key has access to Gemini API

# Test with minimal request
atlas agent register test-agent \
  --provider anthropic \
  --model claude-3-haiku-20240307 \
  --capabilities code_generation \
  --dry-run
```

### "Agent health check failed"

**Symptoms:**
- Agent shows as "unhealthy" or "offline"
- Tasks fail with agent unavailable errors
- Health checks return errors

**Possible Causes:**
1. API service outages
2. Rate limiting
3. Network connectivity issues
4. Invalid model names or endpoints

**Solutions:**

**Solution 1: Check Service Status**
```bash
# Check provider status pages
# Anthropic: https://status.anthropic.com/
# OpenAI: https://status.openai.com/
# Google: https://status.cloud.google.com/

# Test basic connectivity
curl -I https://api.anthropic.com/v1/messages
curl -I https://api.openai.com/v1/models
curl -I https://generativelanguage.googleapis.com/v1/models
```

**Solution 2: Verify Model Names**
```bash
# Correct model names
atlas agent register claude-sonnet-4 \
  --provider anthropic \
  --model claude-sonnet-4.5  # Correct

atlas agent register gpt-4-turbo \
  --provider openai \
  --model gpt-4-turbo  # Correct

atlas agent register gemini-pro \
  --provider google \
  --model gemini-pro  # Correct
```

**Solution 3: Check Rate Limits**
```bash
# Check current usage
atlas metrics agent <agent-id> --period 1h

# Adjust rate limits
atlas agent update <agent-id> --rate-limit 30

# Wait for rate limit reset
sleep 3600  # Wait 1 hour
```

---

## Task Execution Issues

### "Task timed out"

**Symptoms:**
- Tasks fail with timeout errors
- Long-running tasks get cancelled
- Progress stalls at certain points

**Possible Causes:**
1. Task complexity exceeds timeout limits
2. Network latency issues
3. Agent response delays
4. Insufficient timeout configuration

**Solutions:**

**Solution 1: Increase Timeout**
```bash
# Set longer timeout for complex tasks
atlas task submit --type architecture \
  --description "Design complex system" \
  --timeout 600  # 10 minutes

# Update default timeout
atlas config set task.timeout 600
```

**Solution 2: Break Down Complex Tasks**
```bash
# Instead of one complex task, break into smaller ones
atlas task submit --type code_generation \
  --description "Create user model" \
  --timeout 120

atlas task submit --type code_generation \
  --description "Create authentication service" \
  --timeout 120
```

**Solution 3: Check Network Connectivity**
```bash
# Test network speed
speedtest-cli

# Check DNS resolution
nslookup api.anthropic.com

# Test with different network
# Try switching between WiFi and Ethernet
```

### "Task failed: Rate limit exceeded"

**Symptoms:**
- Tasks fail with rate limit errors
- Multiple tasks queue up
- Agent becomes unresponsive

**Possible Causes:**
1. Too many concurrent requests
2. API provider rate limits
3. Insufficient rate limit configuration

**Solutions:**

**Solution 1: Reduce Concurrent Tasks**
```bash
# Check current load
atlas metrics load-balancing

# Reduce max concurrent tasks per agent
atlas agent update <agent-id> --max-concurrent-tasks 2

# Use sequential execution
atlas task submit --type code_generation --description "task 1" --wait
atlas task submit --type code_generation --description "task 2" --wait
```

**Solution 2: Implement Queuing**
```bash
# Enable task queuing
atlas config set task.queue.enabled true
atlas config set task.queue.max_size 50

# Monitor queue status
atlas task list --status queued
```

**Solution 3: Adjust Rate Limits**
```bash
# Check current rate limits
atlas agent show <agent-id> --field rate_limit

# Adjust rate limits based on your API tier
atlas agent update <agent-id> --rate-limit 50  # requests per minute
```

### "Task result quality is poor"

**Symptoms:**
- Generated code has bugs or issues
- Code review misses important problems
- Refactoring introduces errors

**Possible Causes:**
1. Wrong agent selected for task
2. Insufficient task context
3. Task description too vague
4. Agent model limitations

**Solutions:**

**Solution 1: Provide Better Context**
```bash
# Include more context in task description
atlas task submit --type code_generation \
  --description "Create a REST API endpoint for user registration in Express.js with MongoDB, including input validation, password hashing, and JWT authentication" \
  --context language=javascript,framework=express,database=mongodb \
  --files src/models/User.js src/routes/auth.js
```

**Solution 2: Choose Appropriate Agent**
```bash
# Use Claude for complex tasks requiring reasoning
atlas task submit --type architecture \
  --description "Design microservices architecture" \
  --preferred-providers anthropic

# Use GPT-4 for creative tasks
atlas task submit --type code_generation \
  --description "Create innovative UI component" \
  --preferred-providers openai
```

**Solution 3: Break Down Tasks**
```bash
# Instead of "build entire API", break into steps
atlas task submit --type code_generation \
  --description "Create user model with validation"

atlas task submit --type code_generation \
  --description "Create authentication middleware"

atlas task submit --type code_generation \
  --description "Create user registration endpoint"
```

**Solution 4: Use Code Review**
```bash
# Always review AI-generated code
atlas task submit --type code_review \
  --description "Review generated authentication code for security and best practices" \
  --files src/auth.js
```

---

## Repository Analysis Issues

### "Analysis failed: Unable to parse files"

**Symptoms:**
- Repository analysis fails with parsing errors
- Some files are skipped
- Incomplete analysis results

**Possible Causes:**
1. Unsupported file types or languages
2. Syntax errors in code
3. Large files exceeding size limits
4. Binary files included in analysis

**Solutions:**

**Solution 1: Check Supported Languages**
```bash
# ATLAS supports:
# - JavaScript/TypeScript
# - Python
# - Java
# - C#
# - Go
# - Rust
# - PHP

# Check file types in repository
find . -name "*.js" -o -name "*.ts" -o -name "*.py" | head -10
```

**Solution 2: Exclude Problematic Files**
```bash
# Exclude large or binary files
atlas analyze repo . \
  --exclude "**/node_modules/**" \
  --exclude "**/dist/**" \
  --exclude "**/*.min.js" \
  --exclude "**/*.map"

# Set file size limits
atlas config set analysis.max_file_size "1MB"
```

**Solution 3: Fix Syntax Errors**
```bash
# Check for syntax errors before analysis
npm run lint  # or equivalent
# Fix any reported syntax errors

# Run analysis on specific directories
atlas analyze repo src/ --type quick
```

### "Analysis shows no opportunities"

**Symptoms:**
- Analysis completes but finds no refactoring opportunities
- Chaos scores are unexpectedly low
- Missing expected code smells

**Possible Causes:**
1. Code is already well-structured
2. Analysis configuration too restrictive
3. Files not included in analysis scope

**Solutions:**

**Solution 1: Adjust Analysis Scope**
```bash
# Include more file types
atlas analyze repo . --include "**/*.{js,ts,py,java}"

# Analyze specific directories
atlas analyze repo src/ lib/

# Use full analysis instead of quick
atlas analyze repo . --type full
```

**Solution 2: Check Configuration**
```bash
# Review analysis configuration
atlas config show analysis

# Adjust chaos score thresholds
atlas config set analysis.chaos.threshold 30

# Enable more metrics
atlas config set analysis.metrics.complexity true
atlas config set analysis.metrics.duplication true
```

**Solution 3: Verify File Inclusion**
```bash
# Check which files are being analyzed
atlas analyze repo . --dry-run --verbose

# Manually verify file patterns
ls -la src/**/*.js
```

---

## Refactoring Issues

### "Refactoring validation failed"

**Symptoms:**
- Refactoring operations are rejected
- Safety checks fail
- Tests fail after refactoring

**Possible Causes:**
1. Breaking changes detected
2. Test failures
3. Type errors introduced
4. Insufficient test coverage

**Solutions:**

**Solution 1: Review Safety Checks**
```bash
# Check what safety checks failed
atlas refactor status <refactoring-id> --details

# Run safety checks manually
atlas refactor validate <opportunity-id>
```

**Solution 2: Improve Test Coverage**
```bash
# Check current test coverage
npm run test -- --coverage

# Add missing tests before refactoring
atlas task submit --type testing \
  --description "Add tests for function being refactored" \
  --file-path src/utils/helpers.js
```

**Solution 3: Use Conservative Settings**
```bash
# Apply only low-risk refactorings
atlas refactor apply <opportunity-id> --risk-level low

# Skip breaking change validation (use with caution)
atlas refactor apply <opportunity-id> --skip-breaking-checks
```

### "Refactoring created merge conflicts"

**Symptoms:**
- Refactoring creates PR with conflicts
- Automated PR creation fails
- Manual merge required

**Possible Causes:**
1. Concurrent changes to same files
2. Outdated branch
3. File permissions issues

**Solutions:**

**Solution 1: Sync Branch**
```bash
# Update branch before refactoring
git checkout main
git pull origin main
git checkout -b refactor-branch

# Then apply refactoring
atlas refactor apply <opportunity-id> --create-pr
```

**Solution 2: Apply to Clean Branch**
```bash
# Create clean feature branch
git checkout -b feature/clean-refactor
git reset --hard origin/main

# Apply refactoring
atlas refactor apply <opportunity-id> --no-pr

# Manually create PR after review
```

---

## Cost and Performance Issues

### "Unexpected high costs"

**Symptoms:**
- API costs higher than expected
- Budget alerts triggered
- Tasks using expensive agents unnecessarily

**Possible Causes:**
1. Wrong agent selection for tasks
2. Tasks more complex than estimated
3. Fallback chains increasing costs
4. No cost optimization enabled

**Solutions:**

**Solution 1: Enable Cost Optimization**
```bash
# Enable automatic cost optimization
atlas config set routing.cost_optimization true

# Set cost limits
atlas config set cost.max_per_task 1.0
atlas config set cost.max_per_day 50.0
```

**Solution 2: Optimize Agent Selection**
```bash
# Use cost-effective agents for simple tasks
atlas task submit --type code_generation \
  --description "Create simple utility function" \
  --preferred-providers google

# Set cost-based routing rules
atlas config set routing.prefer_low_cost true
```

**Solution 3: Monitor and Alert**
```bash
# Set up cost alerts
atlas config set cost.alert_threshold 0.8

# Monitor cost trends
atlas metrics costs --period 7d --trend

# Generate cost reports
atlas metrics costs --export costs.json
```

### "System performance degraded"

**Symptoms:**
- Slow response times
- High memory usage
- Tasks timing out frequently

**Possible Causes:**
1. Too many concurrent tasks
2. Large repository analysis
3. Memory leaks
4. Insufficient system resources

**Solutions:**

**Solution 1: Adjust Concurrency**
```bash
# Reduce concurrent tasks
atlas config set task.max_concurrent 5

# Check system resources
atlas system health --resources
```

**Solution 2: Optimize Analysis**
```bash
# Use incremental analysis
atlas analyze repo . --incremental

# Limit analysis scope
atlas analyze repo src/ --exclude "**/test/**"
```

**Solution 3: Monitor Resources**
```bash
# Check memory usage
atlas system health --memory

# Monitor performance
atlas metrics performance --period 1h

# Restart if needed
atlas system restart
```

---

## Integration Issues

### "KILO bridge connection failed"

**Symptoms:**
- Bridge tests fail
- Integration features not working
- Governance validation errors

**Possible Causes:**
1. KILO endpoint misconfigured
2. Authentication issues
3. Network connectivity problems
4. Version compatibility issues

**Solutions:**

**Solution 1: Verify Configuration**
```bash
# Check bridge configuration
atlas bridge show a2k

# Test connectivity
atlas bridge test a2k --verbose

# Check KILO status
curl -I https://your-kilo-endpoint/health
```

**Solution 2: Fix Authentication**
```bash
# Verify API key
echo $KILO_API_KEY

# Test authentication
atlas bridge test a2k --auth-only

# Update credentials
atlas bridge configure a2k --api-key $NEW_KILO_KEY
```

**Solution 3: Check Network**
```bash
# Test basic connectivity
ping your-kilo-endpoint.com

# Check firewall rules
telnet your-kilo-endpoint.com 443

# Try different network
# Switch between VPN and direct connection
```

### "CI/CD integration not working"

**Symptoms:**
- Pipeline fails with ATLAS commands
- Authentication issues in CI
- Different behavior between local and CI

**Possible Causes:**
1. Missing environment variables in CI
2. Different Node.js version in CI
3. Permission issues with CI runner

**Solutions:**

**Solution 1: Set CI Environment Variables**
```yaml
# .github/workflows/atlas.yml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_KEY }}
  ATLAS_CONFIG_FILE: .atlas/ci-config.json
```

**Solution 2: Use CI-Specific Configuration**
```bash
# Create CI-specific config
atlas config profile create ci
atlas config profile set ci task.timeout 300
atlas config profile set ci cost.max_per_task 2.0

# Use in CI
atlas --profile ci analyze repo .
```

**Solution 3: Debug CI Issues**
```bash
# Add debug logging to CI
- run: atlas --verbose --debug analyze repo .

# Check CI environment
- run: node --version && npm --version
- run: atlas --version
- run: atlas system health
```

---

## Getting Help

### Diagnostic Commands

```bash
# System health check
atlas system health --detailed

# Configuration validation
atlas config validate

# Log analysis
atlas logs --filter error --since 1h

# Performance diagnostics
atlas system doctor --full
```

### Support Resources

- **Documentation**: [Full Documentation](../README.md)
- **Community**: [Discord Community](https://discord.gg/atlas-platform)
- **GitHub**: [Issue Tracker](https://github.com/atlas-platform/atlas/issues)
- **Enterprise**: [Support Portal](https://support.atlas-platform.com)

### Creating Support Tickets

```bash
# Generate diagnostic report
atlas system doctor --export diagnostics.json

# Include in support request:
# - diagnostics.json
# - atlas --version output
# - Error messages and logs
# - Steps to reproduce
# - Environment details
```

---

This troubleshooting guide covers the most common issues. If you encounter an issue not listed here, please check the [GitHub issues](https://github.com/atlas-platform/atlas/issues) or create a new issue with detailed information about your problem.</instructions>