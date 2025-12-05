# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the ORCHEX-KILO integration, along with diagnostic procedures and preventive measures.

## Quick Diagnosis

### Health Check Commands

```bash
# Check overall integration status
ORCHEX bridge status --health-check

# Test bridge connectivity
ORCHEX bridge test

# Validate configuration
ORCHEX config validate

# Check service availability
ORCHEX doctor
```

### Common Symptoms and Solutions

## Bridge Connection Issues

### Problem: Bridge Not Responding

**Symptoms:**

- Commands hang or timeout
- Bridge status shows "unhealthy"
- Error: "Bridge communication failed"

**Diagnosis:**

```bash
# Check bridge status
ORCHEX bridge status --detailed

# Test individual bridges
ORCHEX bridge test k2a
ORCHEX bridge test a2k

# Check network connectivity
curl -I https://kilo-api.example.com/health
```

**Solutions:**

1. **Restart Bridge Services**

   ```bash
   ORCHEX bridge restart
   ```

2. **Check Configuration**

   ```bash
   ORCHEX config show bridges
   # Verify endpoint URLs and credentials
   ```

3. **Network Issues**

   ```bash
   # Check firewall settings
   # Verify DNS resolution
   # Test TLS certificates
   ```

4. **Service Dependencies**
   ```bash
   # Ensure KILO API is running
   # Check ORCHEX service status
   # Verify database connectivity
   ```

### Problem: Authentication Failures

**Symptoms:**

- HTTP 401/403 errors
- "Invalid credentials" messages
- Bridge status shows authentication errors

**Diagnosis:**

```bash
# Check API key configuration
echo $KILO_API_KEY

# Test authentication
ORCHEX bridge test --auth-only

# Check token expiration
ORCHEX config show kilo.apiKey
```

**Solutions:**

1. **Update API Keys**

   ```bash
   # Set new API key
   ORCHEX config set kilo.apiKey "new-api-key"

   # Or use environment variable
   export KILO_API_KEY="new-api-key"
   ```

2. **Check Token Permissions**
   - Verify API key has required permissions
   - Check organization access
   - Confirm repository permissions

3. **Certificate Issues**

   ```bash
   # Check certificate validity
   openssl s_client -connect kilo-api.example.com:443

   # Update certificates if needed
   ORCHEX config set kilo.certPath "/path/to/cert.pem"
   ```

## Configuration Issues

### Problem: Configuration Not Loading

**Symptoms:**

- "Configuration file not found" errors
- Settings not taking effect
- Default values being used

**Diagnosis:**

```bash
# Check configuration file location
ls -la ORCHEX.config.json

# Validate JSON syntax
ORCHEX config validate ./ORCHEX.config.json

# Check file permissions
stat ORCHEX.config.json

# Test configuration loading
ORCHEX config show --debug
```

**Solutions:**

1. **File Location Issues**

   ```bash
   # Create configuration file
   ORCHEX init --config-only

   # Or specify custom path
   ORCHEX --config ./my-config.json config show
   ```

2. **JSON Syntax Errors**

   ```bash
   # Validate and fix JSON
   ORCHEX config validate ./ORCHEX.config.json --fix

   # Use online JSON validator
   # Check for trailing commas, missing quotes
   ```

3. **Permission Issues**

   ```bash
   # Fix file permissions
   chmod 644 ORCHEX.config.json

   # Check directory permissions
   ls -ld .
   ```

### Problem: Configuration Overrides Not Working

**Symptoms:**

- Environment variables ignored
- Command-line flags not applied
- Configuration precedence issues

**Diagnosis:**

```bash
# Check precedence order
ORCHEX config show --effective --debug

# Test environment variables
env | grep ATLAS_
env | grep KILO_

# Check command-line parsing
ORCHEX --help | grep config
```

**Solutions:**

1. **Environment Variable Issues**

   ```bash
   # Export variables correctly
   export ATLAS_CONFIG_FILE="./ORCHEX.config.json"
   export KILO_API_KEY="your-key"

   # Use .env file
   echo "KILO_API_KEY=your-key" > .env
   ```

2. **Command-line Precedence**

   ```bash
   # Use correct flag syntax
   ORCHEX --config ./config.json command

   # Check flag parsing
   ORCHEX command --help
   ```

## Analysis and Validation Issues

### Problem: Validation Taking Too Long

**Symptoms:**

- Validation timeouts
- Slow response times
- Performance degradation

**Diagnosis:**

```bash
# Check validation performance
ORCHEX bridge test --performance

# Monitor validation metrics
ORCHEX bridge status --metrics

# Check operation complexity
ORCHEX analyze repo . --depth shallow --format json
```

**Solutions:**

1. **Optimize Validation Settings**

   ```bash
   # Adjust timeout settings
   ORCHEX config set bridges.a2k.validation.timeoutMs 120000

   # Change validation strictness
   ORCHEX config set bridges.a2k.validation.strictness lenient

   # Enable caching
   ORCHEX config set bridges.a2k.templates.cacheEnabled true
   ```

2. **Reduce Operation Scope**

   ```bash
   # Use shallower analysis
   ORCHEX analyze repo . --depth shallow

   # Limit file patterns
   ORCHEX analyze repo . --include-patterns "*.ts,*.js"

   # Exclude large directories
   ORCHEX analyze repo . --exclude-patterns "node_modules/**,dist/**"
   ```

3. **Performance Tuning**

   ```bash
   # Increase connection pool
   ORCHEX config set bridges.a2k.connection.poolSize 10

   # Enable batch processing
   ORCHEX config set bridges.a2k.validation.batchSize 5
   ```

### Problem: False Positive Validations

**Symptoms:**

- Valid code flagged as invalid
- Overly strict validation rules
- Policy conflicts

**Diagnosis:**

```bash
# Check validation rules
ORCHEX config show bridges.a2k.validation

# Test with different strictness levels
ORCHEX bridge test --strictness lenient
ORCHEX bridge test --strictness strict

# Review policy configurations
ORCHEX config show kilo.policies
```

**Solutions:**

1. **Adjust Validation Strictness**

   ```bash
   # Use lenient validation for development
   ORCHEX config set bridges.a2k.validation.strictness lenient

   # Gradually increase strictness
   ORCHEX config set bridges.a2k.validation.strictness standard
   ```

2. **Customize Policies**

   ```bash
   # Override specific policies
   ORCHEX config set kilo.policies.overrides.security.maxPasswordLength 256

   # Disable problematic policies
   ORCHEX config set bridges.a2k.compliance.enabledPolicies '["code_quality", "performance"]'
   ```

3. **Policy Conflicts**

   ```bash
   # Review policy precedence
   ORCHEX compliance check . --policies security --debug

   # Resolve conflicts in KILO
   # Update policy definitions
   ```

## Template Issues

### Problem: Template Not Found

**Symptoms:**

- "Template not found" errors
- Empty template responses
- Template list not showing expected items

**Diagnosis:**

```bash
# List available templates
ORCHEX template list --all

# Check template categories
ORCHEX template list cicd

# Test template retrieval
ORCHEX template get cicd/github-actions --dry-run

# Check template cache
ORCHEX template list --cache-info
```

**Solutions:**

1. **Template Repository Issues**

   ```bash
   # Update template repository
   ORCHEX config set kilo.templates.branch main

   # Refresh template cache
   ORCHEX template refresh

   # Check repository access
   ORCHEX config show kilo.templates.repository
   ```

2. **Template Path Issues**

   ```bash
   # Verify template paths
   ORCHEX config set bridges.a2k.templates.basePath "./templates/devops"

   # Check directory structure
   ls -la templates/devops/
   ```

3. **Version Conflicts**

   ```bash
   # Use latest version
   ORCHEX template get cicd/github-actions --version latest

   # List available versions
   ORCHEX template list cicd/github-actions --versions
   ```

### Problem: Template Parameter Errors

**Symptoms:**

- Template generation fails
- Placeholder replacement issues
- Invalid parameter errors

**Diagnosis:**

```bash
# Validate template parameters
ORCHEX template validate cicd/github-actions --parameters ./params.json

# Check parameter format
cat ./params.json

# Test parameter substitution
ORCHEX template get cicd/github-actions --param.test value --dry-run
```

**Solutions:**

1. **Parameter Format Issues**

   ```json
   // Correct parameter format
   {
     "nodeVersion": "18",
     "testCommand": "npm test",
     "buildCommand": "npm run build"
   }
   ```

2. **Missing Required Parameters**

   ```bash
   # Check template requirements
   ORCHEX template get cicd/github-actions --help

   # Use parameter file
   ORCHEX template get cicd/github-actions --parameters ./ci-params.json
   ```

3. **Placeholder Conflicts**

   ```bash
   # Escape special characters
   ORCHEX template get cicd/github-actions --param.command "npm run build"

   # Use parameter file for complex values
   echo '{"command": "npm run build && npm test"}' > params.json
   ```

## Compliance Checking Issues

### Problem: Compliance Score Inconsistent

**Symptoms:**

- Varying compliance scores for same code
- Unexpected violations
- Compliance check failures

**Diagnosis:**

```bash
# Run compliance check with debug
ORCHEX compliance check . --debug --format json

# Check policy versions
ORCHEX config show kilo.policies.version

# Test with different policies
ORCHEX compliance check . --policies security --format detailed
ORCHEX compliance check . --policies code_quality --format detailed
```

**Solutions:**

1. **Policy Version Issues**

   ```bash
   # Update policy versions
   ORCHEX config set kilo.policies.version latest

   # Refresh policy cache
   ORCHEX compliance refresh-policies
   ```

2. **Inconsistent Rule Application**

   ```bash
   # Standardize rule settings
   ORCHEX config set kilo.policies.strictMode true

   # Review rule conflicts
   ORCHEX compliance check . --conflict-report
   ```

3. **Context-Aware Issues**

   ```bash
   # Provide proper context
   ORCHEX compliance check ./src/auth.js --context framework=express

   # Check framework-specific rules
   ORCHEX compliance check . --framework express
   ```

## Performance Issues

### Problem: Slow Operations

**Symptoms:**

- Commands taking too long
- Timeout errors
- High resource usage

**Diagnosis:**

```bash
# Performance profiling
ORCHEX bridge test --performance --duration 60

# Resource monitoring
ORCHEX bridge status --metrics

# Check system resources
top -p $(pgrep ORCHEX)
free -h
```

**Solutions:**

1. **Caching Optimization**

   ```bash
   # Enable all caches
   ORCHEX config set bridges.a2k.templates.cacheEnabled true
   ORCHEX config set bridges.a2k.validation.cacheEnabled true

   # Increase cache sizes
   ORCHEX config set cache.maxSize "512MB"
   ```

2. **Connection Pooling**

   ```bash
   # Optimize connection settings
   ORCHEX config set bridges.a2k.connection.poolSize 20
   ORCHEX config set bridges.a2k.connection.idleTimeoutMs 300000
   ```

3. **Batch Processing**

   ```bash
   # Enable batch operations
   ORCHEX config set bridges.a2k.validation.batchSize 10
   ORCHEX config set bridges.a2k.templates.batchSize 5
   ```

4. **Resource Limits**
   ```bash
   # Adjust resource limits
   ORCHEX config set maxConcurrentOperations 5
   ORCHEX config set memoryLimit "1GB"
   ```

## Logging and Debugging

### Enable Debug Logging

```bash
# Enable debug mode
export ATLAS_DEBUG=true
export KILO_DEBUG=true

# Run command with debug output
ORCHEX analyze repo . --verbose

# Check debug logs
tail -f ~/.orchex/logs/debug.log
```

### Log Analysis

```bash
# Search for errors
grep "ERROR" ~/.orchex/logs/ORCHEX.log

# Check bridge communication
grep "bridge" ~/.orchex/logs/ORCHEX.log

# Analyze performance
grep "duration" ~/.orchex/logs/ORCHEX.log | sort -n
```

### Common Log Messages

| Log Message                | Meaning                    | Action                               |
| -------------------------- | -------------------------- | ------------------------------------ |
| `Bridge connection failed` | Network/connectivity issue | Check network, restart services      |
| `Validation timeout`       | Operation taking too long  | Increase timeout, optimize operation |
| `Template not found`       | Template repository issue  | Update repository, refresh cache     |
| `Configuration invalid`    | Config file problem        | Validate config, check syntax        |
| `Authentication failed`    | Credential issue           | Update API keys, check permissions   |

## Preventive Maintenance

### Regular Health Checks

```bash
# Daily health check script
#!/bin/bash
ORCHEX bridge status --health-check > health.log
ORCHEX config validate >> health.log
ORCHEX bridge test --quick >> health.log

# Alert on failures
if grep -q "unhealthy\|failed\|error" health.log; then
    echo "Health check failed" | mail -s "ORCHEX-KILO Health Alert" admin@company.com
fi
```

### Configuration Backups

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
ORCHEX config backup "config-backup-$DATE.json"
find . -name "config-backup-*.json" -mtime +30 -delete
```

### Performance Monitoring

```bash
# Performance monitoring script
#!/bin/bash
ORCHEX bridge status --metrics > metrics.json

# Check thresholds
RESPONSE_TIME=$(jq '.avgResponseTime' metrics.json)
if [ "$RESPONSE_TIME" -gt 5000 ]; then
    echo "Performance degraded: ${RESPONSE_TIME}ms" | mail -s "Performance Alert" admin@company.com
fi
```

## Getting Help

### Support Resources

1. **Documentation**
   - Check this troubleshooting guide
   - Review configuration examples
   - Consult API reference

2. **Community Support**
   - GitHub issues
   - Community forums
   - Slack channels

3. **Professional Support**
   - Enterprise support contracts
   - Consulting services
   - Training workshops

### Diagnostic Information Collection

When reporting issues, include:

```bash
# System information
ORCHEX --version
uname -a
node --version

# Configuration (redact sensitive data)
ORCHEX config show --safe

# Bridge status
ORCHEX bridge status --detailed

# Recent logs
tail -100 ~/.orchex/logs/ORCHEX.log

# Test results
ORCHEX doctor --report issue-report.json
```

This comprehensive troubleshooting guide should help resolve most issues encountered with the ORCHEX-KILO integration. For persistent problems, consider reaching out to the support community or professional services.
