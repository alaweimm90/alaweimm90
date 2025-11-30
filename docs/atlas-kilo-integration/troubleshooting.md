# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the ATLAS-KILO integration, along with diagnostic procedures and preventive measures.

## Quick Diagnosis

### Health Check Commands

```bash
# Check overall integration status
atlas bridge status --health-check

# Test bridge connectivity
atlas bridge test

# Validate configuration
atlas config validate

# Check service availability
atlas doctor
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
atlas bridge status --detailed

# Test individual bridges
atlas bridge test k2a
atlas bridge test a2k

# Check network connectivity
curl -I https://kilo-api.example.com/health
```

**Solutions:**

1. **Restart Bridge Services**
   ```bash
   atlas bridge restart
   ```

2. **Check Configuration**
   ```bash
   atlas config show bridges
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
   # Check ATLAS service status
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
atlas bridge test --auth-only

# Check token expiration
atlas config show kilo.apiKey
```

**Solutions:**

1. **Update API Keys**
   ```bash
   # Set new API key
   atlas config set kilo.apiKey "new-api-key"

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
   atlas config set kilo.certPath "/path/to/cert.pem"
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
ls -la atlas.config.json

# Validate JSON syntax
atlas config validate ./atlas.config.json

# Check file permissions
stat atlas.config.json

# Test configuration loading
atlas config show --debug
```

**Solutions:**

1. **File Location Issues**
   ```bash
   # Create configuration file
   atlas init --config-only

   # Or specify custom path
   atlas --config ./my-config.json config show
   ```

2. **JSON Syntax Errors**
   ```bash
   # Validate and fix JSON
   atlas config validate ./atlas.config.json --fix

   # Use online JSON validator
   # Check for trailing commas, missing quotes
   ```

3. **Permission Issues**
   ```bash
   # Fix file permissions
   chmod 644 atlas.config.json

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
atlas config show --effective --debug

# Test environment variables
env | grep ATLAS_
env | grep KILO_

# Check command-line parsing
atlas --help | grep config
```

**Solutions:**

1. **Environment Variable Issues**
   ```bash
   # Export variables correctly
   export ATLAS_CONFIG_FILE="./atlas.config.json"
   export KILO_API_KEY="your-key"

   # Use .env file
   echo "KILO_API_KEY=your-key" > .env
   ```

2. **Command-line Precedence**
   ```bash
   # Use correct flag syntax
   atlas --config ./config.json command

   # Check flag parsing
   atlas command --help
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
atlas bridge test --performance

# Monitor validation metrics
atlas bridge status --metrics

# Check operation complexity
atlas analyze repo . --depth shallow --format json
```

**Solutions:**

1. **Optimize Validation Settings**
   ```bash
   # Adjust timeout settings
   atlas config set bridges.a2k.validation.timeoutMs 120000

   # Change validation strictness
   atlas config set bridges.a2k.validation.strictness lenient

   # Enable caching
   atlas config set bridges.a2k.templates.cacheEnabled true
   ```

2. **Reduce Operation Scope**
   ```bash
   # Use shallower analysis
   atlas analyze repo . --depth shallow

   # Limit file patterns
   atlas analyze repo . --include-patterns "*.ts,*.js"

   # Exclude large directories
   atlas analyze repo . --exclude-patterns "node_modules/**,dist/**"
   ```

3. **Performance Tuning**
   ```bash
   # Increase connection pool
   atlas config set bridges.a2k.connection.poolSize 10

   # Enable batch processing
   atlas config set bridges.a2k.validation.batchSize 5
   ```

### Problem: False Positive Validations

**Symptoms:**
- Valid code flagged as invalid
- Overly strict validation rules
- Policy conflicts

**Diagnosis:**
```bash
# Check validation rules
atlas config show bridges.a2k.validation

# Test with different strictness levels
atlas bridge test --strictness lenient
atlas bridge test --strictness strict

# Review policy configurations
atlas config show kilo.policies
```

**Solutions:**

1. **Adjust Validation Strictness**
   ```bash
   # Use lenient validation for development
   atlas config set bridges.a2k.validation.strictness lenient

   # Gradually increase strictness
   atlas config set bridges.a2k.validation.strictness standard
   ```

2. **Customize Policies**
   ```bash
   # Override specific policies
   atlas config set kilo.policies.overrides.security.maxPasswordLength 256

   # Disable problematic policies
   atlas config set bridges.a2k.compliance.enabledPolicies '["code_quality", "performance"]'
   ```

3. **Policy Conflicts**
   ```bash
   # Review policy precedence
   atlas compliance check . --policies security --debug

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
atlas template list --all

# Check template categories
atlas template list cicd

# Test template retrieval
atlas template get cicd/github-actions --dry-run

# Check template cache
atlas template list --cache-info
```

**Solutions:**

1. **Template Repository Issues**
   ```bash
   # Update template repository
   atlas config set kilo.templates.branch main

   # Refresh template cache
   atlas template refresh

   # Check repository access
   atlas config show kilo.templates.repository
   ```

2. **Template Path Issues**
   ```bash
   # Verify template paths
   atlas config set bridges.a2k.templates.basePath "./templates/devops"

   # Check directory structure
   ls -la templates/devops/
   ```

3. **Version Conflicts**
   ```bash
   # Use latest version
   atlas template get cicd/github-actions --version latest

   # List available versions
   atlas template list cicd/github-actions --versions
   ```

### Problem: Template Parameter Errors

**Symptoms:**
- Template generation fails
- Placeholder replacement issues
- Invalid parameter errors

**Diagnosis:**
```bash
# Validate template parameters
atlas template validate cicd/github-actions --parameters ./params.json

# Check parameter format
cat ./params.json

# Test parameter substitution
atlas template get cicd/github-actions --param.test value --dry-run
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
   atlas template get cicd/github-actions --help

   # Use parameter file
   atlas template get cicd/github-actions --parameters ./ci-params.json
   ```

3. **Placeholder Conflicts**
   ```bash
   # Escape special characters
   atlas template get cicd/github-actions --param.command "npm run build"

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
atlas compliance check . --debug --format json

# Check policy versions
atlas config show kilo.policies.version

# Test with different policies
atlas compliance check . --policies security --format detailed
atlas compliance check . --policies code_quality --format detailed
```

**Solutions:**

1. **Policy Version Issues**
   ```bash
   # Update policy versions
   atlas config set kilo.policies.version latest

   # Refresh policy cache
   atlas compliance refresh-policies
   ```

2. **Inconsistent Rule Application**
   ```bash
   # Standardize rule settings
   atlas config set kilo.policies.strictMode true

   # Review rule conflicts
   atlas compliance check . --conflict-report
   ```

3. **Context-Aware Issues**
   ```bash
   # Provide proper context
   atlas compliance check ./src/auth.js --context framework=express

   # Check framework-specific rules
   atlas compliance check . --framework express
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
atlas bridge test --performance --duration 60

# Resource monitoring
atlas bridge status --metrics

# Check system resources
top -p $(pgrep atlas)
free -h
```

**Solutions:**

1. **Caching Optimization**
   ```bash
   # Enable all caches
   atlas config set bridges.a2k.templates.cacheEnabled true
   atlas config set bridges.a2k.validation.cacheEnabled true

   # Increase cache sizes
   atlas config set cache.maxSize "512MB"
   ```

2. **Connection Pooling**
   ```bash
   # Optimize connection settings
   atlas config set bridges.a2k.connection.poolSize 20
   atlas config set bridges.a2k.connection.idleTimeoutMs 300000
   ```

3. **Batch Processing**
   ```bash
   # Enable batch operations
   atlas config set bridges.a2k.validation.batchSize 10
   atlas config set bridges.a2k.templates.batchSize 5
   ```

4. **Resource Limits**
   ```bash
   # Adjust resource limits
   atlas config set maxConcurrentOperations 5
   atlas config set memoryLimit "1GB"
   ```

## Logging and Debugging

### Enable Debug Logging

```bash
# Enable debug mode
export ATLAS_DEBUG=true
export KILO_DEBUG=true

# Run command with debug output
atlas analyze repo . --verbose

# Check debug logs
tail -f ~/.atlas/logs/debug.log
```

### Log Analysis

```bash
# Search for errors
grep "ERROR" ~/.atlas/logs/atlas.log

# Check bridge communication
grep "bridge" ~/.atlas/logs/atlas.log

# Analyze performance
grep "duration" ~/.atlas/logs/atlas.log | sort -n
```

### Common Log Messages

| Log Message | Meaning | Action |
|-------------|---------|--------|
| `Bridge connection failed` | Network/connectivity issue | Check network, restart services |
| `Validation timeout` | Operation taking too long | Increase timeout, optimize operation |
| `Template not found` | Template repository issue | Update repository, refresh cache |
| `Configuration invalid` | Config file problem | Validate config, check syntax |
| `Authentication failed` | Credential issue | Update API keys, check permissions |

## Preventive Maintenance

### Regular Health Checks

```bash
# Daily health check script
#!/bin/bash
atlas bridge status --health-check > health.log
atlas config validate >> health.log
atlas bridge test --quick >> health.log

# Alert on failures
if grep -q "unhealthy\|failed\|error" health.log; then
    echo "Health check failed" | mail -s "ATLAS-KILO Health Alert" admin@company.com
fi
```

### Configuration Backups

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
atlas config backup "config-backup-$DATE.json"
find . -name "config-backup-*.json" -mtime +30 -delete
```

### Performance Monitoring

```bash
# Performance monitoring script
#!/bin/bash
atlas bridge status --metrics > metrics.json

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
atlas --version
uname -a
node --version

# Configuration (redact sensitive data)
atlas config show --safe

# Bridge status
atlas bridge status --detailed

# Recent logs
tail -100 ~/.atlas/logs/atlas.log

# Test results
atlas doctor --report issue-report.json
```

This comprehensive troubleshooting guide should help resolve most issues encountered with the ATLAS-KILO integration. For persistent problems, consider reaching out to the support community or professional services.