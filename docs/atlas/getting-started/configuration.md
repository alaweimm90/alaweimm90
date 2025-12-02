# Configuration Guide

Complete guide to configuring ATLAS for optimal performance and security across different environments and use cases.

---

## Configuration Overview

ATLAS uses a hierarchical configuration system:

1. **Global Configuration** - System-wide settings (`~/.atlas/config.json`)
2. **Project Configuration** - Project-specific settings (`.atlas/config.json`)
3. **Environment Variables** - Runtime overrides
4. **Command-line Flags** - Per-command overrides

---

## Project Initialization

### Basic Initialization

```bash
# Initialize ATLAS in current directory
atlas init

# Initialize with specific settings
atlas init --language typescript --framework nextjs

# Initialize for monorepo
atlas init --monorepo --workspaces packages/
```

### What Gets Created

```bash
.atlas/
├── config.json          # Main configuration
├── agents/              # Agent registry
├── tasks/               # Task history
├── metrics/             # Performance data
└── logs/                # System logs
```

---

## Core Configuration

### Project Settings

```json
{
  "project": {
    "name": "My Project",
    "version": "1.0.0",
    "language": "typescript",
    "framework": "nextjs",
    "repository": "https://github.com/user/project",
    "description": "Project description"
  }
}
```

Configure via CLI:

```bash
atlas config set project.name "My Project"
atlas config set project.language typescript
atlas config set project.framework nextjs
```

### Agent Configuration

```json
{
  "agents": {
    "default_provider": "anthropic",
    "fallback_enabled": true,
    "health_check_interval": 300,
    "max_concurrent_tasks": 5,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 50
    }
  }
}
```

### Task Configuration

```json
{
  "tasks": {
    "timeout": 300,
    "max_retries": 3,
    "default_priority": "medium",
    "auto_cleanup": {
      "enabled": true,
      "max_age_days": 30
    }
  }
}
```

### Cost Management

```json
{
  "cost": {
    "max_per_task": 1.0,
    "max_per_day": 50.0,
    "max_per_month": 1000.0,
    "alert_threshold": 0.8,
    "budget_notifications": true
  }
}
```

### Logging Configuration

```json
{
  "logging": {
    "level": "info",
    "format": "json",
    "file": "./.atlas/logs/atlas.log",
    "max_size": "10m",
    "max_files": 5,
    "console": true
  }
}
```

---

## Environment Variables

### API Keys and Secrets

```bash
# AI Provider API Keys
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENAI_API_KEY="your_openai_key"
export GOOGLE_API_KEY="your_google_key"

# Database Connections (if using)
export ATLAS_DB_URL="postgresql://user:pass@localhost:5432/atlas"
export REDIS_URL="redis://localhost:6379"
```

### Runtime Configuration

```bash
# Logging
export ATLAS_LOG_LEVEL="debug"
export ATLAS_LOG_FORMAT="json"

# Performance
export ATLAS_MAX_CONCURRENT_TASKS="10"
export ATLAS_TASK_TIMEOUT="600"

# Networking
export ATLAS_HTTP_PORT="3000"
export ATLAS_HOST="0.0.0.0"

# Security
export ATLAS_API_KEY="your_atlas_api_key"
export ATLAS_JWT_SECRET="your_jwt_secret"
```

### Development vs Production

```bash
# Development
export NODE_ENV="development"
export ATLAS_DEBUG="true"
export ATLAS_CACHE_ENABLED="false"

# Production
export NODE_ENV="production"
export ATLAS_DEBUG="false"
export ATLAS_CACHE_ENABLED="true"
export ATLAS_METRICS_ENABLED="true"
```

---

## Advanced Configuration

### Custom Routing Rules

```json
{
  "routing": {
    "rules": [
      {
        "condition": "task.type == 'security_analysis'",
        "agent_preference": ["claude-sonnet-4", "gpt-4-turbo"],
        "reason": "Security tasks need high reasoning capability"
      },
      {
        "condition": "task.language == 'python'",
        "agent_preference": ["claude-sonnet-4"],
        "reason": "Claude has excellent Python support"
      },
      {
        "condition": "task.priority == 'high'",
        "max_cost": 2.0,
        "timeout": 300
      }
    ]
  }
}
```

### Custom Validation Rules

```json
{
  "validation": {
    "code_quality": {
      "enabled": true,
      "rules": ["no_console_log", "max_function_length_50", "require_type_hints"]
    },
    "security": {
      "enabled": true,
      "rules": ["no_sql_injection", "secure_headers", "input_validation"]
    }
  }
}
```

### Integration Settings

```json
{
  "integrations": {
    "kilo": {
      "enabled": true,
      "endpoint": "https://kilo-api.company.com",
      "api_key": "${KILO_API_KEY}",
      "sync_interval": 300
    },
    "github": {
      "enabled": true,
      "token": "${GITHUB_TOKEN}",
      "auto_pr": true
    },
    "slack": {
      "enabled": true,
      "webhook_url": "${SLACK_WEBHOOK}",
      "channels": ["#atlas-notifications"]
    }
  }
}
```

### Performance Tuning

```json
{
  "performance": {
    "caching": {
      "enabled": true,
      "ttl": 3600,
      "max_size": "500m"
    },
    "concurrency": {
      "max_tasks": 10,
      "worker_threads": 4
    },
    "optimization": {
      "batch_size": 5,
      "prefetch_enabled": true
    }
  }
}
```

---

## Configuration Management

### Viewing Configuration

```bash
# View all configuration
atlas config show

# View specific section
atlas config show agents

# View global configuration
atlas config show --global

# View effective configuration (merged)
atlas config show --effective
```

### Modifying Configuration

```bash
# Set simple values
atlas config set log.level debug
atlas config set cost.max_per_task 2.0

# Set nested values
atlas config set agents.max_concurrent_tasks 5
atlas config set routing.rules[0].max_cost 3.0

# Set array values
atlas config set project.tags "[\"web\",\"api\",\"microservice\"]"

# Set from file
atlas config set-from-file custom-rules.json
```

### Configuration Validation

```bash
# Validate configuration
atlas config validate

# Validate specific file
atlas config validate /path/to/config.json

# Check for deprecated settings
atlas config validate --strict
```

### Configuration Backup and Restore

```bash
# Backup configuration
atlas config backup backup-2024-01-15.json

# Restore configuration
atlas config restore backup-2024-01-15.json

# Export configuration
atlas config export config.json

# Import configuration
atlas config import config.json
```

---

## Environment-Specific Configuration

### Development Environment

```json
{
  "environment": "development",
  "debug": true,
  "logging": {
    "level": "debug",
    "console": true
  },
  "cost": {
    "max_per_task": 5.0,
    "alert_threshold": 0.5
  },
  "agents": {
    "health_check_interval": 60
  }
}
```

### Staging Environment

```json
{
  "environment": "staging",
  "debug": false,
  "logging": {
    "level": "info"
  },
  "integrations": {
    "monitoring": {
      "enabled": true,
      "endpoint": "https://monitoring.staging.company.com"
    }
  }
}
```

### Production Environment

```json
{
  "environment": "production",
  "debug": false,
  "logging": {
    "level": "warn",
    "file": "/var/log/atlas/atlas.log"
  },
  "security": {
    "api_keys_required": true,
    "rate_limiting": true,
    "audit_logging": true
  },
  "performance": {
    "caching": true,
    "optimization": true
  },
  "monitoring": {
    "enabled": true,
    "metrics": true,
    "alerting": true
  }
}
```

---

## Security Configuration

### API Key Management

```json
{
  "security": {
    "api_keys": {
      "required": true,
      "rotation_period_days": 90,
      "allowed_ips": ["192.168.1.0/24"]
    },
    "authentication": {
      "method": "jwt",
      "token_expiry_hours": 24
    },
    "encryption": {
      "enabled": true,
      "algorithm": "aes-256-gcm"
    }
  }
}
```

### Network Security

```json
{
  "network": {
    "tls": {
      "enabled": true,
      "cert_file": "/etc/ssl/certs/atlas.crt",
      "key_file": "/etc/ssl/private/atlas.key"
    },
    "firewall": {
      "enabled": true,
      "allowed_ports": [3000, 443]
    }
  }
}
```

### Data Protection

```json
{
  "data": {
    "encryption": {
      "at_rest": true,
      "in_transit": true
    },
    "retention": {
      "task_history_days": 90,
      "metrics_days": 365,
      "logs_days": 30
    },
    "backup": {
      "enabled": true,
      "schedule": "0 2 * * *",
      "retention_days": 30
    }
  }
}
```

---

## Monitoring and Alerting

### Metrics Configuration

```json
{
  "monitoring": {
    "metrics": {
      "enabled": true,
      "collection_interval": 60,
      "retention_days": 90
    },
    "alerting": {
      "enabled": true,
      "rules": [
        {
          "name": "high_error_rate",
          "condition": "error_rate > 0.05",
          "severity": "critical",
          "channels": ["slack", "email"]
        },
        {
          "name": "cost_threshold",
          "condition": "daily_cost > 40",
          "severity": "warning",
          "channels": ["email"]
        }
      ]
    }
  }
}
```

### Health Checks

```json
{
  "health": {
    "checks": {
      "database": {
        "enabled": true,
        "interval": 30,
        "timeout": 5
      },
      "agents": {
        "enabled": true,
        "interval": 60,
        "failure_threshold": 3
      },
      "api": {
        "enabled": true,
        "endpoint": "/health",
        "interval": 30
      }
    }
  }
}
```

---

## Troubleshooting Configuration

### Common Issues

**Configuration not loading**

```bash
# Check file permissions
ls -la .atlas/config.json

# Validate JSON syntax
atlas config validate .atlas/config.json

# Check environment variables
env | grep ATLAS
```

**Settings not taking effect**

```bash
# Restart ATLAS services
atlas restart

# Clear configuration cache
atlas config clear-cache

# Check effective configuration
atlas config show --effective
```

**Environment variables ignored**

```bash
# Export variables before running commands
export ATLAS_LOG_LEVEL=debug
atlas task submit ...

# Or use .env file
echo "ATLAS_LOG_LEVEL=debug" > .env
atlas config load-env
```

### Configuration Debugging

```bash
# Enable debug logging
atlas config set log.level debug

# View configuration resolution
atlas config debug

# Test configuration changes
atlas config test-changes new-config.json

# Reset to defaults
atlas config reset
```

---

## Best Practices

### 1. Use Environment Variables for Secrets

```bash
# Good: Use environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
atlas agent register claude-sonnet-4

# Bad: Hardcode in config
atlas config set agents.claude.api_key "sk-ant-..."
```

### 2. Separate Config by Environment

```bash
# Use different config files
atlas --config config.dev.json    # Development
atlas --config config.prod.json   # Production
```

### 3. Version Control Configuration

```bash
# Commit config templates (without secrets)
git add .atlas/config.template.json
git commit -m "Add ATLAS configuration template"

# Ignore actual config with secrets
echo ".atlas/config.json" >> .gitignore
```

### 4. Regular Backups

```bash
# Automate configuration backups
crontab -e
# Add: 0 2 * * * atlas config backup /backups/atlas-config-$(date +\%Y\%m\%d).json
```

### 5. Monitor Configuration Changes

```bash
# Enable audit logging
atlas config set security.audit_config_changes true

# View configuration history
atlas config history

# Rollback configuration
atlas config rollback 2  # Rollback 2 versions
```

---

## Next Steps

- **[Quick Start](../getting-started/quick-start.md)** - Get started with basic configuration
- **[Agent Management](../cli/agents.md)** - Configure AI agents
- **[Integration Guides](../integration/)** - Integrate with other tools
- **[Security Guide](../best-practices/security.md)** - Advanced security configuration</instructions>
