# Configuration Guide

Complete guide to configuring ORCHEX for optimal performance and security across different environments and use cases.

---

## Configuration Overview

ORCHEX uses a hierarchical configuration system:

1. **Global Configuration** - System-wide settings (`~/.orchex/config.json`)
2. **Project Configuration** - Project-specific settings (`.orchex/config.json`)
3. **Environment Variables** - Runtime overrides
4. **Command-line Flags** - Per-command overrides

---

## Project Initialization

### Basic Initialization

```bash
# Initialize ORCHEX in current directory
ORCHEX init

# Initialize with specific settings
ORCHEX init --language typescript --framework nextjs

# Initialize for monorepo
ORCHEX init --monorepo --workspaces packages/
```

### What Gets Created

```bash
.orchex/
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
ORCHEX config set project.name "My Project"
ORCHEX config set project.language typescript
ORCHEX config set project.framework nextjs
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
    "file": "./.orchex/logs/ORCHEX.log",
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
export ATLAS_DB_URL="postgresql://user:pass@localhost:5432/ORCHEX"
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
      "channels": ["#ORCHEX-notifications"]
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
ORCHEX config show

# View specific section
ORCHEX config show agents

# View global configuration
ORCHEX config show --global

# View effective configuration (merged)
ORCHEX config show --effective
```

### Modifying Configuration

```bash
# Set simple values
ORCHEX config set log.level debug
ORCHEX config set cost.max_per_task 2.0

# Set nested values
ORCHEX config set agents.max_concurrent_tasks 5
ORCHEX config set routing.rules[0].max_cost 3.0

# Set array values
ORCHEX config set project.tags "[\"web\",\"api\",\"microservice\"]"

# Set from file
ORCHEX config set-from-file custom-rules.json
```

### Configuration Validation

```bash
# Validate configuration
ORCHEX config validate

# Validate specific file
ORCHEX config validate /path/to/config.json

# Check for deprecated settings
ORCHEX config validate --strict
```

### Configuration Backup and Restore

```bash
# Backup configuration
ORCHEX config backup backup-2024-01-15.json

# Restore configuration
ORCHEX config restore backup-2024-01-15.json

# Export configuration
ORCHEX config export config.json

# Import configuration
ORCHEX config import config.json
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
    "file": "/var/log/ORCHEX/ORCHEX.log"
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
      "cert_file": "/etc/ssl/certs/ORCHEX.crt",
      "key_file": "/etc/ssl/private/ORCHEX.key"
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
ls -la .orchex/config.json

# Validate JSON syntax
ORCHEX config validate .orchex/config.json

# Check environment variables
env | grep ORCHEX
```

**Settings not taking effect**

```bash
# Restart ORCHEX services
ORCHEX restart

# Clear configuration cache
ORCHEX config clear-cache

# Check effective configuration
ORCHEX config show --effective
```

**Environment variables ignored**

```bash
# Export variables before running commands
export ATLAS_LOG_LEVEL=debug
ORCHEX task submit ...

# Or use .env file
echo "ATLAS_LOG_LEVEL=debug" > .env
ORCHEX config load-env
```

### Configuration Debugging

```bash
# Enable debug logging
ORCHEX config set log.level debug

# View configuration resolution
ORCHEX config debug

# Test configuration changes
ORCHEX config test-changes new-config.json

# Reset to defaults
ORCHEX config reset
```

---

## Best Practices

### 1. Use Environment Variables for Secrets

```bash
# Good: Use environment variables
export ANTHROPIC_API_KEY="sk-ant-..."
ORCHEX agent register claude-sonnet-4

# Bad: Hardcode in config
ORCHEX config set agents.claude.api_key "sk-ant-..."
```

### 2. Separate Config by Environment

```bash
# Use different config files
ORCHEX --config config.dev.json    # Development
ORCHEX --config config.prod.json   # Production
```

### 3. Version Control Configuration

```bash
# Commit config templates (without secrets)
git add .orchex/config.template.json
git commit -m "Add ORCHEX configuration template"

# Ignore actual config with secrets
echo ".orchex/config.json" >> .gitignore
```

### 4. Regular Backups

```bash
# Automate configuration backups
crontab -e
# Add: 0 2 * * * ORCHEX config backup /backups/ORCHEX-config-$(date +\%Y\%m\%d).json
```

### 5. Monitor Configuration Changes

```bash
# Enable audit logging
ORCHEX config set security.audit_config_changes true

# View configuration history
ORCHEX config history

# Rollback configuration
ORCHEX config rollback 2  # Rollback 2 versions
```

---

## Next Steps

- **[Quick Start](../getting-started/quick-start.md)** - Get started with basic configuration
- **[Agent Management](../cli/agents.md)** - Configure AI agents
- **[Integration Guides](../integration/)** - Integrate with other tools
- **[Security Guide](../best-practices/security.md)** - Advanced security configuration</instructions>
