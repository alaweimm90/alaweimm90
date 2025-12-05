# Configuration Guide

This guide covers the configuration system for the ORCHEX-KILO integration, including configuration files, environment variables, and runtime configuration management.

## Configuration Overview

The ORCHEX-KILO integration uses a hierarchical configuration system that supports:

- **Global Configuration**: System-wide settings
- **Project Configuration**: Project-specific settings
- **Environment Overrides**: Environment variable overrides
- **Runtime Configuration**: Dynamic configuration changes

## Configuration Files

### Primary Configuration File

The main configuration file is `ORCHEX.config.json` (or `ORCHEX.config.yaml`):

```json
{
  "version": "1.0",
  "integration": {
    "enabled": true,
    "bridges": {
      "k2a": {
        "enabled": true,
        "eventTypes": ["policy_violation", "security_issue", "compliance_failure"],
        "analysis": {
          "autoTrigger": true,
          "priorityMapping": {
            "low": "background",
            "medium": "normal",
            "high": "high",
            "critical": "immediate"
          }
        },
        "filtering": {
          "severityThreshold": "medium",
          "repositoryFilter": ["important-*"],
          "organizationFilter": ["my-org"]
        },
        "retry": {
          "maxAttempts": 3,
          "backoffMs": 1000
        }
      },
      "a2k": {
        "enabled": true,
        "validation": {
          "strictness": "standard",
          "timeoutMs": 30000,
          "enableRollback": true
        },
        "templates": {
          "cacheEnabled": true,
          "cacheTtlMs": 3600000,
          "basePath": "./templates/devops"
        },
        "compliance": {
          "enabledPolicies": ["security", "code_quality", "performance"],
          "failOnWarning": false
        }
      }
    },
    "shared": {
      "policies": ["security", "code_quality"],
      "templates": {
        "cacheEnabled": true,
        "basePath": "./templates"
      }
    }
  },
  "ORCHEX": {
    "analysis": {
      "defaultDepth": "medium",
      "timeoutMs": 60000,
      "maxFileSize": "10MB",
      "excludedPaths": [".git", "node_modules", "dist"]
    },
    "output": {
      "defaultFormat": "table",
      "colors": true,
      "timestamps": true
    }
  },
  "kilo": {
    "endpoint": "https://kilo-api.example.com",
    "apiKey": "${KILO_API_KEY}",
    "timeoutMs": 10000,
    "retry": {
      "maxAttempts": 3,
      "backoffMs": 500
    }
  }
}
```

### Bridge-Specific Configuration

#### K2A Bridge Configuration

```json
{
  "k2aBridge": {
    "enabled": true,
    "eventTypes": ["policy_violation", "security_issue"],
    "analysis": {
      "autoTrigger": true,
      "priorityMapping": {
        "low": "background",
        "medium": "normal",
        "high": "priority",
        "critical": "immediate"
      }
    },
    "filtering": {
      "severityThreshold": "medium",
      "repositoryFilter": ["production-*", "critical-*"],
      "organizationFilter": ["my-company"]
    },
    "queue": {
      "maxSize": 1000,
      "batchSize": 10,
      "processingIntervalMs": 5000
    },
    "retry": {
      "maxAttempts": 3,
      "backoffMultiplier": 2,
      "maxBackoffMs": 30000
    },
    "monitoring": {
      "metricsEnabled": true,
      "healthCheckIntervalMs": 30000
    }
  }
}
```

#### A2K Bridge Configuration

```json
{
  "a2kBridge": {
    "enabled": true,
    "validation": {
      "strictness": "standard",
      "timeoutMs": 30000,
      "enableRollback": true,
      "concurrentValidations": 5
    },
    "templates": {
      "cacheEnabled": true,
      "cacheTtlMs": 3600000,
      "basePath": "./templates/devops",
      "maxCacheSize": "100MB",
      "preloadTemplates": ["cicd/github-actions", "k8s/deployment"]
    },
    "compliance": {
      "enabledPolicies": ["security", "code_quality", "performance", "maintainability"],
      "failOnWarning": false,
      "customPolicies": [],
      "policyOverrides": {}
    },
    "connection": {
      "poolSize": 10,
      "idleTimeoutMs": 60000,
      "maxLifetimeMs": 300000
    },
    "monitoring": {
      "metricsEnabled": true,
      "performanceTracking": true,
      "errorReporting": true
    }
  }
}
```

## Environment Variables

### Core Integration Variables

| Variable             | Description                 | Default                | Required |
| -------------------- | --------------------------- | ---------------------- | -------- |
| `ATLAS_KILO_ENABLED` | Enable/disable integration  | `true`                 | No       |
| `ATLAS_CONFIG_FILE`  | Configuration file path     | `./ORCHEX.config.json` | No       |
| `KILO_ENDPOINT`      | KILO API endpoint           | -                      | Yes      |
| `KILO_API_KEY`       | KILO API authentication key | -                      | Yes      |
| `KILO_TIMEOUT_MS`    | KILO API timeout            | `10000`                | No       |

### Bridge-Specific Variables

#### K2A Bridge Variables

| Variable                  | Description                 | Default                           |
| ------------------------- | --------------------------- | --------------------------------- |
| `K2A_ENABLED`             | Enable K2A bridge           | `true`                            |
| `K2A_EVENT_TYPES`         | Comma-separated event types | `policy_violation,security_issue` |
| `K2A_SEVERITY_THRESHOLD`  | Minimum severity to process | `medium`                          |
| `K2A_QUEUE_SIZE`          | Event queue size            | `1000`                            |
| `K2A_PROCESSING_INTERVAL` | Processing interval (ms)    | `5000`                            |

#### A2K Bridge Variables

| Variable                     | Description             | Default                 |
| ---------------------------- | ----------------------- | ----------------------- |
| `A2K_ENABLED`                | Enable A2K bridge       | `true`                  |
| `A2K_VALIDATION_STRICTNESS`  | Validation strictness   | `standard`              |
| `A2K_TEMPLATE_CACHE_ENABLED` | Enable template caching | `true`                  |
| `A2K_TEMPLATE_CACHE_TTL`     | Cache TTL (ms)          | `3600000`               |
| `A2K_COMPLIANCE_POLICIES`    | Enabled policies        | `security,code_quality` |

### Security Variables

| Variable               | Description                  | Default |
| ---------------------- | ---------------------------- | ------- |
| `ATLAS_ENCRYPTION_KEY` | Configuration encryption key | -       |
| `KILO_CLIENT_CERT`     | Client certificate path      | -       |
| `KILO_CLIENT_KEY`      | Client key path              | -       |
| `TLS_SKIP_VERIFY`      | Skip TLS verification        | `false` |

### Performance Variables

| Variable                 | Description               | Default |
| ------------------------ | ------------------------- | ------- |
| `ATLAS_MAX_CONCURRENT`   | Max concurrent operations | `10`    |
| `BRIDGE_CONNECTION_POOL` | Connection pool size      | `5`     |
| `CACHE_MAX_SIZE`         | Maximum cache size        | `100MB` |
| `METRICS_RETENTION_DAYS` | Metrics retention period  | `30`    |

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest precedence):

1. **Command-line flags**: `--config.key=value`
2. **Environment variables**: `ATLAS_CONFIG_KEY`
3. **Local configuration file**: `./ORCHEX.config.json`
4. **Global configuration file**: `~/.orchex/config.json`
5. **Default values**: Built-in defaults

## Configuration Sections

### Integration Configuration

```json
{
  "integration": {
    "enabled": true,
    "mode": "full",
    "features": {
      "governanceIntegration": true,
      "templateAccess": true,
      "complianceChecking": true,
      "automatedRefactoring": false
    },
    "bridges": {
      // Bridge configurations
    },
    "shared": {
      "policies": ["security", "code_quality"],
      "templates": {
        "cacheEnabled": true,
        "basePath": "./templates"
      },
      "logging": {
        "level": "info",
        "format": "json",
        "outputs": ["console", "file"]
      }
    }
  }
}
```

### ORCHEX Configuration

```json
{
  "ORCHEX": {
    "analysis": {
      "defaultDepth": "medium",
      "timeoutMs": 60000,
      "maxFileSize": "10MB",
      "excludedPaths": [".git", "node_modules"],
      "includedLanguages": ["javascript", "typescript", "python"],
      "customRules": []
    },
    "refactoring": {
      "autoApply": false,
      "riskThreshold": "medium",
      "backupEnabled": true,
      "rollbackEnabled": true
    },
    "output": {
      "defaultFormat": "table",
      "colors": true,
      "timestamps": true,
      "progressBars": true
    },
    "caching": {
      "enabled": true,
      "directory": "~/.orchex/cache",
      "maxSize": "1GB",
      "ttlMs": 86400000
    }
  }
}
```

### KILO Configuration

```json
{
  "kilo": {
    "endpoint": "https://api.kilo.example.com",
    "apiKey": "${KILO_API_KEY}",
    "organization": "my-org",
    "timeoutMs": 10000,
    "retry": {
      "maxAttempts": 3,
      "backoffMs": 500,
      "jitter": true
    },
    "policies": {
      "cacheEnabled": true,
      "cacheTtlMs": 300000,
      "defaultPolicies": ["security", "code_quality"]
    },
    "templates": {
      "repository": "my-org/templates",
      "branch": "main",
      "autoUpdate": true
    }
  }
}
```

## Runtime Configuration

### Dynamic Configuration Updates

The integration supports runtime configuration updates without restart:

```typescript
import { AtlasKiloBridge } from '@ORCHEX/integrations';

// Update bridge configuration
const bridge = new AtlasKiloBridge();
await bridge.updateConfig({
  validation: {
    strictness: 'strict',
    timeoutMs: 45000,
  },
});
```

### Configuration Validation

Configuration files are validated on load:

```typescript
import { ConfigValidator } from '@ORCHEX/config';

const validator = new ConfigValidator();
const result = await validator.validate(config);

if (!result.isValid) {
  console.error('Configuration errors:');
  result.errors.forEach((error) => {
    console.error(`- ${error.path}: ${error.message}`);
  });
}
```

## Configuration Management Commands

### Viewing Configuration

```bash
# Show current configuration
ORCHEX config show

# Show specific section
ORCHEX config show bridges

# Show effective configuration
ORCHEX config show --effective

# Export configuration
ORCHEX config show --format json > config-backup.json
```

### Modifying Configuration

```bash
# Set configuration value
ORCHEX config set bridges.a2k.validation.strictness strict

# Set multiple values
ORCHEX config set bridges.k2a.enabled true
ORCHEX config set kilo.endpoint https://api.kilo.example.com

# Set from file
ORCHEX config set --from-file ./my-config.json
```

### Configuration Validation

```bash
# Validate configuration file
ORCHEX config validate ./ORCHEX.config.json

# Validate with schema
ORCHEX config validate ./ORCHEX.config.json --schema strict

# Check for deprecated options
ORCHEX config validate ./ORCHEX.config.json --check-deprecated
```

### Configuration Backup and Restore

```bash
# Backup current configuration
ORCHEX config backup ./config-backup.json

# Restore configuration
ORCHEX config restore ./config-backup.json

# Reset to defaults
ORCHEX config reset --confirm
```

## Advanced Configuration

### Custom Policy Configuration

```json
{
  "policies": {
    "custom": [
      {
        "name": "company-standards",
        "description": "Company-specific coding standards",
        "rules": [
          {
            "pattern": "console\\.log",
            "severity": "warning",
            "message": "Use proper logging instead of console.log"
          }
        ]
      }
    ],
    "overrides": {
      "security": {
        "maxPasswordLength": 128,
        "requireEncryption": true
      }
    }
  }
}
```

### Template Configuration

```json
{
  "templates": {
    "repositories": [
      {
        "name": "company-templates",
        "url": "https://github.com/company/templates",
        "branch": "main",
        "categories": ["cicd", "iac"]
      }
    ],
    "customParameters": {
      "companyName": "Acme Corp",
      "domain": "acme.com",
      "defaultRegion": "us-west-2"
    },
    "hooks": {
      "preGenerate": "./scripts/pre-template.sh",
      "postGenerate": "./scripts/post-template.sh"
    }
  }
}
```

### Monitoring Configuration

```json
{
  "monitoring": {
    "enabled": true,
    "metrics": {
      "bridge": {
        "requestCount": true,
        "errorCount": true,
        "latency": true,
        "throughput": true
      },
      "analysis": {
        "operationCount": true,
        "successRate": true,
        "averageTime": true
      }
    },
    "alerts": {
      "bridgeDown": {
        "enabled": true,
        "threshold": 300000,
        "channels": ["email", "slack"]
      },
      "highErrorRate": {
        "enabled": true,
        "threshold": 0.05,
        "windowMs": 300000
      }
    },
    "exporters": {
      "prometheus": {
        "enabled": true,
        "endpoint": "/metrics",
        "labels": {
          "service": "ORCHEX-kilo-integration"
        }
      }
    }
  }
}
```

## Configuration Best Practices

### 1. Environment Separation

Use different configurations for different environments:

```bash
# Development
export ATLAS_ENV=development
ORCHEX config set bridges.a2k.validation.strictness lenient

# Production
export ATLAS_ENV=production
ORCHEX config set bridges.a2k.validation.strictness strict
```

### 2. Configuration as Code

Store configuration in version control:

```
project/
├── ORCHEX.config.json
├── ORCHEX.config.production.json
├── ORCHEX.config.staging.json
└── scripts/
    └── configure.sh
```

### 3. Secret Management

Never store secrets in configuration files:

```json
{
  "kilo": {
    "apiKey": "${KILO_API_KEY}",
    "database": {
      "password": "${DB_PASSWORD}"
    }
  }
}
```

### 4. Configuration Validation

Always validate configuration changes:

```bash
# Validate before applying
ORCHEX config validate ./new-config.json

# Test configuration
ORCHEX config test ./new-config.json

# Apply with backup
ORCHEX config backup && ORCHEX config apply ./new-config.json
```

### 5. Monitoring Configuration Changes

Track configuration changes for audit purposes:

```json
{
  "audit": {
    "configChanges": true,
    "changeLog": "./logs/config-changes.log",
    "notifyOnChange": ["admin@company.com"]
  }
}
```

## Troubleshooting Configuration Issues

### Common Issues

**Configuration Not Loading**

```bash
# Check file permissions
ls -la ORCHEX.config.json

# Validate JSON syntax
ORCHEX config validate ./ORCHEX.config.json

# Check file path
ORCHEX config show --config ./ORCHEX.config.json
```

**Environment Variables Not Working**

```bash
# Check variable is set
echo $KILO_API_KEY

# Use debug mode
ATLAS_DEBUG=true ORCHEX config show

# Check variable naming
ORCHEX config show | grep kilo
```

**Bridge Configuration Conflicts**

```bash
# Check for conflicts
ORCHEX bridge status --validate-config

# Reset bridge configuration
ORCHEX bridge configure k2a --reset

# Check configuration precedence
ORCHEX config show --effective --format json
```

This configuration guide provides comprehensive coverage of all configuration options and management procedures for the ORCHEX-KILO integration.
