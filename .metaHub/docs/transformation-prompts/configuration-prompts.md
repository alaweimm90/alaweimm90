# TRAE Configuration Prompts

## Setup Compliance and Cost Controls

**Purpose**: Generate production-ready configuration files that implement compliance frameworks and
cost optimization controls for TRAE routing systems.

**Prerequisites**:

- Repository analysis completed
- Target compliance frameworks identified
- Budget and cost constraints defined
- Geographic requirements understood

---

## CONFIGURATION FRAMEWORK

You are a DevOps and compliance specialist. Your task is to generate comprehensive configuration
files that enable TRAE-level routing capabilities with full compliance and cost control
implementation.

### Configuration Hierarchy

**Base Configuration** → **Environment Overrides** → **Runtime Adjustments**

1. **routing.config.json**: Core system configuration
2. **models.config.json**: Dynamic model registry
3. **budget.config.json**: Cost control and optimization
4. **geographic.config.json**: Data sovereignty and failover
5. **compliance.config.json**: Framework-specific settings

---

## ROUTING.CONFIG.JSON GENERATION

### Enterprise Routing Configuration Prompt

```
You are a system architect configuring TRAE routing for enterprise deployment. Generate routing.config.json that:

CORE SYSTEM SETTINGS:
- Set cost optimization mode (aggressive/balanced/quality_first)
- Configure geographic fallback chains with latency thresholds
- Define emergency controls with automatic triggers
- Set up monitoring and analytics parameters

PERFORMANCE OPTIMIZATION:
- Configure cache settings (TTL, strategy, max size)
- Set timeout and retry parameters
- Define rate limiting and circuit breaker settings
- Configure concurrent request limits

COMPLIANCE INTEGRATION:
- Enable required compliance frameworks (GDPR/HIPAA/SOC2/PCI_DSS)
- Configure audit levels and data handling rules
- Set geographic compliance restrictions
- Define data classification requirements

PRODUCTION HARDENING:
- Configure health check intervals
- Set up alerting thresholds and channels
- Define emergency response procedures
- Configure backup and recovery settings

Generate configuration with:
1. Complete routing.config.json with all sections
2. Environment-specific overrides (dev/staging/prod)
3. Validation rules and constraints
4. Documentation for each setting
5. Migration guides from existing configurations
```

### Generated Configuration Example

```json
{
  "_metadata": {
    "version": "1.0.0",
    "generated": "2025-11-22",
    "target": "enterprise-deployment",
    "compliance": ["GDPR", "HIPAA", "SOC2"]
  },

  "costOptimizationMode": "balanced",

  "geographicFallbacks": [
    {
      "primary": "north_america",
      "fallbacks": ["europe", "asia_pacific"],
      "latencyThreshold": 1000,
      "costMultiplier": 1.5,
      "compliancePriority": true
    }
  ],

  "costBudget": {
    "dailyLimit": 500.0,
    "monthlyLimit": 15000.0,
    "perRequestLimit": 25.0,
    "alertThreshold": 0.8,
    "emergencyThreshold": 0.95
  },

  "emergencyControls": [
    {
      "enabled": true,
      "trigger": "cost_spike",
      "triggerCondition": {
        "threshold": 2.0,
        "timeWindow": 300
      },
      "action": "force_tier_3",
      "cooldownPeriod": 30,
      "notificationChannels": ["alerts", "compliance"]
    }
  ],

  "compliance": {
    "enabledFrameworks": ["GDPR", "HIPAA"],
    "strictMode": true,
    "auditLevel": "detailed",
    "dataHandling": {
      "encryptionRequired": true,
      "anonymizationRequired": true,
      "retentionLimits": {
        "personal": 2555,
        "health": 2555,
        "financial": 2555
      }
    }
  },

  "performance": {
    "targetLatency": 2000,
    "maxConcurrentRequests": 100,
    "rateLimiting": {
      "enabled": true,
      "requestsPerMinute": 300,
      "burstLimit": 50
    }
  }
}
```

---

## MODELS.CONFIG.JSON GENERATION

### Dynamic Model Registry Configuration Prompt

```
You are an AI model registry administrator. Generate models.config.json that:

MODEL REGISTRY MANAGEMENT:
- Define all available AI models with capabilities
- Set tier classifications (tier_1/tier_2/tier_3)
- Configure geographic availability and restrictions
- Set cost parameters and quality scores

PROVIDER CONFIGURATION:
- Configure authentication and API endpoints
- Set rate limits and quota management
- Define retry strategies and error handling
- Configure health monitoring parameters

COMPLIANCE MAPPING:
- Map models to supported compliance frameworks
- Define data handling capabilities
- Configure geographic compliance restrictions
- Set audit and logging requirements

DYNAMIC UPDATES:
- Configure automatic model registry updates
- Set performance monitoring and quality tracking
- Define model deprecation and migration policies
- Configure A/B testing and gradual rollouts

Generate configuration with:
1. Complete model registry with 15+ models
2. Provider configurations for major AI platforms
3. Compliance framework mappings
4. Performance and cost optimization settings
5. Dynamic update and monitoring configurations
```

### Generated Configuration Example

```json
{
  "_metadata": {
    "version": "1.0.0",
    "models": 18,
    "providers": 6,
    "lastUpdated": "2025-11-22"
  },

  "models": [
    {
      "name": "GPT-4",
      "provider": "openai",
      "model": "gpt-4",
      "tier": "tier_1",
      "region": "north_america",
      "maxTokens": 8192,
      "costPerToken": 0.00003,
      "qualityScore": 95,
      "latency": 3000,
      "reliability": 98,
      "specializations": ["reasoning", "code", "analysis"],
      "compliance": ["GDPR", "SOC2"],
      "capabilities": {
        "multimodal": false,
        "streaming": true,
        "functionCalling": true
      }
    }
  ],

  "providers": {
    "openai": {
      "baseUrl": "https://api.openai.com/v1",
      "rateLimitStrategy": "token_bucket",
      "retryStrategy": "exponential_backoff",
      "compliance": ["GDPR", "SOC2"],
      "regions": ["north_america", "europe"]
    }
  },

  "modelGroups": {
    "premium": {
      "models": ["GPT-4", "Claude-3-Opus"],
      "maxCostPerRequest": 5.0
    },
    "balanced": {
      "models": ["GPT-3.5-Turbo", "Claude-3-Sonnet"],
      "maxCostPerRequest": 1.0
    }
  }
}
```

---

## BUDGET.CONFIG.JSON GENERATION

### Cost Control and Budget Management Prompt

```
You are a financial controller for AI operations. Generate budget.config.json that:

MULTI-LAYER BUDGET CONTROLS:
- Set daily, monthly, and yearly spending limits
- Configure per-request cost caps
- Define tier-based budget allocations
- Set provider-specific spending limits

COST OPTIMIZATION RULES:
- Configure automatic cost reduction triggers
- Set optimization mode preferences (aggressive/balanced/quality_first)
- Define emergency cost control measures
- Configure predictive cost analysis

BUDGET MONITORING & ALERTS:
- Set up budget alert thresholds and notifications
- Configure cost anomaly detection
- Define budget compliance reporting
- Set up automated budget reallocation

COMPLIANCE BUDGETS:
- Allocate separate budgets for compliance features
- Configure compliance-specific cost controls
- Set audit and reporting budget limits
- Define compliance violation penalties

Generate configuration with:
1. Comprehensive budget hierarchy
2. Cost optimization strategies
3. Alert and notification system
4. Compliance budget management
5. Reporting and analytics configuration
```

### Generated Configuration Example

```json
{
  "_metadata": {
    "budgetPeriod": "monthly",
    "currency": "USD",
    "optimizationTarget": "7-10x reduction"
  },

  "globalBudget": {
    "monthlyLimit": 15000.0,
    "alertThreshold": 0.8,
    "emergencyThreshold": 0.95
  },

  "tierBudgets": {
    "tier_1": {
      "percentageOfTotal": 0.2,
      "maxCostPerRequest": 5.0
    },
    "tier_2": {
      "percentageOfTotal": 0.5,
      "maxCostPerRequest": 1.0
    },
    "tier_3": {
      "percentageOfTotal": 0.3,
      "maxCostPerRequest": 0.1
    }
  },

  "costOptimizationRules": {
    "mode": "balanced",
    "targetReduction": 0.8,
    "emergencyReduction": 0.9,
    "geographicOptimization": true
  },

  "budgetAlerts": {
    "thresholds": [
      {
        "percentage": 0.75,
        "severity": "warning",
        "actions": ["notify", "optimize"]
      },
      {
        "percentage": 0.95,
        "severity": "critical",
        "actions": ["emergency", "block"]
      }
    ]
  }
}
```

---

## GEOGRAPHIC.CONFIG.JSON GENERATION

### Data Sovereignty and Geographic Configuration Prompt

```
You are a data sovereignty specialist. Generate geographic.config.json that:

GEOGRAPHIC REGION DEFINITIONS:
- Define all supported geographic regions
- Configure data sovereignty requirements
- Set compliance framework mappings
- Define performance and latency baselines

FALLBACK CHAIN CONFIGURATION:
- Create intelligent geographic failover chains
- Set latency thresholds and cost multipliers
- Configure compliance priority routing
- Define health monitoring parameters

DATA LOCALIZATION RULES:
- Configure data residency requirements
- Set cross-border transfer rules
- Define sovereignty compliance frameworks
- Configure audit and reporting requirements

NETWORK HEALTH MONITORING:
- Set up geographic health checks
- Configure latency monitoring and alerting
- Define failover trigger conditions
- Set up performance optimization rules

Generate configuration with:
1. Complete geographic region definitions
2. Intelligent fallback chain configurations
3. Data sovereignty and compliance rules
4. Network health monitoring setup
5. Geographic load balancing configuration
```

### Generated Configuration Example

```json
{
  "_metadata": {
    "regions": 6,
    "complianceFrameworks": ["GDPR", "HIPAA", "SOC2"],
    "dataSovereignty": "strict"
  },

  "regions": {
    "europe": {
      "dataSovereignty": {
        "dataLocalizationRequired": true,
        "gdprCompliantOnly": true
      },
      "compliance": ["GDPR"],
      "performance": {
        "averageLatency": 100,
        "reliability": 0.98
      }
    }
  },

  "fallbackChains": [
    {
      "primary": "north_america",
      "fallbacks": ["europe", "asia_pacific"],
      "latencyThreshold": 1000,
      "compliancePriority": true
    }
  ],

  "dataSovereigntyRules": {
    "GDPR": {
      "restrictedRegions": ["china", "russia"],
      "dataLocalizationRequired": true,
      "sovereignProcessingRequired": true
    }
  }
}
```

---

## COMPLIANCE.CONFIG.JSON GENERATION

### Compliance Framework Configuration Prompt

```
You are a compliance officer. Generate compliance.config.json that:

FRAMEWORK-SPECIFIC CONFIGURATION:
- Configure GDPR data protection rules
- Set HIPAA healthcare compliance requirements
- Define SOC 2 security controls
- Configure PCI DSS payment processing rules

DATA HANDLING POLICIES:
- Define data classification levels
- Set retention and deletion policies
- Configure encryption requirements
- Define cross-border data transfer rules

AUDIT AND MONITORING:
- Configure audit logging levels
- Set compliance monitoring intervals
- Define violation detection rules
- Configure reporting and alerting

ENFORCEMENT MECHANISMS:
- Set automatic compliance enforcement
- Configure violation response procedures
- Define compliance override conditions
- Set up compliance training requirements

Generate configuration with:
1. Complete compliance framework configurations
2. Data handling and privacy policies
3. Audit and monitoring setup
4. Enforcement and response procedures
5. Compliance reporting and documentation
```

### Generated Configuration Example

```json
{
  "_metadata": {
    "frameworks": ["GDPR", "HIPAA", "SOC2"],
    "auditLevel": "comprehensive",
    "dataClassification": "strict"
  },

  "GDPR": {
    "dataProtection": {
      "consentRequired": true,
      "dataMinimization": true,
      "purposeLimitation": true,
      "retentionLimits": {
        "personalData": 2555
      }
    },
    "rights": {
      "access": true,
      "rectification": true,
      "erasure": true,
      "portability": true
    }
  },

  "HIPAA": {
    "privacyRule": {
      "protectedHealthInformation": true,
      "minimumNecessary": true,
      "individualRights": true
    },
    "securityRule": {
      "administrativeSafeguards": true,
      "physicalSafeguards": true,
      "technicalSafeguards": true
    }
  },

  "auditSettings": {
    "logRetention": 2555,
    "realTimeMonitoring": true,
    "violationAlerts": true,
    "complianceReporting": true
  }
}
```

---

## ENVIRONMENT-SPECIFIC CONFIGURATIONS

### Development Configuration

```json
{
  "_environment": "development",
  "costOptimizationMode": "quality_first",
  "monitoringEnabled": true,
  "compliance": {
    "strictMode": false,
    "auditLevel": "basic"
  },
  "budget": {
    "alertThreshold": 0.9,
    "emergencyThreshold": 0.99
  }
}
```

### Staging Configuration

```json
{
  "_environment": "staging",
  "costOptimizationMode": "balanced",
  "monitoringEnabled": true,
  "compliance": {
    "strictMode": true,
    "auditLevel": "detailed"
  },
  "performance": {
    "loadTesting": true,
    "stressTesting": true
  }
}
```

### Production Configuration

```json
{
  "_environment": "production",
  "costOptimizationMode": "aggressive",
  "monitoringEnabled": true,
  "compliance": {
    "strictMode": true,
    "auditLevel": "comprehensive"
  },
  "emergencyControls": {
    "enabled": true,
    "autoMitigation": true
  }
}
```

---

## CONFIGURATION VALIDATION & TESTING

### Validation Rules

**Schema Validation**:

- JSON schema validation for all configuration files
- Type checking and constraint validation
- Cross-reference validation between configurations
- Environment-specific validation rules

**Security Validation**:

- Sensitive data detection and masking
- Access control validation
- Encryption requirement checking
- Compliance rule validation

**Performance Validation**:

- Configuration size and parsing performance
- Memory usage validation
- Startup time impact assessment
- Runtime configuration update validation

### Testing Strategies

**Unit Testing**:

- Individual configuration file validation
- Cross-configuration dependency testing
- Environment override testing
- Configuration update testing

**Integration Testing**:

- Full system configuration loading
- Runtime configuration updates
- Configuration rollback testing
- Multi-environment configuration testing

**Compliance Testing**:

- Compliance rule validation
- Audit logging verification
- Data handling policy testing
- Violation detection testing

---

## CONFIGURATION MANAGEMENT TOOLS

### Configuration Generator Script

```bash
#!/bin/bash
# TRAE Configuration Generator

ENVIRONMENT=$1
FRAMEWORKS=$2
BUDGET=$3

# Generate base configuration
generate_base_config() {
  # Create routing.config.json
  # Create models.config.json
  # Create budget.config.json
}

# Apply environment overrides
apply_environment_overrides() {
  # Merge environment-specific settings
  # Validate configuration integrity
}

# Validate compliance requirements
validate_compliance() {
  # Check framework compatibility
  # Validate data handling rules
}
```

### Configuration Validator

```typescript
class ConfigurationValidator {
  validateConfig(config: any, schema: any): ValidationResult {
    // Schema validation
    // Cross-reference checking
    // Security validation
    // Performance validation
  }

  validateCompliance(config: any): ComplianceResult {
    // Framework validation
    // Data handling validation
    // Geographic compliance validation
  }
}
```

---

**These configuration prompts generate production-ready configuration files that implement
comprehensive compliance frameworks and cost optimization controls for TRAE routing systems. Each
configuration includes validation, security, and performance considerations.**</content>
