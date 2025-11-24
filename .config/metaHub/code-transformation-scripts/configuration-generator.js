#!/usr/bin/env node

/**
 * TRAE Configuration Generator
 * Creates production-ready configuration files from repository analysis
 *
 * Usage: node configuration-generator.js [analysis-file] [output-dir]
 */

const fs = require('fs');
const path = require('path');

class ConfigurationGenerator {
  constructor() {
    this.templates = this.loadConfigurationTemplates();
  }

  async generateConfigurations(analysisFile, outputDir = './trae-config') {
    console.log('⚙️  Generating TRAE Configurations...\n');

    try {
      // Load analysis data
      const analysis = this.loadAnalysisData(analysisFile);

      // Create output directory
      const outputPath = path.resolve(outputDir);
      fs.mkdirSync(outputPath, { recursive: true });

      // Generate configurations
      const configs = {
        'routing.config.json': this.generateRoutingConfig(analysis),
        'models.config.json': this.generateModelsConfig(analysis),
        'budget.config.json': this.generateBudgetConfig(analysis),
        'geographic.config.json': this.generateGeographicConfig(analysis),
        'compliance.config.json': this.generateComplianceConfig(analysis),
        'monitoring.config.json': this.generateMonitoringConfig(analysis),
        'environment.env': this.generateEnvironmentConfig(analysis)
      };

      // Write configuration files
      for (const [filename, config] of Object.entries(configs)) {
        const filePath = path.join(outputPath, filename);
        const content = typeof config === 'string' ? config : JSON.stringify(config, null, 2);
        fs.writeFileSync(filePath, content);
        console.log(`   ✅ Generated ${filename}`);
      }

      // Generate validation script
      await this.generateValidationScript(outputPath, analysis);

      // Generate deployment script
      await this.generateDeploymentScript(outputPath, analysis);

      console.log('\n🎉 Configuration generation complete!');
      console.log(`📁 Configurations saved to: ${outputPath}`);
      console.log('\n📋 Next steps:');
      console.log('   1. Review and customize generated configurations');
      console.log('   2. Set environment variables in .env file');
      console.log('   3. Run validation script: node validate-config.js');
      console.log('   4. Deploy using generated deployment script');

    } catch (error) {
      console.error('❌ Configuration generation failed:', error.message);
      process.exit(1);
    }
  }

  loadAnalysisData(analysisFile) {
    if (!analysisFile) {
      // Try to find default analysis file
      const candidates = [
        'trae-analysis-report.json',
        './trae-analysis-report.json',
        'analysis-report.json'
      ];

      for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
          analysisFile = candidate;
          break;
        }
      }

      if (!analysisFile) {
        throw new Error('No analysis file found. Run repository-scanner.js first or specify analysis file path.');
      }
    }

    const analysisPath = path.resolve(analysisFile);
    if (!fs.existsSync(analysisPath)) {
      throw new Error(`Analysis file not found: ${analysisPath}`);
    }

    const analysis = JSON.parse(fs.readFileSync(analysisPath, 'utf8'));
    console.log(`   📊 Loaded analysis from: ${analysisFile}`);
    console.log(`   📈 Repository: ${analysis.executiveSummary.repository}`);
    console.log(`   💰 Est. Monthly Cost: $${analysis.executiveSummary.estimatedMonthlyCost}`);

    return analysis;
  }

  loadConfigurationTemplates() {
    return {
      routing: {
        "_metadata": {
          "version": "1.0.0",
          "generated": new Date().toISOString(),
          "description": "TRAE routing system configuration"
        },
        "costOptimizationMode": "balanced",
        "maxRetries": 3,
        "timeout": 30000,
        "cacheEnabled": true,
        "monitoringEnabled": true,
        "analyticsEnabled": true
      },
      models: {
        "_metadata": {
          "version": "1.0.0",
          "description": "AI model registry and capabilities"
        },
        "models": [],
        "providers": {},
        "modelGroups": {}
      },
      budget: {
        "_metadata": {
          "description": "Cost control and budget management"
        },
        "globalBudget": {},
        "tierBudgets": {},
        "costOptimizationRules": {},
        "budgetAlerts": {}
      },
      geographic: {
        "_metadata": {
          "description": "Geographic routing and failover configuration"
        },
        "regions": [],
        "fallbackChains": [],
        "healthMonitoring": {}
      },
      compliance: {
        "_metadata": {
          "description": "Compliance framework configuration"
        }
      },
      monitoring: {
        "_metadata": {
          "description": "Monitoring and alerting configuration"
        },
        "prometheus": {},
        "alerting": {},
        "logging": {}
      }
    };
  }

  generateRoutingConfig(analysis) {
    const config = { ...this.templates.routing };

    // Adapt based on analysis
    const complexity = analysis.executiveSummary.transformationComplexity;

    if (complexity >= 4) {
      config.costOptimizationMode = 'aggressive';
    } else if (complexity <= 2) {
      config.costOptimizationMode = 'quality_first';
    }

    // Set geographic fallbacks based on detected providers
    config.geographicFallbacks = this.generateGeographicFallbacks(analysis);

    // Set compliance frameworks
    config.compliance = {
      enabledFrameworks: analysis.executiveSummary.requiredComplianceFrameworks,
      strictMode: analysis.executiveSummary.requiredComplianceFrameworks.length > 0,
      auditLevel: analysis.executiveSummary.requiredComplianceFrameworks.length > 0 ? 'detailed' : 'basic'
    };

    // Performance settings based on analysis
    config.performance = {
      targetLatency: 2000,
      maxConcurrentRequests: 100,
      rateLimiting: {
        enabled: true,
        requestsPerMinute: 300
      }
    };

    return config;
  }

  generateGeographicFallbacks(analysis) {
    // Default fallbacks
    const fallbacks = [
      {
        primary: 'north_america',
        fallbacks: ['europe', 'asia_pacific'],
        latencyThreshold: 1000,
        costMultiplier: 1.5,
        compliancePriority: analysis.executiveSummary.requiredComplianceFrameworks.includes('GDPR')
      }
    ];

    // Add more regions if needed
    if (analysis.detailedAnalysis.compliance.includes('GDPR')) {
      fallbacks.push({
        primary: 'europe',
        fallbacks: ['north_america', 'asia_pacific'],
        latencyThreshold: 800,
        costMultiplier: 1.3,
        compliancePriority: true
      });
    }

    return fallbacks;
  }

  generateModelsConfig(analysis) {
    const config = { ...this.templates.models };
    const detectedProviders = this.extractProvidersFromAnalysis(analysis);

    // Generate model configurations based on detected providers
    config.models = this.generateModelList(detectedProviders, analysis);
    config.providers = this.generateProviderConfigs(detectedProviders);
    config.modelGroups = this.generateModelGroups(config.models);

    return config;
  }

  extractProvidersFromAnalysis(analysis) {
    const providers = new Set();

    // Extract from AI integrations
    if (analysis.detailedAnalysis?.aiIntegrations) {
      analysis.detailedAnalysis.aiIntegrations.forEach(integration => {
        integration.integrations.forEach(int => {
          if (int.confidence === 'high') {
            providers.add(int.provider);
          }
        });
      });
    }

    // Default to common providers if none detected
    if (providers.size === 0) {
      providers.add('openai');
      providers.add('anthropic');
    }

    return Array.from(providers);
  }

  generateModelList(providers, analysis) {
    const models = [];

    if (providers.includes('openai')) {
      models.push(
        {
          name: 'GPT-4',
          provider: 'openai',
          model: 'gpt-4',
          tier: 'tier_1',
          region: 'north_america',
          maxTokens: 8192,
          costPerToken: 0.00003,
          qualityScore: 95,
          latency: 3000,
          reliability: 98,
          specializations: ['reasoning', 'code_generation', 'analysis'],
          compliance: this.getComplianceFrameworks(analysis),
          enabled: true
        },
        {
          name: 'GPT-3.5-Turbo',
          provider: 'openai',
          model: 'gpt-3.5-turbo',
          tier: 'tier_2',
          region: 'north_america',
          maxTokens: 4096,
          costPerToken: 0.000002,
          qualityScore: 80,
          latency: 1000,
          reliability: 95,
          specializations: ['general', 'fast_response'],
          compliance: this.getComplianceFrameworks(analysis),
          enabled: true
        }
      );
    }

    if (providers.includes('anthropic')) {
      models.push({
        name: 'Claude-3-Opus',
        provider: 'anthropic',
        model: 'claude-3-opus-20240229',
        tier: 'tier_1',
        region: 'north_america',
        maxTokens: 200000,
        costPerToken: 0.000015,
        qualityScore: 96,
        latency: 2500,
        reliability: 97,
        specializations: ['reasoning', 'long_context', 'analysis', 'safety_critical'],
        compliance: this.getComplianceFrameworks(analysis),
        enabled: true
      });
    }

    if (providers.includes('google')) {
      models.push({
        name: 'Gemini-Pro',
        provider: 'google',
        model: 'gemini-pro',
        tier: 'tier_2',
        region: 'north_america',
        maxTokens: 32768,
        costPerToken: 0.000001,
        qualityScore: 85,
        latency: 1500,
        reliability: 90,
        specializations: ['multimodal', 'analysis'],
        compliance: this.getComplianceFrameworks(analysis),
        enabled: true
      });
    }

    return models;
  }

  getComplianceFrameworks(analysis) {
    return analysis.executiveSummary.requiredComplianceFrameworks || [];
  }

  generateProviderConfigs(providers) {
    const configs = {};

    const providerSettings = {
      openai: {
        baseUrl: 'https://api.openai.com/v1',
        rateLimitStrategy: 'token_bucket',
        retryStrategy: 'exponential_backoff',
        timeout: 30000,
        compliance: ['GDPR', 'SOC2'],
        regions: ['north_america', 'europe']
      },
      anthropic: {
        baseUrl: 'https://api.anthropic.com',
        rateLimitStrategy: 'fixed_window',
        retryStrategy: 'linear_backoff',
        timeout: 60000,
        compliance: ['GDPR', 'SOC2'],
        regions: ['north_america', 'europe']
      },
      google: {
        baseUrl: 'https://generativelanguage.googleapis.com',
        rateLimitStrategy: 'sliding_window',
        retryStrategy: 'exponential_backoff',
        timeout: 30000,
        compliance: ['GDPR', 'SOC2'],
        regions: ['north_america', 'europe', 'asia_pacific']
      }
    };

    providers.forEach(provider => {
      if (providerSettings[provider]) {
        configs[provider] = providerSettings[provider];
      }
    });

    return configs;
  }

  generateModelGroups(models) {
    const groups = {
      premium: {
        models: models.filter(m => m.tier === 'tier_1').map(m => m.name),
        maxCostPerRequest: 5.0,
        description: 'Highest quality models for critical tasks'
      },
      balanced: {
        models: models.filter(m => m.tier === 'tier_2').map(m => m.name),
        maxCostPerRequest: 1.0,
        description: 'Balanced quality and cost for general tasks'
      },
      budget: {
        models: models.filter(m => m.costPerToken < 0.00001).map(m => m.name),
        maxCostPerRequest: 0.1,
        description: 'Cost-optimized models for simple tasks'
      }
    };

    return groups;
  }

  generateBudgetConfig(analysis) {
    const config = { ...this.templates.budget };
    const estimatedCost = analysis.executiveSummary.estimatedMonthlyCost;

    // Set budget limits based on estimated usage
    config.globalBudget = {
      monthlyLimit: Math.max(1000, Math.round(estimatedCost * 1.5)), // 50% buffer
      alertThreshold: 0.8,
      emergencyThreshold: 0.95,
      currency: 'USD',
      resetSchedule: 'monthly'
    };

    // Set tier budgets
    config.tierBudgets = {
      tier_1: {
        percentageOfTotal: 0.3,
        maxCostPerRequest: 5.0,
        monthlyLimit: Math.round(config.globalBudget.monthlyLimit * 0.3)
      },
      tier_2: {
        percentageOfTotal: 0.7,
        maxCostPerRequest: 1.0,
        monthlyLimit: Math.round(config.globalBudget.monthlyLimit * 0.7)
      }
    };

    // Set optimization rules
    config.costOptimizationRules = {
      mode: analysis.executiveSummary.transformationComplexity >= 4 ? 'aggressive' : 'balanced',
      targetReduction: 0.7, // 70% cost reduction target
      maxReduction: 0.9, // Maximum 90% reduction
      geographicOptimization: true,
      timeBasedRouting: true,
      predictiveOptimization: true
    };

    // Set budget alerts
    config.budgetAlerts = {
      thresholds: [
        {
          percentage: 0.75,
          severity: 'warning',
          actions: ['notify', 'optimize'],
          channels: ['email', 'slack']
        },
        {
          percentage: 0.90,
          severity: 'error',
          actions: ['notify', 'throttle', 'optimize'],
          channels: ['email', 'slack', 'pagerduty']
        },
        {
          percentage: 0.95,
          severity: 'critical',
          actions: ['emergency_mode', 'block_non_critical'],
          channels: ['email', 'slack', 'pagerduty', 'sms']
        }
      ],
      notificationChannels: {
        email: {
          enabled: true,
          recipients: ['admin@company.com']
        },
        slack: {
          enabled: true,
          webhookUrl: '${SLACK_WEBHOOK_URL}',
          channel: '#cost-alerts'
        },
        pagerduty: {
          enabled: false,
          integrationKey: '${PAGERDUTY_KEY}'
        }
      }
    };

    return config;
  }

  generateGeographicConfig(analysis) {
    const config = { ...this.templates.geographic };

    // Define regions
    config.regions = [
      {
        name: 'north_america',
        displayName: 'North America',
        providers: ['openai', 'anthropic', 'google'],
        latencyBaseline: 50,
        complianceFrameworks: ['GDPR', 'CCPA'],
        dataLocalization: false,
        enabled: true
      },
      {
        name: 'europe',
        displayName: 'Europe',
        providers: ['openai', 'anthropic', 'google'],
        latencyBaseline: 100,
        complianceFrameworks: ['GDPR'],
        dataLocalization: true,
        enabled: true
      },
      {
        name: 'asia_pacific',
        displayName: 'Asia Pacific',
        providers: ['openai', 'google'],
        latencyBaseline: 200,
        complianceFrameworks: ['GDPR'],
        dataLocalization: false,
        enabled: true
      },
      {
        name: 'south_america',
        displayName: 'South America',
        providers: ['openai'],
        latencyBaseline: 150,
        complianceFrameworks: [],
        dataLocalization: false,
        enabled: true
      },
      {
        name: 'africa',
        displayName: 'Africa',
        providers: ['openai', 'google'],
        latencyBaseline: 250,
        complianceFrameworks: [],
        dataLocalization: false,
        enabled: true
      }
    ];

    // Generate fallback chains
    config.fallbackChains = this.generateFallbackChains(analysis);

    // Health monitoring configuration
    config.healthMonitoring = {
      enabled: true,
      checkInterval: 30000, // 30 seconds
      failureThreshold: 3, // Consecutive failures before marking unhealthy
      recoveryTimeout: 300000, // 5 minutes before attempting recovery
      healthCheckTimeout: 5000, // 5 second timeout for health checks
      regions: config.regions.map(r => r.name)
    };

    return config;
  }

  generateFallbackChains(analysis) {
    const chains = [];

    // Primary chains based on compliance requirements
    if (analysis.executiveSummary.requiredComplianceFrameworks.includes('GDPR')) {
      chains.push({
        primary: 'europe',
        fallbacks: ['north_america', 'asia_pacific'],
        latencyThreshold: 800,
        costMultiplier: 1.3,
        compliancePriority: true,
        description: 'GDPR-compliant primary with EU data localization'
      });
    }

    // Default global chains
    chains.push(
      {
        primary: 'north_america',
        fallbacks: ['europe', 'asia_pacific'],
        latencyThreshold: 1000,
        costMultiplier: 1.5,
        compliancePriority: false,
        description: 'Standard North American routing with global fallbacks'
      },
      {
        primary: 'asia_pacific',
        fallbacks: ['north_america', 'europe'],
        latencyThreshold: 1200,
        costMultiplier: 1.8,
        compliancePriority: false,
        description: 'Asia Pacific routing with trans-Pacific fallbacks'
      }
    );

    return chains;
  }

  generateComplianceConfig(analysis) {
    const config = { ...this.templates.compliance };
    const frameworks = analysis.executiveSummary.requiredComplianceFrameworks;

    frameworks.forEach(framework => {
      config[framework] = this.generateFrameworkConfig(framework, analysis);
    });

    // General compliance settings
    config.general = {
      strictMode: frameworks.length > 0,
      auditLevel: frameworks.length > 2 ? 'comprehensive' : frameworks.length > 0 ? 'detailed' : 'basic',
      dataHandling: {
        encryptionRequired: frameworks.includes('GDPR') || frameworks.includes('HIPAA'),
        anonymizationRequired: frameworks.includes('GDPR'),
        retentionLimits: this.generateRetentionLimits(frameworks),
        crossBorderTransferAllowed: !frameworks.includes('GDPR')
      },
      auditSettings: {
        logRetention: frameworks.includes('SOC2') ? 2555 : 365, // Days
        realTimeMonitoring: frameworks.includes('HIPAA') || frameworks.includes('SOC2'),
        violationAlerts: true,
        complianceReporting: frameworks.length > 0
      }
    };

    return config;
  }

  generateFrameworkConfig(framework, analysis) {
    const configs = {
      GDPR: {
        dataProtection: {
          consentRequired: true,
          dataMinimization: true,
          purposeLimitation: true,
          accuracy: true,
          storageLimitation: true,
          integrity: true,
          accountability: true
        },
        individualRights: {
          access: true,
          rectification: true,
          erasure: true,
          restrictProcessing: true,
          dataPortability: true,
          object: true
        },
        dataBreach: {
          notificationDeadline: 72, // hours
          supervisoryAuthority: true,
          dataSubjects: true
        }
      },
      HIPAA: {
        privacyRule: {
          protectedHealthInformation: true,
          minimumNecessary: true,
          individualRights: true,
          administrativeRequirements: true,
          usesAndDisclosures: true
        },
        securityRule: {
          administrativeSafeguards: true,
          physicalSafeguards: true,
          technicalSafeguards: true,
          securityIncidentProcedures: true
        },
        breachNotification: {
          coveredEntityDeadline: 60, // days
          mediaDeadline: 60,
          individualDeadline: 60
        }
      },
      SOC2: {
        trustServicesCriteria: {
          security: true,
          availability: true,
          processingIntegrity: true,
          confidentiality: true,
          privacy: true
        },
        monitoring: {
          continuousAuditing: true,
          automatedControls: true,
          exceptionReporting: true
        }
      },
      PCI_DSS: {
        dataProtection: {
          cardholderData: true,
          encryption: true,
          accessControls: true,
          networkSecurity: true
        },
        monitoring: {
          logging: true,
          testing: true,
          incidentResponse: true
        },
        compliance: {
          selfAssessmentQuestionnaire: true,
          quarterlyScan: true,
          annualAttestation: true
        }
      }
    };

    return configs[framework] || {};
  }

  generateRetentionLimits(frameworks) {
    const limits = {
      public: 365, // 1 year
      internal: 2555, // 7 years
      confidential: 2555,
      restricted: -1 // Indefinite
    };

    // Stricter limits for compliance frameworks
    if (frameworks.includes('GDPR')) {
      limits.public = 180; // 6 months
      limits.internal = 1800; // 5 years
    }

    if (frameworks.includes('HIPAA')) {
      limits.confidential = 2555; // 7 years minimum
      limits.restricted = 2555;
    }

    return limits;
  }

  generateMonitoringConfig(analysis) {
    const config = { ...this.templates.monitoring };

    config.prometheus = {
      enabled: true,
      port: 9090,
      path: '/metrics',
      collectDefaultMetrics: true,
      prefix: 'trae_'
    };

    config.alerting = {
      enabled: true,
      rules: this.generateAlertRules(analysis),
      notificationChannels: this.generateNotificationChannels()
    };

    config.logging = {
      level: 'info',
      format: 'json',
      files: {
        routing: 'logs/routing.log',
        errors: 'logs/errors.log',
        audit: 'logs/audit.log'
      },
      rotation: {
        maxSize: '10m',
        maxFiles: 5
      }
    };

    config.dashboards = {
      grafana: {
        enabled: true,
        datasource: 'prometheus',
        dashboards: [
          'trae-routing-overview',
          'trae-cost-analysis',
          'trae-compliance-monitoring'
        ]
      }
    };

    return config;
  }

  generateAlertRules(analysis) {
    return [
      {
        name: 'HighLatency',
        condition: 'rate(trae_routing_latency_seconds{quantile="0.95"}[5m]) > 3',
        severity: 'warning',
        description: '95th percentile latency > 3 seconds',
        for: '5m'
      },
      {
        name: 'HighErrorRate',
        condition: 'rate(trae_routing_errors_total[5m]) / rate(trae_routing_requests_total[5m]) > 0.05',
        severity: 'error',
        description: 'Error rate > 5%',
        for: '5m'
      },
      {
        name: 'CostSpike',
        condition: 'rate(trae_routing_cost_total[1h]) > 2 * rate(trae_routing_cost_total[1h] offset 24h)',
        severity: 'warning',
        description: 'Cost increased by 2x compared to yesterday',
        for: '15m'
      },
      {
        name: 'BudgetAlert',
        condition: 'trae_budget_utilization_percentage > 80',
        severity: 'warning',
        description: 'Budget utilization > 80%',
        for: '5m'
      }
    ];
  }

  generateNotificationChannels() {
    return {
      email: {
        enabled: true,
        smtp: {
          host: '${SMTP_HOST}',
          port: 587,
          secure: false,
          auth: {
            user: '${SMTP_USER}',
            pass: '${SMTP_PASS}'
          }
        },
        recipients: ['alerts@company.com']
      },
      slack: {
        enabled: true,
        webhookUrl: '${SLACK_WEBHOOK_URL}',
        channel: '#trae-alerts',
        username: 'TRAE Monitor'
      },
      pagerduty: {
        enabled: false,
        integrationKey: '${PAGERDUTY_INTEGRATION_KEY}',
        severity: 'error'
      }
    };
  }

  generateEnvironmentConfig(analysis) {
    const envVars = [
      '# TRAE Environment Configuration',
      '# Copy this to your .env file and fill in the values',
      '',
      '# Core Settings',
      `NODE_ENV=production`,
      `PORT=3000`,
      `COST_MODE=${analysis.executiveSummary.transformationComplexity >= 4 ? 'aggressive' : 'balanced'}`,
      `PRIMARY_REGION=north_america`,
      '',
      '# Budget Controls',
      `DAILY_LIMIT=${Math.max(50, Math.round(analysis.executiveSummary.estimatedMonthlyCost / 30))}`,
      `MONTHLY_LIMIT=${Math.round(analysis.executiveSummary.estimatedMonthlyCost * 1.2)}`,
      '',
      '# Compliance Frameworks',
      `COMPLIANCE_FRAMEWORKS=${analysis.executiveSummary.requiredComplianceFrameworks.join(',') || 'GDPR'}`,
      `STRICT_COMPLIANCE=${analysis.executiveSummary.requiredComplianceFrameworks.length > 0}`,
      `AUDIT_LEVEL=${analysis.executiveSummary.requiredComplianceFrameworks.length > 2 ? 'comprehensive' : 'detailed'}`,
      '',
      '# AI Provider API Keys',
      '# OpenAI',
      'OPENAI_API_KEY=your_openai_api_key_here',
      'OPENAI_BASE_URL=https://api.openai.com/v1',
      '',
      '# Anthropic',
      'ANTHROPIC_API_KEY=your_anthropic_api_key_here',
      'ANTHROPIC_BASE_URL=https://api.anthropic.com',
      '',
      '# Google AI',
      'GOOGLE_AI_API_KEY=your_google_ai_api_key_here',
      '',
      '# Monitoring & Alerting',
      'PROMETHEUS_PORT=9090',
      'LOG_LEVEL=info',
      '',
      '# Notification Channels',
      'SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
      'SMTP_HOST=smtp.gmail.com',
      'SMTP_USER=your-email@gmail.com',
      'SMTP_PASS=your-app-password',
      '',
      '# Database (for audit logging)',
      'DATABASE_URL=postgresql://user:password@localhost:5432/trae_routing',
      '',
      '# Redis (for caching)',
      'REDIS_URL=redis://localhost:6379',
      '',
      '# Security',
      'JWT_SECRET=your-jwt-secret-here',
      'ENCRYPTION_KEY=your-32-character-encryption-key',
      '',
      '# Advanced Settings',
      'MAX_RETRIES=3',
      'REQUEST_TIMEOUT=30000',
      'CACHE_TTL=3600',
      'RATE_LIMIT_REQUESTS_PER_MINUTE=300'
    ];

    return envVars.join('\n');
  }

  async generateValidationScript(outputPath, analysis) {
    const validationScript = `#!/usr/bin/env node

/**
 * TRAE Configuration Validator
 * Validates generated configuration files
 */

const fs = require('fs');
const path = require('path');

class ConfigurationValidator {
  async validateConfigurations() {
    console.log('🔍 Validating TRAE configurations...');

    const issues = [];
    const configs = this.loadConfigurations();

    // Validate routing config
    issues.push(...this.validateRoutingConfig(configs.routing));

    // Validate models config
    issues.push(...this.validateModelsConfig(configs.models));

    // Validate budget config
    issues.push(...this.validateBudgetConfig(configs.budget));

    // Validate geographic config
    issues.push(...this.validateGeographicConfig(configs.geographic));

    // Validate compliance config
    issues.push(...this.validateComplianceConfig(configs.compliance));

    // Validate environment variables
    issues.push(...await this.validateEnvironmentConfig());

    if (issues.length === 0) {
      console.log('✅ All configurations are valid!');
      return true;
    } else {
      console.log('❌ Configuration validation failed:');
      issues.forEach(issue => {
        console.log(\`   \${issue.severity.toUpperCase()}: \${issue.message}\`);
      });
      return false;
    }
  }

  loadConfigurations() {
    const configDir = path.dirname(__filename);
    const configs = {};

    const configFiles = [
      'routing.config.json',
      'models.config.json',
      'budget.config.json',
      'geographic.config.json',
      'compliance.config.json'
    ];

    configFiles.forEach(file => {
      const filePath = path.join(configDir, file);
      if (fs.existsSync(filePath)) {
        configs[file.replace('.config.json', '')] = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      }
    });

    return configs;
  }

  validateRoutingConfig(config) {
    const issues = [];

    if (!config.costOptimizationMode) {
      issues.push({ severity: 'error', message: 'costOptimizationMode is required' });
    }

    if (!['aggressive', 'balanced', 'quality_first'].includes(config.costOptimizationMode)) {
      issues.push({ severity: 'error', message: 'Invalid costOptimizationMode value' });
    }

    return issues;
  }

  validateModelsConfig(config) {
    const issues = [];

    if (!config.models || config.models.length === 0) {
      issues.push({ severity: 'error', message: 'At least one model must be configured' });
    }

    config.models.forEach((model, index) => {
      if (!model.name || !model.provider) {
        issues.push({ severity: 'error', message: \`Model \${index} missing required fields\` });
      }
    });

    return issues;
  }

  validateBudgetConfig(config) {
    const issues = [];

    if (!config.globalBudget?.monthlyLimit) {
      issues.push({ severity: 'error', message: 'Global budget monthly limit is required' });
    }

    if (config.globalBudget?.monthlyLimit < 100) {
      issues.push({ severity: 'warning', message: 'Monthly budget limit seems low' });
    }

    return issues;
  }

  validateGeographicConfig(config) {
    const issues = [];

    if (!config.regions || config.regions.length === 0) {
      issues.push({ severity: 'error', message: 'At least one region must be configured' });
    }

    if (!config.fallbackChains || config.fallbackChains.length === 0) {
      issues.push({ severity: 'warning', message: 'No fallback chains configured' });
    }

    return issues;
  }

  validateComplianceConfig(config) {
    const issues = [];

    const frameworks = Object.keys(config).filter(key => key !== '_metadata' && key !== 'general');

    if (frameworks.length > 0 && !config.general?.strictMode) {
      issues.push({ severity: 'warning', message: 'Compliance frameworks enabled but strict mode is off' });
    }

    return issues;
  }

  async validateEnvironmentConfig() {
    const issues = [];
    const envPath = path.join(path.dirname(__filename), 'environment.env');

    if (!fs.existsSync(envPath)) {
      issues.push({ severity: 'warning', message: 'Environment file not found' });
      return issues;
    }

    const envContent = fs.readFileSync(envPath, 'utf8');
    const requiredVars = [
      'OPENAI_API_KEY',
      'DAILY_LIMIT',
      'MONTHLY_LIMIT'
    ];

    requiredVars.forEach(varName => {
      if (!envContent.includes(\`\${varName}=\`)) {
        issues.push({ severity: 'error', message: \`Required environment variable \${varName} not found\` });
      }
    });

    return issues;
  }
}

// Run validation if called directly
if (require.main === module) {
  const validator = new ConfigurationValidator();
  validator.validateConfigurations().then(success => {
    process.exit(success ? 0 : 1);
  });
}

module.exports = ConfigurationValidator;
`;

    const validationPath = path.join(outputPath, 'validate-config.js');
    fs.writeFileSync(validationPath, validationScript);
    console.log('   ✅ Generated validation script');
  }

  async generateDeploymentScript(outputPath, analysis) {
    const deploymentScript = `#!/bin/bash

# TRAE Production Deployment Script
# Generated for ${analysis.executiveSummary.repository}

set -e

echo "🚀 Starting TRAE Production Deployment..."

# Configuration
CONFIG_DIR="$(dirname "$0")"
PROJECT_ROOT="$(cd "$CONFIG_DIR/.." && pwd)"
ENV_FILE="$CONFIG_DIR/environment.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "\${GREEN}[INFO]\${NC} \$1"
}

log_warn() {
    echo -e "\${YELLOW}[WARN]\${NC} \$1"
}

log_error() {
    echo -e "\${RED}[ERROR]\${NC} \$1"
}

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check environment file
    if [ ! -f "\$ENV_FILE" ]; then
        log_error "Environment file not found: \$ENV_FILE"
        log_info "Please copy environment.env to your .env file and configure the variables"
        exit 1
    fi

    log_info "Prerequisites check passed"
}

# Validate configurations
validate_configs() {
    log_info "Validating configurations..."

    if [ -f "\$CONFIG_DIR/validate-config.js" ]; then
        if ! node "\$CONFIG_DIR/validate-config.js"; then
            log_error "Configuration validation failed"
            exit 1
        fi
    else
        log_warn "Validation script not found, skipping validation"
    fi

    log_info "Configuration validation passed"
}

# Set up environment
setup_environment() {
    log_info "Setting up environment..."

    # Copy environment file
    cp "\$ENV_FILE" "\$PROJECT_ROOT/.env"

    # Create necessary directories
    mkdir -p "\$PROJECT_ROOT/logs"
    mkdir -p "\$PROJECT_ROOT/data"

    log_info "Environment setup complete"
}

# Deploy services
deploy_services() {
    log_info "Deploying TRAE services..."

    cd "\$PROJECT_ROOT"

    # Build and start services
    docker-compose build
    docker-compose up -d

    # Wait for services to be healthy
    log_info "Waiting for services to start..."
    sleep 30

    # Check service health
    if curl -f http://localhost:3000/health > /dev/null 2>&1; then
        log_info "TRAE Router is healthy"
    else
        log_error "TRAE Router health check failed"
        docker-compose logs trae-router
        exit 1
    fi

    log_info "Service deployment complete"
}

# Run post-deployment tests
run_tests() {
    log_info "Running post-deployment tests..."

    # Test basic routing
    if curl -X POST http://localhost:3000/api/route \\
        -H "Content-Type: application/json" \\
        -d '{"prompt": "Hello, world!", "context": {"requestId": "test-123"}}' \\
        > /dev/null 2>&1; then
        log_info "Basic routing test passed"
    else
        log_error "Basic routing test failed"
        exit 1
    fi

    # Test metrics endpoint
    if curl -f http://localhost:3000/metrics > /dev/null 2>&1; then
        log_info "Metrics endpoint test passed"
    else
        log_error "Metrics endpoint test failed"
        exit 1
    fi

    log_info "Post-deployment tests passed"
}

# Main deployment process
main() {
    echo "TRAE Production Deployment for ${analysis.executiveSummary.repository}"
    echo "=================================================="
    echo "Estimated monthly cost: \$${analysis.executiveSummary.estimatedMonthlyCost}"
    echo "Complexity score: ${analysis.executiveSummary.transformationComplexity}/5"
    echo "Compliance frameworks: ${analysis.executiveSummary.requiredComplianceFrameworks.join(', ') || 'None'}"
    echo ""

    check_prerequisites
    validate_configs
    setup_environment
    deploy_services
    run_tests

    echo ""
    log_info "🎉 TRAE deployment completed successfully!"
    echo ""
    echo "📊 Service URLs:"
    echo "   API: http://localhost:3000"
    echo "   Health: http://localhost:3000/health"
    echo "   Metrics: http://localhost:3000/metrics"
    echo "   Prometheus: http://localhost:9090"
    echo ""
    echo "📈 Monitoring:"
    echo "   View logs: docker-compose logs -f trae-router"
    echo "   Stop services: docker-compose down"
    echo "   Restart services: docker-compose restart"
    echo ""
    echo "💡 Next steps:"
    echo "   1. Configure monitoring alerts"
    echo "   2. Set up log aggregation"
    echo "   3. Configure backup procedures"
    echo "   4. Test with real traffic"
}

# Run main function
main "$@"
`;

    const deploymentPath = path.join(outputPath, 'deploy.sh');
    fs.writeFileSync(deploymentPath, deploymentScript);
    fs.chmodSync(deploymentPath, '755');
    console.log('   ✅ Generated deployment script');
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const analysisFile = args[0];
  const outputDir = args[1] || './trae-config';

  const generator = new ConfigurationGenerator();
  generator.generateConfigurations(analysisFile, outputDir);
}

module.exports = ConfigurationGenerator;</content>
