#!/usr/bin/env node

/**
 * TRAE Template Applicator
 * Injects TRAE routing components into existing repositories
 *
 * Usage: node template-applicator.js [target-path] [options]
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class TemplateApplicator {
  constructor() {
    this.templatesPath = path.join(__dirname, '..', 'routing-templates');
    this.transformations = [];
    this.backupFiles = [];
  }

  async applyTemplates(targetPath = '.', options = {}) {
    console.log('🔧 Starting TRAE Template Application...\n');

    this.targetPath = path.resolve(targetPath);

    try {
      // Validate target repository
      await this.validateTarget();

      // Analyze existing code
      const analysis = await this.analyzeExistingCode();

      // Generate transformation plan
      const plan = this.generateTransformationPlan(analysis);

      // Backup existing files
      await this.createBackups(plan);

      // Apply transformations
      await this.applyTransformations(plan);

      // Update dependencies
      await this.updateDependencies(analysis.language);

      // Generate integration tests
      await this.generateTests(plan);

      // Create documentation
      await this.createDocumentation();

      console.log('✅ Template application complete!');
      console.log(`📁 Modified ${this.transformations.length} files`);
      console.log(`💾 Created ${this.backupFiles.length} backups`);

    } catch (error) {
      console.error('❌ Template application failed:', error.message);
      await this.rollbackChanges();
      process.exit(1);
    }
  }

  async validateTarget() {
    console.log('🔍 Validating target repository...');

    if (!fs.existsSync(this.targetPath)) {
      throw new Error(`Target path does not exist: ${this.targetPath}`);
    }

    // Check for existing TRAE installation
    const traeFiles = [
      'src/routing/IntelligentRouter.js',
      'src/core/IntelligentRouter.ts',
      'routing-templates'
    ];

    const hasTrae = traeFiles.some(file => fs.existsSync(path.join(this.targetPath, file)));

    if (hasTrae) {
      console.warn('⚠️  TRAE components already detected. This may overwrite existing files.');
      if (!process.env.FORCE_OVERWRITE) {
        throw new Error('Use FORCE_OVERWRITE=1 to overwrite existing TRAE installation');
      }
    }

    console.log('✅ Target validation complete');
  }

  async analyzeExistingCode() {
    console.log('📊 Analyzing existing codebase...');

    const analysis = {
      language: this.detectLanguage(),
      framework: this.detectFramework(),
      aiIntegrations: await this.findAIIntegrations(),
      structure: this.analyzeStructure(),
      dependencies: this.getDependencies()
    };

    console.log(`   Language: ${analysis.language}`);
    console.log(`   Framework: ${analysis.framework}`);
    console.log(`   AI Integrations: ${analysis.aiIntegrations.length}`);

    return analysis;
  }

  detectLanguage() {
    const indicators = {
      'JavaScript/TypeScript': ['package.json', 'tsconfig.json'],
      'Python': ['requirements.txt', 'pyproject.toml', 'Pipfile'],
      'Go': ['go.mod', 'go.sum'],
      'Java': ['pom.xml', 'build.gradle'],
      'Rust': ['Cargo.toml', 'Cargo.lock'],
      'PHP': ['composer.json'],
      'Ruby': ['Gemfile']
    };

    for (const [lang, files] of Object.entries(indicators)) {
      if (files.some(file => fs.existsSync(path.join(this.targetPath, file)))) {
        return lang;
      }
    }

    return 'Unknown';
  }

  detectFramework() {
    // JavaScript/TypeScript frameworks
    if (fs.existsSync(path.join(this.targetPath, 'package.json'))) {
      const pkg = JSON.parse(fs.readFileSync(path.join(this.targetPath, 'package.json'), 'utf8'));
      const deps = { ...pkg.dependencies, ...pkg.devDependencies };

      if (deps.express) return 'Express.js';
      if (deps['@nestjs/core']) return 'NestJS';
      if (deps.fastify) return 'Fastify';
      if (deps.next) return 'Next.js';
      if (deps.react) return 'React';
    }

    // Python frameworks
    if (fs.existsSync(path.join(this.targetPath, 'requirements.txt'))) {
      const reqs = fs.readFileSync(path.join(this.targetPath, 'requirements.txt'), 'utf8');
      if (reqs.includes('django')) return 'Django';
      if (reqs.includes('flask')) return 'Flask';
      if (reqs.includes('fastapi')) return 'FastAPI';
    }

    return 'Unknown';
  }

  async findAIIntegrations() {
    const integrations = [];
    const files = this.findFilesRecursively('.', file =>
      ['.js', '.ts', '.py', '.java', '.go'].some(ext => file.endsWith(ext))
    );

    for (const file of files) {
      const content = fs.readFileSync(path.join(this.targetPath, file), 'utf8');
      const aiPatterns = this.detectAIPatterns(content);

      if (aiPatterns.length > 0) {
        integrations.push({
          file,
          patterns: aiPatterns,
          content: content
        });
      }
    }

    return integrations;
  }

  detectAIPatterns(content) {
    const patterns = [
      { name: 'OpenAI', regex: /openai|OpenAI/, provider: 'openai' },
      { name: 'Anthropic', regex: /anthropic|Anthropic/, provider: 'anthropic' },
      { name: 'Google AI', regex: /google.*ai|vertex|gemini|palm/i, provider: 'google' },
      { name: 'API Call', regex: /\.create\(|\.generate\(|\.complete\(/, provider: 'generic' }
    ];

    return patterns.filter(pattern => pattern.regex.test(content));
  }

  analyzeStructure() {
    const structure = {
      hasSrc: fs.existsSync(path.join(this.targetPath, 'src')),
      hasApp: fs.existsSync(path.join(this.targetPath, 'app')),
      hasLib: fs.existsSync(path.join(this.targetPath, 'lib')),
      hasServices: fs.existsSync(path.join(this.targetPath, 'src/services')) ||
                   fs.existsSync(path.join(this.targetPath, 'app/services')),
      hasRoutes: fs.existsSync(path.join(this.targetPath, 'src/routes')) ||
                 fs.existsSync(path.join(this.targetPath, 'app/routes'))
    };

    return structure;
  }

  getDependencies() {
    const deps = [];

    // JavaScript/TypeScript
    if (fs.existsSync(path.join(this.targetPath, 'package.json'))) {
      const pkg = JSON.parse(fs.readFileSync(path.join(this.targetPath, 'package.json'), 'utf8'));
      deps.push(...Object.keys(pkg.dependencies || {}));
    }

    // Python
    if (fs.existsSync(path.join(this.targetPath, 'requirements.txt'))) {
      const reqs = fs.readFileSync(path.join(this.targetPath, 'requirements.txt'), 'utf8');
      deps.push(...reqs.split('\n').map(line => line.split('==')[0].trim()).filter(Boolean));
    }

    return deps;
  }

  generateTransformationPlan(analysis) {
    console.log('📋 Generating transformation plan...');

    const plan = {
      language: analysis.language,
      framework: analysis.framework,
      components: this.selectComponents(analysis),
      integrations: this.planIntegrations(analysis),
      structure: this.planStructure(analysis),
      configurations: this.generateConfigurations(analysis)
    };

    console.log(`   Components to add: ${plan.components.length}`);
    console.log(`   Files to modify: ${plan.integrations.length}`);

    return plan;
  }

  selectComponents(analysis) {
    const components = [
      'IntelligentRouter',
      'ModelSelector',
      'CostController',
      'FallbackManager',
      'PromptAdapter'
    ];

    // Add compliance components if needed
    if (analysis.aiIntegrations.some(int => int.patterns.some(p => p.provider === 'openai'))) {
      components.push('ComplianceManager');
    }

    return components;
  }

  planIntegrations(analysis) {
    const integrations = [];

    // Plan AI integration replacements
    analysis.aiIntegrations.forEach(integration => {
      integrations.push({
        type: 'ai_integration',
        file: integration.file,
        action: 'wrap_with_router',
        patterns: integration.patterns
      });
    });

    // Plan API endpoint additions
    if (analysis.framework === 'Express.js') {
      integrations.push({
        type: 'api_endpoints',
        file: 'src/routes/ai.js',
        action: 'create',
        endpoints: ['POST /api/route', 'GET /api/metrics', 'GET /api/budget']
      });
    }

    return integrations;
  }

  planStructure(analysis) {
    const structure = {
      directories: [],
      files: []
    };

    // Determine directory structure based on existing layout
    if (analysis.structure.hasSrc) {
      structure.directories = [
        'src/routing/core',
        'src/routing/config',
        'src/routing/integration'
      ];
    } else if (analysis.structure.hasApp) {
      structure.directories = [
        'app/routing/core',
        'app/routing/config',
        'app/routing/integration'
      ];
    } else {
      structure.directories = [
        'routing/core',
        'routing/config',
        'routing/integration'
      ];
    }

    return structure;
  }

  generateConfigurations(analysis) {
    return {
      routing: this.generateRoutingConfig(analysis),
      models: this.generateModelsConfig(analysis),
      budget: this.generateBudgetConfig(analysis),
      geographic: this.generateGeographicConfig(analysis)
    };
  }

  generateRoutingConfig(analysis) {
    return {
      costOptimizationMode: 'balanced',
      geographicFallbacks: [
        {
          primary: 'north_america',
          fallbacks: ['europe', 'asia_pacific'],
          latencyThreshold: 1000,
          costMultiplier: 1.5
        }
      ],
      costBudget: {
        dailyLimit: 100,
        monthlyLimit: 3000,
        alertThreshold: 0.8
      },
      compliance: {
        enabledFrameworks: this.detectComplianceNeeds(analysis)
      }
    };
  }

  generateModelsConfig(analysis) {
    const providers = [...new Set(analysis.aiIntegrations.flatMap(int =>
      int.patterns.map(p => p.provider)
    ))];

    const models = [];

    if (providers.includes('openai')) {
      models.push(
        {
          name: 'GPT-4',
          provider: 'openai',
          model: 'gpt-4',
          tier: 'tier_1',
          maxTokens: 8192,
          costPerToken: 0.03,
          qualityScore: 95
        },
        {
          name: 'GPT-3.5-Turbo',
          provider: 'openai',
          model: 'gpt-3.5-turbo',
          tier: 'tier_2',
          maxTokens: 4096,
          costPerToken: 0.002,
          qualityScore: 80
        }
      );
    }

    if (providers.includes('anthropic')) {
      models.push({
        name: 'Claude-3-Opus',
        provider: 'anthropic',
        model: 'claude-3-opus-20240229',
        tier: 'tier_1',
        maxTokens: 200000,
        costPerToken: 0.015,
        qualityScore: 96
      });
    }

    return { models };
  }

  generateBudgetConfig(analysis) {
    // Estimate budget based on existing usage
    const estimatedMonthlyCalls = analysis.aiIntegrations.length * 1000; // Rough estimate
    const estimatedMonthlyCost = estimatedMonthlyCalls * 0.01; // $0.01 per call average

    return {
      dailyLimit: Math.max(50, Math.round(estimatedMonthlyCost / 30)),
      monthlyLimit: Math.round(estimatedMonthlyCost * 1.2), // 20% buffer
      alertThreshold: 0.8,
      emergencyThreshold: 0.95
    };
  }

  generateGeographicConfig(analysis) {
    return {
      regions: [
        { name: 'north_america', latency: 50 },
        { name: 'europe', latency: 100 },
        { name: 'asia_pacific', latency: 200 }
      ],
      fallbackChains: [
        {
          primary: 'north_america',
          fallbacks: ['europe', 'asia_pacific'],
          latencyThreshold: 1000
        }
      ]
    };
  }

  detectComplianceNeeds(analysis) {
    const frameworks = [];

    // Check for data handling patterns
    const hasPersonalData = analysis.aiIntegrations.some(int =>
      /email|phone|address|personal/i.test(int.content)
    );

    if (hasPersonalData) {
      frameworks.push('GDPR');
    }

    // Check for healthcare patterns
    const hasHealthData = analysis.aiIntegrations.some(int =>
      /health|medical|patient|diagnosis/i.test(int.content)
    );

    if (hasHealthData) {
      frameworks.push('HIPAA');
    }

    return frameworks;
  }

  async createBackups(plan) {
    console.log('💾 Creating backups...');

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupDir = path.join(this.targetPath, `backup-trae-${timestamp}`);

    fs.mkdirSync(backupDir, { recursive: true });

    // Backup files that will be modified
    for (const integration of plan.integrations) {
      if (integration.action === 'wrap_with_router' && fs.existsSync(path.join(this.targetPath, integration.file))) {
        const backupPath = path.join(backupDir, integration.file);
        fs.mkdirSync(path.dirname(backupPath), { recursive: true });

        fs.copyFileSync(
          path.join(this.targetPath, integration.file),
          backupPath
        );

        this.backupFiles.push({
          original: integration.file,
          backup: backupPath
        });
      }
    }

    console.log(`   Created ${this.backupFiles.length} backups in ${backupDir}`);
  }

  async applyTransformations(plan) {
    console.log('🔄 Applying transformations...');

    // Create directory structure
    for (const dir of plan.structure.directories) {
      const fullPath = path.join(this.targetPath, dir);
      fs.mkdirSync(fullPath, { recursive: true });
    }

    // Copy and adapt templates
    for (const component of plan.components) {
      await this.installComponent(component, plan.language);
    }

    // Apply integrations
    for (const integration of plan.integrations) {
      await this.applyIntegration(integration, plan);
    }

    // Write configurations
    await this.writeConfigurations(plan.configurations);

    console.log(`   Applied ${plan.components.length} components`);
    console.log(`   Modified ${plan.integrations.length} integrations`);
  }

  async installComponent(component, language) {
    const templatePath = path.join(this.templatesPath, 'core', `${component}.ts`);
    const targetDir = this.getComponentTargetDir(language);
    const targetFile = path.join(targetDir, `${component}.js`);

    if (!fs.existsSync(templatePath)) {
      console.warn(`   Template not found: ${templatePath}`);
      return;
    }

    // Read template
    let content = fs.readFileSync(templatePath, 'utf8');

    // Adapt for target language/framework
    content = this.adaptTemplateForLanguage(content, language);

    // Write adapted component
    fs.writeFileSync(targetFile, content);

    this.transformations.push({
      type: 'component_install',
      component,
      file: targetFile
    });

    console.log(`   Installed ${component} → ${targetFile}`);
  }

  adaptTemplateForLanguage(content, language) {
    // Basic adaptation - in production would be more sophisticated
    if (language.includes('JavaScript')) {
      // Convert TypeScript to JavaScript
      content = content
        .replace(/export class/g, 'class')
        .replace(/export interface/g, '// interface')
        .replace(/export enum/g, '// enum')
        .replace(/: string/g, '') // Remove type annotations
        .replace(/: number/g, '')
        .replace(/: boolean/g, '')
        .replace(/: any/g, '')
        .replace(/import.*from.*;/g, '// import statement converted')
        .replace(/export default/g, 'module.exports =');
    }

    return content;
  }

  getComponentTargetDir(language) {
    const baseDir = this.targetPath;

    if (fs.existsSync(path.join(baseDir, 'src'))) {
      return path.join(baseDir, 'src', 'routing', 'core');
    } else if (fs.existsSync(path.join(baseDir, 'app'))) {
      return path.join(baseDir, 'app', 'routing', 'core');
    } else {
      return path.join(baseDir, 'routing', 'core');
    }
  }

  async applyIntegration(integration, plan) {
    if (integration.type === 'ai_integration') {
      await this.wrapAIIntegration(integration, plan);
    } else if (integration.type === 'api_endpoints') {
      await this.createAPIEndpoints(integration, plan);
    }
  }

  async wrapAIIntegration(integration, plan) {
    const filePath = path.join(this.targetPath, integration.file);
    let content = fs.readFileSync(filePath, 'utf8');

    // Add TRAE router import
    const importStatement = this.generateImportStatement(plan.language);
    content = importStatement + '\n' + content;

    // Wrap AI calls with router
    content = this.wrapAICalls(content, plan.language);

    // Write modified content
    fs.writeFileSync(filePath, content);

    this.transformations.push({
      type: 'integration_wrap',
      file: integration.file,
      changes: 'Added TRAE router integration'
    });

    console.log(`   Wrapped AI integration in ${integration.file}`);
  }

  generateImportStatement(language) {
    if (language.includes('JavaScript')) {
      return "const { IntelligentRouter } = require('./routing/core/IntelligentRouter');";
    }
    return "// TRAE router import would go here";
  }

  wrapAICalls(content, language) {
    // Simple regex replacement - in production would use AST parsing
    const aiCallPatterns = [
      /\.create\(/g,
      /\.generate\(/g,
      /\.complete\(/g
    ];

    let modified = content;

    // Add router initialization if not present
    if (!modified.includes('IntelligentRouter')) {
      modified = modified.replace(
        /(async function|function|const|let|var)\s+\w+\s*\(/,
        (match) => {
          return `const router = new IntelligentRouter(config, aiEngine);\n\n${match}`;
        }
      );
    }

    // Wrap AI calls (simplified example)
    modified = modified.replace(
      /await (\w+)\.create\(([^)]+)\)/g,
      (match, client, args) => {
        return `await router.routeRequest(${args})`;
      }
    );

    return modified;
  }

  async createAPIEndpoints(integration, plan) {
    const filePath = path.join(this.targetPath, integration.file);
    const content = this.generateAPIEndpoints(plan.framework);

    fs.writeFileSync(filePath, content);

    this.transformations.push({
      type: 'api_creation',
      file: integration.file,
      endpoints: integration.endpoints.length
    });

    console.log(`   Created API endpoints in ${integration.file}`);
  }

  generateAPIEndpoints(framework) {
    if (framework === 'Express.js') {
      return `const express = require('express');
const { IntelligentRouter } = require('../core/IntelligentRouter');

const router = express.Router();
const traeRouter = new IntelligentRouter(config, aiEngine);

// Main routing endpoint
router.post('/route', async (req, res) => {
  try {
    const { prompt, context = {} } = req.body;
    const result = await traeRouter.routeRequest(prompt, context);

    res.json({
      response: result.response,
      cost: result.actualCost,
      model: result.routingDecision.selectedModel.name,
      requestId: result.requestId
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Metrics endpoint
router.get('/metrics', (req, res) => {
  const metrics = traeRouter.getMetrics();
  res.json(metrics);
});

// Budget status endpoint
router.get('/budget', (req, res) => {
  const costAnalysis = traeRouter.costController.getCostAnalysis();
  res.json(costAnalysis);
});

module.exports = router;
`;
    }

    return '// API endpoints for ' + framework;
  }

  async writeConfigurations(configurations) {
    const configDir = path.join(this.targetPath, 'src', 'routing', 'config');

    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }

    const configFiles = {
      'routing.config.json': configurations.routing,
      'models.config.json': configurations.models,
      'budget.config.json': configurations.budget,
      'geographic.config.json': configurations.geographic
    };

    for (const [filename, config] of Object.entries(configFiles)) {
      const filePath = path.join(configDir, filename);
      fs.writeFileSync(filePath, JSON.stringify(config, null, 2));

      this.transformations.push({
        type: 'config_creation',
        file: `src/routing/config/${filename}`
      });
    }

    console.log('   Created configuration files');
  }

  async updateDependencies(language) {
    console.log('📦 Updating dependencies...');

    if (language.includes('JavaScript')) {
      await this.updateNpmDependencies();
    } else if (language === 'Python') {
      await this.updatePythonDependencies();
    }

    console.log('   Dependencies updated');
  }

  async updateNpmDependencies() {
    const packagePath = path.join(this.targetPath, 'package.json');

    if (!fs.existsSync(packagePath)) {
      console.warn('   No package.json found, skipping dependency update');
      return;
    }

    let pkg = JSON.parse(fs.readFileSync(packagePath, 'utf8'));

    // Add TRAE dependencies
    pkg.dependencies = pkg.dependencies || {};
    pkg.dependencies['winston'] = '^3.8.2'; // Logging
    pkg.dependencies['ioredis'] = '^5.3.2'; // Caching
    pkg.dependencies['prom-client'] = '^14.2.0'; // Metrics

    fs.writeFileSync(packagePath, JSON.stringify(pkg, null, 2));

    // Run npm install
    try {
      execSync('npm install', { cwd: this.targetPath, stdio: 'inherit' });
    } catch (error) {
      console.warn('   npm install failed, you may need to run it manually');
    }
  }

  async updatePythonDependencies() {
    const requirementsPath = path.join(this.targetPath, 'requirements.txt');

    const newDeps = [
      'redis>=4.5.0',
      'prometheus-client>=0.16.0',
      'pydantic>=2.0.0'
    ];

    if (fs.existsSync(requirementsPath)) {
      let content = fs.readFileSync(requirementsPath, 'utf8');
      content += '\n# TRAE dependencies\n' + newDeps.join('\n') + '\n';
      fs.writeFileSync(requirementsPath, content);
    } else {
      fs.writeFileSync(requirementsPath, newDeps.join('\n') + '\n');
    }
  }

  async generateTests(plan) {
    console.log('🧪 Generating tests...');

    const testDir = path.join(this.targetPath, 'tests', 'routing');
    fs.mkdirSync(testDir, { recursive: true });

    // Generate basic integration test
    const testContent = this.generateIntegrationTest(plan);
    fs.writeFileSync(path.join(testDir, 'routing.integration.test.js'), testContent);

    console.log('   Generated integration tests');
  }

  generateIntegrationTest(plan) {
    return `const { IntelligentRouter } = require('../src/routing/core/IntelligentRouter');

describe('TRAE Routing Integration', () => {
  let router;
  let mockAiEngine;

  beforeEach(() => {
    mockAiEngine = {
      generateText: jest.fn().mockResolvedValue('Mock response'),
      getAvailableModels: jest.fn().mockReturnValue([
        {
          name: 'GPT-3.5-Turbo',
          provider: 'openai',
          costPerToken: 0.002,
          maxTokens: 4096
        }
      ])
    };

    router = new IntelligentRouter({}, mockAiEngine, {});
  });

  test('should route simple request successfully', async () => {
    const result = await router.routeRequest('Hello world');

    expect(result.success).toBe(true);
    expect(result.response).toBe('Mock response');
    expect(result.actualCost).toBeGreaterThan(0);
  });

  test('should apply cost optimization', async () => {
    const result = await router.routeRequest('Complex analysis task');

    expect(result.routingDecision.costSavings).toBeDefined();
    expect(result.routingDecision.selectedModel).toBeDefined();
  });
});
`;
  }

  async createDocumentation() {
    console.log('📚 Creating documentation...');

    const docs = {
      installation: this.generateInstallationDocs(),
      usage: this.generateUsageDocs(),
      api: this.generateAPIDocs(),
      troubleshooting: this.generateTroubleshootingDocs()
    };

    const docsDir = path.join(this.targetPath, 'docs', 'trae');
    fs.mkdirSync(docsDir, { recursive: true });

    for (const [name, content] of Object.entries(docs)) {
      fs.writeFileSync(path.join(docsDir, `${name}.md`), content);
    }

    console.log('   Documentation created');
  }

  generateInstallationDocs() {
    return `# TRAE Installation

## Overview
TRAE (Transparent Routing for AI Efficiency) has been successfully installed in your repository.

## What's Been Added
- Intelligent routing system with 7-step pipeline
- Cost optimization with automatic model selection
- Geographic failover across regions
- Comprehensive monitoring and metrics

## Next Steps
1. Configure your AI provider API keys
2. Test the routing system
3. Monitor cost savings and performance
4. Scale to production deployment

## Configuration
Update the following files with your settings:
- \`src/routing/config/routing.config.json\`
- \`src/routing/config/models.config.json\`
- Environment variables for API keys
`;
  }

  generateUsageDocs() {
    return `# TRAE Usage Guide

## Basic Usage

\`\`\`javascript
const { IntelligentRouter } = require('./src/routing/core/IntelligentRouter');

// Initialize router
const router = new IntelligentRouter(config, aiEngine);

// Route a request
const result = await router.routeRequest('Your prompt here', {
  userId: 'user123',
  priority: 'high'
});

console.log('Response:', result.response);
console.log('Cost:', result.actualCost);
console.log('Model used:', result.routingDecision.selectedModel.name);
\`\`\`

## Advanced Features

### Cost Control
\`\`\`javascript
// Check budget status
const budget = router.costController.getCostAnalysis();
console.log('Savings:', budget.currentSavings + '%');
\`\`\`

### Geographic Routing
\`\`\`javascript
// Force region failover
router.forceGeographicFailover('north_america', 'europe');
\`\`\`

### Monitoring
\`\`\`javascript
// Get metrics
const metrics = router.getMetrics();
console.log('Success rate:', metrics.successRate + '%');
\`\`\`
`;
  }

  generateAPIDocs() {
    return `# TRAE API Reference

## Endpoints

### POST /api/route
Route an AI request through the TRAE system.

**Request:**
\`\`\`json
{
  "prompt": "Your AI prompt",
  "context": {
    "userId": "optional-user-id",
    "priority": "normal|high|critical",
    "clientRegion": "north_america|europe|asia_pacific"
  }
}
\`\`\`

**Response:**
\`\`\`json
{
  "response": "AI response",
  "cost": 0.012,
  "model": "GPT-3.5-Turbo",
  "requestId": "req_123456"
}
\`\`\`

### GET /api/metrics
Get routing system metrics.

### GET /api/budget
Get cost analysis and budget status.
`;
  }

  generateTroubleshootingDocs() {
    return `# TRAE Troubleshooting

## Common Issues

### High Costs
- Check cost optimization mode in configuration
- Verify model selection is working
- Review budget limits

### Slow Responses
- Check geographic routing configuration
- Verify region health status
- Review latency thresholds

### Errors
- Check API key configuration
- Verify provider status
- Review error logs

## Support
- Check logs in \`logs/routing.log\`
- Review metrics at \`/api/metrics\`
- Test basic routing at \`/api/route\`
`;
  }

  async rollbackChanges() {
    console.log('🔄 Rolling back changes...');

    // Restore backup files
    for (const backup of this.backupFiles) {
      try {
        fs.copyFileSync(backup.backup, path.join(this.targetPath, backup.original));
        console.log(`   Restored ${backup.original}`);
      } catch (error) {
        console.error(`   Failed to restore ${backup.original}:`, error.message);
      }
    }

    // Remove created files
    for (const transformation of this.transformations) {
      try {
        const filePath = path.join(this.targetPath, transformation.file);
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
          console.log(`   Removed ${transformation.file}`);
        }
      } catch (error) {
        console.error(`   Failed to remove ${transformation.file}:`, error.message);
      }
    }

    console.log('   Rollback complete');
  }

  findFilesRecursively(dir, filter) {
    const results = [];

    function walk(currentDir) {
      try {
        const files = fs.readdirSync(path.join(this.targetPath, currentDir));

        for (const file of files) {
          const filePath = path.join(currentDir, file);
          const fullPath = path.join(this.targetPath, filePath);
          const stat = fs.statSync(fullPath);

          if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
            walk(filePath);
          } else if (stat.isFile() && filter(file)) {
            results.push(filePath);
          }
        }
      } catch (error) {
        // Skip inaccessible directories
      }
    }

    walk.call(this, dir);
    return results;
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const targetPath = args[0] || '.';
  const options = {};

  const applicator = new TemplateApplicator();
  applicator.applyTemplates(targetPath, options);
}

module.exports = TemplateApplicator;</content>
