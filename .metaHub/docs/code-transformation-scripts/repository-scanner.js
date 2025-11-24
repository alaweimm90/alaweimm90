#!/usr/bin/env node

/**
 * TRAE Repository Scanner
 * Analyzes existing AI integrations and generates transformation recommendations
 *
 * Usage: node repository-scanner.js [path] [options]
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class RepositoryScanner {
  constructor() {
    this.results = {
      repository: {},
      aiIntegrations: [],
      costAnalysis: {},
      architecture: {},
      compliance: {},
      transformation: {},
      recommendations: []
    };

    this.scanPatterns = {
      providers: {
        openai: {
          patterns: [/openai|OpenAI/, /gpt-|GPT-/],
          files: ['package.json', 'requirements.txt', 'go.mod']
        },
        anthropic: {
          patterns: [/anthropic|Anthropic/, /claude|Claude/],
          files: ['package.json', 'requirements.txt', 'go.mod']
        },
        google: {
          patterns: [/google|Google/, /gemini|Gemini|palm|PaLM|vertex|Vertex/],
          files: ['package.json', 'requirements.txt', 'go.mod']
        },
        local: {
          patterns: [/ollama|Ollama/, /local.*llm|together|Together/],
          files: ['docker-compose.yml', 'Dockerfile']
        }
      },
      apiCalls: [
        /\.create\(/,
        /\.generate\(/,
        /\.complete\(/,
        /api\.openai\.com/,
        /api\.anthropic\.com/,
        /generativelanguage\.googleapis\.com/
      ],
      costTracking: [
        /cost.*=|=.*cost/,
        /price.*=|=.*price/,
        /billing/,
        /usage.*token/
      ],
      errorHandling: [
        /try.*catch/,
        /error.*handling/,
        /\.catch\(/,
        /throw new/
      ]
    };
  }

  async scanRepository(scanPath = '.', options = {}) {
    console.log('🔍 Starting TRAE Repository Analysis...\n');

    this.results.repository.path = path.resolve(scanPath);
    this.results.repository.name = path.basename(this.results.repository.path);

    try {
      // Basic repository info
      await this.analyzeRepositoryBasics();

      // AI integration analysis
      await this.analyzeAIIntegrations();

      // Cost and usage analysis
      await this.analyzeCostAndUsage();

      // Architecture assessment
      await this.analyzeArchitecture();

      // Compliance requirements
      await this.analyzeComplianceRequirements();

      // Generate transformation recommendations
      this.generateTransformationRecommendations();

      // Generate report
      this.generateReport();

      console.log('✅ Repository analysis complete!');
      console.log(`📊 Found ${this.results.aiIntegrations.length} AI integrations`);
      console.log(`💰 Estimated monthly cost: $${this.estimateMonthlyCost()}`);
      console.log(`🎯 Transformation complexity: ${this.assessComplexity()}/5`);

    } catch (error) {
      console.error('❌ Analysis failed:', error.message);
      process.exit(1);
    }
  }

  async analyzeRepositoryBasics() {
    console.log('📁 Analyzing repository structure...');

    const packageJson = this.readFileIfExists('package.json');
    const requirementsTxt = this.readFileIfExists('requirements.txt');
    const goMod = this.readFileIfExists('go.mod');

    this.results.repository.language = this.detectLanguage();
    this.results.repository.framework = this.detectFramework();
    this.results.repository.dependencies = this.extractDependencies();

    // Get repository statistics
    const stats = this.getRepositoryStats();
    this.results.repository.stats = stats;

    console.log(`   Language: ${this.results.repository.language}`);
    console.log(`   Framework: ${this.results.repository.framework}`);
    console.log(`   Files: ${stats.fileCount}, Lines: ${stats.lineCount}`);
  }

  detectLanguage() {
    if (fs.existsSync('package.json')) return 'JavaScript/TypeScript';
    if (fs.existsSync('requirements.txt') || fs.existsSync('pyproject.toml')) return 'Python';
    if (fs.existsSync('go.mod')) return 'Go';
    if (fs.existsSync('Cargo.toml')) return 'Rust';
    if (fs.existsSync('pom.xml') || fs.existsSync('build.gradle')) return 'Java';
    return 'Unknown';
  }

  detectFramework() {
    const packageJson = this.readFileIfExists('package.json');
    if (packageJson) {
      const deps = JSON.parse(packageJson).dependencies || {};
      if (deps.express) return 'Express.js';
      if (deps['@nestjs/core']) return 'NestJS';
      if (deps.react) return 'React';
      if (deps.next) return 'Next.js';
    }
    return 'Unknown';
  }

  extractDependencies() {
    const deps = [];

    // JavaScript/TypeScript
    const packageJson = this.readFileIfExists('package.json');
    if (packageJson) {
      const pkg = JSON.parse(packageJson);
      deps.push(...Object.keys(pkg.dependencies || {}));
      deps.push(...Object.keys(pkg.devDependencies || {}));
    }

    // Python
    const requirements = this.readFileIfExists('requirements.txt');
    if (requirements) {
      deps.push(...requirements.split('\n').map(line => line.split('==')[0].trim()).filter(Boolean));
    }

    return deps;
  }

  getRepositoryStats() {
    const stats = {
      fileCount: 0,
      lineCount: 0,
      aiRelatedFiles: 0
    };

    function walkDir(dir) {
      const files = fs.readdirSync(dir);

      for (const file of files) {
        const filePath = path.join(dir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
          walkDir(filePath);
        } else if (stat.isFile() && this.isCodeFile(file)) {
          stats.fileCount++;

          try {
            const content = fs.readFileSync(filePath, 'utf8');
            stats.lineCount += content.split('\n').length;

            if (this.containsAIPatterns(content)) {
              stats.aiRelatedFiles++;
            }
          } catch (error) {
            // Skip binary files
          }
        }
      }
    }

    walkDir.call(this, '.');

    return stats;
  }

  isCodeFile(filename) {
    const extensions = ['.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.go', '.rs', '.php', '.rb'];
    return extensions.some(ext => filename.endsWith(ext));
  }

  containsAIPatterns(content) {
    const aiKeywords = ['openai', 'anthropic', 'claude', 'gpt', 'ai', 'llm', 'model'];
    return aiKeywords.some(keyword => content.toLowerCase().includes(keyword));
  }

  async analyzeAIIntegrations() {
    console.log('🤖 Analyzing AI integrations...');

    const files = this.findFilesRecursively('.', file => this.isCodeFile(file));

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');
      const integrations = this.detectAIIntegrations(content, file);

      if (integrations.length > 0) {
        this.results.aiIntegrations.push({
          file,
          integrations,
          lineCount: content.split('\n').length,
          lastModified: fs.statSync(file).mtime
        });
      }
    }

    console.log(`   Found ${this.results.aiIntegrations.length} files with AI integrations`);
  }

  detectAIIntegrations(content, filePath) {
    const integrations = [];

    // Check for provider-specific patterns
    for (const [provider, config] of Object.entries(this.scanPatterns.providers)) {
      const hasProvider = config.patterns.some(pattern => pattern.test(content));
      const hasDependency = config.files.some(depFile => {
        const depContent = this.readFileIfExists(depFile);
        return depContent && config.patterns.some(pattern => pattern.test(depContent));
      });

      if (hasProvider || hasDependency) {
        const usage = this.analyzeUsage(content, provider);
        integrations.push({
          provider,
          usage,
          confidence: hasProvider && hasDependency ? 'high' : 'medium'
        });
      }
    }

    return integrations;
  }

  analyzeUsage(content, provider) {
    const usage = {
      apiCalls: 0,
      errorHandling: false,
      costTracking: false,
      asyncPatterns: false
    };

    // Count API calls
    usage.apiCalls = this.scanPatterns.apiCalls.filter(pattern =>
      pattern.test(content)
    ).length;

    // Check error handling
    usage.errorHandling = this.scanPatterns.errorHandling.some(pattern =>
      pattern.test(content)
    );

    // Check cost tracking
    usage.costTracking = this.scanPatterns.costTracking.some(pattern =>
      pattern.test(content)
    );

    // Check async patterns
    usage.asyncPatterns = /async|await|Promise|callback/.test(content);

    return usage;
  }

  async analyzeCostAndUsage() {
    console.log('💰 Analyzing cost and usage patterns...');

    const totalCalls = this.results.aiIntegrations.reduce((sum, file) =>
      sum + file.integrations.reduce((fileSum, integration) =>
        fileSum + integration.usage.apiCalls, 0
      ), 0
    );

    // Estimate monthly usage (rough approximation)
    const estimatedMonthlyCalls = totalCalls * 30; // Assume daily usage
    const estimatedMonthlyCost = this.calculateEstimatedCost(estimatedMonthlyCalls);

    this.results.costAnalysis = {
      totalAICalls: totalCalls,
      estimatedMonthlyCalls,
      estimatedMonthlyCost,
      costTrackingImplemented: this.results.aiIntegrations.some(file =>
        file.integrations.some(int => int.usage.costTracking)
      ),
      providers: this.getProviderBreakdown()
    };

    console.log(`   Estimated monthly cost: $${estimatedMonthlyCost}`);
  }

  calculateEstimatedCost(monthlyCalls) {
    // Rough cost estimation based on GPT-4 pricing
    const avgCostPerCall = 0.03; // $0.03 per call average
    return Math.round(monthlyCalls * avgCostPerCall);
  }

  getProviderBreakdown() {
    const breakdown = {};

    this.results.aiIntegrations.forEach(file => {
      file.integrations.forEach(integration => {
        if (!breakdown[integration.provider]) {
          breakdown[integration.provider] = { files: 0, calls: 0 };
        }
        breakdown[integration.provider].files++;
        breakdown[integration.provider].calls += integration.usage.apiCalls;
      });
    });

    return breakdown;
  }

  async analyzeArchitecture() {
    console.log('🏗️ Analyzing architecture...');

    const architecture = {
      centralization: this.assessCentralization(),
      errorHandling: this.assessErrorHandling(),
      asyncPatterns: this.assessAsyncPatterns(),
      modularity: this.assessModularity()
    };

    this.results.architecture = architecture;

    console.log(`   Centralization: ${architecture.centralization}/5`);
    console.log(`   Error handling: ${architecture.errorHandling}/5`);
  }

  assessCentralization() {
    const aiFiles = this.results.aiIntegrations.length;
    const totalFiles = this.results.repository.stats.fileCount;

    // Lower ratio = more centralized
    const ratio = aiFiles / totalFiles;
    if (ratio < 0.1) return 5; // Very centralized
    if (ratio < 0.2) return 4;
    if (ratio < 0.3) return 3;
    if (ratio < 0.4) return 2;
    return 1; // Very scattered
  }

  assessErrorHandling() {
    const filesWithErrorHandling = this.results.aiIntegrations.filter(file =>
      file.integrations.some(int => int.usage.errorHandling)
    ).length;

    const ratio = filesWithErrorHandling / this.results.aiIntegrations.length;
    return Math.round(ratio * 5);
  }

  assessAsyncPatterns() {
    const filesWithAsync = this.results.aiIntegrations.filter(file =>
      file.integrations.some(int => int.usage.asyncPatterns)
    ).length;

    const ratio = filesWithAsync / this.results.aiIntegrations.length;
    return Math.round(ratio * 5);
  }

  assessModularity() {
    // Check for service/repository patterns
    const hasServices = this.fileExists('src/services') || this.fileExists('app/services');
    const hasRepositories = this.fileExists('src/repositories') || this.fileExists('app/repositories');
    const hasModules = this.fileExists('src/modules') || this.fileExists('app/modules');

    let score = 1;
    if (hasServices) score += 2;
    if (hasRepositories) score += 1;
    if (hasModules) score += 1;

    return Math.min(score, 5);
  }

  async analyzeComplianceRequirements() {
    console.log('🔒 Analyzing compliance requirements...');

    const compliance = {
      gdpr: this.detectGDPRRequirements(),
      hipaa: this.detectHIPAARequirements(),
      soc2: this.detectSOC2Requirements(),
      pci: this.detectPCIRequirements()
    };

    this.results.compliance = compliance;

    const requiredFrameworks = Object.entries(compliance)
      .filter(([_, detected]) => detected)
      .map(([framework, _]) => framework);

    console.log(`   Required frameworks: ${requiredFrameworks.join(', ') || 'None detected'}`);
  }

  detectGDPRRequirements() {
    const files = this.findFilesRecursively('.', file => this.isCodeFile(file));

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for EU-related terms or data handling
      if (/\b(eu|european|gdpr|personal.*data|consent)\b/i.test(content)) {
        return true;
      }

      // Check for data collection/storage patterns
      if (/\b(email|phone|address|name|user.*data)\b/i.test(content)) {
        return true;
      }
    }

    return false;
  }

  detectHIPAARequirements() {
    const files = this.findFilesRecursively('.', file => this.isCodeFile(file));

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for healthcare-related terms
      if (/\b(health|medical|patient|diagnosis|treatment|phi|hipaa)\b/i.test(content)) {
        return true;
      }
    }

    return false;
  }

  detectSOC2Requirements() {
    // SOC 2 is typically required for SaaS companies
    const packageJson = this.readFileIfExists('package.json');
    if (packageJson) {
      const pkg = JSON.parse(packageJson);
      // Check if it's a web service or API
      if (pkg.scripts && (pkg.scripts.start || pkg.scripts.dev)) {
        return true;
      }
    }

    return false;
  }

  detectPCIRequirements() {
    const files = this.findFilesRecursively('.', file => this.isCodeFile(file));

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf8');

      // Check for payment-related terms
      if (/\b(payment|credit.*card|transaction|pci|billing)\b/i.test(content)) {
        return true;
      }
    }

    return false;
  }

  generateTransformationRecommendations() {
    console.log('🎯 Generating transformation recommendations...');

    const recommendations = [];

    // Cost optimization recommendations
    if (this.results.costAnalysis.estimatedMonthlyCost > 1000) {
      recommendations.push({
        type: 'cost_optimization',
        priority: 'high',
        title: 'Implement TRAE Cost Optimization',
        description: `Expected 7-10x cost reduction from $${this.results.costAnalysis.estimatedMonthlyCost}/month`,
        effort: 'medium',
        impact: 'high'
      });
    }

    // Architecture recommendations
    if (this.results.architecture.centralization < 3) {
      recommendations.push({
        type: 'architecture',
        priority: 'high',
        title: 'Centralize AI Integrations',
        description: 'Consolidate scattered AI calls into centralized routing system',
        effort: 'medium',
        impact: 'high'
      });
    }

    // Compliance recommendations
    const requiredFrameworks = Object.entries(this.results.compliance)
      .filter(([_, required]) => required)
      .map(([framework, _]) => framework.toUpperCase());

    if (requiredFrameworks.length > 0) {
      recommendations.push({
        type: 'compliance',
        priority: 'high',
        title: 'Implement Compliance Frameworks',
        description: `Add support for: ${requiredFrameworks.join(', ')}`,
        effort: 'high',
        impact: 'high'
      });
    }

    // Error handling recommendations
    if (this.results.architecture.errorHandling < 3) {
      recommendations.push({
        type: 'reliability',
        priority: 'medium',
        title: 'Enhance Error Handling',
        description: 'Implement comprehensive error handling and fallback mechanisms',
        effort: 'medium',
        impact: 'medium'
      });
    }

    this.results.recommendations = recommendations;
  }

  assessComplexity() {
    let score = 1;

    // AI integration complexity
    if (this.results.aiIntegrations.length > 10) score += 1;
    if (Object.keys(this.results.costAnalysis.providers || {}).length > 2) score += 1;

    // Architecture complexity
    if (this.results.architecture.centralization < 3) score += 1;
    if (this.results.architecture.modularity < 3) score += 1;

    // Compliance complexity
    const complianceCount = Object.values(this.results.compliance).filter(Boolean).length;
    score += Math.min(complianceCount, 2);

    return Math.min(score, 5);
  }

  estimateMonthlyCost() {
    return this.results.costAnalysis.estimatedMonthlyCost || 0;
  }

  generateReport() {
    const report = {
      executiveSummary: {
        repository: this.results.repository.name,
        language: this.results.repository.language,
        aiIntegrations: this.results.aiIntegrations.length,
        estimatedMonthlyCost: this.estimateMonthlyCost(),
        transformationComplexity: this.assessComplexity(),
        requiredComplianceFrameworks: Object.entries(this.results.compliance)
          .filter(([_, required]) => required)
          .map(([framework, _]) => framework.toUpperCase())
      },
      detailedAnalysis: this.results,
      transformationRoadmap: this.generateRoadmap(),
      generatedAt: new Date().toISOString()
    };

    // Save report
    fs.writeFileSync('trae-analysis-report.json', JSON.stringify(report, null, 2));
    console.log('📄 Analysis report saved to trae-analysis-report.json');

    // Print summary
    this.printSummary(report);
  }

  generateRoadmap() {
    const complexity = this.assessComplexity();
    const weeks = complexity <= 2 ? 4 : complexity <= 4 ? 6 : 8;

    return {
      totalWeeks: weeks,
      phases: [
        {
          name: 'Foundation',
          weeks: 2,
          tasks: [
            'Set up TRAE routing infrastructure',
            'Implement basic model selection',
            'Add cost tracking'
          ]
        },
        {
          name: 'Core Features',
          weeks: Math.ceil(weeks * 0.4),
          tasks: [
            'Add geographic routing',
            'Implement cost optimization',
            'Set up monitoring'
          ]
        },
        {
          name: 'Compliance & Security',
          weeks: Math.ceil(weeks * 0.3),
          tasks: [
            'Add compliance frameworks',
            'Implement security controls',
            'Set up audit logging'
          ]
        },
        {
          name: 'Production & Optimization',
          weeks: Math.ceil(weeks * 0.3),
          tasks: [
            'Performance optimization',
            'Production deployment',
            'Monitoring and alerting'
          ]
        }
      ]
    };
  }

  printSummary(report) {
    console.log('\n📊 TRAE ANALYSIS SUMMARY');
    console.log('========================');
    console.log(`Repository: ${report.executiveSummary.repository}`);
    console.log(`Language: ${report.executiveSummary.language}`);
    console.log(`AI Integrations: ${report.executiveSummary.aiIntegrations}`);
    console.log(`Est. Monthly Cost: $${report.executiveSummary.estimatedMonthlyCost}`);
    console.log(`Complexity Score: ${report.executiveSummary.transformationComplexity}/5`);
    console.log(`Compliance: ${report.executiveSummary.requiredComplianceFrameworks.join(', ') || 'None'}`);
    console.log('\n🎯 TRANSFORMATION ROADMAP');
    console.log(`Total Duration: ${report.transformationRoadmap.totalWeeks} weeks`);

    report.transformationRoadmap.phases.forEach(phase => {
      console.log(`\n${phase.name} (${phase.weeks} weeks):`);
      phase.tasks.forEach(task => console.log(`  • ${task}`));
    });

    console.log('\n💡 RECOMMENDATIONS');
    this.results.recommendations.forEach(rec => {
      console.log(`  ${rec.priority.toUpperCase()}: ${rec.title}`);
      console.log(`    ${rec.description}`);
    });
  }

  // Utility methods
  readFileIfExists(filePath) {
    try {
      return fs.readFileSync(filePath, 'utf8');
    } catch (error) {
      return null;
    }
  }

  fileExists(filePath) {
    try {
      fs.accessSync(filePath);
      return true;
    } catch (error) {
      return false;
    }
  }

  findFilesRecursively(dir, filter) {
    const results = [];

    function walk(currentDir) {
      const files = fs.readdirSync(currentDir);

      for (const file of files) {
        const filePath = path.join(currentDir, file);
        const stat = fs.statSync(filePath);

        if (stat.isDirectory() && !file.startsWith('.') && file !== 'node_modules') {
          walk(filePath);
        } else if (stat.isFile() && filter(file)) {
          results.push(filePath);
        }
      }
    }

    walk(dir);
    return results;
  }
}

// CLI interface
if (require.main === module) {
  const args = process.argv.slice(2);
  const scanPath = args[0] || '.';
  const options = {};

  const scanner = new RepositoryScanner();
  scanner.scanRepository(scanPath, options);
}

module.exports = RepositoryScanner;</content>
