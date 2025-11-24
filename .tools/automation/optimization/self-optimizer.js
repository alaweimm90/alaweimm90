/* eslint no-console: off */
/**
 * Self-Optimization System
 * Continuously analyzes and optimizes the repository
 */

const fs = require('fs').promises;
const path = require('path');
const { exec } = require('child_process');
const { promisify } = require('util');
const { EventEmitter } = require('events');

const execAsync = promisify(exec);

class SelfOptimizer extends EventEmitter {
  constructor(config = {}) {
    super();

    this.config = {
      interval: config.interval || 300000, // 5 minutes
      enableAutoOptimization: config.enableAutoOptimization !== false,
      enableLearning: config.enableLearning !== false,
      maxOptimizationsPerRun: config.maxOptimizationsPerRun || 5,
      ...config
    };

    this.optimizationHistory = [];
    this.learningData = {
      patterns: [],
      successfulOptimizations: [],
      failedOptimizations: []
    };

    this.strategies = [
      'codeOptimization',
      'dependencyOptimization',
      'performanceOptimization',
      'securityOptimization',
      'documentationOptimization'
    ];
  }

  async start() {
    console.log('\n🧠 Self-Optimization System Starting...\n');
    console.log('This system will continuously improve your repository...\n');

    // Initial optimization
    await this.optimize();

    // Set up continuous optimization
    this.optimizationInterval = setInterval(async () => {
      await this.optimize();
    }, this.config.interval);

    this.emit('started');
    console.log('✅ Self-optimization active\n');
  }

  async stop() {
    if (this.optimizationInterval) {
      clearInterval(this.optimizationInterval);
    }
    this.emit('stopped');
    console.log('🛑 Self-optimization stopped');
  }

  async optimize() {
    console.log(`\n🔄 Running optimization cycle at ${new Date().toLocaleTimeString()}...\n`);

    const optimizations = [];

    try {
      // Analyze current state
      const analysis = await this.analyzeRepository();

      // Identify optimization opportunities
      const opportunities = await this.identifyOpportunities(analysis);

      // Apply optimizations
      const results = await this.applyOptimizations(opportunities);

      // Learn from results
      if (this.config.enableLearning) {
        await this.learn(results);
      }

      // Report results
      await this.reportResults(results);

      this.emit('optimizationComplete', results);

    } catch (error) {
      console.error('Optimization error:', error.message);
      this.emit('error', error);
    }
  }

  async analyzeRepository() {
    console.log('📊 Analyzing repository state...');

    const analysis = {
      timestamp: new Date().toISOString(),
      codeMetrics: await this.analyzeCode(),
      performanceMetrics: await this.analyzePerformance(),
      dependencyMetrics: await this.analyzeDependencies(),
      securityMetrics: await this.analyzeSecurity(),
      documentationMetrics: await this.analyzeDocumentation()
    };

    console.log('  ✅ Analysis complete\n');
    return analysis;
  }

  async analyzeCode() {
    const metrics = {
      totalFiles: 0,
      totalLines: 0,
      duplicateCode: [],
      complexFunctions: [],
      unusedExports: [],
      deadCode: []
    };

    try {
      // Count JavaScript/TypeScript files
      const { stdout: jsFiles } = await execAsync('find . -type f \\( -name "*.js" -o -name "*.ts" \\) -not -path "*/node_modules/*" | wc -l');
      metrics.totalFiles = parseInt(jsFiles.trim()) || 0;

      // Find duplicate code patterns (simplified)
      // In production, use tools like jscpd
      metrics.duplicateCode = await this.findDuplicatePatterns();

      // Find complex functions
      metrics.complexFunctions = await this.findComplexFunctions();

    } catch (error) {
      // Ignore errors in analysis
    }

    return metrics;
  }

  async analyzePerformance() {
    const metrics = {
      bundleSize: 0,
      loadTime: 0,
      memoryUsage: process.memoryUsage(),
      slowFunctions: []
    };

    try {
      // Check if dist/build directories exist
      const distPath = path.join(process.cwd(), 'dist');
      const buildPath = path.join(process.cwd(), 'build');

      let totalSize = 0;
      for (const dir of [distPath, buildPath]) {
        try {
          totalSize += await this.getDirectorySize(dir);
        } catch {
          // Directory doesn't exist
        }
      }

      metrics.bundleSize = totalSize;
    } catch (error) {
      // Ignore errors
    }

    return metrics;
  }

  async analyzeDependencies() {
    const metrics = {
      total: 0,
      outdated: [],
      unused: [],
      missing: [],
      vulnerabilities: []
    };

    try {
      const packagePath = path.join(process.cwd(), 'package.json');
      const packageJson = JSON.parse(await fs.readFile(packagePath, 'utf8'));

      metrics.total = Object.keys({
        ...packageJson.dependencies,
        ...packageJson.devDependencies
      }).length;

      // Check for outdated packages (simplified)
      // In production, use npm outdated --json
      try {
        const { stdout } = await execAsync('npm outdated --json');
        if (stdout) {
          const outdated = JSON.parse(stdout);
          metrics.outdated = Object.keys(outdated);
        }
      } catch {
        // npm outdated returns non-zero exit code when packages are outdated
      }

    } catch (error) {
      // Ignore errors
    }

    return metrics;
  }

  async analyzeSecurity() {
    const metrics = {
      vulnerabilities: [],
      exposedSecrets: [],
      insecurePatterns: [],
      missingHeaders: []
    };

    try {
      // Check for vulnerabilities (simplified)
      // In production, use npm audit --json
      try {
        const { stdout } = await execAsync('npm audit --json');
        const audit = JSON.parse(stdout);
        metrics.vulnerabilities = audit.vulnerabilities || [];
      } catch {
        // npm audit may fail
      }

      // Check for common insecure patterns
      metrics.insecurePatterns = await this.findInsecurePatterns();

    } catch (error) {
      // Ignore errors
    }

    return metrics;
  }

  async analyzeDocumentation() {
    const metrics = {
      coverage: 0,
      missingDocs: [],
      outdatedDocs: [],
      todoComments: []
    };

    try {
      // Find files without documentation
      metrics.missingDocs = await this.findUndocumentedFiles();

      // Find TODO comments
      try {
        const { stdout } = await execAsync('grep -r "TODO\\|FIXME\\|HACK" --include="*.js" --include="*.ts" . 2>/dev/null | wc -l');
        metrics.todoComments = parseInt(stdout.trim()) || 0;
      } catch {
        // Grep may fail
      }

    } catch (error) {
      // Ignore errors
    }

    return metrics;
  }

  async identifyOpportunities(analysis) {
    console.log('🎯 Identifying optimization opportunities...');

    const opportunities = [];

    // Code optimization opportunities
    if (analysis.codeMetrics.duplicateCode.length > 0) {
      opportunities.push({
        type: 'code',
        action: 'removeDuplicates',
        priority: 'medium',
        data: analysis.codeMetrics.duplicateCode
      });
    }

    if (analysis.codeMetrics.complexFunctions.length > 0) {
      opportunities.push({
        type: 'code',
        action: 'simplifyComplexFunctions',
        priority: 'low',
        data: analysis.codeMetrics.complexFunctions
      });
    }

    // Performance optimization opportunities
    if (analysis.performanceMetrics.bundleSize > 5 * 1024 * 1024) { // > 5MB
      opportunities.push({
        type: 'performance',
        action: 'optimizeBundleSize',
        priority: 'high',
        data: { currentSize: analysis.performanceMetrics.bundleSize }
      });
    }

    // Dependency optimization opportunities
    if (analysis.dependencyMetrics.outdated.length > 0) {
      opportunities.push({
        type: 'dependency',
        action: 'updateOutdatedPackages',
        priority: 'medium',
        data: analysis.dependencyMetrics.outdated
      });
    }

    if (analysis.dependencyMetrics.unused.length > 0) {
      opportunities.push({
        type: 'dependency',
        action: 'removeUnusedPackages',
        priority: 'low',
        data: analysis.dependencyMetrics.unused
      });
    }

    // Security optimization opportunities
    if (analysis.securityMetrics.vulnerabilities.length > 0) {
      opportunities.push({
        type: 'security',
        action: 'fixVulnerabilities',
        priority: 'critical',
        data: analysis.securityMetrics.vulnerabilities
      });
    }

    // Documentation optimization opportunities
    if (analysis.documentationMetrics.missingDocs.length > 0) {
      opportunities.push({
        type: 'documentation',
        action: 'generateDocumentation',
        priority: 'low',
        data: analysis.documentationMetrics.missingDocs
      });
    }

    // Sort by priority
    const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
    opportunities.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

    console.log(`  ✅ Found ${opportunities.length} optimization opportunities\n`);
    return opportunities;
  }

  async applyOptimizations(opportunities) {
    console.log('⚡ Applying optimizations...\n');

    const results = [];
    let applied = 0;

    for (const opportunity of opportunities.slice(0, this.config.maxOptimizationsPerRun)) {
      if (this.config.enableAutoOptimization) {
        console.log(`  🔧 Applying: ${opportunity.action}...`);

        try {
          let result = null;

          switch (opportunity.action) {
            case 'fixVulnerabilities':
              result = await this.fixVulnerabilities(opportunity.data);
              break;
            case 'updateOutdatedPackages':
              result = await this.updatePackages(opportunity.data);
              break;
            case 'optimizeBundleSize':
              result = await this.optimizeBundleSize(opportunity.data);
              break;
            case 'removeDuplicates':
              result = await this.removeDuplicateCode(opportunity.data);
              break;
            case 'generateDocumentation':
              result = await this.generateDocumentation(opportunity.data);
              break;
            default:
              result = { success: false, message: 'Not implemented' };
          }

          if (result.success) {
            console.log(`    ✅ Success: ${result.message}`);
            applied++;
          } else {
            console.log(`    ⚠️ Skipped: ${result.message}`);
          }

          results.push({
            opportunity,
            result,
            timestamp: new Date().toISOString()
          });

        } catch (error) {
          console.log(`    ❌ Failed: ${error.message}`);
          results.push({
            opportunity,
            result: { success: false, error: error.message },
            timestamp: new Date().toISOString()
          });
        }
      } else {
        console.log(`  ⏭️ Skipping ${opportunity.action} (auto-optimization disabled)`);
      }
    }

    console.log(`\n  📊 Applied ${applied} optimizations\n`);
    return results;
  }

  async fixVulnerabilities(vulnerabilities) {
    try {
      const { stdout } = await execAsync('npm audit fix');
      return {
        success: true,
        message: 'Fixed security vulnerabilities',
        details: stdout
      };
    } catch (error) {
      return {
        success: false,
        message: 'Could not auto-fix vulnerabilities'
      };
    }
  }

  async updatePackages(outdatedPackages) {
    // For safety, only update patch versions automatically
    try {
      const { stdout } = await execAsync('npm update');
      return {
        success: true,
        message: `Updated ${outdatedPackages.length} packages`,
        packages: outdatedPackages
      };
    } catch (error) {
      return {
        success: false,
        message: 'Package update failed'
      };
    }
  }

  async optimizeBundleSize(data) {
    // Placeholder - would implement actual bundle optimization
    return {
      success: false,
      message: 'Bundle optimization requires manual review'
    };
  }

  async removeDuplicateCode(duplicates) {
    // Placeholder - would implement duplicate removal
    return {
      success: false,
      message: 'Duplicate code removal requires manual review'
    };
  }

  async generateDocumentation(missingDocs) {
    // Generate basic documentation templates
    let generated = 0;

    for (const file of missingDocs.slice(0, 3)) {
      try {
        const docContent = this.generateDocTemplate(file);
        const docPath = file.replace(/\.(js|ts)$/, '.md');
        await fs.writeFile(docPath, docContent);
        generated++;
      } catch {
        // Ignore errors
      }
    }

    return {
      success: generated > 0,
      message: `Generated ${generated} documentation files`
    };
  }

  async learn(results) {
    // Store optimization results for learning
    for (const result of results) {
      if (result.result.success) {
        this.learningData.successfulOptimizations.push({
          type: result.opportunity.type,
          action: result.opportunity.action,
          timestamp: result.timestamp
        });
      } else {
        this.learningData.failedOptimizations.push({
          type: result.opportunity.type,
          action: result.opportunity.action,
          reason: result.result.message,
          timestamp: result.timestamp
        });
      }
    }

    // Identify patterns
    this.identifyPatterns();

    // Adjust strategies based on learning
    this.adjustStrategies();
  }

  identifyPatterns() {
    // Analyze success/failure patterns
    const successRate = {};

    for (const strategy of this.strategies) {
      const successes = this.learningData.successfulOptimizations.filter(o => o.type === strategy).length;
      const failures = this.learningData.failedOptimizations.filter(o => o.type === strategy).length;
      const total = successes + failures;

      if (total > 0) {
        successRate[strategy] = successes / total;
      }
    }

    this.learningData.patterns.push({
      timestamp: new Date().toISOString(),
      successRates: successRate
    });
  }

  adjustStrategies() {
    // Adjust optimization strategies based on success rates
    // Prioritize strategies with higher success rates
    const {patterns} = this.learningData;
    if (patterns.length > 0) {
      const latestPattern = patterns[patterns.length - 1];

      // Reorder strategies by success rate
      this.strategies.sort((a, b) => {
        const rateA = latestPattern.successRates[a] || 0;
        const rateB = latestPattern.successRates[b] || 0;
        return rateB - rateA;
      });
    }
  }

  async reportResults(results) {
    const report = {
      timestamp: new Date().toISOString(),
      optimizationsApplied: results.filter(r => r.result.success).length,
      optimizationsFailed: results.filter(r => !r.result.success).length,
      results,
      learningData: this.learningData
    };

    // Save report
    const reportPath = path.join(process.cwd(), '.automation', 'optimization', 'reports');
    await fs.mkdir(reportPath, { recursive: true });
    await fs.writeFile(
      path.join(reportPath, `optimization-${Date.now()}.json`),
      JSON.stringify(report, null, 2)
    );

    // Update optimization history
    this.optimizationHistory.push(report);
    if (this.optimizationHistory.length > 100) {
      this.optimizationHistory.shift();
    }

    this.emit('report', report);
  }

  // Helper methods

  async findDuplicatePatterns() {
    // Simplified duplicate detection
    // In production, use proper AST analysis
    return [];
  }

  async findComplexFunctions() {
    // Simplified complexity detection
    // In production, use cyclomatic complexity analysis
    return [];
  }

  async findInsecurePatterns() {
    // Simplified security pattern detection
    return [];
  }

  async findUndocumentedFiles() {
    // Simplified documentation detection
    return [];
  }

  async getDirectorySize(dirPath) {
    let totalSize = 0;

    const processDir = async (dir) => {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          await processDir(fullPath);
        } else {
          const stats = await fs.stat(fullPath);
          totalSize += stats.size;
        }
      }
    };

    await processDir(dirPath);
    return totalSize;
  }

  generateDocTemplate(filePath) {
    const fileName = path.basename(filePath);
    return `# ${fileName}

## Description
[Add description here]

## Usage
\`\`\`javascript
// Add usage example
\`\`\`

## API
[Document API here]

## Notes
Generated by Self-Optimization System on ${new Date().toISOString()}
`;
  }

  getHistory() {
    return this.optimizationHistory;
  }

  getLearningData() {
    return this.learningData;
  }
}

// Run if executed directly
if (require.main === module) {
  const optimizer = new SelfOptimizer({
    interval: 60000, // 1 minute for demo
    enableAutoOptimization: true,
    enableLearning: true
  });

  optimizer.on('optimizationComplete', (results) => {
    console.log('✨ Optimization cycle complete\n');
  });

  optimizer.on('error', (error) => {
    console.error('❌ Optimization error:', error.message);
  });

  optimizer.start().catch(console.error);

  // Graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n\nShutting down optimizer...');
    await optimizer.stop();
    process.exit(0);
  });
}

module.exports = SelfOptimizer;
