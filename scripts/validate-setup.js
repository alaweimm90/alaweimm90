#!/usr/bin/env node

/**
 * Validation Script - Verify MCP & Agent infrastructure setup
 * Usage: node scripts/validate-setup.js [--verbose]
 */

const fs = require('fs');
const path = require('path');

class SetupValidator {
  constructor(options = {}) {
    this.verbose = options.verbose || false;
    this.rootDir = process.cwd();
    this.results = {
      passed: 0,
      failed: 0,
      warnings: 0,
      errors: [],
      warnings_list: [],
    };
  }

  log(message, type = 'info') {
    const prefix = {
      info: '[ℹ️  INFO]',
      success: '[✅ PASS]',
      warning: '[⚠️  WARN]',
      error: '[❌ FAIL]',
    }[type];
    console.log(`${prefix} ${message}`);
  }

  /**
   * Check if file exists
   * @param filepath
   * @param description
   */
  checkFileExists(filepath, description) {
    const fullPath = path.join(this.rootDir, filepath);
    if (fs.existsSync(fullPath)) {
      this.log(`${description}: ${filepath}`, 'success');
      this.results.passed++;
      return true;
    } 
      this.log(`${description}: ${filepath} NOT FOUND`, 'error');
      this.results.failed++;
      this.results.errors.push(`Missing: ${filepath}`);
      return false;
    
  }

  /**
   * Validate JSON file
   * @param filepath
   * @param description
   */
  validateJSON(filepath, description) {
    const fullPath = path.join(this.rootDir, filepath);
    try {
      if (!fs.existsSync(fullPath)) {
        this.log(`${description}: File not found`, 'error');
        this.results.failed++;
        return false;
      }
      const content = fs.readFileSync(fullPath, 'utf-8');
      JSON.parse(content);
      this.log(`${description}: Valid JSON`, 'success');
      this.results.passed++;
      return true;
    } catch (error) {
      this.log(`${description}: Invalid JSON - ${error.message}`, 'error');
      this.results.failed++;
      this.results.errors.push(`Invalid JSON in ${filepath}: ${error.message}`);
      return false;
    }
  }

  /**
   * Check package.json
   * @param packageDir
   * @param packageName
   */
  checkPackageJson(packageDir, packageName) {
    const filepath = path.join(packageDir, 'package.json');
    return this.validateJSON(filepath, `Package ${packageName}`);
  }

  /**
   * Validate MCP configuration
   */
  validateMCPConfig() {
    this.log('\n=== MCP Configuration ===', 'info');

    const configPath = '.claude/mcp-config.json';
    if (!this.validateJSON(configPath, 'MCP Config')) {
      return false;
    }

    const fullPath = path.join(this.rootDir, configPath);
    const config = JSON.parse(fs.readFileSync(fullPath, 'utf-8'));

    // Check structure
    if (!config.mcpServers) {
      this.log('MCP Config: Missing mcpServers object', 'error');
      this.results.failed++;
      return false;
    }

    if (!Array.isArray(config.enabled)) {
      this.log('MCP Config: Missing enabled array', 'error');
      this.results.failed++;
      return false;
    }

    const serverCount = Object.keys(config.mcpServers).length;
    const enabledCount = config.enabled.length;
    this.log(`MCP Config: ${serverCount} servers defined, ${enabledCount} enabled`, 'success');
    this.results.passed++;

    return true;
  }

  /**
   * Validate agent configuration
   */
  validateAgentConfig() {
    this.log('\n=== Agent Configuration ===', 'info');

    const configPath = '.claude/agents.json';
    if (!this.validateJSON(configPath, 'Agents Config')) {
      return false;
    }

    const fullPath = path.join(this.rootDir, configPath);
    const config = JSON.parse(fs.readFileSync(fullPath, 'utf-8'));

    if (!Array.isArray(config.agents)) {
      this.log('Agents Config: Missing agents array', 'error');
      this.results.failed++;
      return false;
    }

    const agentCount = config.agents.length;
    this.log(`Agents Config: ${agentCount} agents defined`, 'success');
    this.results.passed++;

    return true;
  }

  /**
   * Validate orchestration rules
   */
  validateOrchestrationRules() {
    this.log('\n=== Orchestration Rules ===', 'info');

    const configPath = '.claude/orchestration.json';
    if (!this.validateJSON(configPath, 'Orchestration Config')) {
      return false;
    }

    const fullPath = path.join(this.rootDir, configPath);
    const config = JSON.parse(fs.readFileSync(fullPath, 'utf-8'));

    if (!Array.isArray(config.rules)) {
      this.log('Orchestration Config: Missing rules array', 'error');
      this.results.failed++;
      return false;
    }

    const ruleCount = config.rules.length;
    this.log(`Orchestration Config: ${ruleCount} rules defined`, 'success');
    this.results.passed++;

    return true;
  }

  /**
   * Check package structure
   */
  checkPackageStructure() {
    this.log('\n=== Package Structure ===', 'info');

    const packages = [
      'packages/mcp-core',
      'packages/agent-core',
      'packages/context-provider',
      'packages/issue-library',
      'packages/workflow-templates',
    ];

    for (const pkg of packages) {
      this.checkFileExists(`${pkg}/package.json`, `Package ${path.basename(pkg)}`);
      this.checkFileExists(`${pkg}/tsconfig.json`, `TypeScript config for ${path.basename(pkg)}`);
      this.checkFileExists(`${pkg}/src/index.ts`, `Source index for ${path.basename(pkg)}`);
    }
  }

  /**
   * Check documentation
   */
  checkDocumentation() {
    this.log('\n=== Documentation ===', 'info');

    const docs = [
      'GETTING_STARTED.md',
      'IMPLEMENTATION_SUMMARY.md',
      'MCP_SERVERS_GUIDE.md',
      'docs/MCP_AGENTS_ORCHESTRATION.md',
      'docs/QUICK_START.md',
      'docs/ARCHITECTURE.md',
    ];

    for (const doc of docs) {
      this.checkFileExists(doc, `Documentation: ${path.basename(doc)}`);
    }
  }

  /**
   * Check infrastructure
   */
  checkInfrastructure() {
    this.log('\n=== Infrastructure ===', 'info');

    this.checkFileExists('.devcontainer/Dockerfile', 'DevContainer Dockerfile');
    this.checkFileExists('.devcontainer/devcontainer.json', 'DevContainer config');
    this.checkFileExists('scripts/mcp-setup.js', 'MCP setup script');
    this.checkFileExists('scripts/agent-setup.js', 'Agent setup script');
  }

  /**
   * Check Claude Code configuration
   */
  checkClaudeCodeConfig() {
    this.log('\n=== Claude Code Configuration ===', 'info');

    this.validateJSON('.claude/mcp-config.json', '.claude/mcp-config.json');

    if (fs.existsSync(path.join(this.rootDir, '.claude/agents'))) {
      this.log('.claude/agents/ directory exists', 'success');
      this.results.passed++;
    } else {
      this.log('.claude/agents/ directory not found', 'warning');
      this.results.warnings++;
    }

    if (fs.existsSync(path.join(this.rootDir, '.claude/workflows'))) {
      this.log('.claude/workflows/ directory exists', 'success');
      this.results.passed++;
    } else {
      this.log('.claude/workflows/ directory not found', 'warning');
      this.results.warnings++;
    }
  }

  /**
   * Run all validations
   */
  async validate() {
    this.log('╔════════════════════════════════════════╗');
    this.log('║     MCP & Agent Setup Validator        ║');
    this.log('╚════════════════════════════════════════╝');
    this.log(`Validating setup in: ${this.rootDir}\n`);

    // Run all checks
    this.checkPackageStructure();
    this.validateMCPConfig();
    this.validateAgentConfig();
    this.validateOrchestrationRules();
    this.checkDocumentation();
    this.checkInfrastructure();
    this.checkClaudeCodeConfig();

    // Print summary
    this.printSummary();

    return this.results.failed === 0;
  }

  /**
   * Print summary
   */
  printSummary() {
    const total = this.results.passed + this.results.failed + this.results.warnings;

    this.log('\n╔════════════════════════════════════════╗');
    this.log('║           Validation Summary            ║');
    this.log('╚════════════════════════════════════════╝');
    this.log(`Total Checks: ${total}`);
    this.log(`Passed: ${this.results.passed}`, 'success');
    this.log(`Failed: ${this.results.failed}`, this.results.failed > 0 ? 'error' : 'info');
    this.log(`Warnings: ${this.results.warnings}`, this.results.warnings > 0 ? 'warning' : 'info');

    if (this.results.errors.length > 0) {
      this.log('\n=== Errors ===', 'error');
      this.results.errors.forEach(error => this.log(`  • ${error}`, 'error'));
    }

    if (this.results.warnings_list.length > 0) {
      this.log('\n=== Warnings ===', 'warning');
      this.results.warnings_list.forEach(warning => this.log(`  • ${warning}`, 'warning'));
    }

    // Status
    this.log('\n');
    if (this.results.failed === 0) {
      this.log('✅ Setup validation PASSED!', 'success');
      this.log('\nNext steps:');
      this.log('1. Run: pnpm install');
      this.log('2. Run: pnpm build');
      this.log('3. Try a workflow: node scripts/agent-setup.js');
      return true;
    } 
      this.log('❌ Setup validation FAILED!', 'error');
      this.log('\nFix the errors above and run validation again.');
      return false;
    
  }
}

// Run validator
const args = process.argv.slice(2);
const options = {
  verbose: args.includes('--verbose'),
};

const validator = new SetupValidator(options);
validator.validate().then(success => {
  process.exit(success ? 0 : 1);
});
