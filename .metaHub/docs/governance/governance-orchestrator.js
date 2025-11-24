#!/usr/bin/env node

/**
 * Governance Orchestrator
 * Central control system for all governance operations
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const RepositoryValidator = require('./validators/repository-validator');
const FileTrackingSystem = require('./registry/file-tracking-system');
const AutomatedCleanup = require('./scripts/automated-cleanup');
const AuditSystem = require('./audit/audit-system');

class GovernanceOrchestrator {
  constructor(options = {}) {
    this.rootPath = options.rootPath || process.cwd();
    this.configPath = path.join(this.rootPath, '.governance', 'governance-config.json');

    // Initialize subsystems
    this.audit = new AuditSystem({ rootPath: this.rootPath });
    this.validator = new RepositoryValidator(this.rootPath);
    this.tracker = new FileTrackingSystem(this.rootPath);
    this.cleanup = new AutomatedCleanup(this.rootPath);

    // Load configuration
    this.config = this.loadConfig();

    // Set up audit event listeners
    this.setupAuditListeners();
  }

  loadConfig() {
    if (fs.existsSync(this.configPath)) {
      return JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
    }
    console.error('⚠️ Governance configuration not found');
    return {};
  }

  setupAuditListeners() {
    // Listen for critical events
    this.audit.on('alert', entry => {
      console.error(
        `🚨 ALERT: ${entry.eventType} - ${entry.details.message || 'Critical event occurred'}`
      );
      // In production, this would send notifications
    });
  }

  // Main orchestration function
  async orchestrate(command = 'full', options = {}) {
    console.log(`🎯 Running Governance Orchestration: ${command}\n`);

    const commands = {
      full: this.runFullGovernance.bind(this),
      validate: this.runValidation.bind(this),
      track: this.runTracking.bind(this),
      cleanup: this.runCleanup.bind(this),
      audit: this.runAudit.bind(this),
      fix: this.runAutoFix.bind(this),
      report: this.generateReport.bind(this),
      install: this.install.bind(this),
      status: this.getStatus.bind(this),
    };

    if (commands[command]) {
      return await commands[command](options);
    } else {
      console.error(`❌ Unknown command: ${command}`);
      this.showHelp();
      return null;
    }
  }

  // Run full governance check
  async runFullGovernance(options) {
    console.log('🛡️ Running Full Governance Check\n');

    const results = {
      validation: null,
      tracking: null,
      cleanup: null,
      audit: null,
      status: 'unknown',
    };

    try {
      // 1. Validation
      console.log('Step 1/4: Validation');
      results.validation = await this.runValidation(options);

      // 2. File Tracking
      console.log('\nStep 2/4: File Tracking');
      results.tracking = await this.runTracking(options);

      // 3. Cleanup
      console.log('\nStep 3/4: Cleanup');
      results.cleanup = await this.runCleanup({ ...options, autoFix: false });

      // 4. Audit
      console.log('\nStep 4/4: Audit');
      results.audit = await this.runAudit(options);

      // Determine overall status
      results.status = this.determineOverallStatus(results);

      // Log to audit
      await this.audit.logAudit(this.audit.eventTypes.AUTOMATION_RUN, {
        action: 'full-governance-check',
        result: results.status,
        details: results,
      });
    } catch (error) {
      console.error(`❌ Governance check failed: ${error.message}`);
      await this.audit.logAudit(
        this.audit.eventTypes.ERROR,
        {
          action: 'full-governance-check',
          result: 'failed',
          details: { error: error.message },
        },
        { severity: this.audit.severityLevels.HIGH }
      );
    }

    return results;
  }

  // Run validation
  async runValidation(options) {
    const report = await this.validator.validate();

    // Log to audit
    await this.audit.logAudit(this.audit.eventTypes.VALIDATION_RUN, {
      action: 'repository-validation',
      target: this.rootPath,
      result: report.status,
      details: {
        score: report.score,
        violations: report.violations.length,
        warnings: report.warnings.length,
      },
    });

    // Auto-fix if requested
    if (options.autoFix) {
      const fixed = await this.validator.autoFix();
      console.log(`🔧 Auto-fixed ${fixed} issues`);
    }

    return report;
  }

  // Run file tracking
  async runTracking(options) {
    const tracking = await this.tracker.performFullScan();

    // Log to audit
    await this.audit.logAudit(this.audit.eventTypes.AUTOMATION_RUN, {
      action: 'file-tracking-scan',
      target: this.rootPath,
      result: 'success',
      details: tracking.statistics,
    });

    // Generate tracking report
    const report = this.tracker.generateReport();

    if (report.issues.length > 0) {
      console.log(`⚠️ Found ${report.issues.length} tracking issues`);
    }

    return tracking;
  }

  // Run cleanup
  async runCleanup(options) {
    const stats = await this.cleanup.performCleanup({
      ...options,
      report: true,
    });

    // Log to audit
    await this.audit.logAudit(this.audit.eventTypes.CLEANUP_RUN, {
      action: 'automated-cleanup',
      target: this.rootPath,
      result: 'success',
      details: stats,
    });

    return stats;
  }

  // Run audit report
  async runAudit(options) {
    const report = await this.audit.generateReport(options);

    console.log('📊 Audit Report Summary:');
    console.log(`  Total Events: ${report.summary.totalEvents}`);
    console.log(`  Compliance Status: ${report.summary.complianceStatus}`);
    console.log(`  Critical Events: ${report.summary.criticalEvents.length}`);

    return report;
  }

  // Run auto-fix
  async runAutoFix(options) {
    console.log('🔧 Running Auto-Fix\n');

    const results = {
      validation: 0,
      formatting: 0,
      structure: 0,
    };

    // 1. Fix validation issues
    console.log('Fixing validation issues...');
    results.validation = await this.validator.autoFix();

    // 2. Fix formatting
    if (fs.existsSync('package.json')) {
      console.log('Fixing formatting issues...');
      try {
        execSync('npm run format', { stdio: 'pipe' });
        results.formatting = 1;
      } catch (e) {
        console.log('  ⚠️ Formatting fix failed');
      }
    }

    // 3. Fix structure
    console.log('Fixing structure issues...');
    const mandatoryDirs = this.config.repositoryStandards?.structure?.mandatoryDirectories || [];
    mandatoryDirs.forEach(dir => {
      const dirPath = path.join(this.rootPath, dir);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        results.structure++;
      }
    });

    // Log to audit
    await this.audit.logAudit(this.audit.eventTypes.AUTOMATION_RUN, {
      action: 'auto-fix',
      result: 'success',
      details: results,
    });

    console.log(`\n✅ Fixed ${results.validation + results.formatting + results.structure} issues`);
    return results;
  }

  // Generate comprehensive report
  async generateReport(options) {
    console.log('📊 Generating Comprehensive Governance Report\n');

    const report = {
      generated: new Date().toISOString(),
      repository: this.rootPath,
      compliance: {},
      validation: {},
      tracking: {},
      audit: {},
      recommendations: [],
    };

    // Get validation status
    report.validation = await this.runValidation({ ...options, silent: true });

    // Get tracking status
    report.tracking = await this.runTracking({ ...options, silent: true });

    // Get audit summary
    report.audit = await this.runAudit({ ...options, silent: true });

    // Determine compliance
    report.compliance = {
      status: report.validation.score >= 80 ? 'compliant' : 'non-compliant',
      score: report.validation.score,
      frameworks: this.config.governance?.complianceFrameworks || [],
    };

    // Generate recommendations
    if (report.validation.score < 80) {
      report.recommendations.push('Improve repository structure to meet compliance standards');
    }
    if (report.tracking.statistics.incompleteFiles > 10) {
      report.recommendations.push('Complete documentation for incomplete files');
    }
    if (report.audit.summary.criticalEvents.length > 0) {
      report.recommendations.push('Review and address critical security events');
    }

    // Save report
    const reportPath = path.join(
      this.rootPath,
      '.governance',
      'reports',
      `governance-report-${new Date().toISOString().split('T')[0]}.json`
    );

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`📄 Report saved to: ${reportPath}`);

    // Display summary
    this.displayReportSummary(report);

    return report;
  }

  // Install governance
  async install(options) {
    console.log('🚀 Installing Governance Framework\n');

    try {
      execSync('node .governance/setup/install-governance.js', { stdio: 'inherit' });
    } catch (error) {
      console.error('❌ Installation failed:', error.message);
      return false;
    }

    return true;
  }

  // Get current status
  async getStatus(options) {
    console.log('📊 Governance Status\n');

    const status = {
      config: fs.existsSync(this.configPath) ? 'configured' : 'not-configured',
      git: fs.existsSync('.git') ? 'initialized' : 'not-initialized',
      hooks: fs.existsSync('.husky') ? 'installed' : 'not-installed',
      lastValidation: null,
      lastCleanup: null,
      compliance: 'unknown',
    };

    // Check last validation
    const validationReports = fs
      .readdirSync(path.join(this.rootPath, '.governance', 'reports'))
      .filter(f => f.startsWith('validation-'))
      .sort()
      .reverse();

    if (validationReports.length > 0) {
      const lastReport = JSON.parse(
        fs.readFileSync(
          path.join(this.rootPath, '.governance', 'reports', validationReports[0]),
          'utf8'
        )
      );
      status.lastValidation = {
        date: lastReport.timestamp,
        score: lastReport.score,
        status: lastReport.status,
      };
      status.compliance = lastReport.score >= 80 ? 'compliant' : 'non-compliant';
    }

    // Display status
    console.log('Configuration:', status.config);
    console.log('Git Repository:', status.git);
    console.log('Git Hooks:', status.hooks);
    console.log('Compliance:', status.compliance);

    if (status.lastValidation) {
      console.log('\nLast Validation:');
      console.log('  Date:', status.lastValidation.date);
      console.log('  Score:', status.lastValidation.score + '%');
      console.log('  Status:', status.lastValidation.status);
    }

    return status;
  }

  // Determine overall status
  determineOverallStatus(results) {
    if (!results.validation || !results.tracking) return 'unknown';

    if (results.validation.score >= 90 && results.tracking.statistics.incompleteFiles === 0) {
      return 'excellent';
    } else if (results.validation.score >= 80) {
      return 'good';
    } else if (results.validation.score >= 60) {
      return 'needs-improvement';
    } else {
      return 'poor';
    }
  }

  // Display report summary
  displayReportSummary(report) {
    console.log('\n' + '═'.repeat(50));
    console.log('GOVERNANCE REPORT SUMMARY');
    console.log('═'.repeat(50));
    console.log(`Repository: ${report.repository}`);
    console.log(`Generated: ${report.generated}`);
    console.log(`Compliance Status: ${report.compliance.status.toUpperCase()}`);
    console.log(`Compliance Score: ${report.compliance.score}%`);
    console.log(`Validation Score: ${report.validation.score}%`);
    console.log(`Total Files: ${report.tracking.statistics.totalFiles}`);
    console.log(`Incomplete Files: ${report.tracking.statistics.incompleteFiles}`);
    console.log(`Critical Events: ${report.audit.summary.criticalEvents.length}`);

    if (report.recommendations.length > 0) {
      console.log('\n📌 Recommendations:');
      report.recommendations.forEach((rec, i) => {
        console.log(`  ${i + 1}. ${rec}`);
      });
    }

    console.log('═'.repeat(50));
  }

  // Show help
  showHelp() {
    console.log('\nGovernance Orchestrator Commands:');
    console.log('  full      - Run full governance check');
    console.log('  validate  - Run repository validation');
    console.log('  track     - Run file tracking scan');
    console.log('  cleanup   - Run automated cleanup');
    console.log('  audit     - Generate audit report');
    console.log('  fix       - Run auto-fix for issues');
    console.log('  report    - Generate comprehensive report');
    console.log('  install   - Install governance framework');
    console.log('  status    - Get current governance status');
    console.log('\nUsage: node governance-orchestrator.js [command] [options]');
  }
}

// Export for use
module.exports = GovernanceOrchestrator;

// Run if called directly
if (require.main === module) {
  const orchestrator = new GovernanceOrchestrator();
  const command = process.argv[2] || 'status';
  const options = {
    autoFix: process.argv.includes('--fix'),
    silent: process.argv.includes('--silent'),
  };

  orchestrator.orchestrate(command, options).then(results => {
    if (results) {
      console.log('\n✅ Governance orchestration completed');
    }
  });
}
