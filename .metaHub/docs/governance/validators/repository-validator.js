/**
 * Repository Validator
 * Enforces repository structure standards and naming conventions
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

const DEFAULT_IGNORES = [
  '**/node_modules/**',
  '**/.git/**',
  '**/.next/**',
  '**/dist/**',
  '**/build/**',
  '**/.archive/**',
  '**/.archieve/**',
  '**/organizations/**',
  '**/vendor/**',
  '**/tmp/**',
  '**/__pycache__/**',
  '**/compliance/evidence-collection/**',
  '**/alaweimm90/**',
  '**/config/otel-nodejs-init.ts',
  '**/tmp-experiments/**',
  '**/archive/**',
];

class RepositoryValidator {
  constructor(repoPath = process.cwd()) {
    this.repoPath = repoPath;
    this.config = this.loadConfig();
    this.violations = [];
    this.warnings = [];
    this.passed = [];
  }

  loadConfig() {
    const configPath = path.join(this.repoPath, '.governance', 'governance-config.json');
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }
    throw new Error('Governance configuration not found. Please run governance setup first.');
  }

  // Main validation function
  async validate() {
    console.log('🔍 Starting repository validation...\n');

    // Reset results
    this.violations = [];
    this.warnings = [];
    this.passed = [];

    // Run all validations
    await this.validateStructure();
    await this.validateNaming();
    await this.validateDocumentation();
    await this.validateSecurity();
    await this.validateCompliance();

    // Generate report
    return this.generateReport();
  }

  // Validate repository structure
  async validateStructure() {
    console.log('📁 Validating repository structure...');
    const { mandatoryFiles, mandatoryDirectories } = this.config.repositoryStandards.structure;

    // Check mandatory files
    mandatoryFiles.forEach(file => {
      const filePath = path.join(this.repoPath, file);
      if (fs.existsSync(filePath)) {
        this.passed.push({
          type: 'structure',
          item: file,
          message: `Required file exists: ${file}`,
        });
      } else {
        this.violations.push({
          type: 'structure',
          severity: 'high',
          item: file,
          message: `Missing required file: ${file}`,
          fix: `Create ${file} using template`,
        });
      }
    });

    // Check mandatory directories
    mandatoryDirectories.forEach(dir => {
      const dirPath = path.join(this.repoPath, dir);
      if (fs.existsSync(dirPath) && fs.statSync(dirPath).isDirectory()) {
        this.passed.push({
          type: 'structure',
          item: dir,
          message: `Required directory exists: ${dir}`,
        });
      } else {
        this.violations.push({
          type: 'structure',
          severity: 'high',
          item: dir,
          message: `Missing required directory: ${dir}`,
          fix: `Create directory: mkdir ${dir}`,
        });
      }
    });

    // Check for proper .gitignore
    const gitignorePath = path.join(this.repoPath, '.gitignore');
    if (fs.existsSync(gitignorePath)) {
      const gitignore = fs.readFileSync(gitignorePath, 'utf8');
      const requiredEntries = ['node_modules/', '.env', '*.log', 'dist/', 'build/'];
      requiredEntries.forEach(entry => {
        if (!gitignore.includes(entry)) {
          this.warnings.push({
            type: 'structure',
            severity: 'medium',
            item: '.gitignore',
            message: `Missing entry in .gitignore: ${entry}`,
            fix: `Add "${entry}" to .gitignore`,
          });
        }
      });
    }
  }

  // Validate naming conventions
  async validateNaming() {
    console.log('📝 Validating naming conventions...');
    const namingRules = this.config.repositoryStandards.naming;

    // Check directory names
    const directories = glob.sync('**/', {
      cwd: this.repoPath,
      ignore: DEFAULT_IGNORES,
    });

    directories.forEach(dir => {
      const dirName = path.basename(dir.slice(0, -1));
      if (dirName && !/^[a-z0-9-]+$/.test(dirName)) {
        this.violations.push({
          type: 'naming',
          severity: 'medium',
          item: dir,
          message: `Directory name violates convention: ${dirName}`,
          fix: `Rename to lowercase: ${dirName.toLowerCase()}`,
        });
      }
    });

    // Check file names
    const files = glob.sync('**/*.*', {
      cwd: this.repoPath,
      ignore: DEFAULT_IGNORES,
    });

    const exceptions = namingRules.files.exceptions || [];
    files.forEach(file => {
      const fileName = path.basename(file);
      const ext = path.extname(file).toLowerCase();

      // Only enforce for JS/TS app code; skip other artifact types
      if (!['.js', '.jsx', '.ts', '.tsx'].includes(ext)) return;

      // Skip documentation/report artifacts from naming enforcement
      if (['.md', '.txt', '.pdf'].includes(ext)) return;
      if (exceptions.includes(fileName)) return;
      if (fileName.includes('.config.')) return;
      const allowList = [
        'App.tsx',
        'App.css',
        'App.test.tsx',
        'setupTests.ts',
        'react-app-env.d.ts',
        'tailwind.config.js',
        'postcss.config.js',
        'commitlint.config.js',
        'jest.setup.ts',
        'next-env.d.ts',
        'smoke.spec.ts',
      ];
      if (allowList.includes(fileName)) return;

      // Allow common patterns/locations
      if (/\b(test|spec)\.[jt]sx?$/.test(fileName)) return;
      if (file.startsWith('sample-project/components/')) return;
      if (file.startsWith('sample-project/lib/')) return;
      if (file.startsWith('sample-project/tests/')) return;
      if (file.startsWith('sample-project/e2e/')) return;
      if (file.startsWith('config/pci-dss/')) return;
      if (/\.service\.ts$/.test(fileName) || /\.example\.ts$/.test(fileName)) return;

      const nameWithoutExt = path.basename(file, ext);
      if (!/^[a-z0-9-]+$/.test(nameWithoutExt)) {
        this.warnings.push({
          type: 'naming',
          severity: 'low',
          item: file,
          message: `File name should be kebab-case: ${fileName}`,
          fix: `Rename to: ${nameWithoutExt.toLowerCase().replace(/[^a-z0-9]/g, '-')}${ext}`,
        });
      }
    });
  }

  // Validate documentation
  async validateDocumentation() {
    console.log('📚 Validating documentation...');
    const docConfig = this.config.repositoryStandards.documentation;

    // Check README.md content
    const readmePath = path.join(this.repoPath, 'README.md');
    if (fs.existsSync(readmePath)) {
      const readme = fs.readFileSync(readmePath, 'utf8');
      const requiredSections = [
        '## Installation',
        '## Usage',
        '## Configuration',
        '## API',
        '## Contributing',
        '## License',
      ];

      requiredSections.forEach(section => {
        if (!readme.includes(section)) {
          this.warnings.push({
            type: 'documentation',
            severity: 'medium',
            item: 'README.md',
            message: `Missing section: ${section}`,
            fix: `Add ${section} section to README.md`,
          });
        }
      });
    }

    // Check for API documentation
    const srcFiles = glob.sync('src/**/*.{js,ts}', {
      cwd: this.repoPath,
      ignore: ['**/*.test.*', '**/*.spec.*', ...DEFAULT_IGNORES],
    });

    let undocumentedFiles = 0;
    srcFiles.forEach(file => {
      const filePath = path.join(this.repoPath, file);
      const content = fs.readFileSync(filePath, 'utf8');

      // Check for JSDoc/TSDoc comments
      if (!content.includes('/**') && !content.includes('///')) {
        undocumentedFiles++;
      }
    });

    if (undocumentedFiles > 0) {
      const coverage = Math.round(((srcFiles.length - undocumentedFiles) / srcFiles.length) * 100);
      if (coverage < docConfig.minCoverage) {
        this.violations.push({
          type: 'documentation',
          severity: 'medium',
          item: 'src/',
          message: `Documentation coverage ${coverage}% is below minimum ${docConfig.minCoverage}%`,
          fix: 'Add JSDoc/TSDoc comments to all public functions',
        });
      }
    }
  }

  // Validate security
  async validateSecurity() {
    console.log('🔒 Validating security...');

    // Check for security policy
    const securityPath = path.join(this.repoPath, 'SECURITY.md');
    if (!fs.existsSync(securityPath)) {
      this.violations.push({
        type: 'security',
        severity: 'high',
        item: 'SECURITY.md',
        message: 'Missing security policy',
        fix: 'Create SECURITY.md with vulnerability reporting process',
      });
    }

    // Check for exposed secrets
    const dangerousPatterns = [
      /api[_-]?key\s*=\s*["'][^"']+["']/gi,
      /password\s*=\s*["'][^"']+["']/gi,
      /secret\s*=\s*["'][^"']+["']/gi,
      /token\s*=\s*["'][^"']+["']/gi,
      /private[_-]?key\s*=\s*["'][^"']+["']/gi,
    ];

    const codeFiles = glob.sync('**/*.{js,ts,jsx,tsx,py,java,go}', {
      cwd: this.repoPath,
      ignore: DEFAULT_IGNORES,
    });

    codeFiles.forEach(file => {
      const filePath = path.join(this.repoPath, file);

      // Skip if it's a directory
      if (fs.statSync(filePath).isDirectory()) {
        return;
      }
      // Skip self-referential security tooling to reduce false positives
      if (file.includes('advanced_gitleaks_scanner.py')) return;

      const content = fs.readFileSync(filePath, 'utf8');

      dangerousPatterns.forEach(pattern => {
        if (pattern.test(content)) {
          this.violations.push({
            type: 'security',
            severity: 'critical',
            item: file,
            message: 'Possible exposed secret detected',
            fix: 'Move secrets to environment variables',
          });
        }
      });
    });

    // Check for vulnerable dependencies
    const packageJsonPath = path.join(this.repoPath, 'package.json');
    if (fs.existsSync(packageJsonPath)) {
      const packageLockPath = path.join(this.repoPath, 'package-lock.json');
      if (!fs.existsSync(packageLockPath)) {
        this.warnings.push({
          type: 'security',
          severity: 'medium',
          item: 'package-lock.json',
          message: 'Missing package-lock.json',
          fix: 'Run npm install to generate lock file',
        });
      }
    }
  }

  // Validate compliance
  async validateCompliance() {
    console.log('⚖️ Validating compliance...');

    // Check for license
    const licensePath = path.join(this.repoPath, 'LICENSE');
    if (!fs.existsSync(licensePath)) {
      this.violations.push({
        type: 'compliance',
        severity: 'high',
        item: 'LICENSE',
        message: 'Missing license file',
        fix: 'Add appropriate license file',
      });
    }

    // Check for contributing guide
    const contributingPath = path.join(this.repoPath, 'CONTRIBUTING.md');
    if (!fs.existsSync(contributingPath)) {
      this.warnings.push({
        type: 'compliance',
        severity: 'medium',
        item: 'CONTRIBUTING.md',
        message: 'Missing contributing guide',
        fix: 'Create CONTRIBUTING.md with contribution guidelines',
      });
    }

    // Check for code of conduct
    const cocPath = path.join(this.repoPath, 'CODE_OF_CONDUCT.md');
    if (!fs.existsSync(cocPath)) {
      this.warnings.push({
        type: 'compliance',
        severity: 'medium',
        item: 'CODE_OF_CONDUCT.md',
        message: 'Missing code of conduct',
        fix: 'Create CODE_OF_CONDUCT.md',
      });
    }

    // Check for data privacy compliance
    const privacyIndicators = ['GDPR', 'PRIVACY', 'DATA_PROCESSING'];
    const hasPrivacyDocs = privacyIndicators.some(indicator => {
      return (
        glob.sync(`**/*${indicator}*`, {
          cwd: this.repoPath,
          ignore: DEFAULT_IGNORES,
        }).length > 0
      );
    });

    if (!hasPrivacyDocs && this.config.governance.complianceFrameworks.includes('GDPR')) {
      this.warnings.push({
        type: 'compliance',
        severity: 'medium',
        item: 'Privacy',
        message: 'No GDPR/privacy documentation found',
        fix: 'Add privacy policy and data processing documentation',
      });
    }
  }

  // Generate validation report
  generateReport() {
    const totalChecks = this.passed.length + this.violations.length + this.warnings.length;
    const score = Math.round((this.passed.length / totalChecks) * 100);

    const report = {
      timestamp: new Date().toISOString(),
      repository: this.repoPath,
      score: score,
      status:
        this.violations.filter(v => v.severity === 'critical').length > 0
          ? 'failed'
          : this.violations.filter(v => v.severity === 'high').length > 0
            ? 'needs-attention'
            : 'passed',
      summary: {
        totalChecks: totalChecks,
        passed: this.passed.length,
        violations: this.violations.length,
        warnings: this.warnings.length,
      },
      violations: this.violations,
      warnings: this.warnings,
      passed: this.passed,
    };

    // Save report
    const reportPath = path.join(
      this.repoPath,
      '.governance',
      'reports',
      `validation-${new Date().toISOString().split('T')[0]}.json`
    );

    const reportDir = path.dirname(reportPath);
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

    // Print summary
    console.log('\n' + '='.repeat(50));
    console.log('📊 Validation Report Summary');
    console.log('='.repeat(50));
    console.log(`Score: ${score}% (${report.status})`);
    console.log(`✅ Passed: ${this.passed.length}`);
    console.log(`❌ Violations: ${this.violations.length}`);
    console.log(`⚠️ Warnings: ${this.warnings.length}`);

    if (this.violations.length > 0) {
      console.log('\n❌ Critical Issues:');
      this.violations
        .filter(v => v.severity === 'critical')
        .forEach(v => {
          console.log(`  - ${v.message}`);
          console.log(`    Fix: ${v.fix}`);
        });

      console.log('\n⚠️ High Priority Issues:');
      this.violations
        .filter(v => v.severity === 'high')
        .forEach(v => {
          console.log(`  - ${v.message}`);
          console.log(`    Fix: ${v.fix}`);
        });
    }

    console.log(`\nFull report saved to: ${reportPath}`);
    console.log('='.repeat(50));

    return report;
  }

  // Auto-fix violations where possible
  async autoFix() {
    console.log('🔧 Attempting to auto-fix violations...\n');
    let fixedCount = 0;

    for (const violation of this.violations) {
      if (violation.type === 'structure' && violation.item.endsWith('/')) {
        // Create missing directory
        const dirPath = path.join(this.repoPath, violation.item);
        if (!fs.existsSync(dirPath)) {
          fs.mkdirSync(dirPath, { recursive: true });
          console.log(`✅ Created directory: ${violation.item}`);
          fixedCount++;
        }
      } else if (violation.type === 'structure' && violation.severity === 'high') {
        // Create missing file from template
        const filePath = path.join(this.repoPath, violation.item);
        if (!fs.existsSync(filePath)) {
          // Create basic file
          const content = this.getFileTemplate(violation.item);
          fs.writeFileSync(filePath, content);
          console.log(`✅ Created file: ${violation.item}`);
          fixedCount++;
        }
      }
    }

    console.log(`\n🔧 Fixed ${fixedCount} violations automatically`);
    return fixedCount;
  }

  getFileTemplate(filename) {
    const templates = {
      'README.md': `# Project Name

## Description
Brief description of your project.

## Installation
\`\`\`bash
npm install
\`\`\`

## Usage
\`\`\`bash
npm start
\`\`\`

## Configuration
Describe configuration options.

## API
API documentation.

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License
See [LICENSE](LICENSE) file.`,

      LICENSE: `MIT License

Copyright (c) ${new Date().getFullYear()}

Permission is hereby granted, free of charge...`,

      'SECURITY.md': `# Security Policy

## Reporting Security Vulnerabilities

Please report security vulnerabilities to security@example.com.

## Security Update Policy

We release security updates as soon as possible after discovering and fixing vulnerabilities.`,

      'CONTRIBUTING.md': `# Contributing Guidelines

## How to Contribute

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Code Style

Follow the project's coding standards.

## Testing

All code must include tests.`,

      'CODE_OF_CONDUCT.md': `# Code of Conduct

## Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone.

## Our Standards

Examples of behavior that contributes to a positive environment include:
- Being respectful
- Being inclusive
- Being collaborative

## Enforcement

Violations may result in temporary or permanent ban.`,

      '.gitignore': `# Dependencies
node_modules/
vendor/

# Build outputs
dist/
build/
*.min.js
*.min.css

# Environment
.env
.env.local
.env.*.local

# Logs
*.log
logs/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
coverage/
.nyc_output/

# Temporary
tmp/
temp/`,

      'CHANGELOG.md': `# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project setup

### Changed
- None

### Deprecated
- None

### Removed
- None

### Fixed
- None

### Security
- None`,
    };

    return templates[filename] || `# ${filename}\n\nTODO: Add content`;
  }
}

// Export for use in other scripts
module.exports = RepositoryValidator;

// Run if called directly
if (require.main === module) {
  const validator = new RepositoryValidator();
  validator.validate().then(report => {
    if (report.violations.length > 0) {
      console.log(
        '\n🔧 Would you like to auto-fix violations? Run: node repository-validator.js --fix'
      );
    }
  });

  // Check for --fix flag
  if (process.argv.includes('--fix')) {
    validator.autoFix();
  }
}
