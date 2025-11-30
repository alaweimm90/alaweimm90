#!/usr/bin/env node

/**
 * ATLAS Validation Tests
 * Comprehensive test suite to validate ATLAS functionality
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class AtlasValidator {
    constructor() {
        this.testResults = [];
        this.demoDir = path.join(__dirname, '..');
        this.repoDir = path.join(this.demoDir, 'test-repos');
        this.logDir = path.join(this.demoDir, 'logs');
    }

    log(message, level = 'INFO') {
        const timestamp = new Date().toISOString();
        console.log(`[${timestamp}] ${level}: ${message}`);
    }

    runCommand(command, description) {
        this.log(`Running: ${description}`);
        try {
            const result = execSync(command, {
                cwd: this.demoDir,
                encoding: 'utf8',
                timeout: 30000
            });
            this.log(`✓ ${description} completed successfully`);
            return { success: true, output: result };
        } catch (error) {
            this.log(`✗ ${description} failed: ${error.message}`, 'ERROR');
            return { success: false, error: error.message };
        }
    }

    validateCliInstallation() {
        this.log('Validating ATLAS CLI installation...');

        const result = this.runCommand('atlas --version', 'Check ATLAS CLI version');
        if (!result.success) {
            throw new Error('ATLAS CLI not found or not working');
        }

        this.testResults.push({
            test: 'CLI Installation',
            status: 'PASS',
            details: result.output.trim()
        });
    }

    validateTestRepositories() {
        this.log('Validating test repositories...');

        const repos = ['messy-python', 'complex-js', 'spaghetti-ts'];
        for (const repo of repos) {
            const repoPath = path.join(this.repoDir, repo);
            if (!fs.existsSync(repoPath)) {
                throw new Error(`Test repository ${repo} not found`);
            }

            // Check for expected files
            const expectedFiles = {
                'messy-python': ['main.py'],
                'complex-js': ['app.js'],
                'spaghetti-ts': ['service.ts']
            };

            const files = expectedFiles[repo] || [];
            for (const file of files) {
                const filePath = path.join(repoPath, file);
                if (!fs.existsSync(filePath)) {
                    throw new Error(`Expected file ${file} not found in ${repo}`);
                }
            }
        }

        this.testResults.push({
            test: 'Test Repositories',
            status: 'PASS',
            details: 'All test repositories and files present'
        });
    }

    validateAnalysisCommands() {
        this.log('Validating analysis commands...');

        const testRepo = path.join(this.repoDir, 'messy-python');

        // Test repo analysis
        const repoResult = this.runCommand(
            `atlas analyze repo "${testRepo}" --format json`,
            'Repository analysis'
        );

        if (!repoResult.success) {
            this.testResults.push({
                test: 'Repository Analysis',
                status: 'FAIL',
                details: repoResult.error
            });
        } else {
            try {
                const data = JSON.parse(repoResult.output);
                this.testResults.push({
                    test: 'Repository Analysis',
                    status: 'PASS',
                    details: `Analyzed ${data.filesAnalyzed || 0} files`
                });
            } catch (e) {
                this.testResults.push({
                    test: 'Repository Analysis',
                    status: 'FAIL',
                    details: 'Invalid JSON output'
                });
            }
        }

        // Test complexity analysis
        const complexityResult = this.runCommand(
            `atlas analyze complexity "${testRepo}"`,
            'Complexity analysis'
        );

        this.testResults.push({
            test: 'Complexity Analysis',
            status: complexityResult.success ? 'PASS' : 'FAIL',
            details: complexityResult.success ? 'Completed' : complexityResult.error
        });

        // Test chaos analysis
        const chaosResult = this.runCommand(
            `atlas analyze chaos "${testRepo}"`,
            'Chaos analysis'
        );

        this.testResults.push({
            test: 'Chaos Analysis',
            status: chaosResult.success ? 'PASS' : 'FAIL',
            details: chaosResult.success ? 'Completed' : chaosResult.error
        });

        // Test quick scan
        const scanResult = this.runCommand(
            `atlas analyze scan "${testRepo}"`,
            'Quick scan'
        );

        this.testResults.push({
            test: 'Quick Scan',
            status: scanResult.success ? 'PASS' : 'FAIL',
            details: scanResult.success ? 'Completed' : scanResult.error
        });
    }

    validateOutputFormats() {
        this.log('Validating output formats...');

        const testRepo = path.join(this.repoDir, 'messy-python');
        const formats = ['table', 'summary', 'json'];

        for (const format of formats) {
            const result = this.runCommand(
                `atlas analyze repo "${testRepo}" --format ${format}`,
                `Output format: ${format}`
            );

            if (format === 'json' && result.success) {
                try {
                    JSON.parse(result.output);
                    this.testResults.push({
                        test: `Output Format: ${format}`,
                        status: 'PASS',
                        details: 'Valid JSON output'
                    });
                } catch (e) {
                    this.testResults.push({
                        test: `Output Format: ${format}`,
                        status: 'FAIL',
                        details: 'Invalid JSON output'
                    });
                }
            } else {
                this.testResults.push({
                    test: `Output Format: ${format}`,
                    status: result.success ? 'PASS' : 'FAIL',
                    details: result.success ? 'Completed' : result.error
                });
            }
        }
    }

    validatePerformance() {
        this.log('Validating performance...');

        const testRepo = path.join(this.repoDir, 'messy-python');
        const startTime = Date.now();

        const result = this.runCommand(
            `atlas analyze repo "${testRepo}" --format json`,
            'Performance test'
        );

        const endTime = Date.now();
        const duration = endTime - startTime;

        // Performance should complete within 30 seconds
        const acceptableDuration = 30000;

        this.testResults.push({
            test: 'Performance',
            status: duration < acceptableDuration ? 'PASS' : 'WARN',
            details: `Completed in ${duration}ms (${duration < acceptableDuration ? 'acceptable' : 'slow'})`
        });
    }

    generateReport() {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const reportPath = path.join(this.logDir, `validation-report-${timestamp}.json`);

        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                total: this.testResults.length,
                passed: this.testResults.filter(t => t.status === 'PASS').length,
                failed: this.testResults.filter(t => t.status === 'FAIL').length,
                warnings: this.testResults.filter(t => t.status === 'WARN').length
            },
            results: this.testResults
        };

        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        console.log('\n' + '='.repeat(60));
        console.log('VALIDATION REPORT');
        console.log('='.repeat(60));
        console.log(`Total Tests: ${report.summary.total}`);
        console.log(`Passed: ${report.summary.passed}`);
        console.log(`Failed: ${report.summary.failed}`);
        console.log(`Warnings: ${report.summary.warnings}`);
        console.log(`Report saved: ${reportPath}`);

        if (report.summary.failed > 0) {
            console.log('\nFAILED TESTS:');
            report.results.filter(t => t.status === 'FAIL').forEach(test => {
                console.log(`- ${test.test}: ${test.details}`);
            });
        }

        return report;
    }

    async runAllTests() {
        this.log('Starting ATLAS validation tests...');

        try {
            this.validateCliInstallation();
            this.validateTestRepositories();
            this.validateAnalysisCommands();
            this.validateOutputFormats();
            this.validatePerformance();

            const report = this.generateReport();

            if (report.summary.failed === 0) {
                this.log('All validation tests passed!', 'SUCCESS');
                return true;
            } else {
                this.log(`${report.summary.failed} validation tests failed`, 'ERROR');
                return false;
            }

        } catch (error) {
            this.log(`Validation failed: ${error.message}`, 'ERROR');
            return false;
        }
    }
}

// Run validation if called directly
if (require.main === module) {
    const validator = new AtlasValidator();
    validator.runAllTests().then(success => {
        process.exit(success ? 0 : 1);
    });
}

module.exports = AtlasValidator;