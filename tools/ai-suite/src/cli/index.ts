#!/usr/bin/env node

/**
 * AI Tools Suite - Unified CLI
 *
 * Provides command-line access to all AI-powered development tools
 * with seamless ATLAS integration.
 */

import { Command } from 'commander';
import chalk from 'chalk';
import { AITools } from '../core/ai-tools.js';
import { ATLASIntegration } from '../core/atlas-integration.js';

const program = new Command();

program
  .name('ai-tools')
  .description('AI-powered development tools suite leveraging ATLAS architecture')
  .version('1.0.0');

// Global options
program
  .option('--atlas-url <url>', 'ATLAS server URL', 'http://localhost:8000')
  .option('--atlas-key <key>', 'ATLAS API key')
  .option('--verbose', 'Enable verbose output')
  .option('--json', 'Output results as JSON');

// Test Generator commands
const testCmd = program
  .command('test')
  .description('AI-powered test generation and analysis');

testCmd
  .command('generate')
  .description('Generate comprehensive test suite')
  .option('--path <path>', 'Source code path', './src')
  .option('--framework <framework>', 'Test framework (jest, vitest, mocha, etc.)', 'jest')
  .option('--language <language>', 'Programming language', 'javascript')
  .option('--output <dir>', 'Output directory', './tests')
  .option('--coverage <target>', 'Coverage target (0-100)', '80')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.test.generate({
        framework: options.framework,
        language: options.language,
        output: { directory: options.output }
      });

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Test suite generated successfully!'));
          console.log(`Generated ${result.data?.files?.length || 0} test files`);
        } else {
          console.log(chalk.red('❌ Test generation failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Documentation Generator commands
const docsCmd = program
  .command('docs')
  .description('AI-powered documentation generation');

docsCmd
  .command('generate')
  .description('Generate API documentation')
  .option('--path <path>', 'Source code path', './src')
  .option('--format <format>', 'Output format (markdown, html, pdf)', 'markdown')
  .option('--output <file>', 'Output file', './docs/api.md')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.docs.generate({
        format: options.format,
        output: options.output
      });

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Documentation generated successfully!'));
        } else {
          console.log(chalk.red('❌ Documentation generation failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Code Review commands
const reviewCmd = program
  .command('review')
  .description('AI-powered code review and analysis');

reviewCmd
  .command('analyze')
  .description('Analyze code for issues and improvements')
  .option('--path <path>', 'Source code path', './src')
  .option('--files <files>', 'Specific files to review (comma-separated)')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const files = options.files ? options.files.split(',') : undefined;
      const result = await aiTools.review.analyze(files || [options.path]);

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Code review completed!'));
          console.log(`Found ${result.data?.length || 0} issues`);
        } else {
          console.log(chalk.red('❌ Code review failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Architecture Analysis commands
const archCmd = program
  .command('arch')
  .description('AI-powered architecture analysis');

archCmd
  .command('analyze')
  .description('Analyze system architecture and dependencies')
  .option('--path <path>', 'Source code path', './src')
  .option('--output <file>', 'Output file for architecture report')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.arch.analyze(options.path);

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Architecture analysis completed!'));
        } else {
          console.log(chalk.red('❌ Architecture analysis failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Performance Profiling commands
const perfCmd = program
  .command('perf')
  .description('AI-powered performance profiling');

perfCmd
  .command('profile')
  .description('Profile application performance')
  .option('--path <path>', 'Source code path', './src')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.perf.analyze(options.path);

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Performance profiling completed!'));
        } else {
          console.log(chalk.red('❌ Performance profiling failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Security Scanning commands
const securityCmd = program
  .command('security')
  .description('AI-powered security scanning');

securityCmd
  .command('scan')
  .description('Scan for security vulnerabilities')
  .option('--path <path>', 'Source code path', './src')
  .option('--report <file>', 'Output report file')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.security.analyze(options.path);

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Security scan completed!'));
        } else {
          console.log(chalk.red('❌ Security scan failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Unified analysis command
program
  .command('analyze')
  .description('Run comprehensive analysis across all tools')
  .option('--path <path>', 'Source code path', './src')
  .option('--tools <tools>', 'Tools to run (comma-separated)', 'test,docs,review,arch,perf,security')
  .option('--output <file>', 'Output report file')
  .action(async (options) => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const enabledTools = options.tools.split(',');

      const result = await aiTools.analyze(options.path, {
        tools: enabledTools,
        output: options.output
      });

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ Comprehensive analysis completed!'));
          console.log(`Tools run: ${enabledTools.join(', ')}`);
        } else {
          console.log(chalk.red('❌ Analysis failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Health check command
program
  .command('health')
  .description('Check system and tool health')
  .action(async () => {
    try {
      const atlas = new ATLASIntegration({
        url: program.opts().atlasUrl,
        apiKey: program.opts().atlasKey
      });

      const aiTools = new AITools({ atlas });
      const result = await aiTools.health();

      if (program.opts().json) {
        console.log(JSON.stringify(result, null, 2));
      } else {
        if (result.success) {
          console.log(chalk.green('✅ System health check passed!'));
        } else {
          console.log(chalk.red('❌ Health check failed:'), result.error);
        }
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
    }
  });

// Parse and execute
program.parse();
