/**
 * ATLAS CLI Analyze Commands
 *
 * Repository analysis and metrics commands
 */

import { Command } from 'commander';
import { cliContext, output, progress, errorHandler, format } from '../utils.js';
import { RepositoryAnalyzer } from '@atlas/analysis/analyzer.js';
// Bridge integrations - reserved for future KILO-validated operations
// import { AtlasKiloBridge, ComplianceCheck } from '@atlas/integrations/atlas-kilo-bridge.js';
// import { KiloAtlasBridge } from '@atlas/integrations/kilo-bridge.js';

export function registerAnalyzeCommands(program: Command): void {
  const analyzeCmd = program.command('analyze').description('Repository analysis commands');

  // Analyze repository command
  analyzeCmd
    .command('repo <path>')
    .description('Analyze a repository for code quality metrics')
    .option('--format <format>', 'Output format: json, table, summary', 'table')
    .option('--depth <depth>', 'Analysis depth: shallow, medium, deep', 'medium')
    .option('--include-patterns <patterns>', 'File patterns to include (comma-separated)')
    .option('--exclude-patterns <patterns>', 'File patterns to exclude (comma-separated)')
    .option('--governance-check', 'Validate results against KILO governance policies')
    .option('--auto-refactor', 'Apply KILO-validated refactoring operations')
    .action(async (path: string, options: any) => {
      await errorHandler.handleAsync(async () => {
        const spinner = progress.start('Analyzing repository...');

        try {
          const _services = await cliContext.getServices();
          const analyzer = new RepositoryAnalyzer();

          const results = await analyzer.analyzeRepository(path, {
            depth: options.depth,
            includePatterns: options.includePatterns?.split(','),
            excludePatterns: options.excludePatterns?.split(','),
          });

          progress.succeed(spinner, 'Analysis complete');

          switch (options.format) {
            case 'json':
              console.log(JSON.stringify(results, null, 2));
              break;
            case 'summary':
              displayAnalysisSummary(results);
              break;
            default:
              displayAnalysisTable(results);
          }
        } catch (error) {
          progress.fail(spinner, 'Analysis failed');
          throw error;
        }
      });
    });

  // Analyze complexity command
  analyzeCmd
    .command('complexity <path>')
    .description('Analyze code complexity metrics')
    .option('--threshold <threshold>', 'Complexity threshold', parseFloat, 10)
    .action(async (path: string, options: any) => {
      await errorHandler.handleAsync(async () => {
        const spinner = progress.start('Analyzing complexity...');

        try {
          const _services = await cliContext.getServices();
          const analyzer = new RepositoryAnalyzer();

          const results = await analyzer.analyzeComplexity(path, {
            threshold: options.threshold,
          });

          progress.succeed(spinner, 'Complexity analysis complete');
          displayComplexityResults(results);
        } catch (error) {
          progress.fail(spinner, 'Complexity analysis failed');
          throw error;
        }
      });
    });

  // Analyze chaos command
  analyzeCmd
    .command('chaos <path>')
    .description('Analyze code chaos and maintainability')
    .option('--detailed', 'Show detailed chaos breakdown')
    .action(async (path: string, options: any) => {
      await errorHandler.handleAsync(async () => {
        const spinner = progress.start('Analyzing chaos...');

        try {
          const _services = await cliContext.getServices();
          const analyzer = new RepositoryAnalyzer();

          const results = await analyzer.analyzeChaos(path);

          progress.succeed(spinner, 'Chaos analysis complete');
          displayChaosResults(results, options.detailed);
        } catch (error) {
          progress.fail(spinner, 'Chaos analysis failed');
          throw error;
        }
      });
    });

  // Quick scan command
  analyzeCmd
    .command('scan <path>')
    .description('Quick repository scan for basic metrics')
    .action(async (path: string) => {
      await errorHandler.handleAsync(async () => {
        const spinner = progress.start('Quick scanning...');

        try {
          const _services = await cliContext.getServices();
          const analyzer = new RepositoryAnalyzer();

          const results = await analyzer.quickScan(path);

          progress.succeed(spinner, 'Quick scan complete');
          displayQuickScanResults(results);
        } catch (error) {
          progress.fail(spinner, 'Quick scan failed');
          throw error;
        }
      });
    });
}

/**
 * Display analysis results in table format
 */
function displayAnalysisTable(results: any): void {
  output.header('Repository Analysis Results');

  const headers = ['Metric', 'Value', 'Status'];
  const rows = [
    ['Files Analyzed', results.filesAnalyzed?.toString() || '0', 'info'],
    ['Total Lines', results.totalLines?.toString() || '0', 'info'],
    [
      'Complexity Score',
      results.complexityScore?.toFixed(2) || '0.00',
      getStatusColor(results.complexityScore),
    ],
    ['Chaos Level', results.chaosLevel?.toFixed(2) || '0.00', getStatusColor(results.chaosLevel)],
    [
      'Maintainability',
      results.maintainabilityIndex?.toFixed(2) || '0.00',
      getStatusColor(results.maintainabilityIndex, true),
    ],
  ];

  output.table(
    headers,
    rows.map(([metric, value, status]) => [metric, value, status])
  );
}

/**
 * Display analysis summary
 */
function displayAnalysisSummary(results: any): void {
  output.header('Analysis Summary');

  console.log(`Files Analyzed: ${results.filesAnalyzed || 0}`);
  console.log(`Total Lines: ${results.totalLines || 0}`);
  console.log(`Complexity Score: ${results.complexityScore?.toFixed(2) || '0.00'}`);
  console.log(`Chaos Level: ${results.chaosLevel?.toFixed(2) || '0.00'}`);
  console.log(`Maintainability Index: ${results.maintainabilityIndex?.toFixed(2) || '0.00'}`);

  if (results.recommendations?.length > 0) {
    console.log('\nRecommendations:');
    results.recommendations.forEach((rec: string, i: number) => {
      console.log(`  ${i + 1}. ${rec}`);
    });
  }
}

/**
 * Display complexity analysis results
 */
function displayComplexityResults(results: any): void {
  output.header('Complexity Analysis');

  if (results.functions?.length > 0) {
    const headers = ['Function', 'Complexity', 'File'];
    const rows = results.functions.map((fn: any) => [fn.name, fn.complexity.toString(), fn.file]);

    output.table(headers, rows);
  }

  console.log(`\nAverage Complexity: ${results.averageComplexity?.toFixed(2) || '0.00'}`);
  console.log(`High Complexity Functions: ${results.highComplexityCount || 0}`);
}

/**
 * Display chaos analysis results
 */
function displayChaosResults(results: any, detailed: boolean = false): void {
  output.header('Chaos Analysis');

  console.log(`Overall Chaos Level: ${results.chaosLevel?.toFixed(2) || '0.00'}`);
  console.log(`Maintainability Score: ${results.maintainabilityScore?.toFixed(2) || '0.00'}`);

  if (detailed && results.breakdown) {
    console.log('\nDetailed Breakdown:');
    Object.entries(results.breakdown).forEach(([metric, value]: [string, any]) => {
      console.log(`  ${metric}: ${value}`);
    });
  }

  if (results.riskFactors?.length > 0) {
    console.log('\nRisk Factors:');
    results.riskFactors.forEach((factor: string) => {
      console.log(`  • ${factor}`);
    });
  }
}

/**
 * Display quick scan results
 */
function displayQuickScanResults(results: any): void {
  output.header('Quick Scan Results');

  console.log(`Repository: ${results.repository || 'Unknown'}`);
  console.log(`Files: ${results.fileCount || 0}`);
  console.log(`Languages: ${results.languages?.join(', ') || 'Unknown'}`);
  console.log(`Total Size: ${format.bytes(results.totalSize || 0)}`);
  console.log(
    `Last Modified: ${results.lastModified ? format.timestamp(new Date(results.lastModified)) : 'Unknown'}`
  );

  if (results.healthScore !== undefined) {
    console.log(`Health Score: ${results.healthScore.toFixed(1)}/10`);
  }
}

/**
 * Get status color based on value
 */
function getStatusColor(value: number, invert: boolean = false): string {
  if (value === undefined || value === null) return 'unknown';

  // For inverted metrics (higher is better), flip the logic
  if (invert) {
    if (value >= 8) return 'good';
    if (value >= 6) return 'warning';
    return 'bad';
  }

  // For normal metrics (lower is better)
  if (value <= 2) return 'good';
  if (value <= 5) return 'warning';
  return 'bad';
}
