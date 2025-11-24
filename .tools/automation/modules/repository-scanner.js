/**
 * Repository Scanner Module
 * Scans the repository for various metrics and issues
 */

const fs = require('fs').promises;
const path = require('path');
const { BaseModule } = require('../core/framework');

class RepositoryScannerModule extends BaseModule {
  constructor(framework) {
    super(framework);
    this.name = 'repository-scanner';

    // Register tasks
    this.registerTask('scan-files', this.scanFiles);
    this.registerTask('find-todos', this.findTodos);
    this.registerTask('check-dependencies', this.checkDependencies);
    this.registerTask('analyze-structure', this.analyzeStructure);
  }

  async init() {
    await super.init();
    this.logger.info('Repository Scanner Module initialized');
  }

  /**
   * Scan all files in the repository
   * @param {{directory?:string,extensions?:string[]}} params Scan options
   * @returns {Promise<{totalFiles:number,byExtension:object,largeFiles:Array,totalSize:number}>}
   */
  async scanFiles(params = {}) {
    const { directory = process.cwd(), extensions = ['.js', '.ts', '.jsx', '.tsx', '.json', '.md'] } = params;

    this.logger.info(`Scanning files in ${directory}`);

    const stats = {
      totalFiles: 0,
      byExtension: {},
      largeFiles: [],
      totalSize: 0,
    };

    const scanDir = async (dir) => {
      try {
        const entries = await fs.readdir(dir, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);

          // Skip node_modules and .git
          if (entry.name === 'node_modules' || entry.name === '.git') continue;

          if (entry.isDirectory()) {
            await scanDir(fullPath);
          } else if (entry.isFile()) {
            const ext = path.extname(entry.name);

            if (extensions.includes(ext)) {
              stats.totalFiles++;
              stats.byExtension[ext] = (stats.byExtension[ext] || 0) + 1;

              const stat = await fs.stat(fullPath);
              stats.totalSize += stat.size;

              if (stat.size > 100000) { // Files larger than 100KB
                stats.largeFiles.push({
                  path: fullPath,
                  size: stat.size,
                });
              }
            }
          }
        }
      } catch (error) {
        this.logger.error(`Error scanning directory ${dir}: ${error.message}`);
      }
    };

    await scanDir(directory);

    this.logger.info('File scan complete', stats);
    return stats;
  }

  /**
   * Find TODO comments in code files
   * @param {{directory?:string}} params Directory to scan
   * @returns {Promise<{count:number,todos:Array}>}
   */
  async findTodos(params = {}) {
    const { directory = process.cwd() } = params;

    this.logger.info('Searching for TODO comments');

    const todos = [];
    const patterns = [/TODO:/gi, /FIXME:/gi, /HACK:/gi, /NOTE:/gi];

    const searchFile = async (filePath) => {
      try {
        const content = await fs.readFile(filePath, 'utf8');
        const lines = content.split('\n');

        lines.forEach((line, index) => {
          patterns.forEach(pattern => {
            if (pattern.test(line)) {
              todos.push({
                file: filePath,
                line: index + 1,
                text: line.trim(),
                type: pattern.source.replace(':', '').toUpperCase(),
              });
            }
          });
        });
      } catch (error) {
        // Ignore read errors
      }
    };

    const scanDir = async (dir) => {
      try {
        const entries = await fs.readdir(dir, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);

          if (entry.name === 'node_modules' || entry.name === '.git') continue;

          if (entry.isDirectory()) {
            await scanDir(fullPath);
          } else if (entry.isFile()) {
            const ext = path.extname(entry.name);
            if (['.js', '.ts', '.jsx', '.tsx'].includes(ext)) {
              await searchFile(fullPath);
            }
          }
        }
      } catch (error) {
        this.logger.error(`Error scanning for TODOs: ${error.message}`);
      }
    };

    await scanDir(directory);

    this.logger.info(`Found ${todos.length} TODO comments`);
    return { count: todos.length, todos };
  }

  /**
   * Check package.json dependencies
   * @param {{packagePath?:string}} params Target package.json path
   * @returns {Promise<{dependencies:number,devDependencies:number,scripts:number,outdated:Array,security:Array}>}
   */
  async checkDependencies(params = {}) {
    const { packagePath = path.join(process.cwd(), 'package.json') } = params;

    this.logger.info('Checking dependencies');

    try {
      const packageContent = await fs.readFile(packagePath, 'utf8');
      const packageJson = JSON.parse(packageContent);

      const report = {
        dependencies: Object.keys(packageJson.dependencies || {}).length,
        devDependencies: Object.keys(packageJson.devDependencies || {}).length,
        scripts: Object.keys(packageJson.scripts || {}).length,
        outdated: [],
        security: [],
      };

      // Check for common security issues
      const riskyPackages = ['eval', 'child_process'];
      const allDeps = {
        ...packageJson.dependencies,
        ...packageJson.devDependencies,
      };

      for (const [name, version] of Object.entries(allDeps)) {
        if (riskyPackages.some(risky => name.includes(risky))) {
          report.security.push({
            package: name,
            reason: 'Potentially risky package',
          });
        }
      }

      this.logger.info('Dependency check complete', report);
      return report;
    } catch (error) {
      this.logger.error(`Failed to check dependencies: ${error.message}`);
      throw error;
    }
  }

  /**
   * Analyze repository structure
   * @param {{directory?:string}} params Root directory
   * @returns {Promise<object>}
   */
  async analyzeStructure(params = {}) {
    const { directory = process.cwd() } = params;

    this.logger.info('Analyzing repository structure');

    const structure = {
      hasReadme: false,
      hasLicense: false,
      hasPackageJson: false,
      hasGitignore: false,
      hasSrcFolder: false,
      hasTestFolder: false,
      hasDocsFolder: false,
      directories: [],
      recommendations: [],
    };

    try {
      const entries = await fs.readdir(directory);

      // Check for important files
      structure.hasReadme = entries.some(e => e.toLowerCase() === 'readme.md');
      structure.hasLicense = entries.some(e => e.toLowerCase() === 'license');
      structure.hasPackageJson = entries.some(e => e === 'package.json');
      structure.hasGitignore = entries.some(e => e === '.gitignore');

      // Check for important directories
      structure.hasSrcFolder = entries.includes('src');
      structure.hasTestFolder = entries.some(e => ['test', 'tests', '__tests__'].includes(e));
      structure.hasDocsFolder = entries.includes('docs');

      // List all directories
      for (const entry of entries) {
        const fullPath = path.join(directory, entry);
        const stat = await fs.stat(fullPath);
        if (stat.isDirectory() && !entry.startsWith('.')) {
          structure.directories.push(entry);
        }
      }

      // Generate recommendations
      if (!structure.hasReadme) {
        structure.recommendations.push('Add a README.md file');
      }
      if (!structure.hasLicense) {
        structure.recommendations.push('Add a LICENSE file');
      }
      if (!structure.hasTestFolder) {
        structure.recommendations.push('Add a test directory');
      }
      if (!structure.hasDocsFolder) {
        structure.recommendations.push('Consider adding a docs directory');
      }

      this.logger.info('Structure analysis complete', structure);
      return structure;
    } catch (error) {
      this.logger.error(`Failed to analyze structure: ${error.message}`);
      throw error;
    }
  }
}

module.exports = RepositoryScannerModule;
