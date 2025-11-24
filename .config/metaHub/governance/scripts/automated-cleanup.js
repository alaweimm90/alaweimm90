#!/usr/bin/env node

/**
 * Automated Cleanup System
 * Performs intelligent file organization, archival, and cleanup
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class AutomatedCleanup {
  constructor(rootPath = process.cwd()) {
    this.rootPath = rootPath;
    this.config = this.loadConfig();
    this.stats = {
      filesProcessed: 0,
      filesArchived: 0,
      filesDeleted: 0,
      filesOrganized: 0,
      spaceReclaimed: 0,
    };
  }

  loadConfig() {
    const configPath = path.join(this.rootPath, '.governance', 'governance-config.json');
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }
    return {
      fileTracking: {
        lifecycle: {
          archiveAfterDays: 90,
          warnAfterDays: 60,
          deleteAfterDays: 365,
        },
      },
    };
  }

  // Main cleanup function
  async performCleanup(options = {}) {
    console.log('🧹 Starting automated cleanup...\n');

    const actions = {
      organizeFiles: options.organize !== false,
      archiveOldFiles: options.archive !== false,
      removeTempFiles: options.temp !== false,
      cleanDependencies: options.dependencies !== false,
      optimizeStorage: options.optimize !== false,
      generateReport: options.report !== false,
    };

    // Execute cleanup actions
    if (actions.organizeFiles) await this.organizeFiles();
    if (actions.archiveOldFiles) await this.archiveOldFiles();
    if (actions.removeTempFiles) await this.removeTempFiles();
    if (actions.cleanDependencies) await this.cleanDependencies();
    if (actions.optimizeStorage) await this.optimizeStorage();

    // Generate and display report
    if (actions.generateReport) {
      const report = this.generateReport();
      this.displayReport(report);
      this.saveReport(report);
    }

    return this.stats;
  }

  // Organize files by type and status
  async organizeFiles() {
    console.log('📂 Organizing files by type and status...');

    const fileCategories = {
      documentation: ['.md', '.txt', '.rst', '.adoc'],
      source: ['.js', '.ts', '.jsx', '.tsx', '.py', '.java', '.go'],
      config: ['.json', '.yml', '.yaml', '.toml', '.ini'],
      tests: ['.test.js', '.spec.js', '.test.ts', '.spec.ts'],
      assets: ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico'],
      styles: ['.css', '.scss', '.sass', '.less'],
    };

    // Count files by category
    const categoryCounts = {};
    for (const [category, extensions] of Object.entries(fileCategories)) {
      categoryCounts[category] = 0;

      const files = this.findFilesByExtensions(extensions);
      categoryCounts[category] = files.length;

      // Ensure proper organization
      files.forEach(file => {
        this.ensureProperLocation(file, category);
      });
    }

    console.log('  File organization summary:');
    for (const [category, count] of Object.entries(categoryCounts)) {
      console.log(`    ${category}: ${count} files`);
    }

    return categoryCounts;
  }

  // Find files by extensions
  findFilesByExtensions(extensions, dir = this.rootPath) {
    const files = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        // Skip certain directories
        if (['node_modules', '.git', 'dist', 'build', '.next'].includes(entry.name)) {
          continue;
        }
        files.push(...this.findFilesByExtensions(extensions, fullPath));
      } else {
        const ext = path.extname(entry.name);
        if (extensions.includes(ext) || extensions.some(e => entry.name.endsWith(e))) {
          files.push(fullPath);
        }
      }
    }

    return files;
  }

  // Ensure file is in proper location
  ensureProperLocation(filePath, category) {
    const relativePath = path.relative(this.rootPath, filePath);
    const fileName = path.basename(filePath);

    // Define proper locations for each category
    const properLocations = {
      documentation: 'docs/',
      tests: 'tests/',
      assets: 'assets/',
      styles: 'src/styles/',
    };

    if (properLocations[category]) {
      const properDir = path.join(this.rootPath, properLocations[category]);

      // Check if file is already in proper location
      if (!relativePath.startsWith(properLocations[category])) {
        // Check for special cases (don't move root documentation files)
        if (
          category === 'documentation' &&
          [
            'README.md',
            'LICENSE',
            'CHANGELOG.md',
            'SECURITY.md',
            'CONTRIBUTING.md',
            'CODE_OF_CONDUCT.md',
          ].includes(fileName)
        ) {
          return;
        }

        console.log(`  ℹ️ Consider moving ${relativePath} to ${properLocations[category]}`);
        this.stats.filesOrganized++;
      }
    }
  }

  // Archive old files
  async archiveOldFiles() {
    console.log('📦 Archiving old files...');

    const archiveDate = new Date();
    archiveDate.setDate(
      archiveDate.getDate() - this.config.fileTracking.lifecycle.archiveAfterDays
    );

    const archiveDir = path.join(this.rootPath, '.archive', new Date().toISOString().split('T')[0]);

    const filesToArchive = this.findOldFiles(this.rootPath, archiveDate);

    if (filesToArchive.length > 0) {
      // Create archive directory
      if (!fs.existsSync(archiveDir)) {
        fs.mkdirSync(archiveDir, { recursive: true });
      }

      // Create archive manifest
      const manifest = {
        date: new Date().toISOString(),
        reason: 'automated-cleanup',
        files: [],
      };

      // Archive each file
      for (const file of filesToArchive) {
        const relativePath = path.relative(this.rootPath, file);
        const archivePath = path.join(archiveDir, relativePath);
        const archiveFileDir = path.dirname(archivePath);

        // Create directory structure in archive
        if (!fs.existsSync(archiveFileDir)) {
          fs.mkdirSync(archiveFileDir, { recursive: true });
        }

        // Move file to archive
        fs.renameSync(file, archivePath);

        manifest.files.push({
          original: relativePath,
          archived: path.relative(this.rootPath, archivePath),
          size: fs.statSync(archivePath).size,
        });

        this.stats.filesArchived++;
      }

      // Save manifest
      fs.writeFileSync(path.join(archiveDir, 'manifest.json'), JSON.stringify(manifest, null, 2));

      console.log(
        `  Archived ${filesToArchive.length} files to ${path.relative(this.rootPath, archiveDir)}`
      );
    } else {
      console.log('  No files old enough to archive');
    }

    return filesToArchive.length;
  }

  // Find old files
  findOldFiles(dir, cutoffDate) {
    const oldFiles = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      // Skip certain directories and files
      if (entry.isDirectory()) {
        if (['node_modules', '.git', 'dist', 'build', '.next', '.archive'].includes(entry.name)) {
          continue;
        }
        oldFiles.push(...this.findOldFiles(fullPath, cutoffDate));
      } else {
        // Skip certain files
        if (['README.md', 'LICENSE', 'package.json', 'package-lock.json'].includes(entry.name)) {
          continue;
        }

        const stats = fs.statSync(fullPath);
        if (stats.mtime < cutoffDate) {
          oldFiles.push(fullPath);
        }
      }
    }

    return oldFiles;
  }

  // Remove temporary files
  async removeTempFiles() {
    console.log('🗑️ Removing temporary files...');

    const tempPatterns = [
      '*.tmp',
      '*.temp',
      '*.log',
      '*.cache',
      '.DS_Store',
      'Thumbs.db',
      '*.swp',
      '*.swo',
      '*~',
      '*.bak',
      '*.old',
    ];

    let totalSize = 0;
    let filesRemoved = 0;

    tempPatterns.forEach(pattern => {
      const files = this.findFilesByPattern(pattern);
      files.forEach(file => {
        const size = fs.statSync(file).size;
        fs.unlinkSync(file);
        totalSize += size;
        filesRemoved++;
        this.stats.filesDeleted++;
      });
    });

    this.stats.spaceReclaimed += totalSize;

    console.log(`  Removed ${filesRemoved} temporary files`);
    console.log(`  Space reclaimed: ${this.formatBytes(totalSize)}`);

    return filesRemoved;
  }

  // Find files by pattern
  findFilesByPattern(pattern, dir = this.rootPath) {
    const files = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        if (['node_modules', '.git'].includes(entry.name)) {
          continue;
        }
        files.push(...this.findFilesByPattern(pattern, fullPath));
      } else {
        // Simple pattern matching (could be enhanced with proper glob)
        const regex = new RegExp(pattern.replace('*', '.*'));
        if (regex.test(entry.name)) {
          files.push(fullPath);
        }
      }
    }

    return files;
  }

  // Clean dependencies
  async cleanDependencies() {
    console.log('📦 Cleaning dependencies...');

    const commands = [
      { name: 'npm cache', cmd: 'npm cache clean --force', safe: true },
      { name: 'npm prune', cmd: 'npm prune', safe: true },
      { name: 'npm dedupe', cmd: 'npm dedupe', safe: true },
    ];

    for (const { name, cmd, safe } of commands) {
      if (safe) {
        try {
          console.log(`  Running ${name}...`);
          execSync(cmd, { cwd: this.rootPath, stdio: 'pipe' });
        } catch (error) {
          console.log(`    Warning: ${name} failed - ${error.message}`);
        }
      }
    }

    // Check node_modules size
    const nodeModulesPath = path.join(this.rootPath, 'node_modules');
    if (fs.existsSync(nodeModulesPath)) {
      const size = this.getDirectorySize(nodeModulesPath);
      console.log(`  node_modules size: ${this.formatBytes(size)}`);

      if (size > 500 * 1024 * 1024) {
        // > 500MB
        console.log('  ⚠️ Consider reviewing large dependencies');
      }
    }
  }

  // Optimize storage
  async optimizeStorage() {
    console.log('💾 Optimizing storage...');

    // Find and report large files
    const largeFiles = this.findLargeFiles(this.rootPath, 10 * 1024 * 1024); // > 10MB
    if (largeFiles.length > 0) {
      console.log('  Large files found:');
      largeFiles.slice(0, 5).forEach(({ path: filePath, size }) => {
        console.log(`    ${path.relative(this.rootPath, filePath)}: ${this.formatBytes(size)}`);
      });
    }

    // Find duplicate files (simple implementation based on size and name)
    const duplicates = this.findDuplicates();
    if (duplicates.length > 0) {
      console.log(`  Found ${duplicates.length} potential duplicate files`);
    }

    // Compress old logs
    const logFiles = this.findFilesByPattern('*.log');
    console.log(`  Found ${logFiles.length} log files that could be compressed`);

    return { largeFiles: largeFiles.length, duplicates: duplicates.length };
  }

  // Find large files
  findLargeFiles(dir, threshold) {
    const largeFiles = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        if (['node_modules', '.git', '.next', 'dist', 'build'].includes(entry.name)) {
          continue;
        }
        largeFiles.push(...this.findLargeFiles(fullPath, threshold));
      } else {
        const stats = fs.statSync(fullPath);
        if (stats.size > threshold) {
          largeFiles.push({ path: fullPath, size: stats.size });
        }
      }
    }

    return largeFiles.sort((a, b) => b.size - a.size);
  }

  // Find duplicate files (simplified version)
  findDuplicates() {
    const fileMap = new Map();
    const duplicates = [];

    const processDirectory = dir => {
      const entries = fs.readdirSync(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        if (entry.isDirectory()) {
          if (['node_modules', '.git'].includes(entry.name)) {
            continue;
          }
          processDirectory(fullPath);
        } else {
          const stats = fs.statSync(fullPath);
          const key = `${entry.name}-${stats.size}`;

          if (fileMap.has(key)) {
            duplicates.push({
              original: fileMap.get(key),
              duplicate: fullPath,
              size: stats.size,
            });
          } else {
            fileMap.set(key, fullPath);
          }
        }
      }
    };

    processDirectory(this.rootPath);
    return duplicates;
  }

  // Get directory size
  getDirectorySize(dir) {
    let size = 0;
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        size += this.getDirectorySize(fullPath);
      } else {
        size += fs.statSync(fullPath).size;
      }
    }

    return size;
  }

  // Format bytes to human readable
  formatBytes(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let unitIndex = 0;
    let value = bytes;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(2)} ${units[unitIndex]}`;
  }

  // Generate cleanup report
  generateReport() {
    return {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      recommendations: this.generateRecommendations(),
    };
  }

  // Generate recommendations
  generateRecommendations() {
    const recommendations = [];

    if (this.stats.filesArchived > 50) {
      recommendations.push('Consider reviewing archive policy - many files were archived');
    }

    if (this.stats.spaceReclaimed > 100 * 1024 * 1024) {
      recommendations.push('Significant space reclaimed - consider regular cleanup schedule');
    }

    if (this.stats.filesOrganized > 20) {
      recommendations.push('Many files need reorganization - review file structure guidelines');
    }

    return recommendations;
  }

  // Display report
  displayReport(report) {
    console.log('\n' + '='.repeat(50));
    console.log('📊 Cleanup Report');
    console.log('='.repeat(50));
    console.log(`Files Processed: ${report.stats.filesProcessed}`);
    console.log(`Files Archived: ${report.stats.filesArchived}`);
    console.log(`Files Deleted: ${report.stats.filesDeleted}`);
    console.log(`Files Organized: ${report.stats.filesOrganized}`);
    console.log(`Space Reclaimed: ${this.formatBytes(report.stats.spaceReclaimed)}`);

    if (report.recommendations.length > 0) {
      console.log('\n📌 Recommendations:');
      report.recommendations.forEach(rec => {
        console.log(`  - ${rec}`);
      });
    }

    console.log('='.repeat(50));
  }

  // Save report
  saveReport(report) {
    const reportPath = path.join(
      this.rootPath,
      '.governance',
      'reports',
      `cleanup-${new Date().toISOString().split('T')[0]}.json`
    );

    const reportDir = path.dirname(reportPath);
    if (!fs.existsSync(reportDir)) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`\nReport saved to: ${path.relative(this.rootPath, reportPath)}`);
  }
}

// Export for use in other scripts
module.exports = AutomatedCleanup;

// Run if called directly
if (require.main === module) {
  const cleanup = new AutomatedCleanup();

  // Parse command line arguments
  const args = process.argv.slice(2);
  const options = {
    organize: !args.includes('--no-organize'),
    archive: !args.includes('--no-archive'),
    temp: !args.includes('--no-temp'),
    dependencies: !args.includes('--no-dependencies'),
    optimize: !args.includes('--no-optimize'),
    report: !args.includes('--no-report'),
  };

  cleanup.performCleanup(options).then(stats => {
    console.log('\n✅ Cleanup completed successfully!');
  });
}
