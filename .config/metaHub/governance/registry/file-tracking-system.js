/**
 * File Tracking System
 * Automated tracking of all files with metadata and lifecycle management
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { promisify } = require('util');

class FileTrackingSystem {
  constructor(rootPath = process.cwd()) {
    this.rootPath = rootPath;
    this.trackingDb = path.join(rootPath, '.governance', 'registry', 'tracking.json');
    this.config = this.loadConfig();
    this.tracking = this.loadTracking();
  }

  loadConfig() {
    const configPath = path.join(this.rootPath, '.governance', 'governance-config.json');
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }
    return {};
  }

  loadTracking() {
    if (fs.existsSync(this.trackingDb)) {
      return JSON.parse(fs.readFileSync(this.trackingDb, 'utf8'));
    }
    return {
      version: '1.0.0',
      lastScan: null,
      files: {},
      statistics: {
        totalFiles: 0,
        activeFiles: 0,
        archivedFiles: 0,
        deletedFiles: 0,
      },
    };
  }

  saveTracking() {
    fs.writeFileSync(this.trackingDb, JSON.stringify(this.tracking, null, 2));
  }

  generateFileId(filePath) {
    return crypto.createHash('md5').update(filePath).digest('hex');
  }

  async scanFile(filePath) {
    const stats = fs.statSync(filePath);
    const fileId = this.generateFileId(filePath);
    const relativePath = path.relative(this.rootPath, filePath);

    // Determine file status
    const daysSinceModified = (Date.now() - stats.mtime) / (1000 * 60 * 60 * 24);
    let status = 'active';
    if (daysSinceModified > 90) status = 'archived';
    else if (daysSinceModified > 60) status = 'warning';

    // Get or create file record
    if (!this.tracking.files[fileId]) {
      this.tracking.files[fileId] = {
        id: fileId,
        path: relativePath,
        createdAt: stats.birthtime.toISOString(),
        firstSeen: new Date().toISOString(),
        owner: null,
        documentationLink: null,
        tags: [],
        compliance: {
          hasTests: false,
          hasDocumentation: false,
          lastSecurityScan: null,
          violations: [],
        },
      };
    }

    // Update file record
    const fileRecord = this.tracking.files[fileId];
    fileRecord.modifiedAt = stats.mtime.toISOString();
    fileRecord.lastAccessed = stats.atime.toISOString();
    fileRecord.size = stats.size;
    fileRecord.status = status;
    fileRecord.lastScanned = new Date().toISOString();

    // Check for associated documentation
    fileRecord.hasDocumentation = this.checkDocumentation(relativePath);

    // Check for associated tests
    fileRecord.hasTests = this.checkTests(relativePath);

    // Determine completeness
    fileRecord.completeness = this.calculateCompleteness(fileRecord);

    return fileRecord;
  }

  checkDocumentation(filePath) {
    const docPaths = [
      filePath.replace(/\.[^.]+$/, '.md'),
      path.join('docs', path.basename(filePath, path.extname(filePath)) + '.md'),
      path.join('docs', 'api', path.basename(filePath, path.extname(filePath)) + '.md'),
    ];

    return docPaths.some(docPath => fs.existsSync(path.join(this.rootPath, docPath)));
  }

  checkTests(filePath) {
    if (!filePath.startsWith('src/')) return false;

    const testPaths = [
      filePath.replace('src/', 'tests/').replace(/\.[^.]+$/, '.test.js'),
      filePath.replace('src/', 'tests/').replace(/\.[^.]+$/, '.spec.js'),
      filePath.replace('src/', 'tests/unit/').replace(/\.[^.]+$/, '.test.js'),
      filePath.replace('src/', 'tests/integration/').replace(/\.[^.]+$/, '.test.js'),
    ];

    return testPaths.some(testPath => fs.existsSync(path.join(this.rootPath, testPath)));
  }

  calculateCompleteness(fileRecord) {
    let score = 0;
    let total = 0;

    // Check various completeness criteria
    const criteria = {
      hasDocumentation: 25,
      hasTests: 25,
      hasOwner: 10,
      hasCompliance: 20,
      recentlyUpdated: 20,
    };

    for (const [key, weight] of Object.entries(criteria)) {
      total += weight;
      if (key === 'hasDocumentation' && fileRecord.hasDocumentation) score += weight;
      if (key === 'hasTests' && fileRecord.hasTests) score += weight;
      if (key === 'hasOwner' && fileRecord.owner) score += weight;
      if (key === 'hasCompliance' && !fileRecord.compliance.violations.length) score += weight;
      if (key === 'recentlyUpdated') {
        const daysSinceModified =
          (Date.now() - new Date(fileRecord.modifiedAt)) / (1000 * 60 * 60 * 24);
        if (daysSinceModified < 30) score += weight;
      }
    }

    return Math.round((score / total) * 100);
  }

  async scanDirectory(dirPath = this.rootPath, recursive = true) {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    const results = [];

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);

      // Skip certain directories
      if (entry.isDirectory()) {
        if (['.git', 'node_modules', '.next', 'dist', 'build'].includes(entry.name)) {
          continue;
        }
        if (recursive) {
          results.push(...(await this.scanDirectory(fullPath, recursive)));
        }
      } else {
        // Skip certain files
        if (entry.name.startsWith('.') && entry.name !== '.gitignore') continue;

        const fileRecord = await this.scanFile(fullPath);
        results.push(fileRecord);
      }
    }

    return results;
  }

  async performFullScan() {
    console.log('🔍 Starting full repository scan...');
    const startTime = Date.now();

    // Reset statistics
    this.tracking.statistics = {
      totalFiles: 0,
      activeFiles: 0,
      warningFiles: 0,
      archivedFiles: 0,
      completeFiles: 0,
      incompleteFiles: 0,
    };

    // Scan all files
    const files = await this.scanDirectory();

    // Update statistics
    files.forEach(file => {
      this.tracking.statistics.totalFiles++;
      if (file.status === 'active') this.tracking.statistics.activeFiles++;
      if (file.status === 'warning') this.tracking.statistics.warningFiles++;
      if (file.status === 'archived') this.tracking.statistics.archivedFiles++;
      if (file.completeness >= 80) this.tracking.statistics.completeFiles++;
      else this.tracking.statistics.incompleteFiles++;
    });

    // Update tracking metadata
    this.tracking.lastScan = new Date().toISOString();
    this.tracking.scanDuration = Date.now() - startTime;

    // Save tracking data
    this.saveTracking();

    console.log(
      `✅ Scan complete! Processed ${files.length} files in ${this.tracking.scanDuration}ms`
    );
    return this.tracking;
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.tracking.statistics,
      issues: [],
      recommendations: [],
    };

    // Find issues
    Object.values(this.tracking.files).forEach(file => {
      if (file.status === 'warning') {
        report.issues.push({
          type: 'warning',
          file: file.path,
          message: `File hasn't been updated in ${Math.floor((Date.now() - new Date(file.modifiedAt)) / (1000 * 60 * 60 * 24))} days`,
        });
      }

      if (!file.hasDocumentation) {
        report.issues.push({
          type: 'documentation',
          file: file.path,
          message: 'Missing documentation',
        });
      }

      if (!file.hasTests && file.path.startsWith('src/')) {
        report.issues.push({
          type: 'testing',
          file: file.path,
          message: 'Missing tests',
        });
      }

      if (file.completeness < 50) {
        report.issues.push({
          type: 'completeness',
          file: file.path,
          message: `Low completeness score: ${file.completeness}%`,
        });
      }
    });

    // Generate recommendations
    if (report.issues.length > 10) {
      report.recommendations.push('Consider scheduling a code cleanup sprint');
    }

    if (this.tracking.statistics.incompleteFiles > this.tracking.statistics.completeFiles) {
      report.recommendations.push('Focus on improving file completeness scores');
    }

    if (this.tracking.statistics.archivedFiles > 0) {
      report.recommendations.push(
        `Archive or remove ${this.tracking.statistics.archivedFiles} stale files`
      );
    }

    return report;
  }

  // Archive old files
  async archiveOldFiles() {
    const archiveDir = path.join(this.rootPath, '.archive', new Date().toISOString().split('T')[0]);
    if (!fs.existsSync(archiveDir)) {
      fs.mkdirSync(archiveDir, { recursive: true });
    }

    const archived = [];
    Object.values(this.tracking.files).forEach(file => {
      if (file.status === 'archived') {
        const srcPath = path.join(this.rootPath, file.path);
        const destPath = path.join(archiveDir, file.path);

        if (fs.existsSync(srcPath)) {
          const destDir = path.dirname(destPath);
          if (!fs.existsSync(destDir)) {
            fs.mkdirSync(destDir, { recursive: true });
          }

          fs.renameSync(srcPath, destPath);
          archived.push(file.path);

          // Update tracking
          file.archivedAt = new Date().toISOString();
          file.archivePath = path.relative(this.rootPath, destPath);
        }
      }
    });

    if (archived.length > 0) {
      console.log(`📦 Archived ${archived.length} files`);
      this.saveTracking();
    }

    return archived;
  }
}

// Export for use in other scripts
module.exports = FileTrackingSystem;

// Run if called directly
if (require.main === module) {
  const tracker = new FileTrackingSystem();
  tracker.performFullScan().then(result => {
    const report = tracker.generateReport();
    console.log('\n📊 Tracking Report:');
    console.log(JSON.stringify(report, null, 2));
  });
}
