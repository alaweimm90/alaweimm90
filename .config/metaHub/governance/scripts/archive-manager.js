/**
 * Archive Manager
 * Manages archival of old projects and cleanup of legacy code
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class ArchiveManager {
  constructor() {
    this.rootPath = path.join(__dirname, '..', '..');
    this.archivePath = path.join(this.rootPath, '.archive');
    this.config = this.loadConfig();
    this.stats = {
      filesArchived: 0,
      directoriesArchived: 0,
      bytesArchived: 0,
      errors: []
    };
  }

  loadConfig() {
    try {
      const configPath = path.join(__dirname, '..', 'governance-config.json');
      return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    } catch (error) {
      console.error('Failed to load config:', error.message);
      return { archivePolicy: { daysBeforeArchive: 90, patterns: [] } };
    }
  }

  async run(options = {}) {
    console.log('\n🗄️  Archive Manager Starting...\n');

    // Ensure archive directory exists
    if (!fs.existsSync(this.archivePath)) {
      fs.mkdirSync(this.archivePath, { recursive: true });
      console.log('✅ Created archive directory');
    }

    // Archive old projects
    await this.archiveOldProjects();

    // Archive orphaned files
    await this.archiveOrphanedFiles();

    // Archive legacy patterns
    await this.archiveLegacyPatterns();

    // Generate report
    this.generateReport();

    return this.stats;
  }

  async archiveOldProjects() {
    console.log('\n📦 Archiving old projects...');

    const oldProjects = [
      'alaweimm90',
      'metaHub',
      'marketing-automation',
      'ui-components',
      'vercel-templates',
      'sample-project',
      'branding'
    ];

    for (const project of oldProjects) {
      const sourcePath = path.join(this.rootPath, project);
      if (fs.existsSync(sourcePath)) {
        await this.archiveDirectory(sourcePath, `projects/${project}`);
      }
    }
  }

  async archiveOrphanedFiles() {
    console.log('\n🔍 Archiving orphaned files...');

    const orphanedPatterns = [
      '*.py',
      '*.ps1',
      '*.sh',
      '*.exe',
      'test*',
      'sample*',
      '*.bak',
      '*.old',
      '*.tmp'
    ];

    const files = fs.readdirSync(this.rootPath);

    for (const file of files) {
      const filePath = path.join(this.rootPath, file);
      const stat = fs.statSync(filePath);

      if (stat.isFile() && this.matchesPattern(file, orphanedPatterns)) {
        const shouldArchive = await this.confirmArchive(file);
        if (shouldArchive) {
          await this.archiveFile(filePath, `orphaned/${file}`);
        }
      }
    }
  }

  async archiveLegacyPatterns() {
    console.log('\n🔧 Archiving legacy code patterns...');

    const legacyDirs = [
      '__pycache__',
      'node_modules.old',
      'dist.old',
      'build.old',
      '.old',
      '.backup'
    ];

    for (const dir of legacyDirs) {
      const dirPath = path.join(this.rootPath, dir);
      if (fs.existsSync(dirPath)) {
        await this.archiveDirectory(dirPath, `legacy/${dir}`);
      }
    }
  }

  async archiveFile(sourcePath, archiveName) {
    try {
      const destPath = path.join(this.archivePath, archiveName);
      const destDir = path.dirname(destPath);

      // Create destination directory
      if (!fs.existsSync(destDir)) {
        fs.mkdirSync(destDir, { recursive: true });
      }

      // Get file stats
      const stats = fs.statSync(sourcePath);

      // Move file to archive
      fs.renameSync(sourcePath, destPath);

      // Update stats
      this.stats.filesArchived++;
      this.stats.bytesArchived += stats.size;

      console.log(`  ✅ Archived: ${path.basename(sourcePath)} (${this.formatBytes(stats.size)})`);

      // Create archive manifest
      this.updateManifest(archiveName, {
        originalPath: sourcePath,
        archivedAt: new Date().toISOString(),
        size: stats.size,
        reason: 'legacy/orphaned'
      });
    } catch (error) {
      console.error(`  ❌ Failed to archive ${sourcePath}:`, error.message);
      this.stats.errors.push({ file: sourcePath, error: error.message });
    }
  }

  async archiveDirectory(sourcePath, archiveName) {
    try {
      const destPath = path.join(this.archivePath, archiveName);

      // Create destination directory
      if (!fs.existsSync(path.dirname(destPath))) {
        fs.mkdirSync(path.dirname(destPath), { recursive: true });
      }

      // Get directory size
      const size = this.getDirectorySize(sourcePath);

      // Move directory to archive
      fs.renameSync(sourcePath, destPath);

      // Update stats
      this.stats.directoriesArchived++;
      this.stats.bytesArchived += size;

      console.log(`  ✅ Archived: ${path.basename(sourcePath)} (${this.formatBytes(size)})`);

      // Create archive manifest
      this.updateManifest(archiveName, {
        originalPath: sourcePath,
        archivedAt: new Date().toISOString(),
        size: size,
        type: 'directory',
        reason: 'old project'
      });
    } catch (error) {
      console.error(`  ❌ Failed to archive ${sourcePath}:`, error.message);
      this.stats.errors.push({ directory: sourcePath, error: error.message });
    }
  }

  getDirectorySize(dirPath) {
    let size = 0;

    try {
      const files = fs.readdirSync(dirPath);

      for (const file of files) {
        const filePath = path.join(dirPath, file);
        const stats = fs.statSync(filePath);

        if (stats.isDirectory()) {
          size += this.getDirectorySize(filePath);
        } else {
          size += stats.size;
        }
      }
    } catch (error) {
      console.error(`Error calculating size for ${dirPath}:`, error.message);
    }

    return size;
  }

  matchesPattern(filename, patterns) {
    for (const pattern of patterns) {
      if (pattern.startsWith('*')) {
        const ext = pattern.slice(1);
        if (filename.endsWith(ext)) return true;
      } else if (pattern.endsWith('*')) {
        const prefix = pattern.slice(0, -1);
        if (filename.startsWith(prefix)) return true;
      } else if (filename === pattern) {
        return true;
      }
    }
    return false;
  }

  async confirmArchive(filename) {
    // Auto-archive certain patterns without confirmation
    const autoArchive = [
      '*.bak',
      '*.old',
      '*.tmp',
      '*.pyc',
      '__pycache__'
    ];

    return this.matchesPattern(filename, autoArchive);
  }

  updateManifest(archiveName, metadata) {
    const manifestPath = path.join(this.archivePath, 'MANIFEST.json');

    let manifest = {};
    if (fs.existsSync(manifestPath)) {
      try {
        manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
      } catch (error) {
        console.error('Failed to read manifest:', error.message);
      }
    }

    manifest[archiveName] = metadata;

    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
  }

  formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  generateReport() {
    console.log('\n📊 Archive Manager Report');
    console.log('═══════════════════════════════════════');
    console.log(`  Files Archived: ${this.stats.filesArchived}`);
    console.log(`  Directories Archived: ${this.stats.directoriesArchived}`);
    console.log(`  Total Size Archived: ${this.formatBytes(this.stats.bytesArchived)}`);
    console.log(`  Errors: ${this.stats.errors.length}`);

    if (this.stats.errors.length > 0) {
      console.log('\n❌ Errors:');
      this.stats.errors.forEach(err => {
        console.log(`  - ${err.file || err.directory}: ${err.error}`);
      });
    }

    // Save report
    const reportPath = path.join(this.archivePath, 'ARCHIVE_REPORT.json');
    fs.writeFileSync(reportPath, JSON.stringify({
      timestamp: new Date().toISOString(),
      stats: this.stats
    }, null, 2));

    console.log(`\n✅ Archive complete! Report saved to: ${reportPath}`);
    console.log('💡 Tip: Review archived items in .archive/ directory');
  }
}

// Run if executed directly
if (require.main === module) {
  const manager = new ArchiveManager();
  manager.run().catch(error => {
    console.error('Archive Manager failed:', error);
    process.exit(1);
  });
}

module.exports = ArchiveManager;