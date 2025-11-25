const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');

class GitHubRepositoryAuditor {
  constructor() {
    this.files = new Map();
    this.directories = new Map();
    this.languages = new Map();
    this.projects = new Map();
    this.dependencies = new Map();
    this.security = new Map();
    this.metrics = {
      totalFiles: 0,
      totalSize: 0,
      totalLines: 0,
      languages: {},
      projects: {},
      security: { issues: 0, score: 0 }
    };
  }

  async auditRepository(rootPath = process.cwd()) {
    console.log('🔍 Starting full GitHub repository audit...\n');
    
    await this.scanRepository(rootPath);
    await this.analyzeProjects();
    await this.assessSecurity();
    const report = this.generateReport();
    
    await this.saveReport(report);
    this.displayReport(report);
    
    return report;
  }

  async scanRepository(rootPath) {
    await this.scanDirectory(rootPath, '');
  }

  async scanDirectory(dirPath, relativePath) {
    try {
      const entries = await fs.readdir(dirPath, { withFileTypes: true });
      
      for (const entry of entries) {
        const fullPath = path.join(dirPath, entry.name);
        const relPath = path.join(relativePath, entry.name);
        
        if (this.shouldSkip(entry.name)) continue;
        
        if (entry.isDirectory()) {
          await this.analyzeDirectory(fullPath, relPath);
          await this.scanDirectory(fullPath, relPath);
        } else {
          await this.analyzeFile(fullPath, relPath);
        }
      }
    } catch (error) {
      // Skip inaccessible directories
    }
  }

  shouldSkip(name) {
    const skipPatterns = [
      'node_modules', '.git', '.vscode', 'dist', 'build', 'coverage',
      '.nyc_output', 'logs', '*.log', '.DS_Store', 'Thumbs.db'
    ];
    
    return skipPatterns.some(pattern => {
      if (pattern.includes('*')) {
        return name.match(pattern.replace('*', '.*'));
      }
      return name === pattern || name.startsWith('.');
    });
  }

  async analyzeDirectory(fullPath, relativePath) {
    try {
      const stats = await fs.stat(fullPath);
      const files = await fs.readdir(fullPath);
      
      const dirInfo = {
        path: relativePath,
        fullPath,
        fileCount: files.length,
        created: stats.birthtime,
        modified: stats.mtime,
        type: this.inferDirectoryType(relativePath, files),
        project: this.inferProject(relativePath)
      };
      
      this.directories.set(relativePath, dirInfo);
      
      // Track projects
      if (dirInfo.type === 'project') {
        this.projects.set(relativePath, {
          name: path.basename(relativePath),
          path: relativePath,
          files: [],
          languages: new Set(),
          size: 0,
          complexity: 0
        });
      }
    } catch (error) {
      // Skip inaccessible directories
    }
  }

  async analyzeFile(fullPath, relativePath) {
    try {
      const stats = await fs.stat(fullPath);
      const ext = path.extname(relativePath).toLowerCase();
      const language = this.detectLanguage(ext, relativePath);
      
      let content = '';
      let lines = 0;
      let complexity = 0;
      let security = { score: 10, issues: [] };
      
      if (this.isTextFile(ext) && stats.size < 1024 * 1024) { // < 1MB
        try {
          content = await fs.readFile(fullPath, 'utf8');
          lines = content.split('\n').length;
          complexity = this.calculateComplexity(content, language);
          security = this.assessFileSecurity(content, ext);
        } catch (error) {
          // Binary or unreadable file
        }
      }
      
      const fileInfo = {
        path: relativePath,
        fullPath,
        name: path.basename(relativePath),
        extension: ext,
        language,
        size: stats.size,
        lines,
        complexity,
        security,
        created: stats.birthtime,
        modified: stats.mtime,
        project: this.findProject(relativePath)
      };
      
      this.files.set(relativePath, fileInfo);
      this.updateMetrics(fileInfo);
      this.trackLanguage(language, fileInfo);
      this.trackDependencies(content, relativePath);
      
    } catch (error) {
      // Skip inaccessible files
    }
  }

  inferDirectoryType(relativePath, files) {
    const hasPackageJson = files.includes('package.json');
    const hasPomXml = files.includes('pom.xml');
    const hasCargoToml = files.includes('Cargo.toml');
    const hasRequirementsTxt = files.includes('requirements.txt');
    const hasDockerfile = files.includes('Dockerfile');
    
    if (hasPackageJson) return 'nodejs-project';
    if (hasPomXml) return 'java-project';
    if (hasCargoToml) return 'rust-project';
    if (hasRequirementsTxt) return 'python-project';
    if (hasDockerfile) return 'docker-project';
    if (files.some(f => f.endsWith('.sln'))) return 'dotnet-project';
    if (files.some(f => f.endsWith('.go'))) return 'go-project';
    
    const pathParts = relativePath.split(path.sep);
    if (pathParts.includes('automation')) return 'automation';
    if (pathParts.includes('docs')) return 'documentation';
    if (pathParts.includes('test') || pathParts.includes('tests')) return 'testing';
    if (pathParts.includes('config')) return 'configuration';
    
    return 'directory';
  }

  inferProject(relativePath) {
    const parts = relativePath.split(path.sep);
    if (parts.length > 0) return parts[0];
    return 'root';
  }

  findProject(filePath) {
    const parts = filePath.split(path.sep);
    for (let i = parts.length - 1; i >= 0; i--) {
      const projectPath = parts.slice(0, i + 1).join(path.sep);
      if (this.projects.has(projectPath)) {
        return projectPath;
      }
    }
    return this.inferProject(filePath);
  }

  detectLanguage(ext, filePath) {
    const languageMap = {
      '.js': 'JavaScript',
      '.ts': 'TypeScript',
      '.py': 'Python',
      '.java': 'Java',
      '.go': 'Go',
      '.rs': 'Rust',
      '.cpp': 'C++',
      '.c': 'C',
      '.cs': 'C#',
      '.php': 'PHP',
      '.rb': 'Ruby',
      '.swift': 'Swift',
      '.kt': 'Kotlin',
      '.scala': 'Scala',
      '.sh': 'Shell',
      '.bash': 'Bash',
      '.ps1': 'PowerShell',
      '.sql': 'SQL',
      '.html': 'HTML',
      '.css': 'CSS',
      '.scss': 'SCSS',
      '.json': 'JSON',
      '.xml': 'XML',
      '.yaml': 'YAML',
      '.yml': 'YAML',
      '.md': 'Markdown',
      '.txt': 'Text',
      '.dockerfile': 'Docker',
      '.sol': 'Solidity'
    };
    
    if (filePath.includes('Dockerfile')) return 'Docker';
    if (filePath.includes('Makefile')) return 'Makefile';
    
    return languageMap[ext] || 'Unknown';
  }

  isTextFile(ext) {
    const textExts = [
      '.js', '.ts', '.py', '.java', '.go', '.rs', '.cpp', '.c', '.cs',
      '.php', '.rb', '.swift', '.kt', '.scala', '.sh', '.bash', '.ps1',
      '.sql', '.html', '.css', '.scss', '.json', '.xml', '.yaml', '.yml',
      '.md', '.txt', '.dockerfile', '.sol', '.toml', '.ini', '.cfg'
    ];
    return textExts.includes(ext);
  }

  calculateComplexity(content, language) {
    const lines = content.split('\n').length;
    const functions = (content.match(/function|def |func |fn /g) || []).length;
    const conditions = (content.match(/if|else|switch|case|while|for|try|catch/g) || []).length;
    const classes = (content.match(/class |interface |struct /g) || []).length;
    
    return Math.min(10, Math.floor((functions + conditions + classes) / Math.max(lines, 1) * 100));
  }

  assessFileSecurity(content, ext) {
    const issues = [];
    let score = 10;
    
    // Common security issues
    if (content.includes('eval(')) { issues.push('eval-usage'); score -= 2; }
    if (content.includes('innerHTML')) { issues.push('innerHTML-usage'); score -= 1; }
    if (content.includes('document.write')) { issues.push('document-write'); score -= 1; }
    if (content.match(/password\s*=\s*["'][^"']+["']/i)) { issues.push('hardcoded-password'); score -= 3; }
    if (content.match(/api[_-]?key\s*=\s*["'][^"']+["']/i)) { issues.push('hardcoded-api-key'); score -= 3; }
    if (content.includes('SELECT * FROM')) { issues.push('sql-select-all'); score -= 1; }
    if (content.match(/\$\{[^}]*\}/)) { issues.push('template-injection-risk'); score -= 1; }
    
    // Language-specific issues
    if (ext === '.js' || ext === '.ts') {
      if (content.includes('dangerouslySetInnerHTML')) { issues.push('react-xss-risk'); score -= 2; }
      if (content.includes('new Function(')) { issues.push('function-constructor'); score -= 2; }
    }
    
    if (ext === '.py') {
      if (content.includes('pickle.loads')) { issues.push('pickle-deserialization'); score -= 2; }
      if (content.includes('exec(')) { issues.push('exec-usage'); score -= 2; }
    }
    
    return { score: Math.max(0, score), issues };
  }

  updateMetrics(fileInfo) {
    this.metrics.totalFiles++;
    this.metrics.totalSize += fileInfo.size;
    this.metrics.totalLines += fileInfo.lines;
    
    if (!this.metrics.languages[fileInfo.language]) {
      this.metrics.languages[fileInfo.language] = { files: 0, size: 0, lines: 0 };
    }
    this.metrics.languages[fileInfo.language].files++;
    this.metrics.languages[fileInfo.language].size += fileInfo.size;
    this.metrics.languages[fileInfo.language].lines += fileInfo.lines;
    
    if (fileInfo.security.issues.length > 0) {
      this.metrics.security.issues += fileInfo.security.issues.length;
    }
  }

  trackLanguage(language, fileInfo) {
    if (!this.languages.has(language)) {
      this.languages.set(language, { files: [], totalSize: 0, totalLines: 0 });
    }
    
    const langInfo = this.languages.get(language);
    langInfo.files.push(fileInfo.path);
    langInfo.totalSize += fileInfo.size;
    langInfo.totalLines += fileInfo.lines;
  }

  trackDependencies(content, filePath) {
    if (!content) return;
    
    // JavaScript/TypeScript dependencies
    const jsImports = content.match(/(?:import|require)\s*\(?['"`]([^'"`]+)['"`]\)?/g);
    if (jsImports) {
      jsImports.forEach(imp => {
        const match = imp.match(/['"`]([^'"`]+)['"`]/);
        if (match) this.addDependency(match[1], filePath);
      });
    }
    
    // Python imports
    const pyImports = content.match(/(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g);
    if (pyImports) {
      pyImports.forEach(imp => {
        const match = imp.match(/(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)/);
        if (match) this.addDependency(match[1], filePath);
      });
    }
  }

  addDependency(dep, filePath) {
    if (!this.dependencies.has(dep)) {
      this.dependencies.set(dep, { count: 0, files: [] });
    }
    this.dependencies.get(dep).count++;
    this.dependencies.get(dep).files.push(filePath);
  }

  async analyzeProjects() {
    for (const [projectPath, project] of this.projects) {
      const projectFiles = Array.from(this.files.values())
        .filter(file => file.path.startsWith(projectPath));
      
      project.files = projectFiles.map(f => f.path);
      project.languages = new Set(projectFiles.map(f => f.language));
      project.size = projectFiles.reduce((sum, f) => sum + f.size, 0);
      project.complexity = projectFiles.reduce((sum, f) => sum + f.complexity, 0) / Math.max(projectFiles.length, 1);
      project.security = {
        score: projectFiles.reduce((sum, f) => sum + f.security.score, 0) / Math.max(projectFiles.length, 1),
        issues: projectFiles.reduce((sum, f) => sum + f.security.issues.length, 0)
      };
    }
  }

  async assessSecurity() {
    const allFiles = Array.from(this.files.values());
    const totalSecurityScore = allFiles.reduce((sum, f) => sum + f.security.score, 0);
    this.metrics.security.score = totalSecurityScore / Math.max(allFiles.length, 1);
  }

  generateReport() {
    const topLanguages = Array.from(this.languages.entries())
      .sort((a, b) => b[1].totalSize - a[1].totalSize)
      .slice(0, 10);
    
    const topProjects = Array.from(this.projects.values())
      .sort((a, b) => b.size - a.size)
      .slice(0, 10);
    
    const topDependencies = Array.from(this.dependencies.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 15);
    
    const securityIssues = Array.from(this.files.values())
      .filter(f => f.security.issues.length > 0)
      .sort((a, b) => b.security.issues.length - a.security.issues.length);
    
    return {
      metadata: {
        timestamp: new Date().toISOString(),
        auditor: 'GitHub Repository Auditor v1.0',
        repository: path.basename(process.cwd())
      },
      overview: {
        totalFiles: this.metrics.totalFiles,
        totalDirectories: this.directories.size,
        totalSize: this.metrics.totalSize,
        totalLines: this.metrics.totalLines,
        languages: Object.keys(this.metrics.languages).length,
        projects: this.projects.size,
        healthScore: this.calculateHealthScore()
      },
      languages: Object.fromEntries(topLanguages),
      projects: topProjects,
      dependencies: Object.fromEntries(topDependencies),
      security: {
        overallScore: this.metrics.security.score,
        totalIssues: this.metrics.security.issues,
        riskyFiles: securityIssues.slice(0, 20)
      },
      recommendations: this.generateRecommendations()
    };
  }

  calculateHealthScore() {
    const securityWeight = 0.4;
    const complexityWeight = 0.3;
    const organizationWeight = 0.3;
    
    const securityScore = this.metrics.security.score;
    const complexityScore = 10 - (Array.from(this.files.values())
      .reduce((sum, f) => sum + f.complexity, 0) / Math.max(this.files.size, 1));
    const organizationScore = this.projects.size > 0 ? 8 : 5; // Bonus for organized projects
    
    return Math.round(
      securityScore * securityWeight +
      complexityScore * complexityWeight +
      organizationScore * organizationWeight
    );
  }

  generateRecommendations() {
    const recommendations = [];
    
    if (this.metrics.security.issues > 0) {
      recommendations.push({
        type: 'security',
        priority: 'high',
        message: `${this.metrics.security.issues} security issues found across repository`,
        action: 'Review and fix security vulnerabilities'
      });
    }
    
    const largeFiles = Array.from(this.files.values())
      .filter(f => f.size > 1024 * 1024); // > 1MB
    if (largeFiles.length > 0) {
      recommendations.push({
        type: 'performance',
        priority: 'medium',
        message: `${largeFiles.length} large files detected`,
        action: 'Consider splitting or optimizing large files'
      });
    }
    
    const complexFiles = Array.from(this.files.values())
      .filter(f => f.complexity > 8);
    if (complexFiles.length > 0) {
      recommendations.push({
        type: 'maintainability',
        priority: 'medium',
        message: `${complexFiles.length} highly complex files`,
        action: 'Refactor complex code for better maintainability'
      });
    }
    
    return recommendations;
  }

  async saveReport(report) {
    const reportsDir = path.join(process.cwd(), '.tools');
    await fs.mkdir(reportsDir, { recursive: true });
    
    const reportPath = path.join(reportsDir, 'github-repository-audit.json');
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`📄 Full repository audit saved to: ${reportPath}\n`);
  }

  displayReport(report) {
    console.log('📊 GITHUB REPOSITORY AUDIT REPORT');
    console.log('==================================');
    console.log(`Repository: ${report.metadata.repository}`);
    console.log(`Generated: ${report.metadata.timestamp}`);
    console.log(`Health Score: ${report.overview.healthScore}/10\n`);
    
    console.log('📈 OVERVIEW:');
    console.log(`   Files: ${report.overview.totalFiles.toLocaleString()}`);
    console.log(`   Directories: ${report.overview.totalDirectories.toLocaleString()}`);
    console.log(`   Size: ${Math.round(report.overview.totalSize / 1024 / 1024)}MB`);
    console.log(`   Lines of Code: ${report.overview.totalLines.toLocaleString()}`);
    console.log(`   Languages: ${report.overview.languages}`);
    console.log(`   Projects: ${report.overview.projects}\n`);
    
    console.log('🔤 TOP LANGUAGES:');
    Object.entries(report.languages).slice(0, 8).forEach(([lang, data]) => {
      console.log(`   ${lang}: ${data.files.length} files, ${Math.round(data.totalSize / 1024)}KB`);
    });
    console.log('');
    
    console.log('📂 TOP PROJECTS:');
    report.projects.slice(0, 8).forEach(project => {
      console.log(`   ${project.name}: ${project.files.length} files, ${Math.round(project.size / 1024)}KB`);
    });
    console.log('');
    
    console.log('📦 TOP DEPENDENCIES:');
    Object.entries(report.dependencies).slice(0, 10).forEach(([dep, data]) => {
      console.log(`   ${dep}: used in ${data.count} files`);
    });
    console.log('');
    
    console.log('🔒 SECURITY ANALYSIS:');
    console.log(`   Overall Score: ${report.security.overallScore.toFixed(1)}/10`);
    console.log(`   Total Issues: ${report.security.totalIssues}`);
    console.log(`   Risky Files: ${report.security.riskyFiles.length}\n`);
    
    if (report.recommendations.length > 0) {
      console.log('⚠️  RECOMMENDATIONS:');
      report.recommendations.forEach(rec => {
        console.log(`   ${rec.priority.toUpperCase()}: ${rec.message}`);
        console.log(`      Action: ${rec.action}`);
      });
      console.log('');
    }
    
    console.log('✅ REPOSITORY AUDIT COMPLETE');
  }
}

// CLI interface
if (require.main === module) {
  const auditor = new GitHubRepositoryAuditor();
  const targetPath = process.argv[2] || process.cwd();
  
  auditor.auditRepository(targetPath)
    .then(report => {
      console.log(`\n🎉 Audit complete! Health Score: ${report.overview.healthScore}/10`);
    })
    .catch(error => {
      console.error('❌ Audit failed:', error.message);
      process.exit(1);
    });
}

module.exports = GitHubRepositoryAuditor;