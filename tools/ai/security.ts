/**
 * AI Security Scanner
 * Comprehensive security scanning: secrets, vulnerabilities, licenses
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';
import { saveJson } from './utils/file-persistence.js';

const ROOT = process.cwd();
const AI_DIR = path.join(ROOT, '.ai');
export const SECURITY_REPORT_FILE = path.join(AI_DIR, 'security-report.json');

// ============================================================================
// Types
// ============================================================================

interface SecretPattern {
  name: string;
  pattern: RegExp;
  severity: 'critical' | 'high' | 'medium';
  description: string;
}

export interface SecurityFinding {
  id: string;
  timestamp: string;
  type: 'secret' | 'vulnerability' | 'license' | 'sensitive-file';
  severity: 'critical' | 'high' | 'medium' | 'low' | 'info';
  file?: string;
  line?: number;
  description: string;
  recommendation: string;
  resolved: boolean;
}

export interface VulnerabilityInfo {
  name: string;
  severity: string;
  range: string;
  fixAvailable: boolean;
}

export interface LicenseInfo {
  package: string;
  license: string;
  compatible: boolean;
}

export interface SecurityReport {
  timestamp: string;
  summary: {
    totalFindings: number;
    critical: number;
    high: number;
    medium: number;
    low: number;
    secrets: number;
    vulnerabilities: number;
    licenseIssues: number;
  };
  findings: SecurityFinding[];
  vulnerabilities: VulnerabilityInfo[];
  licenses: LicenseInfo[];
  score: number;
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
}

// ============================================================================
// Secret Patterns
// ============================================================================

const SECRET_PATTERNS: SecretPattern[] = [
  {
    name: 'AWS Access Key',
    pattern: /AKIA[0-9A-Z]{16}/g,
    severity: 'critical',
    description: 'AWS Access Key ID detected',
  },
  {
    name: 'AWS Secret Key',
    pattern: /[A-Za-z0-9/+=]{40}/g,
    severity: 'critical',
    description: 'Potential AWS Secret Access Key',
  },
  {
    name: 'GitHub Token',
    pattern: /gh[pousr]_[A-Za-z0-9_]{36,}/g,
    severity: 'critical',
    description: 'GitHub Personal Access Token detected',
  },
  {
    name: 'Generic API Key',
    pattern: /api[_-]?key\s*[:=]\s*['"][A-Za-z0-9_-]{20,}['"]/gi,
    severity: 'high',
    description: 'Generic API key pattern detected',
  },
  {
    name: 'Generic Secret',
    pattern: /secret\s*[:=]\s*['"][^'"]{8,}['"]/gi,
    severity: 'high',
    description: 'Generic secret pattern detected',
  },
  {
    name: 'Password',
    pattern: /password\s*[:=]\s*['"][^'"]{8,}['"]/gi,
    severity: 'high',
    description: 'Hardcoded password detected',
  },
  {
    name: 'Private Key',
    pattern: /-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----/g,
    severity: 'critical',
    description: 'Private key detected',
  },
  {
    name: 'JWT Token',
    pattern: /eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/g,
    severity: 'medium',
    description: 'JWT token detected',
  },
  {
    name: 'Bearer Token',
    pattern: /bearer\s+[A-Za-z0-9_.-]+/gi,
    severity: 'medium',
    description: 'Bearer token detected',
  },
  {
    name: 'Connection String',
    pattern: /mongodb(\+srv)?:\/\/[^\s'"]+/gi,
    severity: 'high',
    description: 'Database connection string detected',
  },
  {
    name: 'Slack Token',
    pattern: /xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}/g,
    severity: 'high',
    description: 'Slack token detected',
  },
];

// ============================================================================
// Sensitive File Patterns
// ============================================================================

const SENSITIVE_FILE_PATTERNS = [
  { pattern: /\.env(\..+)?$/i, severity: 'high' as const, description: 'Environment file' },
  { pattern: /\.pem$/i, severity: 'critical' as const, description: 'PEM certificate/key' },
  { pattern: /\.key$/i, severity: 'critical' as const, description: 'Key file' },
  { pattern: /\.p12$/i, severity: 'critical' as const, description: 'PKCS12 certificate' },
  { pattern: /\.pfx$/i, severity: 'critical' as const, description: 'PFX certificate' },
  { pattern: /id_rsa$/i, severity: 'critical' as const, description: 'SSH private key' },
  { pattern: /id_ed25519$/i, severity: 'critical' as const, description: 'SSH private key' },
  { pattern: /credentials/i, severity: 'high' as const, description: 'Credentials file' },
  {
    pattern: /secrets?\.(json|ya?ml|env)/i,
    severity: 'high' as const,
    description: 'Secrets file',
  },
  { pattern: /\.htpasswd$/i, severity: 'high' as const, description: 'Apache password file' },
];

// ============================================================================
// Allowed Licenses
// ============================================================================

const ALLOWED_LICENSES = [
  'MIT',
  'ISC',
  'Apache-2.0',
  'BSD-2-Clause',
  'BSD-3-Clause',
  'CC0-1.0',
  'Unlicense',
  '0BSD',
];

// ============================================================================
// Security Scanner Implementation
// ============================================================================

export class SecurityScanner {
  private findings: SecurityFinding[] = [];
  private vulnerabilities: VulnerabilityInfo[] = [];
  private licenses: LicenseInfo[] = [];

  private generateId(): string {
    return `sec-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  // Scan for secrets in files
  async scanSecrets(paths: string[] = ['.']): Promise<SecurityFinding[]> {
    const findings: SecurityFinding[] = [];
    const excludeDirs = ['node_modules', '.git', 'dist', 'build', 'coverage', '.ai', '.ORCHEX'];

    const scanFile = (filePath: string): void => {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        for (const pattern of SECRET_PATTERNS) {
          for (let i = 0; i < lines.length; i++) {
            const matches = lines[i].match(pattern.pattern);
            if (matches) {
              findings.push({
                id: this.generateId(),
                timestamp: new Date().toISOString(),
                type: 'secret',
                severity: pattern.severity,
                file: filePath,
                line: i + 1,
                description: pattern.description,
                recommendation: `Remove ${pattern.name} from source code and use environment variables`,
                resolved: false,
              });
            }
          }
        }
      } catch {
        // Skip files that can't be read
      }
    };

    const scanDirectory = (dir: string): void => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);

          if (entry.isDirectory()) {
            if (!excludeDirs.includes(entry.name)) {
              scanDirectory(fullPath);
            }
          } else if (entry.isFile()) {
            // Only scan text files
            const ext = path.extname(entry.name).toLowerCase();
            const textExtensions = [
              '.ts',
              '.js',
              '.json',
              '.yaml',
              '.yml',
              '.md',
              '.txt',
              '.env',
              '.sh',
              '.py',
            ];
            if (textExtensions.includes(ext) || entry.name.startsWith('.')) {
              scanFile(fullPath);
            }
          }
        }
      } catch {
        // Skip directories that can't be read
      }
    };

    for (const p of paths) {
      const fullPath = path.resolve(ROOT, p);
      if (fs.existsSync(fullPath)) {
        if (fs.statSync(fullPath).isDirectory()) {
          scanDirectory(fullPath);
        } else {
          scanFile(fullPath);
        }
      }
    }

    this.findings.push(...findings);
    return findings;
  }

  // Scan for sensitive files
  scanSensitiveFiles(paths: string[] = ['.']): SecurityFinding[] {
    const findings: SecurityFinding[] = [];
    const excludeDirs = ['node_modules', '.git'];

    const scanDirectory = (dir: string): void => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });

        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);

          if (entry.isDirectory()) {
            if (!excludeDirs.includes(entry.name)) {
              scanDirectory(fullPath);
            }
          } else if (entry.isFile()) {
            for (const pattern of SENSITIVE_FILE_PATTERNS) {
              if (pattern.pattern.test(entry.name)) {
                findings.push({
                  id: this.generateId(),
                  timestamp: new Date().toISOString(),
                  type: 'sensitive-file',
                  severity: pattern.severity,
                  file: fullPath,
                  description: `Sensitive file detected: ${pattern.description}`,
                  recommendation: 'Add to .gitignore and remove from version control',
                  resolved: false,
                });
                break;
              }
            }
          }
        }
      } catch {
        // Skip directories that can't be read
      }
    };

    for (const p of paths) {
      const fullPath = path.resolve(ROOT, p);
      if (fs.existsSync(fullPath) && fs.statSync(fullPath).isDirectory()) {
        scanDirectory(fullPath);
      }
    }

    this.findings.push(...findings);
    return findings;
  }

  // Scan npm vulnerabilities
  scanVulnerabilities(): VulnerabilityInfo[] {
    try {
      const result = execSync('npm audit --json', {
        encoding: 'utf8',
        cwd: ROOT,
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      const audit = JSON.parse(result);
      const vulnerabilities: VulnerabilityInfo[] = [];

      if (audit.vulnerabilities) {
        for (const [name, info] of Object.entries(audit.vulnerabilities)) {
          const vuln = info as { severity: string; range: string; fixAvailable: boolean };
          vulnerabilities.push({
            name,
            severity: vuln.severity,
            range: vuln.range,
            fixAvailable: vuln.fixAvailable || false,
          });

          // Add as finding
          this.findings.push({
            id: this.generateId(),
            timestamp: new Date().toISOString(),
            type: 'vulnerability',
            severity:
              vuln.severity === 'critical'
                ? 'critical'
                : vuln.severity === 'high'
                  ? 'high'
                  : 'medium',
            description: `Vulnerability in ${name}: ${vuln.severity}`,
            recommendation: vuln.fixAvailable ? 'Run npm audit fix' : 'Update or replace package',
            resolved: false,
          });
        }
      }

      this.vulnerabilities = vulnerabilities;
      return vulnerabilities;
    } catch {
      // npm audit returns non-zero when vulnerabilities exist
      try {
        const result = execSync('npm audit --json 2>&1 || true', {
          encoding: 'utf8',
          cwd: ROOT,
        });
        const audit = JSON.parse(result);

        if (audit.metadata?.vulnerabilities) {
          const meta = audit.metadata.vulnerabilities;
          if (meta.critical > 0 || meta.high > 0) {
            this.findings.push({
              id: this.generateId(),
              timestamp: new Date().toISOString(),
              type: 'vulnerability',
              severity: meta.critical > 0 ? 'critical' : 'high',
              description: `npm audit: ${meta.critical} critical, ${meta.high} high, ${meta.moderate} moderate`,
              recommendation: 'Run npm audit fix to address vulnerabilities',
              resolved: false,
            });
          }
        }
      } catch {
        // Ignore
      }
      return [];
    }
  }

  // Scan for license compliance
  scanLicenses(): LicenseInfo[] {
    const packageJsonPath = path.join(ROOT, 'package.json');
    const packageLockPath = path.join(ROOT, 'package-lock.json');

    if (!fs.existsSync(packageJsonPath)) {
      return [];
    }

    const licenses: LicenseInfo[] = [];

    try {
      if (fs.existsSync(packageLockPath)) {
        const lockfile = JSON.parse(fs.readFileSync(packageLockPath, 'utf8'));
        const packages = lockfile.packages || {};

        for (const [pkgPath, info] of Object.entries(packages)) {
          if (pkgPath === '' || !pkgPath.includes('node_modules')) continue;

          const pkgInfo = info as { license?: string };
          const pkgName = pkgPath.replace(/^node_modules\//, '');
          const license = pkgInfo.license || 'Unknown';
          const compatible = ALLOWED_LICENSES.includes(license) || license === 'Unknown';

          licenses.push({
            package: pkgName,
            license,
            compatible,
          });

          if (!compatible && license !== 'Unknown') {
            this.findings.push({
              id: this.generateId(),
              timestamp: new Date().toISOString(),
              type: 'license',
              severity: 'medium',
              description: `Package ${pkgName} uses ${license} license`,
              recommendation: 'Review license compatibility for commercial use',
              resolved: false,
            });
          }
        }
      }
    } catch {
      // Ignore parse errors
    }

    this.licenses = licenses;
    return licenses;
  }

  // Generate full security report
  generateReport(): SecurityReport {
    const summary = {
      totalFindings: this.findings.length,
      critical: this.findings.filter((f) => f.severity === 'critical').length,
      high: this.findings.filter((f) => f.severity === 'high').length,
      medium: this.findings.filter((f) => f.severity === 'medium').length,
      low: this.findings.filter((f) => f.severity === 'low').length,
      secrets: this.findings.filter((f) => f.type === 'secret').length,
      vulnerabilities: this.findings.filter((f) => f.type === 'vulnerability').length,
      licenseIssues: this.findings.filter((f) => f.type === 'license').length,
    };

    // Calculate security score
    const deductions =
      summary.critical * 25 + summary.high * 15 + summary.medium * 5 + summary.low * 2;
    const score = Math.max(0, 100 - deductions);

    const grade: SecurityReport['grade'] =
      score >= 90 ? 'A' : score >= 80 ? 'B' : score >= 70 ? 'C' : score >= 60 ? 'D' : 'F';

    const report: SecurityReport = {
      timestamp: new Date().toISOString(),
      summary,
      findings: this.findings,
      vulnerabilities: this.vulnerabilities,
      licenses: this.licenses.filter((l) => !l.compatible),
      score,
      grade,
    };

    // Save report
    saveJson(SECURITY_REPORT_FILE, report);

    return report;
  }

  // Run full scan
  async fullScan(paths: string[] = ['.']): Promise<SecurityReport> {
    this.findings = [];
    this.vulnerabilities = [];
    this.licenses = [];

    console.log('🔍 Scanning for secrets...');
    await this.scanSecrets(paths);

    console.log('📁 Scanning for sensitive files...');
    this.scanSensitiveFiles(paths);

    console.log('📦 Scanning npm vulnerabilities...');
    this.scanVulnerabilities();

    console.log('📜 Scanning licenses...');
    this.scanLicenses();

    return this.generateReport();
  }
}
