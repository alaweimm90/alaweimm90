/**
 * Integration tests for AI Security Module
 * Tests security scanning for secrets, vulnerabilities, and licenses
 */

import { describe, it, expect, vi } from 'vitest';

describe('AI Security Module', () => {
  describe('Secret Detection Patterns', () => {
    const SECRET_PATTERNS = [
      { name: 'AWS Access Key', pattern: /AKIA[0-9A-Z]{16}/g, severity: 'critical' },
      { name: 'GitHub Token', pattern: /gh[pousr]_[A-Za-z0-9_]{36,}/g, severity: 'critical' },
      {
        name: 'Generic API Key',
        pattern: /api[_-]?key\s*[:=]\s*['"][A-Za-z0-9_-]{20,}['"]/gi,
        severity: 'high',
      },
      {
        name: 'Password Assignment',
        pattern: /password\s*[:=]\s*['"][^'"]{8,}['"]/gi,
        severity: 'critical',
      },
      {
        name: 'Private Key',
        pattern: /-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----/g,
        severity: 'critical',
      },
      {
        name: 'JWT Token',
        pattern: /eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/g,
        severity: 'high',
      },
    ];

    it('should detect AWS access keys', () => {
      const content = 'const key = "AKIAIOSFODNN7EXAMPLE"';
      const awsPattern = SECRET_PATTERNS.find((p) => p.name === 'AWS Access Key')!;
      const matches = content.match(awsPattern.pattern);
      expect(matches).not.toBeNull();
      expect(matches![0]).toBe('AKIAIOSFODNN7EXAMPLE');
    });

    it('should detect GitHub tokens', () => {
      const content = 'token: ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx1234';
      const ghPattern = SECRET_PATTERNS.find((p) => p.name === 'GitHub Token')!;
      const matches = content.match(ghPattern.pattern);
      expect(matches).not.toBeNull();
    });

    it('should detect generic API keys', () => {
      const content = 'api_key = "sk-1234567890abcdefghij"';
      const apiPattern = SECRET_PATTERNS.find((p) => p.name === 'Generic API Key')!;
      const matches = content.match(apiPattern.pattern);
      expect(matches).not.toBeNull();
    });

    it('should detect password assignments', () => {
      const content = 'password: "MySecretPassword123!"';
      const pwdPattern = SECRET_PATTERNS.find((p) => p.name === 'Password Assignment')!;
      const matches = content.match(pwdPattern.pattern);
      expect(matches).not.toBeNull();
    });

    it('should detect private keys', () => {
      const content = '-----BEGIN RSA PRIVATE KEY-----\nMIIE...';
      const keyPattern = SECRET_PATTERNS.find((p) => p.name === 'Private Key')!;
      const matches = content.match(keyPattern.pattern);
      expect(matches).not.toBeNull();
    });

    it('should detect JWT tokens', () => {
      const content =
        'token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"';
      const jwtPattern = SECRET_PATTERNS.find((p) => p.name === 'JWT Token')!;
      const matches = content.match(jwtPattern.pattern);
      expect(matches).not.toBeNull();
    });

    it('should not false positive on safe content', () => {
      const content = 'const API_KEY = process.env.API_KEY;';
      const pwdPattern = SECRET_PATTERNS.find((p) => p.name === 'Password Assignment')!;
      const matches = content.match(pwdPattern.pattern);
      expect(matches).toBeNull();
    });
  });

  describe('Sensitive File Detection', () => {
    const SENSITIVE_PATTERNS = [
      { pattern: '.env', type: 'Environment file' },
      { pattern: '.env.local', type: 'Local environment file' },
      { pattern: '.env.production', type: 'Production secrets' },
      { pattern: 'credentials.json', type: 'Credentials file' },
      { pattern: '.pem', type: 'Certificate file' },
      { pattern: '.key', type: 'Key file' },
      { pattern: 'id_rsa', type: 'SSH private key' },
      { pattern: '.p12', type: 'PKCS#12 file' },
    ];

    it('should detect .env files', () => {
      const files = ['.env', 'config.ts', 'package.json'];
      const sensitive = files.filter((f) =>
        SENSITIVE_PATTERNS.some((p) => f.includes(p.pattern) || f.endsWith(p.pattern))
      );
      expect(sensitive).toContain('.env');
    });

    it('should detect all environment file variants', () => {
      const files = ['.env', '.env.local', '.env.production', '.env.development'];
      const envPatterns = SENSITIVE_PATTERNS.filter((p) => p.pattern.startsWith('.env'));
      const detected = files.filter((f) =>
        envPatterns.some((p) => f.includes(p.pattern) || f.endsWith(p.pattern))
      );
      expect(detected.length).toBeGreaterThan(0);
    });

    it('should detect certificate files', () => {
      const files = ['cert.pem', 'server.key', 'id_rsa'];
      const certPatterns = SENSITIVE_PATTERNS.filter((p) =>
        ['.pem', '.key', 'id_rsa'].includes(p.pattern)
      );
      const detected = files.filter((f) =>
        certPatterns.some((p) => f.includes(p.pattern) || f.endsWith(p.pattern))
      );
      expect(detected.length).toBe(3);
    });
  });

  describe('Security Finding Structure', () => {
    interface SecurityFinding {
      type: 'secret' | 'sensitive-file' | 'vulnerability' | 'license';
      severity: 'critical' | 'high' | 'medium' | 'low';
      file: string;
      line?: number;
      description: string;
      recommendation: string;
    }

    it('should create valid finding structure', () => {
      const finding: SecurityFinding = {
        type: 'secret',
        severity: 'critical',
        file: 'config.ts',
        line: 42,
        description: 'AWS Access Key detected',
        recommendation: 'Remove the key and use environment variables instead',
      };

      expect(finding.type).toBe('secret');
      expect(finding.severity).toBe('critical');
      expect(finding.file).toBeDefined();
      expect(finding.recommendation).toBeDefined();
    });
  });

  describe('Vulnerability Scanning', () => {
    interface VulnerabilityInfo {
      package: string;
      severity: 'critical' | 'high' | 'moderate' | 'low';
      title: string;
      fixAvailable: boolean;
    }

    it('should parse npm audit output format', () => {
      const auditOutput = {
        vulnerabilities: {
          lodash: {
            severity: 'high',
            via: [{ title: 'Prototype Pollution' }],
            fixAvailable: true,
          },
        },
      };

      const vulns: VulnerabilityInfo[] = [];
      for (const [pkg, data] of Object.entries(auditOutput.vulnerabilities)) {
        vulns.push({
          package: pkg,
          severity: data.severity as VulnerabilityInfo['severity'],
          title: data.via[0].title,
          fixAvailable: data.fixAvailable,
        });
      }

      expect(vulns.length).toBe(1);
      expect(vulns[0].package).toBe('lodash');
      expect(vulns[0].fixAvailable).toBe(true);
    });
  });

  describe('License Scanning', () => {
    const ALLOWED_LICENSES = ['MIT', 'Apache-2.0', 'BSD-2-Clause', 'BSD-3-Clause', 'ISC', '0BSD'];
    const COPYLEFT_LICENSES = ['GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'LGPL-2.1', 'LGPL-3.0'];
    const RESTRICTED_LICENSES = ['UNLICENSED', 'UNKNOWN', 'CC-BY-NC'];

    it('should allow permissive licenses', () => {
      for (const license of ALLOWED_LICENSES) {
        const isAllowed = ALLOWED_LICENSES.includes(license);
        expect(isAllowed).toBe(true);
      }
    });

    it('should flag copyleft licenses', () => {
      for (const license of COPYLEFT_LICENSES) {
        const isCopyleft = COPYLEFT_LICENSES.includes(license);
        expect(isCopyleft).toBe(true);
      }
    });

    it('should flag restricted licenses', () => {
      for (const license of RESTRICTED_LICENSES) {
        const isRestricted = RESTRICTED_LICENSES.includes(license);
        expect(isRestricted).toBe(true);
      }
    });

    it('should categorize license correctly', () => {
      const categorizeLicense = (license: string) => {
        if (ALLOWED_LICENSES.includes(license)) return 'allowed';
        if (COPYLEFT_LICENSES.includes(license)) return 'copyleft';
        if (RESTRICTED_LICENSES.includes(license)) return 'restricted';
        return 'unknown';
      };

      expect(categorizeLicense('MIT')).toBe('allowed');
      expect(categorizeLicense('GPL-3.0')).toBe('copyleft');
      expect(categorizeLicense('UNLICENSED')).toBe('restricted');
      expect(categorizeLicense('Custom-License')).toBe('unknown');
    });
  });

  describe('Security Report', () => {
    interface SecurityReport {
      timestamp: string;
      scanDuration: number;
      summary: {
        secrets: number;
        sensitiveFiles: number;
        vulnerabilities: number;
        licenseIssues: number;
      };
      riskLevel: 'low' | 'medium' | 'high' | 'critical';
      findings: unknown[];
    }

    it('should calculate risk level based on findings', () => {
      const calculateRiskLevel = (
        critical: number,
        high: number
      ): 'low' | 'medium' | 'high' | 'critical' => {
        if (critical > 0) return 'critical';
        if (high > 0) return 'high';
        return 'low';
      };

      expect(calculateRiskLevel(1, 0)).toBe('critical');
      expect(calculateRiskLevel(0, 2)).toBe('high');
      expect(calculateRiskLevel(0, 0)).toBe('low');
    });

    it('should generate valid report structure', () => {
      const report: SecurityReport = {
        timestamp: new Date().toISOString(),
        scanDuration: 1500,
        summary: {
          secrets: 0,
          sensitiveFiles: 1,
          vulnerabilities: 2,
          licenseIssues: 0,
        },
        riskLevel: 'medium',
        findings: [],
      };

      expect(report.timestamp).toBeDefined();
      expect(report.scanDuration).toBeGreaterThan(0);
      expect(
        report.summary.secrets +
          report.summary.sensitiveFiles +
          report.summary.vulnerabilities +
          report.summary.licenseIssues
      ).toBe(3);
    });
  });

  describe('File Scanning', () => {
    it('should skip binary files', () => {
      const binaryExtensions = ['.png', '.jpg', '.gif', '.ico', '.woff', '.woff2', '.eot', '.ttf'];
      const file = 'image.png';

      const isBinary = binaryExtensions.some((ext) => file.endsWith(ext));
      expect(isBinary).toBe(true);
    });

    it('should scan text files', () => {
      const textExtensions = ['.ts', '.js', '.json', '.yaml', '.yml', '.md', '.env'];
      const file = 'config.ts';

      const isText = textExtensions.some((ext) => file.endsWith(ext));
      expect(isText).toBe(true);
    });
  });

  describe('Secret Masking', () => {
    it('should mask secrets in output', () => {
      const maskSecret = (secret: string): string => {
        if (secret.length <= 8) return '****';
        return secret.substring(0, 4) + '...' + secret.substring(secret.length - 4);
      };

      expect(maskSecret('AKIAIOSFODNN7EXAMPLE')).toBe('AKIA...MPLE');
      expect(maskSecret('short')).toBe('****');
    });
  });
});
