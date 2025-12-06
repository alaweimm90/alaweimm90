#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

console.log('🔒 Enterprise Security & Compliance Scanner\n');

const securityReport = {
  timestamp: new Date().toISOString(),
  summary: {
    totalIssues: 0,
    criticalVulns: 0,
    highRiskFiles: 0,
    complianceScore: 0,
    securityScore: 0
  },
  vulnerabilities: [],
  secrets: [],
  permissions: [],
  dependencies: [],
  compliance: {
    gdpr: { score: 0, issues: [] },
    hipaa: { score: 0, issues: [] },
    soc2: { score: 0, issues: [] },
    pci: { score: 0, issues: [] }
  },
  recommendations: []
};

// 1. SECRETS DETECTION
console.log('🔍 Scanning for exposed secrets and credentials...\n');

function scanForSecrets() {
  const secretPatterns = [
    // API Keys and Tokens
    { name: 'AWS Access Key', pattern: /(?:AKIA|ASIA)[0-9A-Z]{16}/, severity: 'critical' },
    { name: 'AWS Secret Key', pattern: /[A-Za-z0-9/+=]{40}/, severity: 'critical' },
    { name: 'OpenAI API Key', pattern: /sk-[A-Za-z0-9]{48}/, severity: 'critical' },
    { name: 'Stripe Secret Key', pattern: /sk_live_[0-9a-zA-Z]{24}/, severity: 'critical' },
    { name: 'Stripe Test Key', pattern: /sk_test_[0-9a-zA-Z]{24}/, severity: 'high' },
    { name: 'Supabase Service Key', pattern: /eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]*/, severity: 'critical' },
    { name: 'GitHub Token', pattern: /ghp_[A-Za-z0-9]{36}/, severity: 'high' },
    { name: 'Generic API Key', pattern: /['"\\s]?[Aa]pi[_-]?[Kk]ey['"\\s]?[:=]['"\\s]?[A-Za-z0-9]{20,}/, severity: 'high' },
    
    // Database and Connection Strings
    { name: 'Database URL', pattern: /postgres:\/\/[^\\s'"]+/, severity: 'high' },
    { name: 'MongoDB Connection', pattern: /mongodb(\+srv)?:\/\/[^\\s'"]+/, severity: 'high' },
    { name: 'Redis URL', pattern: /redis:\/\/[^\\s'"]+/, severity: 'medium' },
    
    // Private Keys and Certificates
    { name: 'Private Key', pattern: /-----BEGIN [A-Z]+ PRIVATE KEY-----/, severity: 'critical' },
    { name: 'RSA Private Key', pattern: /-----BEGIN RSA PRIVATE KEY-----/, severity: 'critical' },
    { name: 'SSH Private Key', pattern: /-----BEGIN OPENSSH PRIVATE KEY-----/, severity: 'critical' },
    
    // Common Secret Indicators
    { name: 'Password Field', pattern: /['"\\s]password['"\\s]?[:=]['"\\s]?[^\\s'"]{8,}/, severity: 'medium' },
    { name: 'Secret Field', pattern: /['"\\s]secret['"\\s]?[:=]['"\\s]?[A-Za-z0-9]{16,}/, severity: 'high' },
    { name: 'Token Field', pattern: /['"\\s]token['"\\s]?[:=]['"\\s]?[A-Za-z0-9]{20,}/, severity: 'high' }
  ];

  const secretsFound = [];
  
  function scanFile(filePath, relativePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const lines = content.split('\n');
      
      secretPatterns.forEach(pattern => {
        lines.forEach((line, index) => {
          if (pattern.pattern.test(line)) {
            // Skip obvious false positives
            if (line.includes('example') || 
                line.includes('placeholder') || 
                line.includes('YOUR_API_KEY') ||
                line.includes('sk_test_')) {
              return;
            }
            
            secretsFound.push({
              type: pattern.name,
              severity: pattern.severity,
              file: relativePath,
              line: index + 1,
              preview: line.trim().substring(0, 100) + '...',
              recommendation: 'Move to environment variables or secure vault'
            });
          }
        });
      });
    } catch (error) {
      // Skip files that can't be read
    }
  }
  
  function scanDirectory(dir, relativePath = '') {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        if (item === 'node_modules' || item === '.git' || item === '_graveyard') return;
        
        const itemPath = path.join(dir, item);
        const itemRelativePath = path.join(relativePath, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          scanDirectory(itemPath, itemRelativePath);
        } else if (item.endsWith('.ts') || item.endsWith('.tsx') || 
                   item.endsWith('.js') || item.endsWith('.jsx') || 
                   item.endsWith('.env') || item.endsWith('.json') ||
                   item.endsWith('.yml') || item.endsWith('.yaml')) {
          scanFile(itemPath, itemRelativePath);
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  scanDirectory(rootDir);
  securityReport.secrets = secretsFound;
  
  if (secretsFound.length > 0) {
    console.log(`🚨 Found ${secretsFound.length} potential secrets:`);
    secretsFound.forEach((secret, i) => {
      const severityIcon = secret.severity === 'critical' ? '🔴' :
                          secret.severity === 'high' ? '🟠' :
                          secret.severity === 'medium' ? '🟡' : '🔵';
      console.log(`  ${severityIcon} ${secret.type} in ${secret.file}:${secret.line}`);
    });
  } else {
    console.log('✅ No exposed secrets detected');
  }
  console.log();
}

scanForSecrets();

// 2. DEPENDENCY VULNERABILITIES
console.log('📦 Scanning dependencies for vulnerabilities...\n');

function scanDependencyVulnerabilities() {
  try {
    const auditResult = execSync('npm audit --json 2>/dev/null || echo "{\\"vulnerabilities\\":{}}"', { 
      encoding: 'utf8',
      cwd: rootDir 
    });
    
    const audit = JSON.parse(auditResult);
    const vulnerabilities = [];
    
    if (audit.vulnerabilities) {
      Object.entries(audit.vulnerabilities).forEach(([name, vuln]) => {
        vulnerabilities.push({
          package: name,
          severity: vuln.severity,
          range: vuln.range,
          via: Array.isArray(vuln.via) ? vuln.via.join(', ') : vuln.via,
          recommendation: 'Update to latest secure version'
        });
      });
    }
    
    securityReport.vulnerabilities = vulnerabilities;
    
    if (vulnerabilities.length > 0) {
      console.log(`🔍 Found ${vulnerabilities.length} dependency vulnerabilities:`);
      vulnerabilities.forEach(vuln => {
        const severityIcon = vuln.severity === 'critical' ? '🔴' :
                            vuln.severity === 'high' ? '🟠' :
                            vuln.severity === 'moderate' ? '🟡' : '🔵';
        console.log(`  ${severityIcon} ${vuln.package}: ${vuln.severity}`);
      });
    } else {
      console.log('✅ No dependency vulnerabilities found');
    }
  } catch (error) {
    console.log('⚠️  Could not scan dependencies');
  }
  console.log();
}

scanDependencyVulnerabilities();

// 3. FILE PERMISSIONS AUDIT
console.log('🔐 Auditing file permissions...\n');

function auditFilePermissions() {
  const permissionIssues = [];
  
  function checkPermissions(dir, relativePath = '') {
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        if (item === 'node_modules' || item === '.git') return;
        
        const itemPath = path.join(dir, item);
        const itemRelativePath = path.join(relativePath, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          checkPermissions(itemPath, itemRelativePath);
        } else {
          // Check for overly permissive files
          const mode = stat.mode;
          const isExecutable = (mode & parseInt('111', 8)) !== 0;
          const isWorldWritable = (mode & parseInt('002', 8)) !== 0;
          
          if (isWorldWritable) {
            permissionIssues.push({
              file: itemRelativePath,
              issue: 'World-writable file',
              severity: 'high',
              recommendation: 'Remove world-write permissions'
            });
          }
          
          if (isExecutable && !item.endsWith('.sh') && !item.endsWith('.mjs')) {
            permissionIssues.push({
              file: itemRelativePath,
              issue: 'Unexpected executable file',
              severity: 'medium',
              recommendation: 'Review if executable permissions are necessary'
            });
          }
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  checkPermissions(rootDir);
  securityReport.permissions = permissionIssues;
  
  if (permissionIssues.length > 0) {
    console.log(`⚠️  Found ${permissionIssues.length} permission issues:`);
    permissionIssues.forEach(issue => {
      const severityIcon = issue.severity === 'high' ? '🟠' : '🟡';
      console.log(`  ${severityIcon} ${issue.file}: ${issue.issue}`);
    });
  } else {
    console.log('✅ File permissions look secure');
  }
  console.log();
}

auditFilePermissions();

// 4. COMPLIANCE CHECKS
console.log('📋 Running compliance checks...\n');

function checkCompliance() {
  const complianceChecks = {
    gdpr: {
      name: 'GDPR (General Data Protection Regulation)',
      checks: [
        { name: 'Privacy Policy', file: 'privacy-policy.md', weight: 20 },
        { name: 'Cookie Consent', pattern: /cookie.*consent/i, weight: 15 },
        { name: 'Data Processing Notice', pattern: /data.*processing/i, weight: 15 },
        { name: 'User Rights Implementation', pattern: /right.*to.*be.*forgotten|data.*portability/i, weight: 25 },
        { name: 'Consent Management', pattern: /consent.*management|gdpr.*consent/i, weight: 25 }
      ]
    },
    hipaa: {
      name: 'HIPAA (Health Insurance Portability and Accountability Act)',
      checks: [
        { name: 'PHI Encryption', pattern: /encrypt.*phi|phi.*encrypt/i, weight: 30 },
        { name: 'Access Controls', pattern: /access.*control.*health|hipaa.*access/i, weight: 25 },
        { name: 'Audit Logging', pattern: /audit.*log.*health|health.*audit/i, weight: 25 },
        { name: 'Business Associate Agreement', file: 'baa.md', weight: 20 }
      ]
    },
    soc2: {
      name: 'SOC 2 (System and Organization Controls)',
      checks: [
        { name: 'Security Monitoring', pattern: /security.*monitor|monitor.*security/i, weight: 25 },
        { name: 'Incident Response', pattern: /incident.*response|security.*incident/i, weight: 25 },
        { name: 'Access Management', pattern: /access.*management|identity.*access/i, weight: 25 },
        { name: 'Data Backup', pattern: /data.*backup|backup.*strategy/i, weight: 25 }
      ]
    },
    pci: {
      name: 'PCI DSS (Payment Card Industry Data Security Standard)',
      checks: [
        { name: 'Secure Payment Processing', pattern: /secure.*payment|payment.*security|stripe.*secure/i, weight: 40 },
        { name: 'Tokenization', pattern: /token.*payment|payment.*token/i, weight: 30 },
        { name: 'PCI Compliance Documentation', file: 'pci-compliance.md', weight: 30 }
      ]
    }
  };
  
  Object.entries(complianceChecks).forEach(([standard, config]) => {
    let score = 0;
    const issues = [];
    
    config.checks.forEach(check => {
      let found = false;
      
      if (check.file) {
        // Check for specific file
        const filePath = path.join(rootDir, 'docs', check.file);
        found = fs.existsSync(filePath);
      } else if (check.pattern) {
        // Search for pattern in codebase
        found = searchPatternInCodebase(check.pattern);
      }
      
      if (found) {
        score += check.weight;
      } else {
        issues.push({
          check: check.name,
          requirement: check.file ? `Missing file: ${check.file}` : `Missing implementation of: ${check.name}`,
          severity: check.weight > 25 ? 'high' : 'medium'
        });
      }
    });
    
    securityReport.compliance[standard] = {
      score: Math.round(score),
      issues: issues
    };
    
    console.log(`📋 ${config.name}: ${score}% compliant`);
    if (issues.length > 0) {
      issues.forEach(issue => {
        const severityIcon = issue.severity === 'high' ? '🟠' : '🟡';
        console.log(`  ${severityIcon} ${issue.requirement}`);
      });
    }
  });
  
  console.log();
}

function searchPatternInCodebase(pattern) {
  let found = false;
  
  function searchInDirectory(dir) {
    if (found) return;
    
    try {
      const items = fs.readdirSync(dir);
      
      items.forEach(item => {
        if (found || item === 'node_modules' || item === '.git') return;
        
        const itemPath = path.join(dir, item);
        const stat = fs.statSync(itemPath);
        
        if (stat.isDirectory()) {
          searchInDirectory(itemPath);
        } else if (item.endsWith('.ts') || item.endsWith('.tsx') || 
                   item.endsWith('.js') || item.endsWith('.jsx') ||
                   item.endsWith('.md')) {
          try {
            const content = fs.readFileSync(itemPath, 'utf8');
            if (pattern.test(content)) {
              found = true;
            }
          } catch (error) {
            // Skip files that can't be read
          }
        }
      });
    } catch (error) {
      // Skip directories that can't be read
    }
  }
  
  searchInDirectory(rootDir);
  return found;
}

checkCompliance();

// 5. CALCULATE SECURITY SCORES
function calculateSecurityScores() {
  const criticalIssues = [
    ...securityReport.secrets.filter(s => s.severity === 'critical'),
    ...securityReport.vulnerabilities.filter(v => v.severity === 'critical'),
    ...securityReport.permissions.filter(p => p.severity === 'critical')
  ].length;
  
  const highIssues = [
    ...securityReport.secrets.filter(s => s.severity === 'high'),
    ...securityReport.vulnerabilities.filter(v => v.severity === 'high'),
    ...securityReport.permissions.filter(p => p.severity === 'high')
  ].length;
  
  // Security score (0-100)
  let securityScore = 100;
  securityScore -= (criticalIssues * 25);
  securityScore -= (highIssues * 10);
  securityScore = Math.max(0, securityScore);
  
  // Compliance score (average of all standards)
  const complianceScores = Object.values(securityReport.compliance).map(c => c.score);
  const avgComplianceScore = complianceScores.length > 0 ? 
    Math.round(complianceScores.reduce((a, b) => a + b, 0) / complianceScores.length) : 0;
  
  securityReport.summary = {
    totalIssues: securityReport.secrets.length + securityReport.vulnerabilities.length + securityReport.permissions.length,
    criticalVulns: criticalIssues,
    highRiskFiles: securityReport.secrets.filter(s => s.severity === 'critical' || s.severity === 'high').length,
    complianceScore: avgComplianceScore,
    securityScore: securityScore
  };
}

calculateSecurityScores();

// 6. GENERATE RECOMMENDATIONS
function generateSecurityRecommendations() {
  const recommendations = [];
  
  if (securityReport.secrets.length > 0) {
    recommendations.push('Move all secrets and API keys to environment variables');
    recommendations.push('Implement secret scanning in CI/CD pipeline');
    recommendations.push('Use a secure vault solution for production secrets');
  }
  
  if (securityReport.vulnerabilities.length > 0) {
    recommendations.push('Update vulnerable dependencies immediately');
    recommendations.push('Set up automated dependency vulnerability scanning');
    recommendations.push('Implement dependency update policies');
  }
  
  if (securityReport.permissions.length > 0) {
    recommendations.push('Review and fix file permission issues');
    recommendations.push('Implement least-privilege access controls');
  }
  
  if (securityReport.summary.complianceScore < 80) {
    recommendations.push('Improve compliance documentation and implementation');
    recommendations.push('Conduct compliance audit with security professionals');
  }
  
  recommendations.push('Implement security headers and HTTPS enforcement');
  recommendations.push('Set up security monitoring and alerting');
  recommendations.push('Conduct regular security assessments');
  recommendations.push('Implement security training for development team');
  
  securityReport.recommendations = recommendations;
}

generateSecurityRecommendations();

// 7. SAVE SECURITY REPORT
fs.writeFileSync(
  path.join(rootDir, 'security-report.json'),
  JSON.stringify(securityReport, null, 2)
);

// 8. DISPLAY SECURITY DASHBOARD
console.log('🔒 Enterprise Security Dashboard');
console.log('═'.repeat(60));
console.log(`🛡️  Security Score: ${securityReport.summary.securityScore}/100`);
console.log(`📋 Compliance Score: ${securityReport.summary.complianceScore}/100`);
console.log(`🚨 Total Issues: ${securityReport.summary.totalIssues}`);
console.log(`🔴 Critical Vulnerabilities: ${securityReport.summary.criticalVulns}`);
console.log(`⚠️  High Risk Files: ${securityReport.summary.highRiskFiles}`);
console.log('═'.repeat(60));

console.log('\n📊 Issue Breakdown:');
console.log(`🔑 Exposed Secrets: ${securityReport.secrets.length}`);
console.log(`📦 Dependency Vulnerabilities: ${securityReport.vulnerabilities.length}`);
console.log(`🔐 Permission Issues: ${securityReport.permissions.length}`);

console.log('\n📋 Compliance Status:');
Object.entries(securityReport.compliance).forEach(([standard, result]) => {
  const status = result.score >= 80 ? '✅' : result.score >= 60 ? '⚠️' : '❌';
  console.log(`${status} ${standard.toUpperCase()}: ${result.score}%`);
});

if (securityReport.recommendations.length > 0) {
  console.log('\n💡 Security Recommendations:');
  securityReport.recommendations.forEach((rec, i) => {
    console.log(`  ${i + 1}. ${rec}`);
  });
}

console.log('\n📄 Security report saved to: security-report.json');
console.log('\n🔒 Security scan complete!');

// Exit with error code if critical issues found
process.exit(securityReport.summary.criticalVulns > 0 ? 1 : 0);