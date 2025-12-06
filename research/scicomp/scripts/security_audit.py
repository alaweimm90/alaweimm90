#!/usr/bin/env python3
"""
SciComp Security Audit

Comprehensive security vulnerability scanner for the SciComp repository.
Identifies potential security issues, code injection risks, and unsafe practices.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import re
import ast
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

class SecurityAuditor:
    """Security vulnerability scanner for SciComp."""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.vulnerabilities = defaultdict(list)
        self.severity_counts = defaultdict(int)
        
        # Security patterns to detect (scientific computing context-aware)
        self.security_patterns = {
            'HIGH': [
                (r'eval\s*\([^)]*(?:input|request|sys\.argv|raw_input)', 'Dangerous eval() with user input'),
                (r'exec\s*\([^)]*(?:input|request|sys\.argv|raw_input)', 'Dangerous exec() with user input'),
                (r'subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True[^)]*[\+\%]', 'Shell injection risk'),
                (r'os\.system\s*\([^)]*[\+\%][^)]*\)', 'Command injection with user input'),
                (r'pickle\.loads?\s*\([^)]*(?:request|input|untrusted)', 'Unsafe pickle deserialization'),
                (r'yaml\.load\s*\([^)]*(?:request|input)[^)]*\)', 'Unsafe YAML loading with user input'),
            ],
            'MEDIUM': [
                (r'random\.seed\s*\(\s*(?:1|123|42|0)\s*\)(?!.*(?:test|demo|example))', 'Predictable random seed'),
                (r'hashlib\.md5\s*\([^)]*(?:password|secret|key)', 'MD5 for sensitive data'),
                (r'hashlib\.sha1\s*\([^)]*(?:password|secret|key)', 'SHA1 for sensitive data'),
                (r'input\s*\([^)]*\)(?!.*(?:int|float|validate|strip|sanitize))', 'Unvalidated user input'),
                (r'tempfile\.mktemp\s*\(', 'Insecure temp file creation'),
            ],
            'LOW': [
                (r'assert\s+[^#]*(?!(?:test_|_test|# test|test case|unit test))', 'Assert in non-test code'),
                (r'print\s*\([^)]*["\'][^"\']*(?:password|secret|key|token|auth)[^"\']*["\']', 'Potential secret in print'),
                (r'logging\.[^(]*\([^)]*["\'][^"\']*(?:password|secret|key|token)[^"\']*["\']', 'Secret in logging'),
                (r'requests\.[a-z]+\([^)]*verify\s*=\s*False', 'SSL verification disabled'),
            ]
        }
        
        # Hardcoded secrets patterns (refined)
        self.secret_patterns = [
            (r'password\s*=\s*[\'"][^\'"]{8,}[\'"](?!.*test|demo|example)', 'Hardcoded password'),
            (r'api_key\s*=\s*[\'"][A-Za-z0-9]{20,}[\'"]', 'Hardcoded API key'),
            (r'secret\s*=\s*[\'"][A-Za-z0-9]{16,}[\'"]', 'Hardcoded secret'),
            (r'token\s*=\s*[\'"][A-Za-z0-9]{20,}[\'"]', 'Hardcoded token'),
            (r'(?:BEGIN|END)\s+(RSA\s+)?PRIVATE\s+KEY', 'Private key in code'),
        ]
    
    def scan_python_security(self) -> Dict[str, Any]:
        """Scan Python files for security vulnerabilities."""
        print("🔒 Scanning Python files for security issues...")
        
        results = {'files_scanned': 0, 'vulnerabilities': defaultdict(list)}
        python_files = list(self.repo_root.glob("**/*.py"))
        
        for py_file in python_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file) or '.backup_' in str(py_file):
                continue
            
            # Skip test files, examples, and demos for certain security checks
            is_test_file = ('test' in py_file.name.lower() or 
                           'example' in str(py_file).lower() or 
                           'demo' in str(py_file).lower() or
                           '/examples/' in str(py_file))
                
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                results['files_scanned'] += 1
                
                # Check for security patterns
                for severity, patterns in self.security_patterns.items():
                    for pattern, description in patterns:
                        # Skip certain patterns for test files
                        if is_test_file and ('random' in description.lower() or 'assert' in description.lower()):
                            continue
                            
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_no = content[:match.start()].count('\n') + 1
                            
                            # Additional context-based filtering
                            line_content = content.split('\n')[line_no-1] if line_no <= len(content.split('\n')) else ""
                            
                            # Skip scientific/mathematical contexts
                            if any(sci_word in line_content.lower() for sci_word in 
                                  ['numpy', 'scipy', 'scientific', 'math', 'random.rand', 'np.random', 'monte carlo', 'simulation']):
                                continue
                            
                            results['vulnerabilities'][severity].append({
                                'file': str(py_file.relative_to(self.repo_root)),
                                'line': line_no,
                                'pattern': description,
                                'code': match.group().strip()
                            })
                            self.severity_counts[severity] += 1
                
                # Check for hardcoded secrets
                for pattern, description in self.secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_no = content[:match.start()].count('\n') + 1
                        results['vulnerabilities']['SECRET'].append({
                            'file': str(py_file.relative_to(self.repo_root)),
                            'line': line_no,
                            'pattern': description,
                            'code': match.group().strip()[:50] + '...' if len(match.group()) > 50 else match.group().strip()
                        })
                        self.severity_counts['SECRET'] += 1
                
            except Exception as e:
                results['vulnerabilities']['ERROR'].append({
                    'file': str(py_file.relative_to(self.repo_root)),
                    'error': str(e)
                })
        
        return results
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check for unsafe file permissions."""
        print("📁 Checking file permissions...")
        
        results = {'issues': [], 'files_checked': 0}
        
        for file_path in self.repo_root.rglob("*"):
            if file_path.is_file() and '.git' not in str(file_path):
                results['files_checked'] += 1
                
                try:
                    stat = file_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    
                    # Check for world-writable files
                    if mode.endswith('2') or mode.endswith('6') or mode.endswith('7'):
                        results['issues'].append({
                            'file': str(file_path.relative_to(self.repo_root)),
                            'issue': f'World-writable permissions: {mode}',
                            'severity': 'HIGH'
                        })
                    
                    # Check for executable files that shouldn't be
                    if file_path.suffix in ['.py', '.md', '.txt', '.json'] and mode.startswith('7'):
                        results['issues'].append({
                            'file': str(file_path.relative_to(self.repo_root)),
                            'issue': f'Unnecessary execute permissions: {mode}',
                            'severity': 'MEDIUM'
                        })
                        
                except Exception as e:
                    results['issues'].append({
                        'file': str(file_path.relative_to(self.repo_root)),
                        'error': str(e),
                        'severity': 'LOW'
                    })
        
        return results
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities."""
        print("📦 Scanning dependencies for vulnerabilities...")
        
        results = {'vulnerable_deps': [], 'total_deps': 0}
        
        # Known vulnerable packages (simplified - in real use, would query CVE databases)
        known_vulnerabilities = {
            'urllib3': ['<1.26.5', 'HTTPS certificate validation bypass'],
            'requests': ['<2.25.1', 'HTTP header injection'],
            'pyyaml': ['<5.4.1', 'Arbitrary code execution'],
            'pillow': ['<8.3.2', 'Buffer overflow'],
            'numpy': ['<1.21.0', 'Memory corruption'],
        }
        
        req_files = ['requirements.txt', 'requirements-dev.txt', 'setup.py']
        
        for req_file in req_files:
            file_path = self.repo_root / req_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    
                    # Extract package names and versions
                    if req_file.endswith('.txt'):
                        packages = re.findall(r'^([a-zA-Z0-9-_]+)([><=!]+[0-9.]+)?', content, re.MULTILINE)
                    else:
                        packages = re.findall(r"['\"]([a-zA-Z0-9-_]+)([><=!]+[0-9.]+)?['\"]", content)
                    
                    for pkg_name, version_spec in packages:
                        if pkg_name and not pkg_name.startswith('#'):
                            results['total_deps'] += 1
                            
                            # Check against known vulnerabilities
                            if pkg_name.lower() in known_vulnerabilities:
                                vuln_version, description = known_vulnerabilities[pkg_name.lower()]
                                results['vulnerable_deps'].append({
                                    'package': pkg_name,
                                    'file': req_file,
                                    'vulnerable_version': vuln_version,
                                    'description': description,
                                    'current_spec': version_spec or 'No version specified'
                                })
                                
                except Exception as e:
                    results['vulnerable_deps'].append({
                        'file': req_file,
                        'error': str(e)
                    })
        
        return results
    
    def check_configuration_security(self) -> Dict[str, Any]:
        """Check configuration files for security issues."""
        print("⚙️ Checking configuration security...")
        
        results = {'issues': [], 'files_checked': 0}
        
        config_files = list(self.repo_root.glob("**/*.json")) + \
                      list(self.repo_root.glob("**/*.yaml")) + \
                      list(self.repo_root.glob("**/*.yml")) + \
                      list(self.repo_root.glob("**/*.toml")) + \
                      list(self.repo_root.glob("**/*.cfg"))
        
        for config_file in config_files:
            if '.git' in str(config_file):
                continue
                
            results['files_checked'] += 1
            
            try:
                content = config_file.read_text(encoding='utf-8', errors='ignore')
                
                # Check for potential secrets in config
                secret_indicators = ['password', 'secret', 'key', 'token', 'api']
                for indicator in secret_indicators:
                    pattern = rf'{indicator}["\']?\s*[:=]\s*["\']?[^"\'\s]{{8,}}'
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_no = content[:match.start()].count('\n') + 1
                        results['issues'].append({
                            'file': str(config_file.relative_to(self.repo_root)),
                            'line': line_no,
                            'issue': f'Potential secret in config: {indicator}',
                            'severity': 'HIGH'
                        })
                
                # Check for debug settings
                if re.search(r'debug\s*[:=]\s*true', content, re.IGNORECASE):
                    results['issues'].append({
                        'file': str(config_file.relative_to(self.repo_root)),
                        'issue': 'Debug mode enabled in config',
                        'severity': 'MEDIUM'
                    })
                    
            except Exception as e:
                results['issues'].append({
                    'file': str(config_file.relative_to(self.repo_root)),
                    'error': str(e),
                    'severity': 'LOW'
                })
        
        return results
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        print("🛡️ Generating security audit report...")
        
        # Run all security checks
        python_security = self.scan_python_security()
        file_permissions = self.check_file_permissions()
        dependencies = self.scan_dependencies()
        config_security = self.check_configuration_security()
        
        # Compile overall report
        report = {
            'summary': {
                'total_files_scanned': (python_security['files_scanned'] + 
                                      file_permissions['files_checked'] + 
                                      config_security['files_checked']),
                'high_severity_issues': self.severity_counts['HIGH'],
                'medium_severity_issues': self.severity_counts['MEDIUM'],
                'low_severity_issues': self.severity_counts['LOW'],
                'secrets_found': self.severity_counts['SECRET'],
                'vulnerable_dependencies': len(dependencies['vulnerable_deps'])
            },
            'python_security': python_security,
            'file_permissions': file_permissions,
            'dependencies': dependencies,
            'configuration_security': config_security
        }
        
        # Calculate overall security score
        total_issues = (self.severity_counts['HIGH'] * 10 + 
                       self.severity_counts['MEDIUM'] * 5 + 
                       self.severity_counts['LOW'] * 1 + 
                       self.severity_counts['SECRET'] * 15)
        
        if total_issues == 0:
            security_score = 100
        elif total_issues <= 5:
            security_score = 90
        elif total_issues <= 15:
            security_score = 75
        elif total_issues <= 30:
            security_score = 60
        else:
            security_score = max(0, 50 - (total_issues - 30))
        
        report['security_score'] = security_score
        
        return report
    
    def print_security_summary(self, report: Dict[str, Any]):
        """Print security audit summary."""
        print("\n" + "="*80)
        print("🛡️  SECURITY AUDIT SUMMARY 🛡️")
        print("="*80)
        
        summary = report['summary']
        score = report['security_score']
        
        print(f"Files Scanned: {summary['total_files_scanned']}")
        print(f"Security Score: {score}/100")
        print()
        
        # Print issues by severity
        if summary['high_severity_issues'] > 0:
            print(f"🚨 HIGH Severity Issues: {summary['high_severity_issues']}")
        if summary['medium_severity_issues'] > 0:
            print(f"⚠️  MEDIUM Severity Issues: {summary['medium_severity_issues']}")
        if summary['low_severity_issues'] > 0:
            print(f"ℹ️  LOW Severity Issues: {summary['low_severity_issues']}")
        if summary['secrets_found'] > 0:
            print(f"🔐 Potential Secrets Found: {summary['secrets_found']}")
        if summary['vulnerable_dependencies'] > 0:
            print(f"📦 Vulnerable Dependencies: {summary['vulnerable_dependencies']}")
        
        # Overall assessment
        print("\n🎯 SECURITY ASSESSMENT:")
        if score >= 90:
            print("✅ EXCELLENT: No significant security issues found")
        elif score >= 75:
            print("⚠️  GOOD: Minor security issues found, review recommended")
        elif score >= 60:
            print("⚠️  FAIR: Some security issues found, fixes recommended")
        else:
            print("❌ POOR: Critical security issues found, immediate action required")
        
        print("\n*Security audit completed*")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='SciComp Security Audit')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--export-report', help='Export detailed report to JSON file')
    parser.add_argument('--show-details', action='store_true', help='Show detailed vulnerability information')
    
    args = parser.parse_args()
    
    auditor = SecurityAuditor(args.repo_root)
    report = auditor.generate_security_report()
    
    auditor.print_security_summary(report)
    
    if args.show_details:
        # Print detailed issues
        for severity in ['HIGH', 'MEDIUM', 'LOW', 'SECRET']:
            if severity in report['python_security']['vulnerabilities']:
                issues = report['python_security']['vulnerabilities'][severity]
                if issues:
                    print(f"\n{severity} Severity Issues:")
                    for issue in issues[:5]:  # Show first 5
                        print(f"  • {issue['file']}:{issue['line']} - {issue['pattern']}")
    
    if args.export_report:
        with open(args.export_report, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Security report exported to: {args.export_report}")
    
    # Exit with error code if critical issues found
    critical_issues = report['summary']['high_severity_issues'] + report['summary']['secrets_found']
    sys.exit(1 if critical_issues > 0 else 0)

if __name__ == "__main__":
    main()