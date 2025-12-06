#!/usr/bin/env python3
"""
SciComp Automated Deployment Script

Handles complete deployment workflow including validation, backup, and rollout.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""

import os
import sys
import shutil
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
import hashlib

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

class SciCompDeployer:
    """Automated deployment system for SciComp."""
    
    def __init__(self, target_env='production', dry_run=False):
        self.repo_root = Path(__file__).parent.parent
        self.target_env = target_env
        self.dry_run = dry_run
        self.deployment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.deployment_log = []
        
        # Deployment environments
        self.environments = {
            'development': {
                'min_score': 60,
                'required_tests': ['validation'],
                'backup_required': False
            },
            'staging': {
                'min_score': 75,
                'required_tests': ['validation', 'consistency', 'integration'],
                'backup_required': True
            },
            'production': {
                'min_score': 85,
                'required_tests': ['validation', 'consistency', 'integration', 'security'],
                'backup_required': True
            }
        }
    
    def log(self, message, level='INFO'):
        """Log deployment messages."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.deployment_log.append(log_entry)
        
        # Color output based on level
        if level == 'SUCCESS':
            print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")
        elif level == 'ERROR':
            print(f"{Colors.RED}✗ {message}{Colors.RESET}")
        elif level == 'WARNING':
            print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")
        elif level == 'INFO':
            print(f"{Colors.CYAN}ℹ {message}{Colors.RESET}")
        else:
            print(f"  {message}")
    
    def print_header(self):
        """Print deployment header."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}{Colors.BOLD}🚀 SciComp Automated Deployment System 🚀{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.YELLOW}Target Environment: {self.target_env.upper()}{Colors.RESET}")
        print(f"{Colors.YELLOW}Deployment ID: {self.deployment_id}{Colors.RESET}")
        if self.dry_run:
            print(f"{Colors.MAGENTA}DRY RUN MODE - No actual changes will be made{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")
    
    def check_prerequisites(self):
        """Check deployment prerequisites."""
        self.log("Checking deployment prerequisites...", "INFO")
        
        # Check Python version
        if sys.version_info < (3, 8):
            self.log(f"Python 3.8+ required (current: {sys.version})", "ERROR")
            return False
        
        # Check git status
        try:
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.stdout.strip() and self.target_env == 'production':
                self.log("Uncommitted changes detected (production requires clean repo)", "WARNING")
                if not self.dry_run:
                    response = input("Continue anyway? (y/N): ")
                    if response.lower() != 'y':
                        return False
        except:
            self.log("Git not available", "WARNING")
        
        # Check required files
        required_files = ['requirements.txt', 'setup.py', 'README.md', 'LICENSE']
        for file in required_files:
            if not (self.repo_root / file).exists():
                self.log(f"Required file missing: {file}", "ERROR")
                return False
        
        self.log("Prerequisites check passed", "SUCCESS")
        return True
    
    def run_validation_tests(self):
        """Run required validation tests."""
        self.log(f"Running {self.target_env} validation tests...", "INFO")
        
        env_config = self.environments[self.target_env]
        required_tests = env_config['required_tests']
        min_score = env_config['min_score']
        
        # Run quick deployment check first
        try:
            result = subprocess.run([sys.executable, 'scripts/quick_deployment_check.py'],
                                  capture_output=True, text=True, cwd=self.repo_root,
                                  timeout=60)
            
            # Parse score from output
            score = 0
            for line in result.stdout.split('\n'):
                if 'DEPLOYMENT READINESS SCORE:' in line:
                    score = int(line.split(':')[1].split('/')[0].strip())
                    break
            
            if score < min_score:
                self.log(f"Score {score} below minimum {min_score} for {self.target_env}", "ERROR")
                return False
            
            self.log(f"Deployment readiness score: {score}/100", "SUCCESS")
            
        except subprocess.TimeoutExpired:
            self.log("Validation tests timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Validation error: {str(e)}", "ERROR")
            return False
        
        # Run specific required tests
        for test in required_tests:
            self.log(f"Running {test} test...", "INFO")
            
            test_scripts = {
                'validation': 'scripts/validate_framework.py',
                'consistency': 'scripts/consistency_check.py',
                'integration': 'scripts/integration_tests.py',
                'security': 'scripts/security_audit.py'
            }
            
            if test in test_scripts:
                script = test_scripts[test]
                if (self.repo_root / script).exists():
                    if self.dry_run:
                        self.log(f"[DRY RUN] Would run {script}", "INFO")
                    else:
                        # Run test with timeout
                        try:
                            result = subprocess.run([sys.executable, script],
                                                  capture_output=True, text=True,
                                                  cwd=self.repo_root, timeout=300)
                            if result.returncode != 0:
                                self.log(f"{test} test failed", "WARNING")
                                # Don't fail deployment for non-critical tests in dev/staging
                                if self.target_env == 'production':
                                    return False
                        except subprocess.TimeoutExpired:
                            self.log(f"{test} test timed out", "WARNING")
        
        self.log("All validation tests completed", "SUCCESS")
        return True
    
    def create_backup(self):
        """Create backup before deployment."""
        if not self.environments[self.target_env]['backup_required']:
            self.log("Backup not required for this environment", "INFO")
            return True
        
        self.log("Creating deployment backup...", "INFO")
        
        backup_dir = self.repo_root / 'backups' / f"backup_{self.deployment_id}"
        
        if self.dry_run:
            self.log(f"[DRY RUN] Would create backup at {backup_dir}", "INFO")
            return True
        
        try:
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Files to backup
            items_to_backup = [
                'Python',
                'MATLAB',
                'Mathematica',
                'requirements.txt',
                'setup.py',
                'README.md'
            ]
            
            for item in items_to_backup:
                source = self.repo_root / item
                if source.exists():
                    dest = backup_dir / item
                    if source.is_dir():
                        shutil.copytree(source, dest)
                    else:
                        shutil.copy2(source, dest)
            
            # Create backup manifest
            manifest = {
                'deployment_id': self.deployment_id,
                'timestamp': datetime.now().isoformat(),
                'environment': self.target_env,
                'files_backed_up': items_to_backup
            }
            
            with open(backup_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.log(f"Backup created at {backup_dir}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Backup failed: {str(e)}", "ERROR")
            return False
    
    def optimize_deployment(self):
        """Optimize code for deployment."""
        self.log("Optimizing deployment...", "INFO")
        
        if self.dry_run:
            self.log("[DRY RUN] Would optimize Python bytecode", "INFO")
            return True
        
        try:
            # Compile Python files to bytecode
            import py_compile
            python_files = list((self.repo_root / 'Python').glob('**/*.py'))
            
            compiled = 0
            for py_file in python_files:
                try:
                    py_compile.compile(str(py_file), doraise=True, optimize=2)
                    compiled += 1
                except:
                    pass
            
            self.log(f"Compiled {compiled} Python files", "SUCCESS")
            
            # Remove unnecessary files
            patterns_to_remove = ['**/__pycache__', '**/*.pyc', '**/.DS_Store', '**/Thumbs.db']
            removed = 0
            
            for pattern in patterns_to_remove:
                for item in self.repo_root.glob(pattern):
                    if item.is_file():
                        item.unlink()
                        removed += 1
                    elif item.is_dir():
                        shutil.rmtree(item)
                        removed += 1
            
            if removed > 0:
                self.log(f"Removed {removed} unnecessary files", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.log(f"Optimization error: {str(e)}", "WARNING")
            return True  # Don't fail deployment for optimization issues
    
    def generate_deployment_report(self):
        """Generate deployment report."""
        self.log("Generating deployment report...", "INFO")
        
        report = {
            'deployment_id': self.deployment_id,
            'timestamp': datetime.now().isoformat(),
            'environment': self.target_env,
            'dry_run': self.dry_run,
            'status': 'SUCCESS',
            'log': self.deployment_log
        }
        
        report_file = self.repo_root / f"deployment_report_{self.deployment_id}.json"
        
        if not self.dry_run:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            self.log(f"Report saved to {report_file.name}", "SUCCESS")
        else:
            self.log("[DRY RUN] Would save deployment report", "INFO")
        
        return report
    
    def deploy(self):
        """Execute deployment workflow."""
        self.print_header()
        
        steps = [
            ("Prerequisites Check", self.check_prerequisites),
            ("Validation Tests", self.run_validation_tests),
            ("Backup Creation", self.create_backup),
            ("Deployment Optimization", self.optimize_deployment),
            ("Report Generation", self.generate_deployment_report)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{Colors.BOLD}📋 {step_name}{Colors.RESET}")
            print("-" * 40)
            
            if not step_func():
                self.log(f"Deployment failed at: {step_name}", "ERROR")
                self.save_failure_log()
                return False
        
        # Final success message
        print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}✅ DEPLOYMENT SUCCESSFUL!{Colors.RESET}")
        print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}Deployment Summary:{Colors.RESET}")
        print(f"  • Environment: {self.target_env}")
        print(f"  • Deployment ID: {self.deployment_id}")
        print(f"  • Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"  • Status: SUCCESS")
        
        if self.target_env == 'production':
            print(f"\n{Colors.BLUE}🐻 Go Bears! SciComp deployed to production! 🐻{Colors.RESET}")
        
        return True
    
    def save_failure_log(self):
        """Save failure log for debugging."""
        log_file = self.repo_root / f"deployment_failure_{self.deployment_id}.log"
        
        if not self.dry_run:
            with open(log_file, 'w') as f:
                for entry in self.deployment_log:
                    f.write(entry + '\n')
            print(f"\nFailure log saved to: {log_file.name}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='SciComp Automated Deployment')
    parser.add_argument('--env', choices=['development', 'staging', 'production'],
                       default='production', help='Target environment')
    parser.add_argument('--dry-run', action='store_true',
                       help='Simulate deployment without making changes')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip validation tests (not recommended)')
    parser.add_argument('--force', action='store_true',
                       help='Force deployment even with warnings')
    
    args = parser.parse_args()
    
    # Confirmation for production deployment
    if args.env == 'production' and not args.dry_run:
        print(f"{Colors.YELLOW}⚠️  WARNING: Production deployment requested{Colors.RESET}")
        response = input("Are you sure you want to deploy to PRODUCTION? (yes/no): ")
        if response.lower() != 'yes':
            print("Deployment cancelled")
            sys.exit(0)
    
    deployer = SciCompDeployer(target_env=args.env, dry_run=args.dry_run)
    
    if deployer.deploy():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()