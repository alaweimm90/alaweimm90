#!/usr/bin/env python3
"""
SciComp Repository Consistency Check
Comprehensive pre-deployment verification script that checks:
- Naming consistency across all files
- Code style and formatting standards
- Documentation completeness and accuracy
- Reference consistency (URLs, citations, etc.)
- File structure and organization
- Dependencies and requirements
- Cross-platform compatibility markers
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""
import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter
import subprocess
class Colors:
    """ANSI color codes for terminal output."""
    BERKELEY_BLUE = '\033[38;5;18m'
    CALIFORNIA_GOLD = '\033[38;5;178m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
class ConsistencyChecker:
    """Main consistency checking class."""
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.issues = defaultdict(list)
        self.stats = defaultdict(int)
        self.warnings = defaultdict(list)
        # Expected naming patterns
        self.correct_names = {
            'project_name': 'SciComp',
            'full_name': 'SciComp: A Cross-Platform Scientific Computing Suite for Research and Education',
            'author': 'Meshal Alawein',
            'email': 'meshal@berkeley.edu',
            'institution': 'University of California, Berkeley',
            'license': 'MIT',
            'tagline': 'Crafted with love, 🐻 energy, and zero sleep.',
            'repo_url': 'https://github.com/berkeley/scicomp',
            'copyright': '© 2025 Meshal Alawein'
        }
        # Deprecated/incorrect naming patterns to flag
        self.deprecated_patterns = [
            r'SciComp',
            r'SciComp',
            r'UC SciComp',
            r'A Cross-Platform Scientific Computing Suite for Research and Education',
            r'Crafted with love, 🐻 energy, and zero sleep.',
        ]
        # File extensions to check
        self.code_extensions = {'.py', '.m', '.nb', '.wl'}
        self.doc_extensions = {'.md', '.rst', '.txt'}
        self.config_extensions = {'.json', '.toml', '.yaml', '.yml', '.cfg'}
    def print_header(self):
        """Print consistency check header."""
        print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{Colors.BOLD}🔍 SciComp Repository Consistency Check 🔍{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}University of California, Berkeley{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}Pre-Deployment Verification System{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}\n")
    def check_naming_consistency(self):
        """Check for consistent naming across all files."""
        print(f"{Colors.BOLD}📝 Checking Naming Consistency...{Colors.RESET}")
        # Find all text files
        text_files = []
        for ext in self.code_extensions | self.doc_extensions | self.config_extensions:
            text_files.extend(self.repo_root.glob(f"**/*{ext}"))
        # Check each file for deprecated patterns
        for file_path in text_files:
            if '.git' in str(file_path) or '__pycache__' in str(file_path):
                continue
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                # Check for deprecated naming patterns
                for pattern in self.deprecated_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        self.issues['naming'].append(
                            f"{file_path.relative_to(self.repo_root)}: Found deprecated '{pattern}'"
                        )
                # Check for correct project name usage
                if 'SciComp' in content:
                    self.stats['files_with_correct_name'] += 1
            except Exception as e:
                self.warnings['file_read'].append(f"Could not read {file_path}: {e}")
        print(f"   ✅ Checked {len(text_files)} files for naming consistency")
    def check_code_style(self):
        """Check code style and formatting."""
        print(f"{Colors.BOLD}🎨 Checking Code Style...{Colors.RESET}")
        # Check Python files for basic style issues
        py_files = list(self.repo_root.glob("**/*.py"))
        style_issues = []
        for py_file in py_files:
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding='utf-8', errors='ignore')
                # Check for basic style issues
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    # Long lines (>100 chars, excluding comments)
                    if len(line) > 100 and not line.strip().startswith('#'):
                        style_issues.append(f"{py_file.relative_to(self.repo_root)}:{i}: Line too long ({len(line)} chars)")
                    # Trailing whitespace
                    if line.endswith(' ') or line.endswith('\t'):
                        style_issues.append(f"{py_file.relative_to(self.repo_root)}:{i}: Trailing whitespace")
                # Check for missing docstrings in functions
                if re.search(r'^def\s+\w+.*:\s*$', content, re.MULTILINE):
                    if not re.search(r'""".*?"""', content, re.DOTALL):
                        self.warnings['docstrings'].append(
                            f"{py_file.relative_to(self.repo_root)}: Missing docstrings"
                        )
            except Exception as e:
                self.warnings['style_check'].append(f"Could not check {py_file}: {e}")
        if len(style_issues) > 20:  # Only show first 20 to avoid spam
            self.issues['style'] = style_issues[:20] + [f"... and {len(style_issues)-20} more style issues"]
        else:
            self.issues['style'] = style_issues
        print(f"   ✅ Checked {len(py_files)} Python files for style issues")
    def check_documentation_consistency(self):
        """Check documentation completeness and consistency."""
        print(f"{Colors.BOLD}📚 Checking Documentation Consistency...{Colors.RESET}")
        # Required files
        required_files = [
            'README.md',
            'LICENSE',
            'CONTRIBUTING.md',
            'CITATION.cff',
            'setup.py',
            'requirements.txt'
        ]
        for req_file in required_files:
            if not (self.repo_root / req_file).exists():
                self.issues['missing_files'].append(f"Missing required file: {req_file}")
            else:
                self.stats['required_files_present'] += 1
        # Check README.md specifically
        readme_path = self.repo_root / 'README.md'
        if readme_path.exists():
            readme_content = readme_path.read_text(encoding='utf-8', errors='ignore')
            # Check for required sections
            required_sections = [
                'Installation', 'Quick Start', 'Citation', 'License', 'Contact'
            ]
            for section in required_sections:
                if f"## {section}" not in readme_content and f"# {section}" not in readme_content:
                    self.issues['readme'].append(f"README missing section: {section}")
            # Check for correct tagline at end
            if not readme_content.strip().endswith(self.correct_names['tagline']):
                self.issues['readme'].append("README doesn't end with correct tagline")
        # Check API documentation exists
        api_docs = list(self.repo_root.glob("docs/api/*.md"))
        if len(api_docs) < 10:  # Should have substantial API docs
            self.warnings['documentation'].append(f"Only {len(api_docs)} API documentation files found")
        print(f"   ✅ Checked documentation structure and completeness")
    def check_reference_consistency(self):
        """Check URLs, emails, and other references."""
        print(f"{Colors.BOLD}🔗 Checking Reference Consistency...{Colors.RESET}")
        # Find all references to check
        all_files = list(self.repo_root.glob("**/*"))
        text_files = [f for f in all_files if f.suffix in self.doc_extensions | self.code_extensions]
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].,)]'
        emails_found = set()
        urls_found = set()
        for file_path in text_files[:50]:  # Limit to avoid overwhelming output
            if '.git' in str(file_path):
                continue
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                # Find emails
                emails = re.findall(email_pattern, content)
                emails_found.update(emails)
                # Find URLs
                urls = re.findall(url_pattern, content)
                urls_found.update(urls)
            except Exception as e:
                continue
        # Check for consistency
        expected_email = self.correct_names['email']
        if expected_email not in emails_found and len(emails_found) > 0:
            self.warnings['references'].append(f"Expected email {expected_email} not found consistently")
        # Check for broken/inconsistent GitHub URLs
        github_urls = [url for url in urls_found if 'github.com' in url]
        if len(github_urls) > 1:
            unique_repos = set(re.findall(r'github\.com/([^/]+/[^/]+)', ' '.join(github_urls)))
            if len(unique_repos) > 1:
                self.warnings['references'].append(f"Multiple GitHub repos referenced: {unique_repos}")
        self.stats['unique_emails'] = len(emails_found)
        self.stats['unique_urls'] = len(urls_found)
        print(f"   ✅ Found {len(emails_found)} unique emails, {len(urls_found)} unique URLs")
    def check_dependencies(self):
        """Check dependency consistency across requirements files."""
        print(f"{Colors.BOLD}📦 Checking Dependencies...{Colors.RESET}")
        req_files = [
            'requirements.txt',
            'requirements-dev.txt',
            'setup.py'
        ]
        dependencies = {}
        for req_file in req_files:
            file_path = self.repo_root / req_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Extract package names (simplified)
                    if req_file.endswith('.txt'):
                        packages = re.findall(r'^([a-zA-Z0-9-_]+)(?:[><=!].*)?$', content, re.MULTILINE)
                    else:  # setup.py
                        packages = re.findall(r"['\"]([a-zA-Z0-9-_]+)(?:[><=!].*)?['\"]", content)
                    dependencies[req_file] = set(pkg.lower() for pkg in packages if pkg and not pkg.startswith('#'))
                except Exception as e:
                    self.warnings['dependencies'].append(f"Could not parse {req_file}: {e}")
        # Check for core dependencies
        core_deps = {'numpy', 'scipy', 'matplotlib'}
        for dep in core_deps:
            found_in = [f for f, deps in dependencies.items() if dep in deps]
            if not found_in:
                self.issues['dependencies'].append(f"Core dependency '{dep}' not found in any requirements file")
        print(f"   ✅ Checked {len(req_files)} dependency files")
    def check_git_status(self):
        """Check git status and repository cleanliness."""
        print(f"{Colors.BOLD}🔄 Checking Git Status...{Colors.RESET}")
        try:
            # Check if there are uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd=self.repo_root)
            if result.stdout.strip():
                self.warnings['git'].append("Repository has uncommitted changes")
                uncommitted = result.stdout.strip().split('\n')
                for change in uncommitted[:10]:  # Show first 10
                    self.warnings['git'].append(f"  {change}")
                if len(uncommitted) > 10:
                    self.warnings['git'].append(f"  ... and {len(uncommitted)-10} more changes")
            # Check current branch
            result = subprocess.run(['git', 'branch', '--show-current'],
                                  capture_output=True, text=True, cwd=self.repo_root)
            current_branch = result.stdout.strip()
            if current_branch != 'main':
                self.warnings['git'].append(f"Currently on branch '{current_branch}', not 'main'")
            # Check if ahead/behind remote
            result = subprocess.run(['git', 'status', '-b', '--porcelain'],
                                  capture_output=True, text=True, cwd=self.repo_root)
            if 'ahead' in result.stdout or 'behind' in result.stdout:
                self.warnings['git'].append("Branch is ahead/behind remote")
        except subprocess.CalledProcessError as e:
            self.warnings['git'].append(f"Git command failed: {e}")
        except FileNotFoundError:
            self.warnings['git'].append("Git not found in PATH")
        print(f"   ✅ Checked git repository status")
    def check_file_structure(self):
        """Check expected file and directory structure."""
        print(f"{Colors.BOLD}📁 Checking File Structure...{Colors.RESET}")
        # Expected directories
        expected_dirs = [
            'Python',
            'MATLAB',
            'Mathematica',
            'examples',
            'notebooks',
            'tests',
            'docs',
            'scripts'
        ]
        for exp_dir in expected_dirs:
            dir_path = self.repo_root / exp_dir
            if not dir_path.exists():
                self.issues['structure'].append(f"Missing expected directory: {exp_dir}")
            elif not dir_path.is_dir():
                self.issues['structure'].append(f"{exp_dir} exists but is not a directory")
            else:
                self.stats['expected_dirs_present'] += 1
        # Check for __init__.py files in Python packages
        python_dirs = list((self.repo_root / 'Python').glob('**/'))
        missing_init = []
        for py_dir in python_dirs:
            if py_dir.name != '__pycache__' and not (py_dir / '__init__.py').exists():
                # Only flag if directory contains .py files
                if list(py_dir.glob('*.py')):
                    missing_init.append(str(py_dir.relative_to(self.repo_root)))
        if missing_init:
            self.issues['structure'].extend([f"Missing __init__.py in: {d}" for d in missing_init[:10]])
        print(f"   ✅ Checked repository structure and Python packages")
    def run_full_check(self):
        """Run all consistency checks."""
        self.print_header()
        # Run all checks
        checks = [
            self.check_naming_consistency,
            self.check_code_style,
            self.check_documentation_consistency,
            self.check_reference_consistency,
            self.check_dependencies,
            self.check_git_status,
            self.check_file_structure
        ]
        for check in checks:
            try:
                check()
            except Exception as e:
                self.issues['check_errors'].append(f"Error in {check.__name__}: {e}")
        # Print results
        self.print_results()
        # Return status code
        return len(self.issues) == 0
    def print_results(self):
        """Print comprehensive results."""
        print(f"\n{Colors.BOLD}📊 CONSISTENCY CHECK RESULTS{Colors.RESET}")
        print("=" * 80)
        # Print statistics
        if self.stats:
            print(f"\n{Colors.CALIFORNIA_GOLD}{Colors.BOLD}📈 Statistics:{Colors.RESET}")
            for key, value in self.stats.items():
                print(f"   {key.replace('_', ' ').title()}: {value}")
        # Print issues (errors)
        total_issues = sum(len(issues) for issues in self.issues.values())
        if total_issues > 0:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ ISSUES FOUND ({total_issues} total):{Colors.RESET}")
            for category, issue_list in self.issues.items():
                if issue_list:
                    print(f"\n{Colors.RED}{Colors.BOLD}{category.upper().replace('_', ' ')}:{Colors.RESET}")
                    for issue in issue_list:
                        print(f"   • {issue}")
        # Print warnings
        total_warnings = sum(len(warnings) for warnings in self.warnings.values())
        if total_warnings > 0:
            print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  WARNINGS ({total_warnings} total):{Colors.RESET}")
            for category, warning_list in self.warnings.items():
                if warning_list:
                    print(f"\n{Colors.YELLOW}{Colors.BOLD}{category.upper().replace('_', ' ')}:{Colors.RESET}")
                    for warning in warning_list[:10]:  # Limit warnings shown
                        print(f"   • {warning}")
                    if len(warning_list) > 10:
                        print(f"   • ... and {len(warning_list)-10} more warnings")
        # Final verdict
        print(f"\n{Colors.BOLD}🎯 FINAL VERDICT:{Colors.RESET}")
        if total_issues == 0 and total_warnings == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ REPOSITORY IS DEPLOYMENT READY! 🚀{Colors.RESET}")
            print(f"{Colors.BERKELEY_BLUE}🐻 Go Bears! Ready for production deployment! 🐻{Colors.RESET}")
        elif total_issues == 0:
            print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  REPOSITORY HAS WARNINGS BUT IS DEPLOYABLE{Colors.RESET}")
            print(f"   Consider addressing warnings before deployment")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ REPOSITORY NEEDS FIXES BEFORE DEPLOYMENT{Colors.RESET}")
            print(f"   Please address all issues before final deployment")
        print(f"\n{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}")
        print(f"{Colors.CALIFORNIA_GOLD}SciComp Consistency Check Complete{Colors.RESET}")
        print(f"{Colors.BERKELEY_BLUE}{'='*80}{Colors.RESET}\n")
def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='SciComp Repository Consistency Checker')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--fix-issues', action='store_true', help='Attempt to fix common issues')
    parser.add_argument('--export-report', help='Export detailed report to JSON file')
    args = parser.parse_args()
    checker = ConsistencyChecker(args.repo_root)
    success = checker.run_full_check()
    # Export report if requested
    if args.export_report:
        report = {
            'issues': dict(checker.issues),
            'warnings': dict(checker.warnings),
            'stats': dict(checker.stats),
            'timestamp': str(datetime.now()),
            'deployment_ready': success
        }
        with open(args.export_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report exported to: {args.export_report}")
    # Exit with appropriate code
    sys.exit(0 if success else 1)
if __name__ == "__main__":
    from datetime import datetime
    main()