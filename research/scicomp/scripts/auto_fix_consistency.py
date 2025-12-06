#!/usr/bin/env python3
"""
SciComp Automated Consistency Fixes
Script to automatically fix common consistency issues found by the consistency checker.
Only fixes safe, non-destructive changes. More complex issues require manual review.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
"""
import os
import re
import sys
from pathlib import Path
from typing import List, Dict
import shutil
from datetime import datetime
class AutoFixer:
    """Automated consistency fixer."""
    def __init__(self, repo_root: str = ".", dry_run: bool = True):
        self.repo_root = Path(repo_root)
        self.dry_run = dry_run
        self.fixes_applied = []
        self.backup_dir = self.repo_root / f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Safe replacement patterns
        self.safe_replacements = [
            # Old project names -> correct name
            (r'SciComp', 'SciComp'),
            (r'SciComp', 'SciComp'),
            (r'UC SciComp', 'SciComp'),
            # Old taglines -> correct tagline
            (r'A Cross-Platform Scientific Computing Suite for Research and Education',
             'A Cross-Platform Scientific Computing Suite for Research and Education'),
            (r'Crafted with love, 🐻 energy, and zero sleep.',
             'Crafted with love, 🐻 energy, and zero sleep.'),
            # Common spacing/formatting issues
            (r'\s+$', ''),  # Remove trailing whitespace
            (r'\n{3,}', '\n\n'),  # Replace multiple newlines with double newline
        ]
        # File extensions to process
        self.text_extensions = {'.py', '.m', '.md', '.txt', '.yml', '.yaml', '.toml', '.json'}
    def create_backup(self, file_path: Path):
        """Create backup of file before modification."""
        if not self.backup_dir.exists():
            self.backup_dir.mkdir()
        relative_path = file_path.relative_to(self.repo_root)
        backup_file = self.backup_dir / relative_path
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, backup_file)
    def fix_file_naming(self):
        """Fix naming consistency issues in files."""
        print("🔧 Fixing naming consistency...")
        # Find all text files
        text_files = []
        for ext in self.text_extensions:
            text_files.extend(self.repo_root.glob(f"**/*{ext}"))
        files_modified = 0
        for file_path in text_files:
            if '.git' in str(file_path) or '__pycache__' in str(file_path):
                continue
            try:
                original_content = file_path.read_text(encoding='utf-8', errors='ignore')
                modified_content = original_content
                changes_made = []
                # Apply safe replacements
                for pattern, replacement in self.safe_replacements:
                    new_content = re.sub(pattern, replacement, modified_content, flags=re.MULTILINE)
                    if new_content != modified_content:
                        changes_made.append(f"Replaced '{pattern}' with '{replacement}'")
                        modified_content = new_content
                # If changes were made
                if modified_content != original_content:
                    if not self.dry_run:
                        self.create_backup(file_path)
                        file_path.write_text(modified_content, encoding='utf-8')
                    files_modified += 1
                    self.fixes_applied.append({
                        'file': str(file_path.relative_to(self.repo_root)),
                        'changes': changes_made
                    })
            except Exception as e:
                print(f"   Warning: Could not process {file_path}: {e}")
        action = "Would modify" if self.dry_run else "Modified"
        print(f"   {action} {files_modified} files for naming consistency")
    def fix_missing_init_files(self):
        """Add missing __init__.py files to Python packages."""
        print("🔧 Adding missing __init__.py files...")
        python_dirs = []
        if (self.repo_root / 'Python').exists():
            python_dirs = list((self.repo_root / 'Python').glob('**/'))
        init_files_added = 0
        for py_dir in python_dirs:
            if py_dir.name == '__pycache__':
                continue
            # Check if directory contains .py files but no __init__.py
            py_files = list(py_dir.glob('*.py'))
            init_file = py_dir / '__init__.py'
            if py_files and not init_file.exists():
                # Create appropriate __init__.py content
                module_name = py_dir.name
                init_content = f'"""{module_name} - SciComp module."""\n'
                if not self.dry_run:
                    init_file.write_text(init_content, encoding='utf-8')
                init_files_added += 1
                self.fixes_applied.append({
                    'file': str(init_file.relative_to(self.repo_root)),
                    'changes': ['Added missing __init__.py file']
                })
        action = "Would add" if self.dry_run else "Added"
        print(f"   {action} {init_files_added} missing __init__.py files")
    def fix_readme_format(self):
        """Fix common README.md formatting issues."""
        print("🔧 Fixing README formatting...")
        readme_path = self.repo_root / 'README.md'
        if not readme_path.exists():
            return
        try:
            content = readme_path.read_text(encoding='utf-8', errors='ignore')
            original_content = content
            changes = []
            # Ensure proper title format
            if not content.startswith('# SciComp:'):
                # Fix title if it's close
                title_pattern = r'^#\s*[^\n]*(?:SciComp|scicomp)[^\n]*'
                if re.search(title_pattern, content, re.IGNORECASE):
                    content = re.sub(title_pattern, '# SciComp: A Cross-Platform Scientific Computing Suite for Research and Education', content, count=1)
                    changes.append('Fixed README title format')
            # Ensure correct tagline at end
            correct_tagline = '*Crafted with love, 🐻 energy, and zero sleep.*'
            if not content.strip().endswith(correct_tagline):
                # Remove incorrect taglines and add correct one
                wrong_taglines = [
                    '*Crafted with love, 🐻 energy, and zero sleep.*',
                    '*Made with love, 🐻 energy, and zero sleep.*'  # Close but wrong
                ]
                for wrong in wrong_taglines:
                    if wrong in content:
                        content = content.replace(wrong, '')
                        changes.append(f'Removed incorrect tagline: {wrong}')
                # Add correct tagline
                if not content.strip().endswith(correct_tagline):
                    content = content.rstrip() + '\n\n' + correct_tagline + '\n'
                    changes.append('Added correct tagline at end')
            # Fix badge consistency (ensure they're on separate lines or proper format)
            badge_pattern = r'(\[!\[.*?\]\(.*?\)\]\(.*?\))'
            badges = re.findall(badge_pattern, content)
            if len(badges) > 3:  # If multiple badges, ensure they're formatted consistently
                # This is a complex fix, so we'll just note it
                changes.append('Note: Check badge formatting manually')
            if content != original_content:
                if not self.dry_run:
                    self.create_backup(readme_path)
                    readme_path.write_text(content, encoding='utf-8')
                self.fixes_applied.append({
                    'file': 'README.md',
                    'changes': changes
                })
        except Exception as e:
            print(f"   Warning: Could not fix README: {e}")
        action = "Would fix" if self.dry_run else "Fixed"
        if any('README.md' in fix['file'] for fix in self.fixes_applied):
            print(f"   {action} README.md formatting")
        else:
            print("   README.md formatting appears correct")
    def fix_citation_format(self):
        """Fix CITATION.cff format."""
        print("🔧 Checking CITATION.cff format...")
        citation_path = self.repo_root / 'CITATION.cff'
        if not citation_path.exists():
            print("   CITATION.cff not found - this should be created manually")
            return
        try:
            content = citation_path.read_text(encoding='utf-8', errors='ignore')
            # Basic validation
            required_fields = ['title:', 'authors:', 'date-released:', 'version:']
            missing_fields = []
            for field in required_fields:
                if field not in content:
                    missing_fields.append(field)
            if missing_fields:
                print(f"   Warning: CITATION.cff missing fields: {', '.join(missing_fields)}")
            else:
                print("   CITATION.cff format appears correct")
        except Exception as e:
            print(f"   Warning: Could not check CITATION.cff: {e}")
    def run_fixes(self):
        """Run all automated fixes."""
        print("🔧 SciComp Automated Consistency Fixes")
        print("=" * 50)
        if self.dry_run:
            print("🔍 DRY RUN MODE - No files will be modified")
        else:
            print("⚠️  LIVE MODE - Files will be modified")
        print()
        # Run all fixes
        fixes = [
            self.fix_file_naming,
            self.fix_missing_init_files,
            self.fix_readme_format,
            self.fix_citation_format
        ]
        for fix in fixes:
            try:
                fix()
            except Exception as e:
                print(f"   Error in {fix.__name__}: {e}")
        # Summary
        print(f"\n📊 Summary:")
        total_fixes = len(self.fixes_applied)
        if total_fixes > 0:
            print(f"   {total_fixes} fixes applied:")
            for fix in self.fixes_applied:
                print(f"   • {fix['file']}: {', '.join(fix['changes'])}")
            if self.backup_dir.exists() and not self.dry_run:
                print(f"\n💾 Backups stored in: {self.backup_dir}")
        else:
            print("   No fixes needed - repository appears consistent!")
        if self.dry_run:
            print("\n🔄 Run with --apply to actually make changes")
        return total_fixes
def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='SciComp Automated Consistency Fixes')
    parser.add_argument('--repo-root', default='.', help='Repository root directory')
    parser.add_argument('--apply', action='store_true', help='Apply fixes (default is dry-run)')
    parser.add_argument('--backup-dir', help='Custom backup directory')
    args = parser.parse_args()
    fixer = AutoFixer(args.repo_root, dry_run=not args.apply)
    if args.backup_dir:
        fixer.backup_dir = Path(args.backup_dir)
    fixes_applied = fixer.run_fixes()
    # Exit with success if fixes were applied or none were needed
    sys.exit(0)
if __name__ == "__main__":
    main()