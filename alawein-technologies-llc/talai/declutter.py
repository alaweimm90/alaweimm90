#!/usr/bin/env python3
"""Repository decluttering script - removes redundant files and optimizes structure"""

import shutil
from pathlib import Path
from typing import List

class RepositoryDeclutterer:
    def __init__(self, root_path: Path = None):
        self.root_path = root_path or Path.cwd()
        self.removed_files = []
        self.removed_dirs = []

    def declutter_all(self) -> None:
        """Run all decluttering operations"""
        print("🧹 Starting repository decluttering...")

        self._remove_duplicate_templates()
        self._remove_empty_directories()
        self._consolidate_configs()
        self._clean_cache_files()

        self._generate_report()

    def _remove_duplicate_templates(self) -> None:
        """Remove duplicate template files"""
        print("\n📄 Removing duplicate templates...")

        # Remove organization-specific README templates that are now redundant
        template_patterns = [
            "ORGANIZATIONS/*/README.template.md",
            "ORGANIZATIONS/*/SECURITY.template.md"
        ]

        for pattern in template_patterns:
            for file_path in self.root_path.glob(pattern):
                if file_path.exists():
                    file_path.unlink()
                    self.removed_files.append(str(file_path.relative_to(self.root_path)))
                    print(f"  Removed: {file_path.relative_to(self.root_path)}")

    def _remove_empty_directories(self) -> None:
        """Remove empty directories"""
        print("\n📁 Removing empty directories...")

        def remove_empty_dirs(path: Path) -> None:
            if not path.is_dir():
                return

            # Remove empty subdirectories first
            for subdir in path.iterdir():
                if subdir.is_dir():
                    remove_empty_dirs(subdir)

            # Remove this directory if it's empty
            try:
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    self.removed_dirs.append(str(path.relative_to(self.root_path)))
                    print(f"  Removed empty dir: {path.relative_to(self.root_path)}")
            except OSError:
                pass  # Directory not empty or permission issue

        # Check common areas for empty directories
        check_paths = [
            self.root_path / ".meta",
            self.root_path / "ORGANIZATIONS",
            self.root_path / "docs",
            self.root_path / "examples"
        ]

        for check_path in check_paths:
            if check_path.exists():
                remove_empty_dirs(check_path)

    def _consolidate_configs(self) -> None:
        """Consolidate configuration files"""
        print("\n⚙️ Consolidating configurations...")

        # Remove redundant pre-commit configs in favor of unified system
        redundant_configs = [
            ".meta/config/user/governance/.pre-commit-config.yaml",
            "tools/dev/governance/.pre-commit-config.yaml"
        ]

        for config_path in redundant_configs:
            file_path = self.root_path / config_path
            if file_path.exists():
                file_path.unlink()
                self.removed_files.append(config_path)
                print(f"  Removed redundant config: {config_path}")

    def _clean_cache_files(self) -> None:
        """Clean cache and temporary files"""
        print("\n🗑️ Cleaning cache files...")

        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/node_modules",
            "**/.DS_Store",
            "**/Thumbs.db"
        ]

        for pattern in cache_patterns:
            for item in self.root_path.glob(pattern):
                try:
                    if item.is_file():
                        item.unlink()
                        self.removed_files.append(str(item.relative_to(self.root_path)))
                    elif item.is_dir():
                        shutil.rmtree(item)
                        self.removed_dirs.append(str(item.relative_to(self.root_path)))
                    print(f"  Cleaned: {item.relative_to(self.root_path)}")
                except (OSError, PermissionError):
                    pass  # Skip files that can't be removed

    def _generate_report(self) -> None:
        """Generate decluttering report"""
        print("\n" + "="*50)
        print("DECLUTTERING REPORT")
        print("="*50)

        print(f"Files removed: {len(self.removed_files)}")
        print(f"Directories removed: {len(self.removed_dirs)}")

        if self.removed_files:
            print("\nRemoved files:")
            for file_path in self.removed_files[:10]:  # Show first 10
                print(f"  - {file_path}")
            if len(self.removed_files) > 10:
                print(f"  ... and {len(self.removed_files) - 10} more")

        if self.removed_dirs:
            print("\nRemoved directories:")
            for dir_path in self.removed_dirs:
                print(f"  - {dir_path}")

        print(f"\n✅ Repository decluttered successfully!")

def main():
    """Main entry point"""
    declutterer = RepositoryDeclutterer()
    declutterer.declutter_all()

if __name__ == "__main__":
    main()
