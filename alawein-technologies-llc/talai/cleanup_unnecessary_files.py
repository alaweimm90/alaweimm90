#!/usr/bin/env python3
"""
Cleanup Unnecessary Files in .meta Directory

Modernized cleanup script with type safety, dataclasses, and enterprise features.
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Set, Optional, Tuple


class CleanupAction(Enum):
    """Types of cleanup actions."""
    REMOVE_FILE = "remove_file"
    REMOVE_DIRECTORY = "remove_directory"
    SKIP = "skip"


@dataclass
class CleanupItem:
    """Represents an item to be cleaned up."""
    path: Path
    action: CleanupAction
    reason: str
    size_bytes: int = 0
    is_safe: bool = True


@dataclass
class CleanupRules:
    """Configuration for cleanup rules."""
    files_to_remove: Set[str] = field(default_factory=lambda: {
        'ULTRATHINK_BASELINE.txt',
        'ULTRATHINK_COMPLETION_SUMMARY.md', 
        'ULTRATHINK_FINALIZATION_COMPLETE.md',
        'ULTRATHINK_KNOWLEDGE.zip',
        'questions_log.json',
        'self_refutation_checkpoint.json',
        'phase2_validation.json',
        'verification_report_*.json',
        'ENTERPRISE_TOOLS_COMPLETE.md',
        'ENTERPRISE_TRANSFORMATION_COMPLETE.md',
        'DIRECTORY_STRATEGY.md',
        'PROJECT_FOLDERS_STRATEGY.md',
    })
    
    dirs_to_check: Set[str] = field(default_factory=lambda: {
        'ai', 'augment-cli', 'components', 'directory_management',
        'knowledge', 'mcp', 'scripts/generated', 'styles/templates'
    })
    
    protected_files: Set[str] = field(default_factory=lambda: {
        'README.md', 'governance', 'tools', 'scripts'
    })
    
    max_file_size_mb: int = 100


class MetaDirectoryCleaner:
    """Cleans up unnecessary files in .meta directory."""

    def __init__(self, meta_dir: Optional[Path] = None, rules: Optional[CleanupRules] = None) -> None:
        self.meta_dir = meta_dir or Path('.meta')
        self.rules = rules or CleanupRules()
        self.cleanup_items: List[CleanupItem] = []
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def scan_for_cleanup(self) -> bool:
        """Scan for files and directories that can be cleaned up."""
        logging.info("🔍 Scanning .meta directory for cleanup opportunities...")
        self.cleanup_items.clear()
        
        if not self.meta_dir.exists():
            logging.warning(f"Meta directory {self.meta_dir} does not exist")
            return False
        
        try:
            self._scan_files()
            self._scan_directories()
            self._validate_cleanup_safety()
        except Exception as e:
            logging.error(f"Error during scan: {e}")
            return False
        
        return len(self.cleanup_items) > 0
    
    def _scan_files(self) -> None:
        """Scan for files to remove."""
        import fnmatch
        
        try:
            for item in self.meta_dir.rglob('*'):
                if not item.is_file():
                    continue
                    
                for pattern in self.rules.files_to_remove:
                    if fnmatch.fnmatch(item.name, pattern) or item.name == pattern:
                        if self._is_safe_to_remove(item):
                            size_bytes = item.stat().st_size
                            self.cleanup_items.append(CleanupItem(
                                path=item,
                                action=CleanupAction.REMOVE_FILE,
                                reason=f"Matches cleanup pattern: {pattern}",
                                size_bytes=size_bytes,
                                is_safe=True
                            ))
                        break
        except Exception as e:
            logging.warning(f"Error scanning files: {e}")
    
    def _scan_directories(self) -> None:
        """Scan for empty directories to remove."""
        for dirname in self.rules.dirs_to_check:
            dir_path = self.meta_dir / dirname
            
            if dir_path.exists() and dir_path.is_dir():
                if self._is_empty_directory(dir_path):
                    self.cleanup_items.append(CleanupItem(
                        path=dir_path,
                        action=CleanupAction.REMOVE_DIRECTORY,
                        reason="Empty directory",
                        is_safe=True
                    ))
    
    def _is_safe_to_remove(self, file_path: Path) -> bool:
        """Check if file is safe to remove."""
        if file_path.name in self.rules.protected_files:
            return False
            
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.rules.max_file_size_mb:
                logging.warning(f"File {file_path.name} is {size_mb:.1f}MB, skipping")
                return False
        except OSError:
            return False
            
        return True
    
    def _validate_cleanup_safety(self) -> None:
        """Validate that cleanup operations are safe."""
        for item in self.cleanup_items:
            if item.path.suffix in {'.py', '.md'} and 'governance' in str(item.path):
                item.is_safe = False
                item.reason += " (governance file - manual review required)"

    def _is_empty_directory(self, dir_path: Path) -> bool:
        """Check if directory is empty."""
        try:
            return not any(dir_path.iterdir())
        except (PermissionError, OSError):
            return False

    def execute_cleanup(self, dry_run: bool = False) -> Tuple[int, int]:
        """Execute cleanup operations."""
        logging.info(f"🧹 {'DRY RUN: ' if dry_run else ''}Executing cleanup operations...")
        
        successful = 0
        failed = 0
        
        safe_items = [item for item in self.cleanup_items if item.is_safe]
        unsafe_items = [item for item in self.cleanup_items if not item.is_safe]
        
        if unsafe_items:
            logging.warning(f"Skipping {len(unsafe_items)} unsafe operations")
        
        for item in safe_items:
            try:
                if self._execute_cleanup_item(item, dry_run):
                    successful += 1
                    action_verb = "Would remove" if dry_run else "Removed"
                    logging.info(f"✅ {action_verb}: {item.path.relative_to(self.meta_dir)}")
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logging.error(f"❌ Failed to process {item.path}: {e}")
        
        return successful, failed
    
    def _execute_cleanup_item(self, item: CleanupItem, dry_run: bool) -> bool:
        """Execute cleanup for a single item."""
        if dry_run:
            return True
            
        try:
            if item.action == CleanupAction.REMOVE_FILE:
                item.path.unlink()
                return True
            elif item.action == CleanupAction.REMOVE_DIRECTORY:
                item.path.rmdir()
                return True
        except OSError as e:
            logging.error(f"OS error removing {item.path}: {e}")
            return False
            
        return False

    def generate_report(self, successful: int, failed: int, dry_run: bool = False) -> None:
        """Generate cleanup report."""
        print("\n📊 CLEANUP SUMMARY")
        print("=" * 50)
        
        total_items = len(self.cleanup_items)
        safe_items = len([item for item in self.cleanup_items if item.is_safe])
        
        print(f"Total items found: {total_items}")
        print(f"Safe operations: {safe_items}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        files = [item for item in self.cleanup_items if item.action == CleanupAction.REMOVE_FILE]
        dirs = [item for item in self.cleanup_items if item.action == CleanupAction.REMOVE_DIRECTORY]
        
        if files:
            print(f"\n📄 Files ({len(files)}):")
            total_size = sum(item.size_bytes for item in files)
            print(f"  Total size: {self._format_bytes(total_size)}")
            for item in files[:10]:
                status = "✅" if item.is_safe else "⚠️"
                size_str = f" ({self._format_bytes(item.size_bytes)})" if item.size_bytes > 0 else ""
                print(f"  {status} {item.path.name}{size_str}")
        
        if dirs:
            print(f"\n📁 Directories ({len(dirs)}):")
            for item in dirs:
                status = "✅" if item.is_safe else "⚠️"
                print(f"  {status} {item.path.name}/")
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.1f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.1f} TB"

    def run(self, dry_run: bool = False) -> int:
        """Main cleanup execution."""
        print("🧹 .meta Directory Cleanup Tool")
        print("=" * 50)
        
        try:
            items_found = self.scan_for_cleanup()
            
            if not items_found:
                print("✅ No cleanup needed - directory is already clean!")
                return 0
            
            successful, failed = self.execute_cleanup(dry_run=dry_run)
            self.generate_report(successful, failed, dry_run=dry_run)
            
            if dry_run and successful > 0:
                print(f"\n💡 Run without --dry-run to apply {successful} cleanup operations")
            
            return 1 if failed > 0 else 0
            
        except KeyboardInterrupt:
            logging.info("\nCleanup cancelled by user")
            return 130
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return 1


def main() -> None:
    """Main entry point for the cleanup tool."""
    parser = argparse.ArgumentParser(description="Clean up unnecessary files in .meta directory")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    parser.add_argument('--meta-dir', type=Path, help='Path to .meta directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        cleaner = MetaDirectoryCleaner(meta_dir=args.meta_dir)
        exit_code = cleaner.run(dry_run=args.dry_run)
        sys.exit(exit_code)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
