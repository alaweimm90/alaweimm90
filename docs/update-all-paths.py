"""Auto-update all paths after consolidation"""
from pathlib import Path
import re

def update_file(file_path, old_pattern, new_pattern):
    """Update paths in a file"""
    try:
        content = Path(file_path).read_text(encoding='utf-8')
        updated = content.replace(old_pattern, new_pattern)
        if content != updated:
            Path(file_path).write_text(updated, encoding='utf-8')
            print(f"[OK] Updated: {file_path}")
            return True
    except Exception as e:
        print(f"[ERROR] {file_path}: {e}")
    return False

def main():
    print("\n[UPDATE] Updating all paths after consolidation...\n")
    
    updates = [
        # Sync tool
        (".ai-system/tools/cross-ide-sync/config.py",
         'Path(__file__).parent.parent.parent / "docs" / "ai-knowledge"',
         'Path(__file__).parent.parent / "knowledge"'),
        
        # Recommendation engine
        (".ai-system/tools/recommendation-engine/recommender.py",
         'Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"',
         'Path(__file__).parent.parent / "knowledge" / "prompts"'),
        
        # Pattern extractor
        (".ai-system/tools/pattern-extractor/extractor.py",
         'Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"',
         'Path(__file__).parent.parent / "knowledge" / "prompts"'),
        
        # Prompt composer
        (".ai-system/tools/prompt-composer/composer.py",
         'Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"',
         'Path(__file__).parent.parent / "knowledge" / "prompts"'),
        
        # Validator
        (".ai-system/tools/prompt-testing/validator.py",
         'Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"',
         'Path(__file__).parent.parent / "knowledge" / "prompts"'),
    ]
    
    updated_count = 0
    for file_path, old, new in updates:
        if update_file(file_path, old, new):
            updated_count += 1
    
    print(f"\n[DONE] Updated {updated_count} files")
    print("\nNext: Test the system:")
    print("  cd .ai-system\\tools\\cross-ide-sync")
    print("  python cli.py sync")

if __name__ == "__main__":
    main()
