"""Install prompts from marketplace"""
from pathlib import Path
import shutil
from typing import Dict
from registry import PromptRegistry

class PromptInstaller:
    def __init__(self, install_dir: str = None):
        if install_dir is None:
            install_dir = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge" / "prompts"
        self.install_dir = Path(install_dir)
        self.registry = PromptRegistry()
    
    def install(self, prompt_id: str, source_path: str = None) -> bool:
        """Install a prompt"""
        prompt = self.registry.get(prompt_id)
        if not prompt:
            return False
        
        # Create category directory
        category = prompt.get('category', 'community')
        target_dir = self.install_dir / category
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy prompt file
        if source_path:
            source = Path(source_path)
            target = target_dir / f"{prompt['name']}.md"
            shutil.copy2(source, target)
        
        # Track download
        self.registry.download(prompt_id)
        
        return True
    
    def uninstall(self, prompt_id: str) -> bool:
        """Uninstall a prompt"""
        prompt = self.registry.get(prompt_id)
        if not prompt:
            return False
        
        category = prompt.get('category', 'community')
        target = self.install_dir / category / f"{prompt['name']}.md"
        
        if target.exists():
            target.unlink()
            return True
        
        return False
    
    def list_installed(self) -> list:
        """List installed prompts"""
        installed = []
        for prompt_file in self.install_dir.rglob("*.md"):
            if prompt_file.name != "README.md":
                installed.append({
                    'name': prompt_file.stem,
                    'path': str(prompt_file.relative_to(self.install_dir)),
                    'category': prompt_file.parent.name
                })
        return installed

if __name__ == "__main__":
    installer = PromptInstaller()
    installed = installer.list_installed()
    print(f"\n[INSTALLED] {len(installed)} prompts")
    for p in installed[:5]:
        print(f"  {p['name']} ({p['category']})")
