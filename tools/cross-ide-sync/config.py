"""IDE configuration paths"""
from pathlib import Path
from typing import Dict, List

class IDEConfig:
    def __init__(self):
        self.home = Path.home()
        self.source = Path(__file__).parent.parent.parent / "docs" / "ai-knowledge"
    
    def get_ide_paths(self) -> Dict[str, List[Path]]:
        """Get prompt paths for each IDE"""
        return {
            "amazonq": [self.home / ".aws" / "amazonq" / "prompts"],
            "claude-desktop": [self.home / "Library" / "Application Support" / "Claude" / "prompts"] if self._is_mac() else [self.home / "AppData" / "Roaming" / "Claude" / "prompts"],
            "windsurf": [self.home / ".windsurf" / "prompts"],
            "cline": [self.home / ".cline" / "prompts"],
            "cursor": [self.home / ".cursor" / "prompts"],
        }
    
    def _is_mac(self) -> bool:
        import platform
        return platform.system() == "Darwin"

if __name__ == "__main__":
    config = IDEConfig()
    paths = config.get_ide_paths()
    for ide, path_list in paths.items():
        print(f"{ide}: {path_list[0]}")
