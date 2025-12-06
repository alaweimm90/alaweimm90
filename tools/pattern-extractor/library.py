"""Pattern library management"""
import json
from pathlib import Path
from typing import List, Dict

class PatternLibrary:
    def __init__(self, library_path: str = "tools/pattern-extractor/patterns.json"):
        self.library_path = Path(library_path)
        self.patterns = self._load()
    
    def _load(self) -> Dict:
        if self.library_path.exists():
            return json.loads(self.library_path.read_text())
        return {}
    
    def save(self, patterns: List) -> None:
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        data = {p.name: {"structure": p.structure, "frequency": p.frequency, "examples": p.examples} for p in patterns}
        self.library_path.write_text(json.dumps(data, indent=2))
        self.patterns = data
    
    def get(self, pattern_name: str) -> Dict:
        return self.patterns.get(pattern_name)
    
    def search(self, query: str) -> List[str]:
        return [name for name in self.patterns.keys() if query.lower() in name.lower()]
    
    def list_all(self) -> List[str]:
        return list(self.patterns.keys())

if __name__ == "__main__":
    from extractor import PatternExtractor
    
    extractor = PatternExtractor()
    patterns = extractor.extract_all()
    
    library = PatternLibrary()
    library.save(patterns)
    
    print(f"[OK] Saved {len(patterns)} patterns to library")
