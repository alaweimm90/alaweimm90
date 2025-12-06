"""Regression testing for prompts"""
from pathlib import Path
from typing import Dict, List
import json
import hashlib

class RegressionTester:
    def __init__(self, baseline_path: str = "baseline.json"):
        self.baseline_path = Path(baseline_path)
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict:
        """Load baseline results"""
        if self.baseline_path.exists():
            return json.loads(self.baseline_path.read_text())
        return {}
    
    def save_baseline(self, results: Dict):
        """Save baseline results"""
        self.baseline_path.write_text(json.dumps(results, indent=2))
        self.baseline = results
    
    def check_regression(self, prompt_name: str, current_score: float) -> Dict:
        """Check for regression"""
        baseline_score = self.baseline.get(prompt_name, {}).get('score', 0)
        
        regression = current_score < baseline_score
        delta = current_score - baseline_score
        
        return {
            'prompt': prompt_name,
            'baseline': baseline_score,
            'current': current_score,
            'delta': delta,
            'regression': regression
        }
    
    def compute_hash(self, prompt_path: Path) -> str:
        """Compute hash of prompt content"""
        content = prompt_path.read_text(encoding='utf-8')
        return hashlib.md5(content.encode()).hexdigest()
    
    def detect_changes(self, prompts_dir: Path) -> List[Dict]:
        """Detect changed prompts"""
        changes = []
        
        for prompt_file in prompts_dir.rglob("*.md"):
            if prompt_file.name == "README.md":
                continue
            
            current_hash = self.compute_hash(prompt_file)
            baseline_hash = self.baseline.get(prompt_file.stem, {}).get('hash')
            
            if baseline_hash and current_hash != baseline_hash:
                changes.append({
                    'prompt': prompt_file.stem,
                    'path': str(prompt_file),
                    'changed': True
                })
        
        return changes

if __name__ == "__main__":
    tester = RegressionTester()
    
    # Check regression
    result = tester.check_regression("code-review", 0.85)
    print(f"\n[REGRESSION] {result['prompt']}")
    print(f"  Baseline: {result['baseline']:.2f}")
    print(f"  Current: {result['current']:.2f}")
    print(f"  Delta: {result['delta']:+.2f}")
    print(f"  Regression: {result['regression']}")
