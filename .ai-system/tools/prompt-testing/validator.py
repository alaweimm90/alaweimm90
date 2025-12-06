"""Validate prompt quality"""
from pathlib import Path
from typing import Dict, List
import re

class PromptValidator:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
        self.prompts_dir = Path(prompts_dir)
    
    def validate(self, prompt_path: str) -> Dict:
        """Validate a prompt file"""
        path = self.prompts_dir / prompt_path if not Path(prompt_path).is_absolute() else Path(prompt_path)
        content = path.read_text(encoding='utf-8')
        
        results = {
            'path': str(path),
            'valid': True,
            'score': 0.0,
            'issues': [],
            'metrics': {}
        }
        
        # Check structure
        has_title = bool(re.search(r'^#\s+.+', content, re.MULTILINE))
        has_sections = len(re.findall(r'^#{2,3}\s+.+', content, re.MULTILINE)) >= 2
        has_code = bool(re.search(r'```', content))
        has_examples = 'example' in content.lower() or 'usage' in content.lower()
        
        # Calculate score
        score = 0
        if has_title: score += 0.25
        if has_sections: score += 0.25
        if has_code: score += 0.25
        if has_examples: score += 0.25
        
        results['score'] = score
        results['metrics'] = {
            'has_title': has_title,
            'has_sections': has_sections,
            'has_code': has_code,
            'has_examples': has_examples,
            'length': len(content),
            'lines': content.count('\n')
        }
        
        # Check for issues
        if not has_title:
            results['issues'].append('Missing title (# heading)')
        if not has_sections:
            results['issues'].append('Needs more sections (## headings)')
        if not has_code:
            results['issues'].append('No code examples')
        if len(content) < 100:
            results['issues'].append('Content too short')
        
        results['valid'] = score >= 0.5
        
        return results
    
    def validate_all(self) -> List[Dict]:
        """Validate all prompts"""
        results = []
        for prompt_file in self.prompts_dir.rglob("*.md"):
            if prompt_file.name != "README.md":
                result = self.validate(str(prompt_file))
                results.append(result)
        return results

if __name__ == "__main__":
    validator = PromptValidator()
    
    # Test single prompt
    result = validator.validate("superprompts/optimization-framework.md")
    print(f"\n[VALIDATE] {Path(result['path']).name}")
    print(f"  Score: {result['score']:.2f}")
    print(f"  Valid: {result['valid']}")
    if result['issues']:
        print(f"  Issues: {', '.join(result['issues'])}")
