"""Extract patterns from successful prompts"""
import re
from pathlib import Path
from typing import List, Dict
from collections import Counter

class Pattern:
    def __init__(self, name: str, structure: str, frequency: int, examples: List):
        self.name = name
        self.structure = structure
        self.frequency = frequency
        self.examples = examples

class PatternExtractor:
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent.parent / "knowledge" / "prompts"
        self.prompts_dir = Path(prompts_dir)
    
    def extract_all(self) -> List[Pattern]:
        patterns = []
        
        sections = self._extract_sections()
        if sections:
            patterns.append(Pattern("section-structure", "Organized sections", sum(sections.values()), list(sections.most_common(5))))
        
        code_blocks = self._extract_code_blocks()
        if code_blocks:
            patterns.append(Pattern("code-examples", "Includes code", len(code_blocks), list(set(code_blocks))[:5]))
        
        instructions = self._extract_instructions()
        if instructions:
            patterns.append(Pattern("instruction-style", "Imperative verbs", sum(instructions.values()), list(instructions.most_common(5))))
        
        contexts = self._extract_contexts()
        if contexts:
            patterns.append(Pattern("context-setting", "Domain context", len(contexts), list(set(contexts))[:5]))
        
        return patterns
    
    def _extract_sections(self) -> Counter:
        sections = Counter()
        for f in self.prompts_dir.glob("**/*.md"):
            content = f.read_text(encoding='utf-8')
            headers = re.findall(r'^#{1,3}\s+(.+)$', content, re.MULTILINE)
            sections.update(headers)
        return sections
    
    def _extract_code_blocks(self) -> List[str]:
        blocks = []
        for f in self.prompts_dir.glob("**/*.md"):
            content = f.read_text(encoding='utf-8')
            code = re.findall(r'```(\w+)?', content)
            blocks.extend(code)
        return blocks
    
    def _extract_instructions(self) -> Counter:
        instructions = Counter()
        verbs = ['create', 'build', 'implement', 'write', 'generate', 'analyze', 'review', 'test', 'optimize', 'refactor']
        for f in self.prompts_dir.glob("**/*.md"):
            content = f.read_text(encoding='utf-8')
            for verb in verbs:
                if re.search(rf'\b{verb}\b', content, re.IGNORECASE):
                    instructions[verb] += 1
        return instructions
    
    def _extract_contexts(self) -> List[str]:
        contexts = []
        markers = ['background:', 'context:', 'domain:', 'scenario:', 'given:']
        for f in self.prompts_dir.glob("**/*.md"):
            content = f.read_text(encoding='utf-8')
            for marker in markers:
                if marker in content.lower():
                    contexts.append(marker.rstrip(':'))
        return contexts
    
    def generate_template(self, pattern_name: str) -> str:
        templates = {
            "section-structure": "# {Title}\n\n## Purpose\n{Description}\n\n## Instructions\n{Steps}\n\n## Examples\n{Code}\n",
            "code-examples": "# {Title}\n\n{Description}\n\n```{language}\n{code}\n```\n",
            "instruction-style": "# {Title}\n\n**Task:** {Instruction}\n\n**Requirements:**\n- {Item}\n",
            "context-setting": "# {Title}\n\n**Context:** {Domain}\n\n**Goal:** {Objective}\n"
        }
        return templates.get(pattern_name, "")

if __name__ == "__main__":
    extractor = PatternExtractor()
    print(f"Looking in: {extractor.prompts_dir}")
    print(f"Exists: {extractor.prompts_dir.exists()}")
    if extractor.prompts_dir.exists():
        files = list(extractor.prompts_dir.glob("**/*.md"))
        print(f"Found {len(files)} markdown files")
    patterns = extractor.extract_all()
    print(f"\n[PATTERNS] Extracted {len(patterns)} patterns\n")
    for p in patterns:
        print(f"  {p.name}: {p.frequency} occurrences")
