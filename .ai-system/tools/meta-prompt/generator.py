#!/usr/bin/env python3
"""Meta-Prompt Generator - Generate prompts from requirements."""

import re
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class PromptSpec:
    title: str
    purpose: str
    domain: str
    use_cases: List[str]
    examples: List[Dict[str, str]]

class PromptGenerator:
    """Generate prompts from natural language requirements."""
    
    def __init__(self, templates_dir: Path):
        self.templates_dir = templates_dir
        self.templates = self._load_templates()
    
    def generate(self, requirement: str) -> str:
        """Generate prompt from requirement."""
        spec = self._parse_requirement(requirement)
        template = self._select_template(spec)
        prompt = self._fill_template(template, spec)
        return self._validate(prompt)
    
    def _parse_requirement(self, requirement: str) -> PromptSpec:
        """Parse natural language requirement into structured spec."""
        # Extract key information
        title = self._extract_title(requirement)
        purpose = self._extract_purpose(requirement)
        domain = self._extract_domain(requirement)
        use_cases = self._extract_use_cases(requirement)
        examples = self._generate_examples(requirement)
        
        return PromptSpec(
            title=title,
            purpose=purpose,
            domain=domain,
            use_cases=use_cases,
            examples=examples
        )
    
    def _extract_title(self, req: str) -> str:
        """Extract title from requirement."""
        # Look for action verbs
        actions = ['optimize', 'refactor', 'design', 'implement', 'test', 'debug']
        for action in actions:
            if action in req.lower():
                # Extract noun phrase after action
                match = re.search(f'{action}\\s+([\\w\\s]+)', req.lower())
                if match:
                    return f"{action.title()} {match.group(1).strip().title()}"
        
        # Fallback: use first few words
        words = req.split()[:5]
        return ' '.join(words).title()
    
    def _extract_purpose(self, req: str) -> str:
        """Extract purpose from requirement."""
        # Clean up requirement
        purpose = req.strip()
        if not purpose.endswith('.'):
            purpose += '.'
        return purpose
    
    def _extract_domain(self, req: str) -> str:
        """Extract domain from requirement."""
        domains = {
            'database': ['database', 'sql', 'query'],
            'optimization': ['optimize', 'performance', 'speed'],
            'testing': ['test', 'qa', 'quality'],
            'architecture': ['design', 'architecture', 'system'],
            'ai': ['ai', 'ml', 'machine learning'],
        }
        
        req_lower = req.lower()
        for domain, keywords in domains.items():
            if any(kw in req_lower for kw in keywords):
                return domain
        
        return 'general'
    
    def _extract_use_cases(self, req: str) -> List[str]:
        """Extract use cases from requirement."""
        # Generate generic use cases based on domain
        return [
            "When you need to accomplish the stated goal",
            "When working on similar problems",
            "When you want to follow best practices"
        ]
    
    def _generate_examples(self, req: str) -> List[Dict[str, str]]:
        """Generate example input/output based on domain."""
        domain = self._extract_domain(req)
        
        examples_map = {
            'optimization': [
                {
                    "input": "Current implementation with performance issues",
                    "output": "Optimized implementation with benchmarks showing improvement"
                }
            ],
            'testing': [
                {
                    "input": "Function or module to test",
                    "output": "Comprehensive test suite with unit, integration, and edge case tests"
                }
            ],
            'architecture': [
                {
                    "input": "System requirements and constraints",
                    "output": "Architecture diagram with component descriptions and data flows"
                }
            ],
            'debugging': [
                {
                    "input": "Bug report with symptoms and reproduction steps",
                    "output": "Root cause analysis and fix with prevention measures"
                }
            ],
            'refactoring': [
                {
                    "input": "Complex or messy code",
                    "output": "Clean, readable code with same behavior"
                }
            ],
        }
        
        return examples_map.get(domain, [
            {
                "input": "Relevant input for the task",
                "output": "Expected high-quality output"
            }
        ])
    
    def _select_template(self, spec: PromptSpec) -> str:
        """Select appropriate template based on spec."""
        template_file = self.templates_dir / f"{spec.domain}.md"
        if template_file.exists():
            return template_file.read_text()
        
        # Fallback to generic template
        return self.templates.get('generic', self._default_template())
    
    def _fill_template(self, template: str, spec: PromptSpec) -> str:
        """Fill template with spec data."""
        # Replace placeholders
        prompt = template.replace('{title}', spec.title)
        prompt = prompt.replace('{purpose}', spec.purpose)
        prompt = prompt.replace('{domain}', spec.domain)
        
        # Format use cases
        use_cases_text = '\n'.join(f"- {uc}" for uc in spec.use_cases)
        prompt = prompt.replace('{use_cases}', use_cases_text)
        
        # Format examples
        examples_text = '\n\n'.join(
            f"### Input\n```\n{ex['input']}\n```\n\n### Output\n```\n{ex['output']}\n```"
            for ex in spec.examples
        )
        prompt = prompt.replace('{examples}', examples_text)
        
        return prompt
    
    def _validate(self, prompt: str) -> str:
        """Validate generated prompt and calculate quality score."""
        # Check minimum length
        if len(prompt) < 100:
            raise ValueError("Generated prompt too short")
        
        # Check required sections
        required = ['# ', '## Purpose', '## When to Use', '## Prompt', '## Examples']
        for section in required:
            if section not in prompt:
                raise ValueError(f"Missing required section: {section}")
        
        # Calculate quality score
        score = self._calculate_quality_score(prompt)
        if score < 0.6:
            print(f"[WARN] Quality score: {score:.2f} (< 0.6)")
        else:
            print(f"[OK] Quality score: {score:.2f}")
        
        return prompt
    
    def _calculate_quality_score(self, prompt: str) -> float:
        """Calculate quality score for generated prompt."""
        score = 0.0
        
        # Length check (0.2 points)
        if len(prompt) > 500:
            score += 0.2
        
        # Section completeness (0.3 points)
        sections = ['Purpose', 'When to Use', 'Prompt', 'Examples', 'Success Criteria']
        present = sum(1 for s in sections if f'## {s}' in prompt)
        score += (present / len(sections)) * 0.3
        
        # Code blocks (0.2 points)
        if '```' in prompt:
            score += 0.2
        
        # Examples quality (0.2 points)
        if 'Input' in prompt and 'Output' in prompt:
            score += 0.2
        
        # Related prompts (0.1 points)
        if '## Related Prompts' in prompt:
            score += 0.1
        
        return min(score, 1.0)
    
    def _load_templates(self) -> Dict[str, str]:
        """Load all templates."""
        templates = {}
        if self.templates_dir.exists():
            for template_file in self.templates_dir.glob('*.md'):
                templates[template_file.stem] = template_file.read_text()
        return templates
    
    def _default_template(self) -> str:
        """Default template if none found."""
        return """# {title}

> **{purpose}**

## Purpose

{purpose}

## When to Use

{use_cases}

## Prompt

Provide detailed instructions for accomplishing the goal.

## Examples

{examples}

## Related Prompts

- [Related prompt 1]
- [Related prompt 2]

---

**Generated by Meta-Prompt Generator**
"""

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate prompts from requirements")
    parser.add_argument('requirement', help='Natural language requirement')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--templates', default='tools/meta-prompt/templates', help='Templates directory')
    
    args = parser.parse_args()
    
    # Generate prompt
    generator = PromptGenerator(Path(args.templates))
    prompt = generator.generate(args.requirement)
    
    # Output
    if args.output:
        Path(args.output).write_text(prompt)
        print(f"[OK] Generated prompt: {args.output}")
    else:
        print(prompt)

if __name__ == '__main__':
    main()
