"""Pattern extraction logic"""

import re
from typing import List

from .models import PromptPattern


class PatternExtractor:
    """Extract prompt patterns using regex"""

    def __init__(self):
        # Regex patterns for common prompt structures
        self.patterns = {
            "instruction": re.compile(
                r'(?:^|\n)(?:instruction|task|goal|objective):\s*(.+?)(?:\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            "role_play": re.compile(
                r'(?:you are|act as|behave as|role:)\s+(?:a |an )?(.+?)(?:\.|,|\n)',
                re.IGNORECASE
            ),
            "format": re.compile(
                r'(?:format|structure|output):\s*(.+?)(?:\n\n|\n-|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            "constraint": re.compile(
                r'(?:constraint|requirement|must|should):\s*(.+?)(?:\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            "example": re.compile(
                r'(?:example|sample|for instance):\s*(.+?)(?:\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
            "step_by_step": re.compile(
                r'(?:step \d+|first|then|next|finally):\s*(.+?)(?:\n|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            "conditional": re.compile(
                r'(?:if|when|given|assuming)\s+(.+?),?\s+(?:then|do|should)\s+(.+?)(?:\.|,|\n)',
                re.IGNORECASE
            ),
            "context": re.compile(
                r'(?:context|background|given):\s*(.+?)(?:\n\n|\Z)',
                re.IGNORECASE | re.DOTALL
            ),
        }

        # Variable detection patterns
        self.variable_patterns = [
            re.compile(r'\{([^}]+)\}'),  # {variable}
            re.compile(r'\[([^\]]+)\]'),  # [variable]
            re.compile(r'<([^>]+)>'),    # <variable>
            re.compile(r'\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?'),  # $variable or ${variable}
        ]

    def extract_from_text(self, text: str, source_file: str = "unknown") -> List[PromptPattern]:
        """Extract prompt patterns from text"""
        patterns = []

        lines = text.split('\n')
        line_map = {i: line for i, line in enumerate(lines, 1)}

        for pattern_type, regex in self.patterns.items():
            matches = regex.finditer(text)

            for match in matches:
                # Find line number
                match_pos = match.start()
                line_num = text[:match_pos].count('\n') + 1

                # Extract matched content
                if len(match.groups()) == 1:
                    content = match.group(1).strip()
                    template = content
                elif len(match.groups()) == 2:
                    # Conditional pattern
                    content = f"IF {match.group(1)} THEN {match.group(2)}"
                    template = content
                else:
                    content = match.group(0).strip()
                    template = content

                # Extract variables
                variables = self._extract_variables(template)

                # Generate pattern name
                pattern_name = self._generate_name(pattern_type, template)

                # Determine confidence based on structure
                confidence = self._calculate_confidence(template, variables)

                # Extract tags
                tags = self._extract_tags(text, match.start(), match.end())

                # Create example from surrounding context
                example = self._extract_example(text, match.start(), match.end())

                pattern = PromptPattern(
                    pattern_type=pattern_type,
                    pattern_name=pattern_name,
                    template=template[:500],  # Truncate long templates
                    variables=variables,
                    example=example[:200],
                    source_file=source_file,
                    line_number=line_num,
                    confidence=round(confidence, 2),
                    tags=tags
                )

                patterns.append(pattern)

        return patterns

    def _extract_variables(self, template: str) -> List[str]:
        """Extract variable placeholders from template"""
        variables = set()

        for var_pattern in self.variable_patterns:
            matches = var_pattern.findall(template)
            variables.update(matches)

        return sorted(list(variables))

    def _generate_name(self, pattern_type: str, template: str) -> str:
        """Generate human-readable pattern name"""
        # Take first few significant words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', template)
        if words:
            name_words = words[:3]
            name = "_".join(name_words).lower()
            return f"{pattern_type}_{name}"
        return f"{pattern_type}_pattern"

    def _calculate_confidence(self, template: str, variables: List[str]) -> float:
        """Calculate confidence score for pattern"""
        score = 0.5  # Base score

        # Bonus for having variables (more reusable)
        if variables:
            score += 0.2

        # Bonus for structured content
        if any(kw in template.lower() for kw in ['step', 'first', 'then', 'finally']):
            score += 0.1

        # Bonus for explicit instructions
        if any(kw in template.lower() for kw in ['must', 'should', 'required']):
            score += 0.1

        # Penalty for very short templates
        if len(template) < 20:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _extract_tags(self, text: str, start: int, end: int) -> List[str]:
        """Extract relevant tags from context"""
        tags = set()

        # Look for hashtags in surrounding context
        context_start = max(0, start - 200)
        context_end = min(len(text), end + 200)
        context = text[context_start:context_end]

        hashtags = re.findall(r'#(\w+)', context)
        tags.update(hashtags[:5])

        # Add semantic tags based on content
        template = text[start:end].lower()
        if 'code' in template or 'function' in template:
            tags.add('coding')
        if 'write' in template or 'generate' in template:
            tags.add('generation')
        if 'analyze' in template or 'review' in template:
            tags.add('analysis')

        return sorted(list(tags))[:5]

    def _extract_example(self, text: str, start: int, end: int) -> str:
        """Extract example from surrounding context"""
        # Get a few lines before and after
        context_start = max(0, start - 100)
        context_end = min(len(text), end + 100)
        context = text[context_start:context_end].strip()

        # Look for explicit examples
        example_match = re.search(r'e\.?g\.?,?\s*(.+?)(?:\n|$)', context, re.IGNORECASE)
        if example_match:
            return example_match.group(1).strip()

        # Otherwise return the match itself
        return text[start:end].strip()
