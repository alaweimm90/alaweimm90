#!/usr/bin/env python3
"""
PromptForge Lite - Offline Prompt Pattern Extraction

Extracts reusable prompt patterns from notes/documentation using regex.
No ML required - works completely offline.

Usage:
    python promptforge.py extract --input notes.md --output prompts.json
    python promptforge.py scan --directory ./notes/ --output all_prompts.json
    python promptforge.py template --input prompts.json --name "system-design"
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class PromptPattern:
    """Extracted prompt pattern"""
    pattern_type: str
    pattern_name: str
    template: str
    variables: List[str]
    example: str
    source_file: str
    line_number: int
    confidence: float  # 0-1
    tags: List[str]


@dataclass
class ExtractionResult:
    """Result of prompt extraction"""
    total_patterns: int
    patterns: List[PromptPattern]
    files_processed: int
    timestamp: str


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


class PromptLibrary:
    """Manage extracted prompt patterns"""

    def __init__(self):
        self.patterns: List[PromptPattern] = []

    def add_patterns(self, patterns: List[PromptPattern]):
        """Add patterns to library"""
        self.patterns.extend(patterns)

    def filter_by_type(self, pattern_type: str) -> List[PromptPattern]:
        """Filter patterns by type"""
        return [p for p in self.patterns if p.pattern_type == pattern_type]

    def filter_by_tag(self, tag: str) -> List[PromptPattern]:
        """Filter patterns by tag"""
        return [p for p in self.patterns if tag in p.tags]

    def filter_by_confidence(self, min_confidence: float) -> List[PromptPattern]:
        """Filter patterns by minimum confidence"""
        return [p for p in self.patterns if p.confidence >= min_confidence]

    def get_template(self, name: str) -> Optional[PromptPattern]:
        """Get pattern by name"""
        matches = [p for p in self.patterns if p.pattern_name == name]
        return matches[0] if matches else None

    def deduplicate(self):
        """Remove duplicate patterns"""
        seen = set()
        unique = []

        for pattern in self.patterns:
            # Create signature from template
            signature = pattern.template.lower().strip()
            if signature not in seen:
                seen.add(signature)
                unique.append(pattern)

        self.patterns = unique

    def rank_by_reusability(self) -> List[PromptPattern]:
        """Rank patterns by reusability score"""
        # Score = confidence * num_variables * len(template)
        scored = []
        for p in self.patterns:
            reusability = p.confidence * (1 + len(p.variables)) * min(len(p.template) / 100, 2.0)
            scored.append((reusability, p))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [p for _, p in scored]


def extract_from_file(file_path: Path, extractor: PatternExtractor) -> List[PromptPattern]:
    """Extract patterns from a single file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return extractor.extract_from_text(content, str(file_path))
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []


def extract_from_directory(directory: Path, extractor: PatternExtractor) -> ExtractionResult:
    """Extract patterns from all markdown/text files in directory"""
    patterns = []
    files_processed = 0

    # Find all markdown and text files
    for ext in ['*.md', '*.txt', '*.markdown']:
        for file_path in directory.rglob(ext):
            print(f"Processing: {file_path}")
            file_patterns = extract_from_file(file_path, extractor)
            patterns.extend(file_patterns)
            files_processed += 1

    result = ExtractionResult(
        total_patterns=len(patterns),
        patterns=patterns,
        files_processed=files_processed,
        timestamp=datetime.now().isoformat()
    )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PromptForge Lite - Offline Prompt Pattern Extraction"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract from single file')
    extract_parser.add_argument('--input', required=True, help='Input file path')
    extract_parser.add_argument('--output', required=True, help='Output JSON file')
    extract_parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory')
    scan_parser.add_argument('--directory', required=True, help='Directory to scan')
    scan_parser.add_argument('--output', required=True, help='Output JSON file')
    scan_parser.add_argument('--min-confidence', type=float, default=0.4, help='Minimum confidence')

    # Template command
    template_parser = subparsers.add_parser('template', help='Get template by name')
    template_parser.add_argument('--input', required=True, help='Patterns JSON file')
    template_parser.add_argument('--name', help='Pattern name to retrieve')
    template_parser.add_argument('--type', help='Filter by pattern type')
    template_parser.add_argument('--tag', help='Filter by tag')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    extractor = PatternExtractor()

    if args.command == 'extract':
        # Extract from single file
        file_path = Path(args.input)
        patterns = extract_from_file(file_path, extractor)

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= args.min_confidence]

        result = ExtractionResult(
            total_patterns=len(patterns),
            patterns=patterns,
            files_processed=1,
            timestamp=datetime.now().isoformat()
        )

        # Save to JSON
        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nExtracted {len(patterns)} patterns from {file_path}")
        print(f"Saved to: {args.output}")

    elif args.command == 'scan':
        # Scan directory
        directory = Path(args.directory)
        result = extract_from_directory(directory, extractor)

        # Filter by confidence
        result.patterns = [p for p in result.patterns if p.confidence >= args.min_confidence]
        result.total_patterns = len(result.patterns)

        # Deduplicate
        library = PromptLibrary()
        library.add_patterns(result.patterns)
        library.deduplicate()
        result.patterns = library.patterns
        result.total_patterns = len(library.patterns)

        # Save to JSON
        with open(args.output, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nScanned {result.files_processed} files")
        print(f"Extracted {result.total_patterns} unique patterns")
        print(f"Saved to: {args.output}")

    elif args.command == 'template':
        # Load patterns
        with open(args.input, 'r') as f:
            data = json.load(f)

        library = PromptLibrary()
        library.patterns = [
            PromptPattern(**p) for p in data['patterns']
        ]

        # Filter patterns
        if args.name:
            pattern = library.get_template(args.name)
            if pattern:
                print(f"\nPattern: {pattern.pattern_name}")
                print(f"Type: {pattern.pattern_type}")
                print(f"Confidence: {pattern.confidence}")
                print(f"Variables: {', '.join(pattern.variables) if pattern.variables else 'None'}")
                print(f"\nTemplate:\n{pattern.template}")
            else:
                print(f"Pattern '{args.name}' not found")

        elif args.type:
            patterns = library.filter_by_type(args.type)
            print(f"\nFound {len(patterns)} patterns of type '{args.type}':")
            for p in patterns:
                print(f"  - {p.pattern_name} (confidence: {p.confidence})")

        elif args.tag:
            patterns = library.filter_by_tag(args.tag)
            print(f"\nFound {len(patterns)} patterns with tag '{args.tag}':")
            for p in patterns:
                print(f"  - {p.pattern_name} (confidence: {p.confidence})")

        else:
            # Show top patterns by reusability
            top_patterns = library.rank_by_reusability()[:10]
            print(f"\nTop 10 most reusable patterns:")
            for i, p in enumerate(top_patterns, 1):
                print(f"{i}. {p.pattern_name} (type: {p.pattern_type}, confidence: {p.confidence})")


if __name__ == "__main__":
    main()
