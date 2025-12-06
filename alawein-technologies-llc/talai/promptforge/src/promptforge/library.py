"""Prompt library management"""

from typing import List, Optional

from .models import PromptPattern


class PromptLibrary:
    """Manage extracted prompt patterns"""

    def __init__(self):
        self.patterns: List[PromptPattern] = []

    def add_patterns(self, patterns: List[PromptPattern]):
        """Add patterns to library"""
        self.patterns.extend(patterns)

    def add_pattern(self, pattern: PromptPattern):
        """Add a single pattern to library"""
        self.patterns.append(pattern)

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

    def get_pattern_types(self) -> List[str]:
        """Get list of unique pattern types"""
        return sorted(list(set(p.pattern_type for p in self.patterns)))

    def get_all_tags(self) -> List[str]:
        """Get list of all unique tags"""
        all_tags = set()
        for p in self.patterns:
            all_tags.update(p.tags)
        return sorted(list(all_tags))

    def count(self) -> int:
        """Get total number of patterns"""
        return len(self.patterns)

    def clear(self):
        """Remove all patterns from library"""
        self.patterns = []
