"""Data models for PromptForge"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptPattern':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class ExtractionResult:
    """Result of prompt extraction"""
    total_patterns: int
    patterns: List[PromptPattern]
    files_processed: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_patterns': self.total_patterns,
            'patterns': [p.to_dict() for p in self.patterns],
            'files_processed': self.files_processed,
            'timestamp': self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create from dictionary"""
        return cls(
            total_patterns=data['total_patterns'],
            patterns=[PromptPattern.from_dict(p) for p in data['patterns']],
            files_processed=data['files_processed'],
            timestamp=data['timestamp']
        )
