from dataclasses import dataclass
from typing import List


@dataclass
class RefactorReport:
    product_name: str
    agent: str
    timestamp: str
    changes_made: List[str]
    issues_found: List[str]
    issues_fixed: List[str]
    warnings: List[str]
    status: str


@dataclass
class QualityMetrics:
    product_name: str
    loc: int
    files_count: int
    has_tests: bool
    has_docs: bool
    has_examples: bool
    docstring_coverage: float
    type_hint_coverage: float
    naming_compliance: float
    structure_compliance: float
    overall_score: float

