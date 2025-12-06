"""
Question Loader

Loads questions from the 200_QUESTION_DATABASE.json file.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from interrogation.core.models import QuestionCategory


class QuestionLoader:
    """
    Loads and manages the 200-question database

    The database contains 10 categories × 20 questions with weights.
    """

    def __init__(self, database_path: Optional[Path] = None):
        """
        Initialize question loader

        Args:
            database_path: Path to 200_QUESTION_DATABASE.json
                          If None, searches in standard locations
        """
        if database_path is None:
            database_path = self._find_database()

        self.database_path = database_path
        self.categories: List[QuestionCategory] = []
        self.database_info: Dict = {}

        self._load_database()

    def _find_database(self) -> Path:
        """Find the database file in standard locations"""
        # Standard locations to search
        search_paths = [
            # Relative to project
            Path(__file__).parent.parent.parent.parent.parent.parent / "01-DOCUMENTATION" / "research" / "turing-challenge" / "FROM QAP" / "200_QUESTION_DATABASE.json",
            # Relative to current directory
            Path.cwd() / "01-DOCUMENTATION" / "research" / "turing-challenge" / "FROM QAP" / "200_QUESTION_DATABASE.json",
            # Direct path
            Path("/home/user/CLAUDE-CODE/01-DOCUMENTATION/research/turing-challenge/FROM QAP/200_QUESTION_DATABASE.json"),
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            "Could not find 200_QUESTION_DATABASE.json. "
            "Please specify database_path explicitly."
        )

    def _load_database(self):
        """Load database from JSON file"""
        with open(self.database_path, 'r') as f:
            data = json.load(f)

        framework = data.get("interrogation_framework", {})
        self.database_info = {
            "version": framework.get("version", "unknown"),
            "total_questions": framework.get("total_questions", 0),
            "categories": framework.get("categories", 0),
            "last_updated": framework.get("last_updated", "unknown"),
            "description": framework.get("description", ""),
        }

        # Load categories
        categories_data = framework.get("categories_list", [])
        for cat_data in categories_data:
            category = QuestionCategory(
                id=cat_data["id"],
                name=cat_data["name"],
                weight=cat_data["weight"],
                description=cat_data["description"],
                question_count=cat_data["question_count"],
                questions=cat_data["questions"],
            )
            self.categories.append(category)

    def get_category(self, name: str) -> Optional[QuestionCategory]:
        """Get category by name"""
        for cat in self.categories:
            if cat.name.lower() == name.lower():
                return cat
        return None

    def get_category_by_id(self, category_id: int) -> Optional[QuestionCategory]:
        """Get category by ID"""
        for cat in self.categories:
            if cat.id == category_id:
                return cat
        return None

    def get_all_categories(self) -> List[QuestionCategory]:
        """Get all categories"""
        return self.categories

    def get_category_names(self) -> List[str]:
        """Get list of all category names"""
        return [cat.name for cat in self.categories]

    def get_questions(
        self,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[str]:
        """
        Get questions from database

        Args:
            category: Filter by category name (None = all categories)
            limit: Limit number of questions per category

        Returns:
            List of questions
        """
        questions = []

        if category:
            cat = self.get_category(category)
            if cat:
                questions.extend(cat.questions[:limit] if limit else cat.questions)
        else:
            for cat in self.categories:
                questions.extend(cat.questions[:limit] if limit else cat.questions)

        return questions

    def get_weighted_categories(self) -> List[tuple]:
        """
        Get categories sorted by weight (descending)

        Returns:
            List of (category_name, weight) tuples
        """
        return sorted(
            [(cat.name, cat.weight) for cat in self.categories],
            key=lambda x: x[1],
            reverse=True
        )

    def validate_database(self) -> Dict[str, Any]:
        """
        Validate database structure

        Returns:
            Dict with validation results
        """
        issues = []
        warnings = []

        # Check total questions
        total_questions = sum(cat.question_count for cat in self.categories)
        if total_questions != 200:
            issues.append(f"Expected 200 questions, found {total_questions}")

        # Check categories
        if len(self.categories) != 10:
            issues.append(f"Expected 10 categories, found {len(self.categories)}")

        # Check each category
        for cat in self.categories:
            if cat.question_count != 20:
                warnings.append(f"{cat.name}: Expected 20 questions, has {cat.question_count}")

            if len(cat.questions) != cat.question_count:
                issues.append(
                    f"{cat.name}: question_count={cat.question_count} "
                    f"but len(questions)={len(cat.questions)}"
                )

            if not cat.description:
                warnings.append(f"{cat.name}: Missing description")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "total_questions": total_questions,
            "total_categories": len(self.categories),
        }

    def get_info(self) -> Dict:
        """Get database information"""
        return {
            **self.database_info,
            "loaded_categories": len(self.categories),
            "database_path": str(self.database_path),
        }

    def __repr__(self) -> str:
        return (
            f"QuestionLoader("
            f"categories={len(self.categories)}, "
            f"total_questions={sum(cat.question_count for cat in self.categories)}"
            f")"
        )
