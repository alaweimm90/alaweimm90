"""
PEDs Playbook - Supplement Templates and Recommendations System.

This module provides core functionality for managing supplement templates,
recommendations, and personalized supplement plans.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class SupplementCategory(Enum):
    """Categories of supplements."""
    PROTEIN = "protein"
    VITAMINS = "vitamins"
    MINERALS = "minerals"
    PRE_WORKOUT = "pre_workout"
    POST_WORKOUT = "post_workout"
    RECOVERY = "recovery"
    ENERGY = "energy"
    HEALTH = "health"


class FitnessGoal(Enum):
    """User fitness goals."""
    MUSCLE_GAIN = "muscle_gain"
    WEIGHT_LOSS = "weight_loss"
    ENDURANCE = "endurance"
    STRENGTH = "strength"
    GENERAL_HEALTH = "general_health"
    RECOVERY = "recovery"


@dataclass
class Supplement:
    """Represents a supplement with its properties."""
    name: str
    category: SupplementCategory
    dosage: str
    timing: str
    price: float
    benefits: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate supplement data."""
        if not self.name or not self.name.strip():
            raise ValueError("Supplement name cannot be empty")
        if self.price < 0:
            raise ValueError("Price cannot be negative")
        if not self.dosage or not self.dosage.strip():
            raise ValueError("Dosage cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert supplement to dictionary."""
        data = asdict(self)
        data['category'] = self.category.value
        return data


@dataclass
class UserProfile:
    """User profile with fitness goals and constraints."""
    user_id: str
    age: int
    weight: float
    height: float
    fitness_goal: FitnessGoal
    allergies: List[str] = field(default_factory=list)
    budget: Optional[float] = None

    def __post_init__(self):
        """Validate user profile."""
        if self.age < 18 or self.age > 120:
            raise ValueError("Age must be between 18 and 120")
        if self.weight <= 0:
            raise ValueError("Weight must be positive")
        if self.height <= 0:
            raise ValueError("Height must be positive")
        if self.budget is not None and self.budget < 0:
            raise ValueError("Budget cannot be negative")

    def calculate_bmi(self) -> float:
        """Calculate BMI (weight in kg, height in cm)."""
        height_m = self.height / 100
        return round(self.weight / (height_m ** 2), 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary."""
        data = asdict(self)
        data['fitness_goal'] = self.fitness_goal.value
        return data


@dataclass
class SupplementPlan:
    """A personalized supplement plan."""
    plan_id: str
    user_id: str
    supplements: List[Supplement] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_cost: float = 0.0

    def add_supplement(self, supplement: Supplement) -> None:
        """Add a supplement to the plan."""
        if not isinstance(supplement, Supplement):
            raise TypeError("Must be a Supplement instance")
        self.supplements.append(supplement)
        self.total_cost = sum(s.price for s in self.supplements)

    def remove_supplement(self, supplement_name: str) -> bool:
        """Remove a supplement by name."""
        for i, supp in enumerate(self.supplements):
            if supp.name == supplement_name:
                self.supplements.pop(i)
                self.total_cost = sum(s.price for s in self.supplements)
                return True
        return False

    def get_supplements_by_category(self, category: SupplementCategory) -> List[Supplement]:
        """Get all supplements in a category."""
        return [s for s in self.supplements if s.category == category]

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary."""
        return {
            'plan_id': self.plan_id,
            'user_id': self.user_id,
            'supplements': [s.to_dict() for s in self.supplements],
            'created_at': self.created_at,
            'total_cost': self.total_cost
        }


class SupplementDatabase:
    """Database of available supplements."""

    def __init__(self):
        """Initialize with empty database."""
        self.supplements: Dict[str, Supplement] = {}

    def add_supplement(self, supplement: Supplement) -> None:
        """Add a supplement to the database."""
        if not isinstance(supplement, Supplement):
            raise TypeError("Must be a Supplement instance")
        self.supplements[supplement.name] = supplement

    def get_supplement(self, name: str) -> Optional[Supplement]:
        """Get a supplement by name."""
        return self.supplements.get(name)

    def search_by_category(self, category: SupplementCategory) -> List[Supplement]:
        """Search supplements by category."""
        return [s for s in self.supplements.values() if s.category == category]

    def search_by_price_range(self, min_price: float, max_price: float) -> List[Supplement]:
        """Search supplements within price range."""
        if min_price < 0 or max_price < 0:
            raise ValueError("Prices cannot be negative")
        if min_price > max_price:
            raise ValueError("Min price cannot exceed max price")
        return [s for s in self.supplements.values()
                if min_price <= s.price <= max_price]

    def get_all_supplements(self) -> List[Supplement]:
        """Get all supplements."""
        return list(self.supplements.values())

    def count(self) -> int:
        """Count total supplements."""
        return len(self.supplements)


class RecommendationEngine:
    """Engine for generating supplement recommendations."""

    def __init__(self, database: SupplementDatabase):
        """Initialize with supplement database."""
        self.database = database

    def recommend_for_goal(self,
                          profile: UserProfile,
                          max_supplements: int = 5) -> SupplementPlan:
        """Generate recommendations based on user goal."""
        if max_supplements < 1:
            raise ValueError("Max supplements must be at least 1")

        plan = SupplementPlan(
            plan_id=f"plan_{profile.user_id}_{datetime.now().timestamp()}",
            user_id=profile.user_id
        )

        # Define supplement priorities based on goals
        goal_categories = self._get_categories_for_goal(profile.fitness_goal)

        # Get supplements from priority categories
        recommended = []
        for category in goal_categories:
            supplements = self.database.search_by_category(category)
            recommended.extend(supplements)

        # Filter by budget if specified
        if profile.budget is not None:
            recommended = [s for s in recommended if s.price <= profile.budget]

        # Filter out allergens
        recommended = self._filter_allergens(recommended, profile.allergies)

        # Sort by price (cheapest first) and limit
        recommended.sort(key=lambda x: x.price)
        recommended = recommended[:max_supplements]

        # Add to plan
        for supplement in recommended:
            plan.add_supplement(supplement)

        return plan

    def _get_categories_for_goal(self, goal: FitnessGoal) -> List[SupplementCategory]:
        """Map fitness goals to supplement categories."""
        goal_mapping = {
            FitnessGoal.MUSCLE_GAIN: [
                SupplementCategory.PROTEIN,
                SupplementCategory.POST_WORKOUT,
                SupplementCategory.RECOVERY
            ],
            FitnessGoal.WEIGHT_LOSS: [
                SupplementCategory.ENERGY,
                SupplementCategory.VITAMINS,
                SupplementCategory.MINERALS
            ],
            FitnessGoal.ENDURANCE: [
                SupplementCategory.PRE_WORKOUT,
                SupplementCategory.ENERGY,
                SupplementCategory.RECOVERY
            ],
            FitnessGoal.STRENGTH: [
                SupplementCategory.PROTEIN,
                SupplementCategory.PRE_WORKOUT,
                SupplementCategory.POST_WORKOUT
            ],
            FitnessGoal.GENERAL_HEALTH: [
                SupplementCategory.VITAMINS,
                SupplementCategory.MINERALS,
                SupplementCategory.HEALTH
            ],
            FitnessGoal.RECOVERY: [
                SupplementCategory.RECOVERY,
                SupplementCategory.POST_WORKOUT,
                SupplementCategory.PROTEIN
            ]
        }
        return goal_mapping.get(goal, [SupplementCategory.HEALTH])

    def _filter_allergens(self, supplements: List[Supplement],
                         allergies: List[str]) -> List[Supplement]:
        """Filter out supplements with allergens."""
        if not allergies:
            return supplements

        filtered = []
        for supplement in supplements:
            has_allergen = False
            for allergen in allergies:
                # Check if allergen is in supplement name or warnings
                if (allergen.lower() in supplement.name.lower() or
                    any(allergen.lower() in w.lower() for w in supplement.warnings)):
                    has_allergen = True
                    break
            if not has_allergen:
                filtered.append(supplement)

        return filtered

    def calculate_plan_value(self, plan: SupplementPlan) -> Dict[str, Any]:
        """Calculate value metrics for a plan."""
        if not plan.supplements:
            return {
                'total_cost': 0.0,
                'avg_cost': 0.0,
                'supplement_count': 0,
                'categories_covered': 0
            }

        categories = set(s.category for s in plan.supplements)

        return {
            'total_cost': plan.total_cost,
            'avg_cost': round(plan.total_cost / len(plan.supplements), 2),
            'supplement_count': len(plan.supplements),
            'categories_covered': len(categories)
        }


class TemplateManager:
    """Manager for supplement plan templates."""

    def __init__(self):
        """Initialize with empty templates."""
        self.templates: Dict[str, SupplementPlan] = {}

    def create_template(self, name: str, plan: SupplementPlan) -> None:
        """Create a new template."""
        if not name or not name.strip():
            raise ValueError("Template name cannot be empty")
        if not isinstance(plan, SupplementPlan):
            raise TypeError("Plan must be a SupplementPlan instance")
        self.templates[name] = plan

    def get_template(self, name: str) -> Optional[SupplementPlan]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all template names."""
        return list(self.templates.keys())

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        if name in self.templates:
            del self.templates[name]
            return True
        return False

    def export_template(self, name: str) -> Optional[str]:
        """Export template as JSON."""
        template = self.templates.get(name)
        if template:
            return json.dumps(template.to_dict(), indent=2)
        return None

    def import_template(self, name: str, json_data: str) -> bool:
        """Import template from JSON."""
        try:
            data = json.loads(json_data)
            # Reconstruct supplements
            supplements = []
            for supp_data in data.get('supplements', []):
                category = SupplementCategory(supp_data['category'])
                supplement = Supplement(
                    name=supp_data['name'],
                    category=category,
                    dosage=supp_data['dosage'],
                    timing=supp_data['timing'],
                    price=supp_data['price'],
                    benefits=supp_data.get('benefits', []),
                    warnings=supp_data.get('warnings', [])
                )
                supplements.append(supplement)

            # Create plan
            plan = SupplementPlan(
                plan_id=data['plan_id'],
                user_id=data['user_id'],
                supplements=supplements,
                created_at=data.get('created_at', datetime.now().isoformat())
            )

            self.templates[name] = plan
            return True
        except (json.JSONDecodeError, KeyError, ValueError):
            return False


def main():
    """Main entry point."""
    print("PEDs Playbook - Supplement Templates System")
    print("Version 0.1.0")
    print("\nInitializing system...")

    # Create database
    db = SupplementDatabase()

    # Add sample supplements
    db.add_supplement(Supplement(
        name="Whey Protein",
        category=SupplementCategory.PROTEIN,
        dosage="25g",
        timing="Post-workout",
        price=29.99,
        benefits=["Muscle growth", "Recovery"],
        warnings=["Contains dairy"]
    ))

    print(f"Loaded {db.count()} supplements into database")
    print("\nSystem ready!")
    return 0


if __name__ == "__main__":
    exit(main())
