"""
Tests for RecommendationEngine class and related functionality.
"""

import pytest
from PEDs_Playbook.main import (
    RecommendationEngine,
    SupplementDatabase,
    UserProfile,
    FitnessGoal,
    SupplementCategory
)


class TestRecommendationEngineCreation:
    """Test recommendation engine creation."""

    def test_create_engine_with_database(self, recommendation_engine):
        """Test creating recommendation engine with database."""
        assert recommendation_engine.database is not None
        assert recommendation_engine.database.count() == 5


class TestRecommendForGoal:
    """Test recommendation generation for different goals."""

    def test_recommend_for_muscle_gain(self, recommendation_engine, sample_user_profile):
        """Test recommendations for muscle gain goal."""
        plan = recommendation_engine.recommend_for_goal(sample_user_profile)
        assert plan is not None
        assert plan.user_id == sample_user_profile.user_id
        assert len(plan.supplements) > 0

    def test_recommend_respects_max_supplements(self, recommendation_engine,
                                               sample_user_profile):
        """Test that recommendations respect max_supplements limit."""
        plan = recommendation_engine.recommend_for_goal(sample_user_profile, max_supplements=2)
        assert len(plan.supplements) <= 2

    def test_recommend_with_zero_max_raises_error(self, recommendation_engine,
                                                  sample_user_profile):
        """Test that max_supplements < 1 raises ValueError."""
        with pytest.raises(ValueError, match="Max supplements must be at least 1"):
            recommendation_engine.recommend_for_goal(sample_user_profile, max_supplements=0)

    def test_recommend_respects_budget(self, recommendation_engine):
        """Test that recommendations respect budget constraint."""
        profile = UserProfile(
            user_id="budget_user",
            age=25,
            weight=70.0,
            height=175.0,
            fitness_goal=FitnessGoal.MUSCLE_GAIN,
            budget=20.0
        )
        plan = recommendation_engine.recommend_for_goal(profile)
        for supplement in plan.supplements:
            assert supplement.price <= 20.0

    def test_recommend_filters_allergens(self, recommendation_engine, user_with_allergies):
        """Test that recommendations filter out allergens."""
        plan = recommendation_engine.recommend_for_goal(user_with_allergies)
        for supplement in plan.supplements:
            # Check no dairy in name or warnings
            assert "dairy" not in supplement.name.lower()
            assert not any("dairy" in w.lower() for w in supplement.warnings)
            # Check no caffeine in name or warnings
            assert "caffeine" not in supplement.name.lower()
            assert not any("caffeine" in w.lower() for w in supplement.warnings)

    def test_recommend_without_budget_constraint(self, recommendation_engine, user_no_budget):
        """Test recommendations without budget constraint."""
        plan = recommendation_engine.recommend_for_goal(user_no_budget)
        assert len(plan.supplements) > 0
        # Can include any price
        total_cost = sum(s.price for s in plan.supplements)
        assert total_cost > 0

    def test_recommend_for_weight_loss_goal(self, recommendation_engine):
        """Test recommendations for weight loss goal."""
        profile = UserProfile(
            user_id="weight_loss_user",
            age=30,
            weight=85.0,
            height=170.0,
            fitness_goal=FitnessGoal.WEIGHT_LOSS,
            budget=100.0
        )
        plan = recommendation_engine.recommend_for_goal(profile)
        assert plan is not None
        assert len(plan.supplements) > 0

    def test_recommend_for_endurance_goal(self, recommendation_engine):
        """Test recommendations for endurance goal."""
        profile = UserProfile(
            user_id="endurance_user",
            age=28,
            weight=70.0,
            height=175.0,
            fitness_goal=FitnessGoal.ENDURANCE,
            budget=100.0
        )
        plan = recommendation_engine.recommend_for_goal(profile)
        assert plan is not None
        assert len(plan.supplements) > 0


class TestRecommendationEngineHelpers:
    """Test helper methods."""

    def test_get_categories_for_muscle_gain(self, recommendation_engine):
        """Test category mapping for muscle gain."""
        categories = recommendation_engine._get_categories_for_goal(FitnessGoal.MUSCLE_GAIN)
        assert SupplementCategory.PROTEIN in categories
        assert SupplementCategory.POST_WORKOUT in categories

    def test_get_categories_for_weight_loss(self, recommendation_engine):
        """Test category mapping for weight loss."""
        categories = recommendation_engine._get_categories_for_goal(FitnessGoal.WEIGHT_LOSS)
        assert SupplementCategory.ENERGY in categories
        assert SupplementCategory.VITAMINS in categories

    def test_filter_allergens_removes_matches(self, recommendation_engine, various_supplements):
        """Test allergen filtering removes matching supplements."""
        allergies = ["fish"]
        filtered = recommendation_engine._filter_allergens(various_supplements, allergies)
        for supplement in filtered:
            assert "fish" not in supplement.name.lower()
            assert not any("fish" in w.lower() for w in supplement.warnings)

    def test_filter_allergens_with_empty_list(self, recommendation_engine, various_supplements):
        """Test allergen filtering with no allergies returns all."""
        filtered = recommendation_engine._filter_allergens(various_supplements, [])
        assert len(filtered) == len(various_supplements)

    def test_calculate_plan_value_empty_plan(self, recommendation_engine, empty_supplement_plan):
        """Test calculating value for empty plan."""
        value = recommendation_engine.calculate_plan_value(empty_supplement_plan)
        assert value['total_cost'] == 0.0
        assert value['avg_cost'] == 0.0
        assert value['supplement_count'] == 0
        assert value['categories_covered'] == 0

    def test_calculate_plan_value_with_supplements(self, recommendation_engine,
                                                   populated_supplement_plan):
        """Test calculating value for plan with supplements."""
        value = recommendation_engine.calculate_plan_value(populated_supplement_plan)
        assert value['supplement_count'] == 2
        assert value['total_cost'] > 0
        assert value['avg_cost'] > 0
        assert value['categories_covered'] > 0
