"""
Tests for UserProfile class and related functionality.
"""

import pytest
from PEDs_Playbook.main import UserProfile, FitnessGoal


class TestUserProfileCreation:
    """Test user profile creation and validation."""

    def test_create_valid_user_profile(self, sample_user_profile):
        """Test creating a valid user profile."""
        assert sample_user_profile.user_id == "user123"
        assert sample_user_profile.age == 25
        assert sample_user_profile.weight == 75.0
        assert sample_user_profile.height == 180.0
        assert sample_user_profile.fitness_goal == FitnessGoal.MUSCLE_GAIN

    def test_user_profile_with_allergies(self, user_with_allergies):
        """Test user profile with allergies."""
        assert len(user_with_allergies.allergies) == 2
        assert "dairy" in user_with_allergies.allergies
        assert "caffeine" in user_with_allergies.allergies

    def test_user_profile_with_budget(self, sample_user_profile):
        """Test user profile with budget constraint."""
        assert sample_user_profile.budget == 100.0

    def test_user_profile_without_budget(self, user_no_budget):
        """Test user profile without budget constraint."""
        assert user_no_budget.budget is None

    def test_user_age_too_young_raises_error(self):
        """Test that age < 18 raises ValueError."""
        with pytest.raises(ValueError, match="Age must be between"):
            UserProfile(
                user_id="test",
                age=15,
                weight=70.0,
                height=175.0,
                fitness_goal=FitnessGoal.MUSCLE_GAIN
            )

    def test_user_age_too_old_raises_error(self):
        """Test that age > 120 raises ValueError."""
        with pytest.raises(ValueError, match="Age must be between"):
            UserProfile(
                user_id="test",
                age=125,
                weight=70.0,
                height=175.0,
                fitness_goal=FitnessGoal.MUSCLE_GAIN
            )

    def test_negative_weight_raises_error(self):
        """Test that negative weight raises ValueError."""
        with pytest.raises(ValueError, match="Weight must be positive"):
            UserProfile(
                user_id="test",
                age=25,
                weight=-70.0,
                height=175.0,
                fitness_goal=FitnessGoal.MUSCLE_GAIN
            )

    def test_negative_height_raises_error(self):
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="Height must be positive"):
            UserProfile(
                user_id="test",
                age=25,
                weight=70.0,
                height=-175.0,
                fitness_goal=FitnessGoal.MUSCLE_GAIN
            )

    def test_negative_budget_raises_error(self):
        """Test that negative budget raises ValueError."""
        with pytest.raises(ValueError, match="Budget cannot be negative"):
            UserProfile(
                user_id="test",
                age=25,
                weight=70.0,
                height=175.0,
                fitness_goal=FitnessGoal.MUSCLE_GAIN,
                budget=-50.0
            )


class TestUserProfileMethods:
    """Test user profile methods."""

    def test_calculate_bmi(self, sample_user_profile):
        """Test BMI calculation."""
        bmi = sample_user_profile.calculate_bmi()
        # 75 kg / (1.8 m)^2 = 23.15
        assert bmi == 23.15

    def test_calculate_bmi_different_values(self):
        """Test BMI calculation with different values."""
        profile = UserProfile(
            user_id="test",
            age=30,
            weight=90.0,
            height=175.0,
            fitness_goal=FitnessGoal.WEIGHT_LOSS
        )
        bmi = profile.calculate_bmi()
        # 90 / (1.75)^2 = 29.39
        assert bmi == 29.39

    def test_user_profile_to_dict(self, sample_user_profile):
        """Test converting user profile to dictionary."""
        result = sample_user_profile.to_dict()
        assert isinstance(result, dict)
        assert result['user_id'] == "user123"
        assert result['age'] == 25
        assert result['fitness_goal'] == "muscle_gain"

    def test_user_profile_to_dict_includes_all_fields(self, user_with_allergies):
        """Test that to_dict includes all fields."""
        result = user_with_allergies.to_dict()
        assert 'user_id' in result
        assert 'age' in result
        assert 'weight' in result
        assert 'height' in result
        assert 'fitness_goal' in result
        assert 'allergies' in result
        assert 'budget' in result
