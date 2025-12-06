"""
Pytest configuration and shared fixtures for PEDs Playbook tests.
"""

import pytest
from PEDs_Playbook.main import (
    Supplement,
    SupplementCategory,
    UserProfile,
    FitnessGoal,
    SupplementPlan,
    SupplementDatabase,
    RecommendationEngine,
    TemplateManager
)


@pytest.fixture
def sample_protein_supplement():
    """Create a sample protein supplement."""
    return Supplement(
        name="Whey Protein",
        category=SupplementCategory.PROTEIN,
        dosage="25g",
        timing="Post-workout",
        price=29.99,
        benefits=["Muscle growth", "Recovery"],
        warnings=["Contains dairy"]
    )


@pytest.fixture
def sample_vitamin_supplement():
    """Create a sample vitamin supplement."""
    return Supplement(
        name="Vitamin D3",
        category=SupplementCategory.VITAMINS,
        dosage="2000 IU",
        timing="Morning",
        price=12.99,
        benefits=["Bone health", "Immune support"],
        warnings=[]
    )


@pytest.fixture
def sample_preworkout_supplement():
    """Create a sample pre-workout supplement."""
    return Supplement(
        name="Pre-Workout Formula",
        category=SupplementCategory.PRE_WORKOUT,
        dosage="1 scoop",
        timing="30 min before workout",
        price=34.99,
        benefits=["Energy", "Focus", "Endurance"],
        warnings=["Contains caffeine"]
    )


@pytest.fixture
def sample_recovery_supplement():
    """Create a sample recovery supplement."""
    return Supplement(
        name="BCAA Complex",
        category=SupplementCategory.RECOVERY,
        dosage="5g",
        timing="During/Post-workout",
        price=24.99,
        benefits=["Muscle recovery", "Reduce soreness"],
        warnings=[]
    )


@pytest.fixture
def sample_energy_supplement():
    """Create a sample energy supplement."""
    return Supplement(
        name="Caffeine Pills",
        category=SupplementCategory.ENERGY,
        dosage="200mg",
        timing="Morning",
        price=9.99,
        benefits=["Energy", "Mental focus"],
        warnings=["Contains caffeine", "May cause jitters"]
    )


@pytest.fixture
def sample_user_profile():
    """Create a sample user profile."""
    return UserProfile(
        user_id="user123",
        age=25,
        weight=75.0,
        height=180.0,
        fitness_goal=FitnessGoal.MUSCLE_GAIN,
        allergies=[],
        budget=100.0
    )


@pytest.fixture
def user_with_allergies():
    """Create a user profile with allergies."""
    return UserProfile(
        user_id="user456",
        age=30,
        weight=68.0,
        height=165.0,
        fitness_goal=FitnessGoal.WEIGHT_LOSS,
        allergies=["dairy", "caffeine"],
        budget=50.0
    )


@pytest.fixture
def user_no_budget():
    """Create a user profile without budget constraint."""
    return UserProfile(
        user_id="user789",
        age=35,
        weight=80.0,
        height=175.0,
        fitness_goal=FitnessGoal.ENDURANCE,
        allergies=[],
        budget=None
    )


@pytest.fixture
def empty_supplement_plan(sample_user_profile):
    """Create an empty supplement plan."""
    return SupplementPlan(
        plan_id="plan001",
        user_id=sample_user_profile.user_id
    )


@pytest.fixture
def populated_supplement_plan(sample_user_profile, sample_protein_supplement,
                             sample_vitamin_supplement):
    """Create a supplement plan with some supplements."""
    plan = SupplementPlan(
        plan_id="plan002",
        user_id=sample_user_profile.user_id
    )
    plan.add_supplement(sample_protein_supplement)
    plan.add_supplement(sample_vitamin_supplement)
    return plan


@pytest.fixture
def supplement_database(sample_protein_supplement, sample_vitamin_supplement,
                       sample_preworkout_supplement, sample_recovery_supplement,
                       sample_energy_supplement):
    """Create a populated supplement database."""
    db = SupplementDatabase()
    db.add_supplement(sample_protein_supplement)
    db.add_supplement(sample_vitamin_supplement)
    db.add_supplement(sample_preworkout_supplement)
    db.add_supplement(sample_recovery_supplement)
    db.add_supplement(sample_energy_supplement)
    return db


@pytest.fixture
def empty_database():
    """Create an empty supplement database."""
    return SupplementDatabase()


@pytest.fixture
def recommendation_engine(supplement_database):
    """Create a recommendation engine with populated database."""
    return RecommendationEngine(supplement_database)


@pytest.fixture
def template_manager():
    """Create an empty template manager."""
    return TemplateManager()


@pytest.fixture
def template_manager_with_templates(template_manager, populated_supplement_plan):
    """Create a template manager with some templates."""
    template_manager.create_template("Muscle Gain Basic", populated_supplement_plan)
    return template_manager


@pytest.fixture
def various_supplements():
    """Create a variety of supplements for testing."""
    return [
        Supplement(
            name="Creatine Monohydrate",
            category=SupplementCategory.POST_WORKOUT,
            dosage="5g",
            timing="Post-workout",
            price=19.99,
            benefits=["Strength", "Power output"],
            warnings=[]
        ),
        Supplement(
            name="Fish Oil",
            category=SupplementCategory.HEALTH,
            dosage="1000mg",
            timing="With meals",
            price=15.99,
            benefits=["Heart health", "Anti-inflammatory"],
            warnings=["Contains fish"]
        ),
        Supplement(
            name="Multivitamin",
            category=SupplementCategory.VITAMINS,
            dosage="1 tablet",
            timing="Morning",
            price=18.99,
            benefits=["Overall health", "Nutrient coverage"],
            warnings=[]
        ),
        Supplement(
            name="Magnesium",
            category=SupplementCategory.MINERALS,
            dosage="400mg",
            timing="Evening",
            price=10.99,
            benefits=["Sleep quality", "Muscle function"],
            warnings=[]
        ),
        Supplement(
            name="Beta-Alanine",
            category=SupplementCategory.PRE_WORKOUT,
            dosage="3g",
            timing="Pre-workout",
            price=16.99,
            benefits=["Endurance", "Delay fatigue"],
            warnings=["May cause tingling sensation"]
        )
    ]
