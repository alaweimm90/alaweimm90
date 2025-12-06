#!/usr/bin/env python3
"""
Comprehensive test suite for HypothesisMatch

Tests cover:
- Profile creation and management
- Hypothesis creation and management
- Swipe functionality
- Match finding algorithms
- Similarity scoring
- Edge cases and error handling
- Integration workflows
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from hypothesis_match.main import (
    ResearcherProfile,
    Hypothesis,
    Match,
    MatchingEngine
)


# Fixtures for test data
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def engine(temp_dir):
    """Create a MatchingEngine instance with temporary files"""
    profiles_file = temp_dir / "profiles.json"
    hypotheses_file = temp_dir / "hypotheses.json"
    return MatchingEngine(
        profiles_file=str(profiles_file),
        hypotheses_file=str(hypotheses_file)
    )


@pytest.fixture
def sample_profile():
    """Sample researcher profile for testing"""
    return {
        "name": "Dr. Alice Smith",
        "institution": "MIT",
        "skills": ["machine learning", "computer vision", "deep learning"],
        "interests": ["medical imaging", "cancer detection"],
        "h_index": 25,
        "looking_for": "complementary"
    }


@pytest.fixture
def sample_profile_2():
    """Second sample researcher profile"""
    return {
        "name": "Dr. Bob Johnson",
        "institution": "Stanford",
        "skills": ["statistics", "clinical trials", "epidemiology"],
        "interests": ["medical imaging", "drug development"],
        "h_index": 30,
        "looking_for": "complementary"
    }


@pytest.fixture
def sample_hypothesis():
    """Sample research hypothesis"""
    return {
        "text": "Deep learning can diagnose cancer better than radiologists",
        "field": "medical imaging",
        "tags": ["AI", "healthcare", "diagnostics"]
    }


# Test Class: Profile Management
class TestProfileManagement:
    """Tests for creating and managing researcher profiles"""

    def test_create_profile_basic(self, engine, sample_profile):
        """Test creating a basic researcher profile"""
        profile = engine.create_profile(**sample_profile)

        assert profile.profile_id == 1
        assert profile.name == sample_profile["name"]
        assert profile.institution == sample_profile["institution"]
        assert len(profile.skills) == 3
        assert profile.h_index == 25
        assert profile.looking_for == "complementary"
        assert len(profile.hypotheses) == 0
        assert len(profile.anti_hypotheses) == 0

    def test_create_profile_incremental_ids(self, engine, sample_profile, sample_profile_2):
        """Test that profile IDs increment correctly"""
        profile1 = engine.create_profile(**sample_profile)
        profile2 = engine.create_profile(**sample_profile_2)

        assert profile1.profile_id == 1
        assert profile2.profile_id == 2

    def test_create_profile_normalizes_skills(self, engine):
        """Test that skills are normalized to lowercase and stripped"""
        profile = engine.create_profile(
            name="Dr. Test",
            institution="TestU",
            skills=["Machine Learning", " Deep Learning ", "CV"],
            interests=["AI"],
            h_index=10
        )

        assert "machine learning" in profile.skills
        assert "deep learning" in profile.skills
        assert "cv" in profile.skills

    def test_create_profile_normalizes_interests(self, engine):
        """Test that interests are normalized to lowercase and stripped"""
        profile = engine.create_profile(
            name="Dr. Test",
            institution="TestU",
            skills=["ML"],
            interests=["Medical Imaging", " Cancer Detection "],
            h_index=10
        )

        assert "medical imaging" in profile.interests
        assert "cancer detection" in profile.interests

    def test_create_profile_with_default_looking_for(self, engine):
        """Test default value for looking_for parameter"""
        profile = engine.create_profile(
            name="Dr. Default",
            institution="TestU",
            skills=["ML"],
            interests=["AI"],
            h_index=5
        )

        assert profile.looking_for == "complementary"

    def test_profile_persistence(self, engine, temp_dir, sample_profile):
        """Test that profiles are saved and can be reloaded"""
        profile = engine.create_profile(**sample_profile)

        # Create new engine instance with same files
        new_engine = MatchingEngine(
            profiles_file=str(temp_dir / "profiles.json"),
            hypotheses_file=str(temp_dir / "hypotheses.json")
        )

        assert profile.profile_id in new_engine.profiles
        assert new_engine.profiles[profile.profile_id].name == sample_profile["name"]

    def test_profile_has_timestamp(self, engine, sample_profile):
        """Test that profiles have creation timestamp"""
        profile = engine.create_profile(**sample_profile)

        assert profile.created_at is not None
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(profile.created_at)


# Test Class: Hypothesis Management
class TestHypothesisManagement:
    """Tests for creating and managing research hypotheses"""

    def test_add_hypothesis_basic(self, engine, sample_profile, sample_hypothesis):
        """Test adding a basic hypothesis"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        assert hypothesis.hypothesis_id == 1
        assert hypothesis.text == sample_hypothesis["text"]
        assert hypothesis.field == sample_hypothesis["field"].lower()
        assert hypothesis.author_id == profile.profile_id
        assert hypothesis.support_count == 0
        assert hypothesis.oppose_count == 0
        assert hypothesis.debate_count == 0

    def test_add_hypothesis_incremental_ids(self, engine, sample_profile):
        """Test that hypothesis IDs increment correctly"""
        profile = engine.create_profile(**sample_profile)

        hyp1 = engine.add_hypothesis(
            author_id=profile.profile_id,
            text="Hypothesis 1",
            field="AI"
        )
        hyp2 = engine.add_hypothesis(
            author_id=profile.profile_id,
            text="Hypothesis 2",
            field="ML"
        )

        assert hyp1.hypothesis_id == 1
        assert hyp2.hypothesis_id == 2

    def test_add_hypothesis_normalizes_field(self, engine, sample_profile):
        """Test that field is normalized to lowercase"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            text="Test hypothesis",
            field="Medical Imaging"
        )

        assert hypothesis.field == "medical imaging"

    def test_add_hypothesis_with_tags(self, engine, sample_profile, sample_hypothesis):
        """Test adding hypothesis with tags"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        assert len(hypothesis.tags) == 3
        assert "AI" in hypothesis.tags
        assert "healthcare" in hypothesis.tags

    def test_add_hypothesis_without_tags(self, engine, sample_profile):
        """Test adding hypothesis without tags"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            text="Test hypothesis",
            field="AI"
        )

        assert hypothesis.tags == []

    def test_add_hypothesis_adds_to_author_profile(self, engine, sample_profile, sample_hypothesis):
        """Test that hypothesis is added to author's supported hypotheses"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        assert hypothesis.text in engine.profiles[profile.profile_id].hypotheses

    def test_hypothesis_persistence(self, engine, temp_dir, sample_profile, sample_hypothesis):
        """Test that hypotheses are saved and can be reloaded"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        # Create new engine instance with same files
        new_engine = MatchingEngine(
            profiles_file=str(temp_dir / "profiles.json"),
            hypotheses_file=str(temp_dir / "hypotheses.json")
        )

        assert hypothesis.hypothesis_id in new_engine.hypotheses
        assert new_engine.hypotheses[hypothesis.hypothesis_id].text == sample_hypothesis["text"]

    def test_hypothesis_has_timestamp(self, engine, sample_profile, sample_hypothesis):
        """Test that hypotheses have creation timestamp"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        assert hypothesis.created_at is not None
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(hypothesis.created_at)


# Test Class: Swipe Functionality
class TestSwipeFunctionality:
    """Tests for swiping on hypotheses"""

    def test_swipe_support(self, engine, sample_profile, sample_profile_2, sample_hypothesis):
        """Test supporting a hypothesis"""
        author = engine.create_profile(**sample_profile)
        swiper = engine.create_profile(**sample_profile_2)
        hypothesis = engine.add_hypothesis(
            author_id=author.profile_id,
            **sample_hypothesis
        )

        engine.swipe(swiper.profile_id, hypothesis.hypothesis_id, "SUPPORT")

        assert engine.hypotheses[hypothesis.hypothesis_id].support_count == 1
        assert hypothesis.text in engine.profiles[swiper.profile_id].hypotheses

    def test_swipe_oppose(self, engine, sample_profile, sample_profile_2, sample_hypothesis):
        """Test opposing a hypothesis"""
        author = engine.create_profile(**sample_profile)
        swiper = engine.create_profile(**sample_profile_2)
        hypothesis = engine.add_hypothesis(
            author_id=author.profile_id,
            **sample_hypothesis
        )

        engine.swipe(swiper.profile_id, hypothesis.hypothesis_id, "OPPOSE")

        assert engine.hypotheses[hypothesis.hypothesis_id].oppose_count == 1
        assert hypothesis.text in engine.profiles[swiper.profile_id].anti_hypotheses

    def test_swipe_debate(self, engine, sample_profile, sample_profile_2, sample_hypothesis):
        """Test debating a hypothesis"""
        author = engine.create_profile(**sample_profile)
        swiper = engine.create_profile(**sample_profile_2)
        hypothesis = engine.add_hypothesis(
            author_id=author.profile_id,
            **sample_hypothesis
        )

        engine.swipe(swiper.profile_id, hypothesis.hypothesis_id, "DEBATE")

        assert engine.hypotheses[hypothesis.hypothesis_id].debate_count == 1

    def test_swipe_invalid_profile(self, engine, sample_profile, sample_hypothesis):
        """Test swiping with invalid profile ID"""
        profile = engine.create_profile(**sample_profile)
        hypothesis = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        with pytest.raises(ValueError, match="Invalid profile or hypothesis ID"):
            engine.swipe(999, hypothesis.hypothesis_id, "SUPPORT")

    def test_swipe_invalid_hypothesis(self, engine, sample_profile):
        """Test swiping with invalid hypothesis ID"""
        profile = engine.create_profile(**sample_profile)

        with pytest.raises(ValueError, match="Invalid profile or hypothesis ID"):
            engine.swipe(profile.profile_id, 999, "SUPPORT")

    def test_swipe_multiple_times(self, engine, sample_profile, sample_profile_2, sample_hypothesis):
        """Test that multiple swipes increment counters"""
        author = engine.create_profile(**sample_profile)
        swiper = engine.create_profile(**sample_profile_2)
        hypothesis = engine.add_hypothesis(
            author_id=author.profile_id,
            **sample_hypothesis
        )

        # Support twice
        engine.swipe(swiper.profile_id, hypothesis.hypothesis_id, "SUPPORT")
        engine.swipe(swiper.profile_id, hypothesis.hypothesis_id, "SUPPORT")

        assert engine.hypotheses[hypothesis.hypothesis_id].support_count == 2


# Test Class: Match Finding
class TestMatchFinding:
    """Tests for finding matches between researchers"""

    def test_find_matches_complementary(self, engine, sample_profile, sample_profile_2):
        """Test finding complementary matches"""
        profile1 = engine.create_profile(**sample_profile)
        profile2 = engine.create_profile(**sample_profile_2)

        matches = engine.find_matches(profile1.profile_id, mode="complementary", limit=10)

        assert len(matches) > 0
        assert matches[0].researcher_1 == profile1.profile_id
        assert matches[0].researcher_2 == profile2.profile_id
        assert matches[0].match_type == "complementary"
        assert matches[0].compatibility_score > 0

    def test_find_matches_similar(self, engine):
        """Test finding similar matches"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML", "CV", "NLP"],
            interests=["AI"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["ML", "CV", "DL"],
            interests=["AI"],
            h_index=22
        )

        matches = engine.find_matches(profile1.profile_id, mode="similar", limit=10)

        assert len(matches) > 0
        assert matches[0].match_type == "similar"

    def test_find_matches_adversarial(self, engine, sample_hypothesis):
        """Test finding adversarial matches"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats"],
            interests=["AI"],
            h_index=25
        )

        # Add hypothesis and create opposing views
        hyp = engine.add_hypothesis(
            author_id=profile1.profile_id,
            **sample_hypothesis
        )
        engine.swipe(profile2.profile_id, hyp.hypothesis_id, "OPPOSE")

        matches = engine.find_matches(profile1.profile_id, mode="adversarial", limit=10)

        assert len(matches) > 0
        assert matches[0].match_type == "adversarial"

    def test_find_matches_invalid_profile(self, engine):
        """Test finding matches with invalid profile ID"""
        with pytest.raises(ValueError, match="Profile not found"):
            engine.find_matches(999, mode="complementary")

    def test_find_matches_excludes_self(self, engine, sample_profile):
        """Test that matches don't include the researcher themselves"""
        profile = engine.create_profile(**sample_profile)
        matches = engine.find_matches(profile.profile_id, mode="complementary")

        for match in matches:
            assert match.researcher_2 != profile.profile_id

    def test_find_matches_respects_limit(self, engine):
        """Test that match limit is respected"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=20
        )

        # Create 15 other profiles
        for i in range(15):
            engine.create_profile(
                name=f"Dr. {i}",
                institution="TestU",
                skills=["ML"],
                interests=["AI"],
                h_index=20
            )

        matches = engine.find_matches(profile1.profile_id, mode="complementary", limit=5)

        assert len(matches) <= 5

    def test_find_matches_minimum_threshold(self, engine):
        """Test that matches below threshold are excluded"""
        # Create two profiles with no overlap
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=10
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["chemistry"],
            interests=["biology"],
            h_index=50
        )

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        # Should have low or no matches due to minimum threshold
        if matches:
            assert matches[0].compatibility_score > 20

    def test_match_shared_interests(self, engine):
        """Test that shared interests are correctly identified"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["medical imaging", "cancer detection"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats"],
            interests=["medical imaging", "drug development"],
            h_index=25
        )

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        assert len(matches) > 0
        assert "medical imaging" in matches[0].shared_interests

    def test_match_has_timestamp(self, engine, sample_profile, sample_profile_2):
        """Test that matches have timestamp"""
        profile1 = engine.create_profile(**sample_profile)
        profile2 = engine.create_profile(**sample_profile_2)

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        assert matches[0].timestamp is not None
        datetime.fromisoformat(matches[0].timestamp)


# Test Class: Scoring Algorithms
class TestScoringAlgorithms:
    """Tests for match scoring and collaboration potential"""

    def test_complementary_scoring_rewards_different_skills(self, engine):
        """Test that complementary mode rewards different skills"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML", "CV"],
            interests=["AI"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats", "clinical"],
            interests=["AI"],
            h_index=20
        )

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        # Should have high score due to complementary skills
        assert matches[0].compatibility_score > 30

    def test_similar_scoring_rewards_overlapping_skills(self, engine):
        """Test that similar mode rewards skill overlap"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML", "CV", "DL"],
            interests=["AI"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["ML", "CV", "NLP"],
            interests=["AI"],
            h_index=20
        )

        matches = engine.find_matches(profile1.profile_id, mode="similar")

        # Should have high score due to overlapping skills (2 skills + 1 interest + h-index)
        assert matches[0].compatibility_score > 30

    def test_h_index_similarity_bonus(self, engine):
        """Test that similar h-index gives bonus points"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=25
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats"],
            interests=["AI"],
            h_index=27
        )

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        # Check that reasons mention h-index
        reasons_text = " ".join(matches[0].reasons)
        assert "h-index" in reasons_text.lower()

    def test_hypothesis_alignment_calculation(self, engine, sample_hypothesis):
        """Test hypothesis alignment calculation"""
        profile1 = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=20
        )
        profile2 = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats"],
            interests=["AI"],
            h_index=20
        )

        # Create and support hypothesis
        hyp = engine.add_hypothesis(
            author_id=profile1.profile_id,
            **sample_hypothesis
        )
        engine.swipe(profile2.profile_id, hyp.hypothesis_id, "SUPPORT")

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        # Should have hypothesis alignment
        assert len(matches) > 0
        if matches[0].hypothesis_alignment:
            assert "agree" in matches[0].hypothesis_alignment.values()

    def test_collaboration_potential_calculation(self, engine, sample_profile, sample_profile_2):
        """Test collaboration potential is calculated"""
        profile1 = engine.create_profile(**sample_profile)
        profile2 = engine.create_profile(**sample_profile_2)

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        assert matches[0].collaboration_potential > 0
        assert matches[0].collaboration_potential <= 100

    def test_match_reasons_provided(self, engine, sample_profile, sample_profile_2):
        """Test that matches include reasons"""
        profile1 = engine.create_profile(**sample_profile)
        profile2 = engine.create_profile(**sample_profile_2)

        matches = engine.find_matches(profile1.profile_id, mode="complementary")

        assert len(matches[0].reasons) > 0
        assert isinstance(matches[0].reasons[0], str)


# Test Class: Edge Cases
class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    def test_empty_engine(self, engine):
        """Test operations on empty engine"""
        assert len(engine.profiles) == 0
        assert len(engine.hypotheses) == 0
        assert len(engine.matches) == 0

    def test_profile_with_empty_skills(self, engine):
        """Test creating profile with empty skills list"""
        profile = engine.create_profile(
            name="Dr. Test",
            institution="TestU",
            skills=[],
            interests=["AI"],
            h_index=10
        )

        assert len(profile.skills) == 0

    def test_profile_with_empty_interests(self, engine):
        """Test creating profile with empty interests list"""
        profile = engine.create_profile(
            name="Dr. Test",
            institution="TestU",
            skills=["ML"],
            interests=[],
            h_index=10
        )

        assert len(profile.interests) == 0

    def test_hypothesis_author_not_exists(self, engine, sample_hypothesis):
        """Test adding hypothesis with non-existent author"""
        # Should still create the hypothesis but not add to profile
        hypothesis = engine.add_hypothesis(
            author_id=999,
            **sample_hypothesis
        )

        assert hypothesis.hypothesis_id == 1
        assert hypothesis.author_id == 999

    def test_no_matches_found(self, engine, sample_profile):
        """Test when no matches are found"""
        profile = engine.create_profile(**sample_profile)
        matches = engine.find_matches(profile.profile_id, mode="complementary")

        assert len(matches) == 0

    def test_load_from_nonexistent_files(self, temp_dir):
        """Test loading when files don't exist"""
        engine = MatchingEngine(
            profiles_file=str(temp_dir / "new_profiles.json"),
            hypotheses_file=str(temp_dir / "new_hypotheses.json")
        )

        assert len(engine.profiles) == 0
        assert len(engine.hypotheses) == 0


# Test Class: Integration Workflows
class TestIntegrationWorkflows:
    """Tests for complete user workflows"""

    def test_complete_matching_workflow(self, engine, sample_hypothesis):
        """Test complete workflow: create profiles, hypotheses, swipe, match"""
        # Create profiles
        alice = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML", "CV"],
            interests=["medical imaging"],
            h_index=25
        )
        bob = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats", "clinical"],
            interests=["medical imaging"],
            h_index=30
        )

        # Add hypothesis
        hyp = engine.add_hypothesis(
            author_id=alice.profile_id,
            **sample_hypothesis
        )

        # Bob supports it
        engine.swipe(bob.profile_id, hyp.hypothesis_id, "SUPPORT")

        # Find matches
        matches = engine.find_matches(alice.profile_id, mode="complementary")

        assert len(matches) > 0
        assert matches[0].researcher_2 == bob.profile_id
        assert matches[0].compatibility_score > 0

    def test_adversarial_matching_workflow(self, engine, sample_hypothesis):
        """Test workflow for finding adversarial collaborators"""
        # Create profiles with similar interests
        alice = engine.create_profile(
            name="Dr. Alice",
            institution="MIT",
            skills=["ML"],
            interests=["AI"],
            h_index=20
        )
        bob = engine.create_profile(
            name="Dr. Bob",
            institution="Stanford",
            skills=["stats"],
            interests=["AI"],
            h_index=22
        )

        # Alice creates hypothesis, Bob opposes
        hyp = engine.add_hypothesis(
            author_id=alice.profile_id,
            **sample_hypothesis
        )
        engine.swipe(bob.profile_id, hyp.hypothesis_id, "OPPOSE")

        # Find adversarial matches
        matches = engine.find_matches(alice.profile_id, mode="adversarial")

        assert len(matches) > 0
        assert matches[0].match_type == "adversarial"

    def test_multi_researcher_ecosystem(self, engine):
        """Test matching in ecosystem with multiple researchers"""
        # Create 5 diverse researchers
        profiles = []
        for i in range(5):
            profile = engine.create_profile(
                name=f"Dr. Researcher {i}",
                institution=f"University {i}",
                skills=[f"skill_{i}", f"skill_{i+1}"],
                interests=["AI", "research"],
                h_index=10 + i * 5
            )
            profiles.append(profile)

        # Each should find matches
        for profile in profiles:
            matches = engine.find_matches(profile.profile_id, mode="complementary")
            # Should find some matches (at least 1)
            assert len(matches) >= 1

    def test_persistence_across_sessions(self, engine, temp_dir, sample_profile, sample_hypothesis):
        """Test data persists across engine restarts"""
        # Session 1: Create data
        profile = engine.create_profile(**sample_profile)
        hyp = engine.add_hypothesis(
            author_id=profile.profile_id,
            **sample_hypothesis
        )

        # Session 2: Load and verify
        engine2 = MatchingEngine(
            profiles_file=str(temp_dir / "profiles.json"),
            hypotheses_file=str(temp_dir / "hypotheses.json")
        )

        assert profile.profile_id in engine2.profiles
        assert hyp.hypothesis_id in engine2.hypotheses
        assert engine2.profiles[profile.profile_id].name == sample_profile["name"]
        assert engine2.hypotheses[hyp.hypothesis_id].text == sample_hypothesis["text"]
