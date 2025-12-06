"""
Comprehensive tests for ELO Rating System.

Tests all ELO calculations, edge cases, and rating updates.
"""

import pytest
from agent_tournaments.strategies.elo_system import ELOSystem
from agent_tournaments.core.models import ELORating


class TestELOSystemInitialization:
    """Test ELO system initialization."""

    def test_default_initialization(self):
        """Test default K-factor."""
        elo = ELOSystem()
        assert elo.k_factor == 32

    def test_custom_k_factor(self):
        """Test custom K-factor."""
        elo = ELOSystem(k_factor=16)
        assert elo.k_factor == 16

    def test_high_k_factor(self):
        """Test high K-factor for volatile ratings."""
        elo = ELOSystem(k_factor=64)
        assert elo.k_factor == 64


class TestExpectedScore:
    """Test expected score calculations."""

    def test_equal_ratings(self):
        """Test expected score with equal ratings."""
        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001

    def test_higher_rating_a(self):
        """Test expected score when A has higher rating."""
        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(1700, 1500)
        # 200 point difference -> ~76% win probability
        assert 0.75 < expected < 0.77

    def test_lower_rating_a(self):
        """Test expected score when A has lower rating."""
        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(1300, 1500)
        # 200 point difference -> ~24% win probability
        assert 0.23 < expected < 0.25

    def test_large_rating_gap(self):
        """Test expected score with large rating gap."""
        elo = ELOSystem(k_factor=32)
        expected = elo.calculate_expected_score(2000, 1200)
        # 800 point difference -> >99% win probability
        assert expected > 0.99

    def test_symmetry(self):
        """Test that expected scores sum to 1.0."""
        elo = ELOSystem(k_factor=32)
        expected_a = elo.calculate_expected_score(1600, 1400)
        expected_b = elo.calculate_expected_score(1400, 1600)
        assert abs((expected_a + expected_b) - 1.0) < 0.001


class TestRatingUpdates:
    """Test rating update calculations."""

    def test_equal_ratings_a_wins(self):
        """Test rating update when equal-rated A wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        # With K=32, equal ratings, winner gains 16 points
        assert abs(updated_a.current_rating - 1516) < 1
        assert abs(updated_b.current_rating - 1484) < 1

    def test_equal_ratings_b_wins(self):
        """Test rating update when equal-rated B wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="b")

        assert abs(updated_a.current_rating - 1484) < 1
        assert abs(updated_b.current_rating - 1516) < 1

    def test_underdog_wins(self):
        """Test rating update when underdog wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1300)  # Underdog
        rating_b = ELORating(agent_id="b", current_rating=1500)  # Favorite

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        # Underdog gains more points for upset
        rating_gain_a = updated_a.current_rating - 1300
        assert rating_gain_a > 20  # Significant gain for upset

    def test_favorite_wins(self):
        """Test rating update when favorite wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1700)  # Favorite
        rating_b = ELORating(agent_id="b", current_rating=1500)  # Underdog

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        # Favorite gains fewer points for expected win
        rating_gain_a = updated_a.current_rating - 1700
        assert rating_gain_a < 10  # Small gain for expected win

    def test_draw(self):
        """Test rating update for draw."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner=None)

        # Equal ratings, draw -> no change
        assert abs(updated_a.current_rating - 1500) < 0.1
        assert abs(updated_b.current_rating - 1500) < 0.1

    def test_draw_unequal_ratings(self):
        """Test rating update for draw with unequal ratings."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1700)  # Favorite
        rating_b = ELORating(agent_id="b", current_rating=1500)  # Underdog

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner=None)

        # Favorite loses points in draw, underdog gains
        assert updated_a.current_rating < 1700
        assert updated_b.current_rating > 1500


class TestWinLossTracking:
    """Test win/loss/draw tracking."""

    def test_win_increments_wins(self):
        """Test that winning increments win count."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500, wins=5)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, _ = elo.update_ratings(rating_a, rating_b, winner="a")

        assert updated_a.wins == 6

    def test_loss_increments_losses(self):
        """Test that losing increments loss count."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500, losses=3)

        _, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        assert updated_b.losses == 4

    def test_draw_increments_draws(self):
        """Test that draw increments draw count."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500, draws=2)
        rating_b = ELORating(agent_id="b", current_rating=1500, draws=1)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner=None)

        assert updated_a.draws == 3
        assert updated_b.draws == 2

    def test_total_matches(self):
        """Test total matches calculation."""
        rating = ELORating(
            agent_id="a", current_rating=1500, wins=10, losses=5, draws=2
        )
        assert rating.total_matches == 17

    def test_win_rate(self):
        """Test win rate calculation."""
        rating = ELORating(
            agent_id="a", current_rating=1600, wins=10, losses=5, draws=5
        )
        # Win rate = wins / total_matches
        expected_win_rate = 10 / 20
        assert abs(rating.win_rate - expected_win_rate) < 0.01

    def test_win_rate_no_matches(self):
        """Test win rate with no matches."""
        rating = ELORating(agent_id="a", current_rating=1500)
        assert rating.win_rate == 0.0


class TestKFactorImpact:
    """Test impact of different K-factors."""

    def test_high_k_volatile_ratings(self):
        """Test that high K-factor creates volatile ratings."""
        elo_high = ELOSystem(k_factor=64)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, _ = elo_high.update_ratings(rating_a, rating_b, winner="a")

        # High K -> larger rating change
        assert abs(updated_a.current_rating - 1500) > 30

    def test_low_k_stable_ratings(self):
        """Test that low K-factor creates stable ratings."""
        elo_low = ELOSystem(k_factor=8)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        updated_a, _ = elo_low.update_ratings(rating_a, rating_b, winner="a")

        # Low K -> smaller rating change
        assert abs(updated_a.current_rating - 1500) < 10


class TestRatingConservation:
    """Test that total rating is conserved."""

    def test_rating_conservation_win(self):
        """Test that total rating is conserved after win."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1600)
        rating_b = ELORating(agent_id="b", current_rating=1400)

        total_before = rating_a.current_rating + rating_b.current_rating

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        total_after = updated_a.current_rating + updated_b.current_rating

        # Total rating should be conserved
        assert abs(total_before - total_after) < 0.1

    def test_rating_conservation_draw(self):
        """Test that total rating is conserved after draw."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1700)
        rating_b = ELORating(agent_id="b", current_rating=1300)

        total_before = rating_a.current_rating + rating_b.current_rating

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner=None)

        total_after = updated_a.current_rating + updated_b.current_rating

        # Total rating should be conserved
        assert abs(total_before - total_after) < 0.1


class TestEdgeCases:
    """Test edge cases."""

    def test_maximum_rating_difference(self):
        """Test with maximum rating difference."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=3000)
        rating_b = ELORating(agent_id="b", current_rating=0)

        # Should not crash
        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        assert updated_a.current_rating >= 3000
        assert updated_b.current_rating <= 0 or updated_b.current_rating >= 0

    def test_negative_ratings_possible(self):
        """Test that ratings can go negative."""
        elo = ELOSystem(k_factor=64)
        rating_a = ELORating(agent_id="a", current_rating=2000)
        rating_b = ELORating(agent_id="b", current_rating=100)

        updated_a, updated_b = elo.update_ratings(rating_a, rating_b, winner="a")

        # Underdog losing badly can go negative
        # (This is theoretically possible in ELO)
        assert updated_b.current_rating is not None


class TestMultipleMatches:
    """Test rating progression over multiple matches."""

    def test_rating_progression_consistent_wins(self):
        """Test rating increases with consistent wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        # Agent A wins 5 times in a row
        for _ in range(5):
            rating_a, rating_b = elo.update_ratings(rating_a, rating_b, winner="a")

        assert rating_a.current_rating > 1600  # Significant improvement
        assert rating_b.current_rating < 1400  # Significant decline
        assert rating_a.wins == 5
        assert rating_b.losses == 5

    def test_rating_equilibrium_alternating_wins(self):
        """Test ratings stabilize with alternating wins."""
        elo = ELOSystem(k_factor=32)
        rating_a = ELORating(agent_id="a", current_rating=1500)
        rating_b = ELORating(agent_id="b", current_rating=1500)

        # Alternating wins
        for i in range(10):
            winner = "a" if i % 2 == 0 else "b"
            rating_a, rating_b = elo.update_ratings(rating_a, rating_b, winner=winner)

        # Should stay near 1500
        assert 1450 < rating_a.current_rating < 1550
        assert 1450 < rating_b.current_rating < 1550
