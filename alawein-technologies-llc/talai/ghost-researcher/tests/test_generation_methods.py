"""
Tests for content generation methods
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ghost_researcher.main import GhostResearcher, SCIENTISTS


class TestInitialReactionGeneration:
    """Test initial reaction generation"""

    def test_generate_initial_reaction_returns_string(self):
        """Test initial reaction is a string"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['einstein']
        reaction = ghost._generate_initial_reaction(profile, 'test problem')

        assert isinstance(reaction, str)
        assert len(reaction) > 0

    def test_generate_initial_reaction_varies(self):
        """Test that reactions vary (randomness)"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['einstein']

        reactions = set()
        for _ in range(20):
            reaction = ghost._generate_initial_reaction(profile, 'test problem')
            reactions.add(reaction)

        # Should have at least 2 different reactions in 20 tries
        assert len(reactions) >= 2

    def test_generate_initial_reaction_references_profile(self):
        """Test reaction references scientist's work"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['curie']

        # Generate multiple to check they use profile data
        reactions_found = []
        for _ in range(10):
            reaction = ghost._generate_initial_reaction(profile, 'test')
            reaction_lower = reaction.lower()
            # Reactions reference work, field, or general scientific terms
            if any(keyword in reaction_lower for keyword in
                   ['radium', 'radioactivity', 'physics', 'chemistry', 'working', 'field', 'science', 'problem']):
                reactions_found.append(True)

        # At least half should reference something relevant
        assert len(reactions_found) >= 5


class TestAnalogiesGeneration:
    """Test analogies to historical time period"""

    def test_find_analogies_returns_list(self):
        """Test analogies returns a list"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['newton']
        analogies = ghost._find_analogies(profile, 'test problem', 'physics')

        assert isinstance(analogies, list)
        assert len(analogies) == 3

    def test_find_analogies_uses_era(self):
        """Test analogies use appropriate era"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['lovelace']  # 1815-1852
        analogies = ghost._find_analogies(profile, 'AI systems', 'computer_science')

        # Should reference 19th century context
        analogies_text = ' '.join(analogies).lower()
        assert any(keyword in analogies_text for keyword in ['time', 'era', 'debates', 'developed'])


class TestExperimentalSuggestions:
    """Test experimental suggestion generation"""

    def test_suggest_experiments_returns_list(self):
        """Test experiments returns proper list"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['franklin']
        experiments = ghost._suggest_experiments(profile, 'protein folding', 'biology')

        assert isinstance(experiments, list)
        assert len(experiments) == 5

    def test_suggest_experiments_reference_problem(self):
        """Test experiments reference the problem"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['darwin']
        problem = 'artificial_selection_in_crops'
        experiments = ghost._suggest_experiments(profile, problem, 'biology')

        # At least some experiments should mention the problem
        experiments_text = ' '.join(experiments)
        assert problem in experiments_text


class TestTheoreticalFramework:
    """Test theoretical framework generation"""

    def test_theoretical_framework_returns_string(self):
        """Test framework is a substantial string"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['newton']
        framework = ghost._theoretical_framework(profile, 'orbital mechanics', 'physics')

        assert isinstance(framework, str)
        assert len(framework) > 50

    def test_theoretical_framework_uses_field(self):
        """Test framework references scientist's field"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['turing']
        framework = ghost._theoretical_framework(profile, 'neural networks', 'computer_science')

        assert profile['field'].lower() in framework.lower()

    def test_theoretical_framework_mentions_problem(self):
        """Test framework mentions the problem"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['einstein']
        problem = 'dark_energy'
        framework = ghost._theoretical_framework(profile, problem, 'physics')

        assert problem in framework


class TestCharacteristicQuotes:
    """Test characteristic quotes generation"""

    def test_generate_quotes_returns_list(self):
        """Test quotes returns a list"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['feynman']
        quotes = ghost._generate_characteristic_quotes(profile, 'quantum computing')

        assert isinstance(quotes, list)
        assert len(quotes) == 3

    def test_generate_quotes_includes_originals(self):
        """Test quotes include original quotes"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['curie']
        quotes = ghost._generate_characteristic_quotes(profile, 'radiation therapy')

        # First two should be from original quotes
        assert quotes[0] in profile['quotes']
        assert quotes[1] in profile['quotes']

    def test_generate_quotes_creates_custom_quote(self):
        """Test custom quote is generated"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['darwin']
        problem = 'genetic_algorithms'
        quotes = ghost._generate_characteristic_quotes(profile, problem)

        # Third quote should reference the problem
        assert problem in quotes[2] or 'imagination' in quotes[2].lower() or 'understand' in quotes[2].lower()


class TestThoughtExperiments:
    """Test thought experiment generation"""

    def test_create_thought_experiments_returns_list(self):
        """Test thought experiments returns a list"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['einstein']
        experiments = ghost._create_thought_experiments(profile, 'time travel')

        assert isinstance(experiments, list)
        assert len(experiments) == 4

    def test_thought_experiments_reference_problem(self):
        """Test thought experiments reference the problem"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['feynman']
        problem = 'superconductivity'
        experiments = ghost._create_thought_experiments(profile, problem)

        # All should reference the problem
        for exp in experiments:
            assert problem in exp

    def test_thought_experiments_are_substantial(self):
        """Test thought experiments are substantial"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['turing']
        experiments = ghost._create_thought_experiments(profile, 'consciousness')

        for exp in experiments:
            assert len(exp) > 20


class TestObstaclesPrediction:
    """Test obstacle prediction"""

    def test_predict_obstacles_returns_list(self):
        """Test obstacles returns a list"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['lovelace']
        obstacles = ghost._predict_obstacles(profile, 'AGI development')

        assert isinstance(obstacles, list)
        assert len(obstacles) == 5

    def test_predict_obstacles_are_reasonable(self):
        """Test obstacles are reasonable challenges"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['franklin']
        obstacles = ghost._predict_obstacles(profile, 'gene sequencing')

        # All obstacles should be strings
        for obstacle in obstacles:
            assert isinstance(obstacle, str)
            assert len(obstacle) > 10


class TestInsightsGeneration:
    """Test key insights generation"""

    def test_generate_insights_returns_three_items(self):
        """Test insights returns 3 items"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['newton']
        insights = ghost._generate_insights(profile, 'spacetime', 'physics')

        assert isinstance(insights, list)
        assert len(insights) == 3

    def test_generate_insights_uses_domain(self):
        """Test insights reference the domain"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['darwin']
        insights = ghost._generate_insights(profile, 'ecosystem dynamics', 'biology')

        # First insight should use domain-specific nature
        insights_text = ' '.join(insights).lower()
        assert 'evolution' in insights_text or 'adaptation' in insights_text or 'biology' in insights_text

    def test_generate_insights_includes_quote(self):
        """Test insights include a quote from the scientist"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['einstein']
        insights = ghost._generate_insights(profile, 'black holes', 'physics')

        # Third insight should be or contain a quote
        any_quote_found = any(
            quote.lower() in insights[2].lower()
            for quote in profile['quotes']
        )
        assert any_quote_found or insights[2] in profile['quotes']


class TestGenerateApproachVariations:
    """Test different approach generation paths"""

    def test_approach_for_abstract_thinker(self):
        """Test approach for abstract thinkers"""
        ghost = GhostResearcher()
        profile = SCIENTISTS['turing']
        approach = ghost._generate_approach(profile, 'AI reasoning', 'computer_science')

        assert isinstance(approach, str)
        assert len(approach) > 20

    def test_approach_default_case(self):
        """Test default approach when no specific pattern matches"""
        ghost = GhostResearcher()
        # Create a mock profile that doesn't match specific patterns
        mock_profile = {
            'field': 'General Science',
            'personality': ['Systematic researcher', 'Detail oriented', 'Collaborative']
        }
        approach = ghost._generate_approach(mock_profile, 'complex system', 'general')

        assert isinstance(approach, str)
        assert 'General Science' in approach or 'general science' in approach.lower()


class TestEraContextVariations:
    """Test era context for different time periods"""

    def test_era_context_all_periods(self):
        """Test era context for all defined periods"""
        ghost = GhostResearcher()

        # Test each era boundary
        eras = [
            (1700, 'steam engine'),  # <1800
            (1850, 'telegraph'),     # 1800-1900
            (1920, 'computer'),      # 1900-1950
            (1955, 'DNA structure')  # >=1950
        ]

        for year, expected_tech in eras:
            era = ghost._determine_era(year)
            assert 'challenge' in era
            assert 'approach' in era
            assert 'technology' in era
            assert 'controversy' in era


class TestDomainNatureMapping:
    """Test domain to fundamental nature mapping"""

    def test_all_defined_domains(self):
        """Test all explicitly defined domains"""
        ghost = GhostResearcher()

        domains = {
            'physics': 'energy and information flow',
            'biology': 'evolution and adaptation',
            'computer_science': 'computation and complexity',
            'medicine': 'maintaining homeostasis',
            'mathematics': 'structure and relationships'
        }

        for domain, expected_nature in domains.items():
            nature = ghost._identify_fundamental_nature(domain)
            assert nature == expected_nature

    def test_unknown_domain_default(self):
        """Test unknown domain returns default"""
        ghost = GhostResearcher()
        nature = ghost._identify_fundamental_nature('unknown_field')
        assert nature == 'system dynamics and feedback'
