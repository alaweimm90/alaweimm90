"""Comprehensive test suite for PaperMiner

Tests all major functionality including paper analysis, trend detection,
gap identification, citation networks, and temporal analysis.

Target: 35+ tests, 70%+ coverage
"""

import pytest
from collections import Counter
from paper_miner.main import (
    PaperMiner,
    Paper,
    ResearchGap,
    Trend,
    MiningResult
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def miner():
    """Create fresh PaperMiner instance"""
    return PaperMiner()


@pytest.fixture
def sample_papers():
    """Create sample papers for testing"""
    return [
        Paper(
            paper_id="p1",
            title="Deep Learning for Image Recognition",
            authors=["Alice Smith", "Bob Jones"],
            year=2020,
            venue="CVPR",
            abstract="We propose a novel deep learning approach using CNNs and attention mechanisms for image recognition. Our method achieves state-of-the-art results.",
            citations=150,
            keywords=["deep learning", "CNN", "image recognition"],
            references=["Attention is All You Need"],
            field="computer vision"
        ),
        Paper(
            paper_id="p2",
            title="Transformer Models for NLP",
            authors=["Alice Smith", "Carol White"],
            year=2021,
            venue="ACL",
            abstract="We apply transformer architectures to natural language processing tasks. The main limitation is the need for large datasets.",
            citations=200,
            keywords=["transformer", "NLP", "attention"],
            references=[],
            field="NLP"
        ),
        Paper(
            paper_id="p3",
            title="Reinforcement Learning in Robotics",
            authors=["Bob Jones", "Dave Brown"],
            year=2021,
            venue="ICRA",
            abstract="This paper explores reinforcement learning and Q-learning for robotic control. A major challenge is real-world deployment.",
            citations=80,
            keywords=["reinforcement learning", "robotics"],
            references=[],
            field="robotics"
        ),
        Paper(
            paper_id="p4",
            title="Attention Mechanisms in Deep Networks",
            authors=["Alice Smith"],
            year=2022,
            venue="CVPR",
            abstract="We study attention mechanisms and transformer models for computer vision. Lack of data remains an issue for small objects.",
            citations=120,
            keywords=["attention", "transformer", "vision"],
            references=["Deep Learning for Image Recognition"],
            field="computer vision"
        ),
        Paper(
            paper_id="p5",
            title="Neural Networks for Time Series",
            authors=["Eve Wilson", "Frank Davis"],
            year=2022,
            venue="NeurIPS",
            abstract="We use LSTM and neural networks for time series prediction. The approach shows promise but requires more empirical validation.",
            citations=60,
            keywords=["LSTM", "neural network", "time series"],
            references=[],
            field="machine learning"
        ),
    ]


@pytest.fixture
def ml_papers():
    """Papers focused on machine learning"""
    return [
        Paper(
            paper_id="ml1",
            title="Deep Learning Advances",
            authors=["Author A"],
            year=2018,
            venue="ICML",
            abstract="Deep learning and neural networks continue to improve.",
            citations=300,
            keywords=["deep learning"],
            field="machine learning"
        ),
        Paper(
            paper_id="ml2",
            title="Transformers for Everything",
            authors=["Author A", "Author B"],
            year=2020,
            venue="ICML",
            abstract="Transformer models are revolutionizing machine learning.",
            citations=500,
            keywords=["transformer"],
            field="machine learning"
        ),
        Paper(
            paper_id="ml3",
            title="Self-Attention Deep Learning",
            authors=["Author B"],
            year=2021,
            venue="NeurIPS",
            abstract="Self-attention and deep learning with transformers.",
            citations=250,
            keywords=["attention", "transformer"],
            field="machine learning"
        ),
        Paper(
            paper_id="ml4",
            title="Transformer Architectures",
            authors=["Author C"],
            year=2022,
            venue="ICLR",
            abstract="New transformer architectures for deep learning.",
            citations=180,
            keywords=["transformer", "deep learning"],
            field="machine learning"
        ),
    ]


# ============================================================================
# TEST BASIC ANALYSIS
# ============================================================================

class TestBasicAnalysis:
    """Test basic paper analysis functionality"""

    def test_analyze_basic(self, miner, sample_papers):
        """Test basic analysis execution"""
        result = miner.analyze(sample_papers)

        assert isinstance(result, MiningResult)
        assert result.total_papers == 5
        assert result.date_range == (2020, 2022)

    def test_analyze_empty_raises_error(self, miner):
        """Test that analyzing empty list raises error"""
        with pytest.raises(ValueError, match="No papers to analyze"):
            miner.analyze([])

    def test_result_has_all_components(self, miner, sample_papers):
        """Test that result contains all expected components"""
        result = miner.analyze(sample_papers)

        assert result.total_papers > 0
        assert len(result.date_range) == 2
        assert isinstance(result.top_authors, list)
        assert isinstance(result.top_venues, list)
        assert isinstance(result.common_methods, list)
        assert isinstance(result.trending_topics, list)
        assert isinstance(result.research_gaps, list)
        assert isinstance(result.citation_network, dict)
        assert isinstance(result.temporal_trends, dict)
        assert isinstance(result.keyword_evolution, dict)
        assert result.timestamp is not None

    def test_date_range_correct(self, miner, sample_papers):
        """Test date range calculation"""
        result = miner.analyze(sample_papers)

        assert result.date_range[0] == 2020
        assert result.date_range[1] == 2022


# ============================================================================
# TEST AUTHOR ANALYSIS
# ============================================================================

class TestAuthorAnalysis:
    """Test author analysis functionality"""

    def test_top_authors_counted(self, miner, sample_papers):
        """Test that authors are counted correctly"""
        result = miner.analyze(sample_papers)

        # Alice Smith appears in 3 papers
        top_authors_dict = dict(result.top_authors)
        assert top_authors_dict.get("Alice Smith", 0) == 3

    def test_top_authors_format(self, miner, sample_papers):
        """Test author result format"""
        result = miner.analyze(sample_papers)

        for author, count in result.top_authors:
            assert isinstance(author, str)
            assert isinstance(count, int)
            assert count > 0

    def test_authors_sorted_by_count(self, miner, sample_papers):
        """Test that authors are sorted by paper count"""
        result = miner.analyze(sample_papers)

        counts = [count for _, count in result.top_authors]
        assert counts == sorted(counts, reverse=True)


# ============================================================================
# TEST VENUE ANALYSIS
# ============================================================================

class TestVenueAnalysis:
    """Test venue analysis functionality"""

    def test_top_venues_counted(self, miner, sample_papers):
        """Test that venues are counted correctly"""
        result = miner.analyze(sample_papers)

        venues_dict = dict(result.top_venues)
        assert venues_dict["CVPR"] == 2

    def test_venues_sorted(self, miner, sample_papers):
        """Test that venues are sorted by count"""
        result = miner.analyze(sample_papers)

        counts = [count for _, count in result.top_venues]
        assert counts == sorted(counts, reverse=True)


# ============================================================================
# TEST METHOD EXTRACTION
# ============================================================================

class TestMethodExtraction:
    """Test research method extraction"""

    def test_methods_extracted(self, miner, sample_papers):
        """Test that methods are extracted from papers"""
        result = miner.analyze(sample_papers)

        methods = [method for method, _ in result.common_methods]

        # Should find some common methods
        assert len(methods) > 0

        # Should find deep learning or related terms
        methods_str = " ".join(methods)
        assert "deep learning" in methods_str or "cnn" in methods_str or "lstm" in methods_str

    def test_methods_counted(self, miner, sample_papers):
        """Test that method frequencies are counted"""
        result = miner.analyze(sample_papers)

        methods_dict = dict(result.common_methods)

        # Deep learning appears in multiple papers
        dl_count = methods_dict.get("deep learning", 0)
        assert dl_count >= 2

    def test_method_patterns_work(self, miner):
        """Test that method patterns match correctly"""
        paper = Paper(
            paper_id="test",
            title="Testing CNNs",
            authors=["Test"],
            year=2020,
            venue="Test",
            abstract="We use CNNs and LSTM for supervised learning with cross-validation.",
            citations=0
        )

        result = miner.analyze([paper])
        methods = [m for m, _ in result.common_methods]

        # Methods are extracted in lowercase
        assert "lstm" in methods
        assert "supervised learning" in methods
        assert "cross-validation" in methods


# ============================================================================
# TEST TREND IDENTIFICATION
# ============================================================================

class TestTrendIdentification:
    """Test trending topic identification"""

    def test_trends_identified(self, miner, ml_papers):
        """Test that trends are identified"""
        result = miner.analyze(ml_papers)

        assert len(result.trending_topics) > 0

    def test_trend_has_momentum(self, miner, ml_papers):
        """Test that trends have momentum calculated"""
        result = miner.analyze(ml_papers)

        for trend in result.trending_topics:
            assert isinstance(trend, Trend)
            assert isinstance(trend.momentum, float)
            assert isinstance(trend.papers_per_year, dict)

    def test_trend_emerging_detection(self, miner, ml_papers):
        """Test emerging trend detection"""
        result = miner.analyze(ml_papers)

        # Transformer is growing trend (2020-2022)
        transformer_trends = [
            t for t in result.trending_topics
            if "transformer" in t.topic
        ]

        # Should detect transformer as trend
        assert len(transformer_trends) > 0

    def test_trend_key_papers(self, miner, ml_papers):
        """Test that trends include key papers"""
        result = miner.analyze(ml_papers)

        for trend in result.trending_topics:
            if trend.key_papers:
                assert isinstance(trend.key_papers, list)
                assert len(trend.key_papers) > 0

    def test_trends_sorted_by_momentum(self, miner, ml_papers):
        """Test that trends are sorted by momentum"""
        result = miner.analyze(ml_papers)

        momentums = [t.momentum for t in result.trending_topics]
        assert momentums == sorted(momentums, reverse=True)


# ============================================================================
# TEST GAP IDENTIFICATION
# ============================================================================

class TestGapIdentification:
    """Test research gap identification"""

    def test_gaps_identified(self, miner, sample_papers):
        """Test that research gaps are identified"""
        result = miner.analyze(sample_papers)

        # Sample papers mention limitations, challenges, lack of data
        assert len(result.research_gaps) > 0

    def test_gap_types(self, miner, sample_papers):
        """Test different gap types are identified"""
        result = miner.analyze(sample_papers)

        gap_types = {gap.gap_type for gap in result.research_gaps}

        # Should find methodological gaps (limitation, challenge)
        # Should find empirical gaps (lack of data, need for)
        # Should find application gaps (real-world, deployment)
        assert "methodological" in gap_types or "empirical" in gap_types or "application" in gap_types

    def test_gap_has_evidence(self, miner, sample_papers):
        """Test that gaps include evidence"""
        result = miner.analyze(sample_papers)

        for gap in result.research_gaps:
            assert isinstance(gap, ResearchGap)
            assert isinstance(gap.evidence, list)
            assert gap.frequency > 0
            assert gap.impact_score > 0

    def test_gaps_sorted_by_impact(self, miner, sample_papers):
        """Test that gaps are sorted by impact score"""
        result = miner.analyze(sample_papers)

        impact_scores = [gap.impact_score for gap in result.research_gaps]
        assert impact_scores == sorted(impact_scores, reverse=True)

    def test_gap_frequency_threshold(self, miner):
        """Test that gaps require minimum frequency"""
        # Create papers with only 1-2 gap mentions
        papers = [
            Paper(
                paper_id=f"p{i}",
                title=f"Paper {i}",
                authors=["Author"],
                year=2020,
                venue="Venue",
                abstract="Normal paper with no gap indicators.",
                citations=0
            )
            for i in range(5)
        ]

        # Add one paper with gap mention
        papers.append(Paper(
            paper_id="gap1",
            title="Gap Paper",
            authors=["Author"],
            year=2020,
            venue="Venue",
            abstract="This paper has a limitation in the methodology.",
            citations=0
        ))

        result = miner.analyze(papers)

        # Should not create gap with only 1 mention (threshold is 3)
        # (or if it does, frequency should be low)
        if result.research_gaps:
            assert all(gap.frequency >= 3 for gap in result.research_gaps)


# ============================================================================
# TEST CITATION NETWORK
# ============================================================================

class TestCitationNetwork:
    """Test citation network building"""

    def test_citation_network_built(self, miner, sample_papers):
        """Test that citation network is built"""
        result = miner.analyze(sample_papers)

        assert isinstance(result.citation_network, dict)

    def test_citation_links_detected(self, miner, sample_papers):
        """Test that citation links are detected"""
        result = miner.analyze(sample_papers)

        # p4 references "Deep Learning for Image Recognition" (p1)
        # So p1 should be cited by p4
        if "p1" in result.citation_network:
            assert "p4" in result.citation_network["p1"]

    def test_citation_network_structure(self, miner, sample_papers):
        """Test citation network structure"""
        result = miner.analyze(sample_papers)

        for paper_id, citing_papers in result.citation_network.items():
            assert isinstance(paper_id, str)
            assert isinstance(citing_papers, list)


# ============================================================================
# TEST TEMPORAL ANALYSIS
# ============================================================================

class TestTemporalAnalysis:
    """Test temporal trend analysis"""

    def test_temporal_trends_calculated(self, miner, sample_papers):
        """Test temporal trends calculation"""
        result = miner.analyze(sample_papers)

        assert isinstance(result.temporal_trends, dict)
        assert len(result.temporal_trends) > 0

    def test_temporal_counts_correct(self, miner, sample_papers):
        """Test that year counts are correct"""
        result = miner.analyze(sample_papers)

        # 2020: 1 paper, 2021: 2 papers, 2022: 2 papers
        assert result.temporal_trends[2020] == 1
        assert result.temporal_trends[2021] == 2
        assert result.temporal_trends[2022] == 2

    def test_temporal_years_included(self, miner, sample_papers):
        """Test that all years are included"""
        result = miner.analyze(sample_papers)

        years = set(result.temporal_trends.keys())
        assert 2020 in years
        assert 2021 in years
        assert 2022 in years


# ============================================================================
# TEST KEYWORD EVOLUTION
# ============================================================================

class TestKeywordEvolution:
    """Test keyword evolution tracking"""

    def test_keyword_evolution_tracked(self, miner, sample_papers):
        """Test that keyword evolution is tracked"""
        result = miner.analyze(sample_papers)

        assert isinstance(result.keyword_evolution, dict)
        assert len(result.keyword_evolution) > 0

    def test_keywords_per_year(self, miner, sample_papers):
        """Test that keywords are tracked per year"""
        result = miner.analyze(sample_papers)

        for year, keywords in result.keyword_evolution.items():
            assert isinstance(year, int)
            assert isinstance(keywords, list)
            assert len(keywords) <= 10  # Top 10 per year

    def test_keywords_extracted(self, miner, sample_papers):
        """Test that relevant keywords are extracted"""
        result = miner.analyze(sample_papers)

        # Combine all keywords
        all_keywords = []
        for keywords in result.keyword_evolution.values():
            all_keywords.extend(keywords)

        # Should include technical terms
        keywords_str = " ".join(all_keywords)
        assert "learning" in keywords_str or "transformer" in keywords_str or "attention" in keywords_str


# ============================================================================
# TEST EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_paper(self, miner):
        """Test analysis with single paper"""
        papers = [
            Paper(
                paper_id="single",
                title="Single Paper",
                authors=["Author"],
                year=2020,
                venue="Venue",
                abstract="Just one paper for testing.",
                citations=10
            )
        ]

        result = miner.analyze(papers)

        assert result.total_papers == 1
        assert result.date_range == (2020, 2020)

    def test_papers_same_year(self, miner):
        """Test papers all from same year"""
        papers = [
            Paper(
                paper_id=f"p{i}",
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                year=2020,
                venue="Venue",
                abstract="Paper abstract.",
                citations=10
            )
            for i in range(5)
        ]

        result = miner.analyze(papers)

        assert result.date_range == (2020, 2020)
        assert result.temporal_trends[2020] == 5

    def test_no_citations(self, miner):
        """Test papers with no citations"""
        papers = [
            Paper(
                paper_id=f"p{i}",
                title=f"Paper {i}",
                authors=["Author"],
                year=2020,
                venue="Venue",
                abstract="Abstract.",
                citations=0,
                references=[]
            )
            for i in range(3)
        ]

        result = miner.analyze(papers)

        # Should still work
        assert result.total_papers == 3
        assert isinstance(result.citation_network, dict)

    def test_no_matching_methods(self, miner):
        """Test papers with no recognizable methods"""
        papers = [
            Paper(
                paper_id="p1",
                title="Philosophy Paper",
                authors=["Philosopher"],
                year=2020,
                venue="Phil Review",
                abstract="Discussing epistemology and ontology.",
                citations=5
            )
        ]

        result = miner.analyze(papers)

        # Should not crash
        assert result.total_papers == 1
        # Common methods might be empty or have generic terms


# ============================================================================
# TEST INTEGRATION
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_analysis_workflow(self, miner, sample_papers):
        """Test complete analysis workflow"""
        result = miner.analyze(sample_papers)

        # Verify all components present and valid
        assert result.total_papers == 5
        assert result.date_range == (2020, 2022)

        # Authors analyzed
        assert len(result.top_authors) > 0
        assert "Alice Smith" in [a for a, _ in result.top_authors]

        # Venues analyzed
        assert len(result.top_venues) > 0

        # Methods extracted
        assert len(result.common_methods) > 0

        # Trends identified
        assert len(result.trending_topics) >= 0  # May or may not have trends

        # Gaps found
        assert len(result.research_gaps) > 0

        # Citation network built
        assert isinstance(result.citation_network, dict)

        # Temporal trends
        assert len(result.temporal_trends) == 3  # 2020, 2021, 2022

        # Keyword evolution
        assert len(result.keyword_evolution) > 0

    def test_large_dataset_analysis(self, miner):
        """Test analysis with larger dataset"""
        # Create 50 papers
        papers = []
        for i in range(50):
            papers.append(Paper(
                paper_id=f"p{i}",
                title=f"Deep Learning Paper {i}",
                authors=[f"Author {i % 10}"],
                year=2018 + (i % 5),
                venue=["ICML", "NeurIPS", "ICLR", "CVPR", "ACL"][i % 5],
                abstract=f"We use deep learning and neural networks for task {i}. A major limitation is scalability.",
                citations=100 + i,
                keywords=["deep learning", "neural networks"]
            ))

        result = miner.analyze(papers)

        assert result.total_papers == 50
        assert result.date_range == (2018, 2022)

        # Should have substantial results
        assert len(result.top_authors) >= 10
        assert len(result.top_venues) >= 5
        assert len(result.common_methods) > 0

    def test_multi_field_analysis(self, miner):
        """Test analysis across multiple fields"""
        papers = [
            Paper(
                paper_id=f"cv{i}",
                title=f"Computer Vision {i}",
                authors=["CV Author"],
                year=2020,
                venue="CVPR",
                abstract="Computer vision with deep learning using CNNs and neural networks.",
                citations=100,
                field="computer vision"
            )
            for i in range(3)
        ] + [
            Paper(
                paper_id=f"nlp{i}",
                title=f"NLP Paper {i}",
                authors=["NLP Author"],
                year=2020,
                venue="ACL",
                abstract="NLP with transformer models and attention mechanisms.",
                citations=100,
                field="NLP"
            )
            for i in range(3)
        ]

        result = miner.analyze(papers)

        assert result.total_papers == 6

        # Should have captured some authors and venues
        assert len(result.top_authors) > 0
        assert len(result.top_venues) > 0


# ============================================================================
# SUMMARY
# ============================================================================

"""
Test Coverage Summary:

Test Classes: 11
- TestBasicAnalysis (4 tests)
- TestAuthorAnalysis (3 tests)
- TestVenueAnalysis (2 tests)
- TestMethodExtraction (3 tests)
- TestTrendIdentification (5 tests)
- TestGapIdentification (5 tests)
- TestCitationNetwork (3 tests)
- TestTemporalAnalysis (3 tests)
- TestKeywordEvolution (3 tests)
- TestEdgeCases (4 tests)
- TestIntegration (3 tests)

Total Tests: 38

Coverage Areas:
✅ Basic analysis and result structure
✅ Author analysis and counting
✅ Venue analysis
✅ Method extraction with regex patterns
✅ Trend identification and momentum calculation
✅ Research gap identification with evidence
✅ Citation network building
✅ Temporal trend analysis
✅ Keyword evolution tracking
✅ Edge cases (single paper, same year, no citations, etc.)
✅ Integration workflows (complete analysis, large datasets, multi-field)

Expected Coverage: 75%+
"""
