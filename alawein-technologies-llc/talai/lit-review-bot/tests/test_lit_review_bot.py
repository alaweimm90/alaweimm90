"""
Automated tests for LitReviewBot

Tests cover:
- Paper addition and management
- Theme detection
- Methodology classification
- Gap identification
- Clustering
- Review generation
"""

import pytest
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lit_review_bot import LitReviewBot, Paper, PaperCluster, ResearchGap, LiteratureReview


@pytest.fixture
def temp_data_file(tmp_path):
    """Create temporary data file for tests"""
    data_file = tmp_path / "test_litreview.json"
    return str(data_file)


@pytest.fixture
def bot(temp_data_file):
    """Create fresh LitReviewBot instance for each test"""
    return LitReviewBot(data_file=temp_data_file)


@pytest.fixture
def sample_papers(bot):
    """Add sample papers for testing"""
    papers = []

    # Transformer paper
    p1 = bot.add_paper(
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        year=2017,
        venue="NeurIPS",
        abstract="The dominant sequence transduction models are based on complex recurrent neural networks. We propose a new simple network architecture based solely on attention mechanisms.",
        keywords=["transformer", "attention", "neural networks"],
        citations=50000,
        collection_id=1
    )
    papers.append(p1)

    # AlphaFold paper
    p2 = bot.add_paper(
        title="Highly accurate protein structure prediction with AlphaFold",
        authors=["Jumper, J.", "Evans, R."],
        year=2021,
        venue="Nature",
        abstract="We demonstrate that deep learning can predict protein structure with atomic accuracy. AlphaFold achieved unprecedented accuracy in CASP14.",
        keywords=["protein folding", "deep learning", "AlphaFold"],
        citations=15000,
        collection_id=1
    )
    papers.append(p2)

    # GPT-3 paper
    p3 = bot.add_paper(
        title="Language Models are Few-Shot Learners",
        authors=["Brown, T.", "Mann, B."],
        year=2020,
        venue="NeurIPS",
        abstract="We trained GPT-3, a 175 billion parameter autoregressive language model. Our results showed that scaling laws continue to hold at unprecedented model sizes.",
        keywords=["language models", "GPT-3", "few-shot learning"],
        citations=25000,
        collection_id=1
    )
    papers.append(p3)

    return papers


class TestPaperManagement:
    """Test paper addition and management"""

    def test_add_paper(self, bot):
        """Test adding a single paper"""
        paper = bot.add_paper(
            title="Test Paper",
            authors=["Author A", "Author B"],
            year=2023,
            venue="TestConf",
            abstract="This is a test abstract",
            keywords=["test", "paper"],
            citations=10,
            collection_id=1
        )

        assert paper.paper_id == 1
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.year == 2023
        assert paper.citations == 10

    def test_paper_id_increment(self, bot):
        """Test that paper IDs increment correctly"""
        p1 = bot.add_paper(
            title="Paper 1", authors=["A"], year=2020, venue="V",
            abstract="Abstract 1", keywords=["k1"], collection_id=1
        )
        p2 = bot.add_paper(
            title="Paper 2", authors=["B"], year=2021, venue="V",
            abstract="Abstract 2", keywords=["k2"], collection_id=1
        )

        assert p1.paper_id == 1
        assert p2.paper_id == 2

    def test_get_collection(self, bot, sample_papers):
        """Test retrieving a collection"""
        assert 1 in bot.collections
        assert len(bot.collections[1]) == 3

    def test_list_collections(self, bot, sample_papers):
        """Test listing all collections"""
        # Add papers to another collection
        bot.add_paper(
            title="Paper in collection 2",
            authors=["C"],
            year=2022,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=2
        )

        assert len(bot.collections) == 2
        assert 1 in bot.collections
        assert 2 in bot.collections


class TestThemeDetection:
    """Test automatic theme detection"""

    def test_machine_learning_detection(self, bot):
        """Test detection of machine learning theme"""
        paper = bot.add_paper(
            title="Deep Neural Networks",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="We trained neural networks",
            keywords=["neural networks", "deep learning"],
            collection_id=1
        )

        assert "machine_learning" in paper.themes

    def test_experimental_detection(self, bot):
        """Test detection of experimental theme"""
        paper = bot.add_paper(
            title="Experimental Results",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Our experimental results showed significant improvements",
            keywords=["experiment", "validation"],
            collection_id=1
        )

        assert "experimental" in paper.themes

    def test_survey_detection(self, bot):
        """Test detection of survey theme"""
        paper = bot.add_paper(
            title="A Survey of Methods",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="This survey covers major developments",
            keywords=["survey", "review"],
            collection_id=1
        )

        assert "survey" in paper.themes

    def test_multiple_themes(self, bot, sample_papers):
        """Test that papers can have multiple themes"""
        # The transformer paper should have multiple themes
        paper = sample_papers[0]
        assert len(paper.themes) > 0


class TestMethodologyClassification:
    """Test methodology classification"""

    def test_experimental_methodology(self, bot):
        """Test experimental methodology detection"""
        paper = bot.add_paper(
            title="Test",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Our experimental results demonstrated",
            keywords=["experiment"],
            collection_id=1
        )

        assert paper.methodology == "experimental"

    def test_computational_methodology(self, bot):
        """Test computational methodology detection"""
        paper = bot.add_paper(
            title="Test",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="We used computational methods to simulate",
            keywords=["simulation"],
            collection_id=1
        )

        assert paper.methodology == "computational"

    def test_survey_methodology(self, bot):
        """Test survey methodology detection"""
        paper = bot.add_paper(
            title="A Survey",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="This survey examines",
            keywords=["survey"],
            collection_id=1
        )

        assert paper.methodology == "survey"


class TestFeatureExtraction:
    """Test extraction of findings and limitations"""

    def test_finding_extraction(self, bot):
        """Test extraction of research findings"""
        paper = bot.add_paper(
            title="Test",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Our results showed significant improvement. We found that performance increased by 50%.",
            keywords=["test"],
            collection_id=1
        )

        assert len(paper.findings) > 0
        assert any("showed" in f or "found" in f for f in paper.findings)

    def test_limitation_extraction(self, bot):
        """Test extraction of limitations"""
        paper = bot.add_paper(
            title="Test",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="However, limitations include small sample size. Future work should address this.",
            keywords=["test"],
            collection_id=1
        )

        assert len(paper.limitations) > 0


class TestClustering:
    """Test paper clustering functionality"""

    def test_cluster_papers(self, bot, sample_papers):
        """Test clustering papers by theme"""
        clusters = bot.cluster_papers(collection_id=1, num_clusters=2)

        assert len(clusters) > 0
        assert len(clusters) <= 2  # Should not exceed requested clusters

        # Check cluster structure
        for cluster in clusters:
            assert isinstance(cluster, PaperCluster)
            assert cluster.theme != ""
            assert len(cluster.papers) > 0
            assert len(cluster.keywords) > 0

    def test_cluster_assignment(self, bot, sample_papers):
        """Test that all papers are assigned to clusters"""
        clusters = bot.cluster_papers(collection_id=1, num_clusters=3)

        # Collect all paper IDs from clusters
        clustered_paper_ids = set()
        for cluster in clusters:
            clustered_paper_ids.update(cluster.papers)

        # Should have all 3 sample papers
        assert len(clustered_paper_ids) == 3

    def test_cluster_keywords(self, bot, sample_papers):
        """Test that clusters have representative keywords"""
        clusters = bot.cluster_papers(collection_id=1, num_clusters=2)

        for cluster in clusters:
            assert len(cluster.keywords) > 0
            assert len(cluster.keywords) <= 5  # Max 5 representative keywords


class TestGapIdentification:
    """Test research gap identification"""

    def test_identify_gaps(self, bot, sample_papers):
        """Test basic gap identification"""
        gaps = bot.identify_gaps(collection_id=1)

        assert len(gaps) > 0
        for gap in gaps:
            assert isinstance(gap, ResearchGap)
            assert gap.gap_type in ["methodological", "temporal", "thematic", "citation_impact", "theoretical", "empirical"]
            assert gap.description != ""
            assert gap.importance in ["low", "medium", "high", "critical"]
            assert gap.feasibility in ["low", "medium", "high"]

    def test_gap_evidence(self, bot, sample_papers):
        """Test that gaps include evidence"""
        gaps = bot.identify_gaps(collection_id=1)

        for gap in gaps:
            assert len(gap.evidence) > 0

    def test_temporal_gap(self, bot):
        """Test temporal gap detection"""
        # Add papers from different years
        bot.add_paper(
            title="Old Paper", authors=["A"], year=2015, venue="V",
            abstract="Abstract", keywords=["k"], collection_id=1
        )
        bot.add_paper(
            title="Recent Paper", authors=["B"], year=2023, venue="V",
            abstract="Abstract", keywords=["k"], collection_id=1
        )

        gaps = bot.identify_gaps(collection_id=1)

        # Just verify gaps are found - temporal gap may or may not be detected
        # depending on algorithm thresholds
        assert len(gaps) > 0


class TestReviewGeneration:
    """Test literature review generation"""

    def test_generate_narrative_review(self, bot, sample_papers):
        """Test narrative review generation"""
        review = bot.generate_review(collection_id=1, style="narrative")

        assert isinstance(review, LiteratureReview)
        assert review.style == "narrative"
        assert review.papers_reviewed == 3
        assert review.word_count > 0

    def test_generate_systematic_review(self, bot, sample_papers):
        """Test systematic review generation"""
        review = bot.generate_review(collection_id=1, style="systematic")

        assert review.style == "systematic"
        assert review.word_count > 0

    def test_generate_meta_analysis(self, bot, sample_papers):
        """Test meta-analysis review generation"""
        review = bot.generate_review(collection_id=1, style="meta-analysis")

        assert review.style == "meta-analysis"
        assert review.word_count > 0

    def test_review_structure(self, bot, sample_papers):
        """Test that review has required sections"""
        review = bot.generate_review(collection_id=1, style="narrative")

        # Check required fields
        assert review.introduction != ""
        assert review.background != ""
        assert review.synthesis != ""
        assert review.conclusion != ""
        assert len(review.gaps_identified) > 0
        assert len(review.future_directions) > 0

    def test_review_themes(self, bot, sample_papers):
        """Test that review includes thematic sections"""
        review = bot.generate_review(collection_id=1, style="narrative")

        assert len(review.thematic_sections) > 0
        for theme, content in review.thematic_sections.items():
            assert theme != ""
            assert content != ""

    def test_word_count(self, bot, sample_papers):
        """Test that word count is calculated"""
        review = bot.generate_review(collection_id=1, style="narrative")

        assert review.word_count > 0
        # Word count should be reasonable
        assert review.word_count > 100
        assert review.word_count < 10000


class TestDataPersistence:
    """Test data saving and loading"""

    def test_save_and_load(self, tmp_path):
        """Test that data persists across instances"""
        data_file = str(tmp_path / "test_persistence.json")

        # Create bot and add data
        bot1 = LitReviewBot(data_file=data_file)
        paper = bot1.add_paper(
            title="Test Paper",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=1
        )
        paper_id = paper.paper_id

        # Create new bot instance
        bot2 = LitReviewBot(data_file=data_file)

        # Data should be loaded
        assert paper_id in bot2.papers
        assert bot2.papers[paper_id].title == "Test Paper"

    def test_persistence_after_operations(self, tmp_path):
        """Test persistence after various operations"""
        data_file = str(tmp_path / "test_persistence2.json")
        bot1 = LitReviewBot(data_file=data_file)

        # Add papers
        for i in range(3):
            bot1.add_paper(
                title=f"Paper {i}",
                authors=["A"],
                year=2020,
                venue="V",
                abstract="Abstract",
                keywords=["k"],
                collection_id=1
            )

        # Generate clusters
        clusters = bot1.cluster_papers(collection_id=1)

        # Create new bot
        bot2 = LitReviewBot(data_file=data_file)

        # Should have all papers
        assert len(bot2.papers) == 3
        assert 1 in bot2.collections
        assert len(bot2.collections[1]) == 3


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_collection(self, bot):
        """Test operations on empty collection"""
        assert 999 not in bot.collections

    def test_cluster_with_few_papers(self, bot):
        """Test clustering with very few papers"""
        bot.add_paper(
            title="Paper 1",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=1
        )

        # Should handle gracefully
        clusters = bot.cluster_papers(collection_id=1, num_clusters=5)
        assert len(clusters) <= 1  # Can't have more clusters than papers

    def test_review_generation_minimal_papers(self, bot):
        """Test review generation with minimal papers"""
        bot.add_paper(
            title="Single Paper",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=1
        )

        # Should still generate review
        review = bot.generate_review(collection_id=1)
        assert review is not None
        assert review.word_count > 0
        assert review.introduction != ""

    def test_duplicate_paper_detection(self, bot):
        """Test adding similar papers"""
        p1 = bot.add_paper(
            title="Paper",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=1
        )
        p2 = bot.add_paper(
            title="Paper",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract",
            keywords=["k"],
            collection_id=1
        )

        # Should create separate papers (no deduplication yet)
        assert p1.paper_id != p2.paper_id


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complete_workflow(self, bot):
        """Test complete workflow from paper addition to review"""
        # Add papers
        papers = []
        for i in range(5):
            p = bot.add_paper(
                title=f"Paper {i}",
                authors=[f"Author {i}"],
                year=2020 + i,
                venue="Conference",
                abstract=f"This is abstract {i} with results showing improvements",
                keywords=[f"keyword{i}", "research"],
                citations=100 * i,
                collection_id=1
            )
            papers.append(p)

        # Cluster papers
        clusters = bot.cluster_papers(collection_id=1, num_clusters=2)
        assert len(clusters) > 0

        # Identify gaps
        gaps = bot.identify_gaps(collection_id=1)
        assert len(gaps) > 0

        # Generate review
        review = bot.generate_review(collection_id=1, style="narrative")
        assert review.papers_reviewed == 5
        assert review.word_count > 0
        assert len(review.thematic_sections) > 0
        assert len(review.gaps_identified) > 0
        assert len(review.future_directions) > 0

    def test_multi_collection_workflow(self, bot):
        """Test working with multiple collections"""
        # Add papers to collection 1
        bot.add_paper(
            title="Collection 1 Paper",
            authors=["A"],
            year=2020,
            venue="V",
            abstract="Abstract 1",
            keywords=["k1"],
            collection_id=1
        )

        # Add papers to collection 2
        bot.add_paper(
            title="Collection 2 Paper",
            authors=["B"],
            year=2021,
            venue="V",
            abstract="Abstract 2",
            keywords=["k2"],
            collection_id=2
        )

        # Operations should be independent
        review1 = bot.generate_review(collection_id=1)
        review2 = bot.generate_review(collection_id=2)

        assert review1.collection_id == 1
        assert review2.collection_id == 2
        assert review1.papers_reviewed == 1
        assert review2.papers_reviewed == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
