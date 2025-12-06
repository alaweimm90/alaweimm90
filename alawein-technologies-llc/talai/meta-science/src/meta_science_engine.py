"""
Meta-Science Analytics Engine for TalAI
========================================

Advanced analytics system for understanding the science of science itself,
including citation networks, research trends, and scientific impact prediction.

References:
- Wang, D., Song, C., & Barabási, A. L. (2013). Quantifying long-term scientific impact. Science.
- Fortunato, S., et al. (2018). Science of science. Science.
- Clauset, A., Larremore, D. B., & Sinatra, R. (2017). Data-driven predictions in science.
- Wais, K. (2016). Gender prediction methods based on first names with genderizeR.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import re
from scipy import stats, sparse
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import warnings
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchField(Enum):
    """Major research fields for classification."""
    PHYSICS = "physics"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    MEDICINE = "medicine"
    COMPUTER_SCIENCE = "computer_science"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    SOCIAL_SCIENCES = "social_sciences"
    INTERDISCIPLINARY = "interdisciplinary"


@dataclass
class Paper:
    """Represents a scientific paper with metadata."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    field: ResearchField
    journal: str
    citations: int = 0
    references: List[str] = field(default_factory=list)
    abstract: str = ""
    keywords: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    funding_amount: float = 0.0
    is_retracted: bool = False
    altmetric_score: float = 0.0
    open_access: bool = False

    def age(self, current_year: int = 2024) -> int:
        """Calculate paper age in years."""
        return current_year - self.year

    def citations_per_year(self) -> float:
        """Calculate average citations per year."""
        age = self.age()
        return self.citations / age if age > 0 else 0

    def h_index_contribution(self, author: str) -> int:
        """Calculate this paper's contribution to an author's h-index."""
        # Simplified: 1 if citations >= author's paper count, 0 otherwise
        return 1 if self.citations > 0 and author in self.authors else 0


@dataclass
class Author:
    """Represents a researcher with publication history."""
    author_id: str
    name: str
    affiliation: str
    papers: List[str] = field(default_factory=list)
    h_index: int = 0
    total_citations: int = 0
    years_active: int = 0
    fields: List[ResearchField] = field(default_factory=list)
    gender: Optional[str] = None
    country: Optional[str] = None
    career_age: int = 0
    collaboration_network: Set[str] = field(default_factory=set)

    def productivity_rate(self) -> float:
        """Calculate papers per year."""
        return len(self.papers) / max(self.years_active, 1)

    def collaboration_index(self) -> float:
        """Calculate average number of collaborators per paper."""
        return len(self.collaboration_network) / max(len(self.papers), 1)


@dataclass
class Journal:
    """Represents a scientific journal."""
    journal_id: str
    name: str
    impact_factor: float
    field: ResearchField
    publisher: str
    open_access: bool = False
    publication_fee: float = 0.0
    review_time_days: int = 90
    acceptance_rate: float = 0.3
    is_predatory: bool = False
    founded_year: int = 1900


@dataclass
class ResearchTrend:
    """Represents an emerging research trend or topic."""
    trend_id: str
    keywords: List[str]
    papers: List[str]
    start_year: int
    growth_rate: float
    predicted_peak_year: int
    field: ResearchField
    is_emerging: bool = True
    breakthrough_probability: float = 0.0
    interdisciplinary_score: float = 0.0


@dataclass
class CollaborationNetwork:
    """Represents a research collaboration network."""
    nodes: List[Author]
    edges: List[Tuple[str, str, int]]  # (author1, author2, num_collaborations)
    graph: Optional[nx.Graph] = None

    def __post_init__(self):
        """Initialize NetworkX graph."""
        self.graph = nx.Graph()
        for author in self.nodes:
            self.graph.add_node(author.author_id, author=author)
        for edge in self.edges:
            self.graph.add_edge(edge[0], edge[1], weight=edge[2])

    def get_communities(self) -> List[Set[str]]:
        """Detect research communities using Louvain algorithm."""
        # Simplified community detection using connected components
        return list(nx.connected_components(self.graph))

    def centrality_scores(self) -> Dict[str, float]:
        """Calculate various centrality measures."""
        return {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }


class MetaScienceEngine:
    """
    Main engine for meta-science analytics and scientometrics.

    Analyzes research ecosystems, predicts scientific impact, and identifies trends.
    """

    def __init__(self, random_state: int = 42):
        """Initialize the meta-science engine."""
        self.random_state = random_state
        np.random.seed(random_state)
        self.papers: Dict[str, Paper] = {}
        self.authors: Dict[str, Author] = {}
        self.journals: Dict[str, Journal] = {}
        self.citation_network: Optional[nx.DiGraph] = None
        self.collaboration_network: Optional[CollaborationNetwork] = None
        self.trends: List[ResearchTrend] = []

    def load_papers(self, papers: List[Paper]):
        """Load papers into the system."""
        for paper in papers:
            self.papers[paper.paper_id] = paper
        self._build_citation_network()
        logger.info(f"Loaded {len(papers)} papers")

    def load_authors(self, authors: List[Author]):
        """Load authors into the system."""
        for author in authors:
            self.authors[author.author_id] = author
        self._build_collaboration_network()
        logger.info(f"Loaded {len(authors)} authors")

    def _build_citation_network(self):
        """Build directed citation network from papers."""
        self.citation_network = nx.DiGraph()

        for paper_id, paper in self.papers.items():
            self.citation_network.add_node(paper_id, paper=paper)

            for ref_id in paper.references:
                if ref_id in self.papers:
                    # Edge from citing to cited paper
                    self.citation_network.add_edge(paper_id, ref_id)

    def _build_collaboration_network(self):
        """Build collaboration network from author data."""
        edges = []

        # Build edges from co-authorship
        for paper in self.papers.values():
            for i, author1 in enumerate(paper.authors[:-1]):
                for author2 in paper.authors[i+1:]:
                    if author1 in self.authors and author2 in self.authors:
                        edges.append((author1, author2, 1))

        # Aggregate collaboration counts
        edge_counts = defaultdict(int)
        for a1, a2, _ in edges:
            key = tuple(sorted([a1, a2]))
            edge_counts[key] += 1

        aggregated_edges = [(a1, a2, count) for (a1, a2), count in edge_counts.items()]

        self.collaboration_network = CollaborationNetwork(
            nodes=list(self.authors.values()),
            edges=aggregated_edges
        )

    def predict_paper_impact(self, paper: Paper, horizon_years: int = 5) -> Dict[str, Any]:
        """
        Predict future citations and impact of a paper.

        Based on Wang et al. (2013) preferential attachment model.
        """
        features = []

        # Paper features
        features.extend([
            paper.age(),
            len(paper.authors),
            len(paper.references),
            len(paper.keywords),
            1 if paper.open_access else 0,
            paper.funding_amount / 1e6,  # Millions
        ])

        # Author features (average h-index, max h-index)
        author_h_indices = []
        for author_name in paper.authors:
            for author in self.authors.values():
                if author.name == author_name:
                    author_h_indices.append(author.h_index)
                    break

        if author_h_indices:
            features.extend([
                np.mean(author_h_indices),
                np.max(author_h_indices),
                np.min(author_h_indices)
            ])
        else:
            features.extend([0, 0, 0])

        # Journal impact factor (if available)
        journal_if = 0
        for journal in self.journals.values():
            if journal.name == paper.journal:
                journal_if = journal.impact_factor
                break
        features.append(journal_if)

        # Early citation trajectory (fitness parameter from Wang et al.)
        early_citations = paper.citations
        features.append(early_citations)

        # Predict using preferential attachment with fitness
        features = np.array(features).reshape(1, -1)

        # Simple model: Citations follow lognormal distribution
        # C(t) = exp(μ + σ * fitness) * (t + 1)^β

        fitness = np.log(early_citations + 1) / max(paper.age(), 1)
        mu = 2.0  # Base growth rate
        sigma = 0.5
        beta = 0.8  # Aging exponent

        predicted_citations = {}
        confidence_intervals = {}

        for year in range(1, horizon_years + 1):
            t = paper.age() + year
            mean_citations = np.exp(mu + sigma * fitness) * (t ** beta)

            # Add noise for confidence interval
            std_dev = mean_citations * 0.3
            ci_lower = max(0, mean_citations - 1.96 * std_dev)
            ci_upper = mean_citations + 1.96 * std_dev

            predicted_citations[year] = mean_citations
            confidence_intervals[year] = (ci_lower, ci_upper)

        # Classify impact level
        total_predicted = sum(predicted_citations.values())
        if total_predicted > 1000:
            impact_class = "Highly Influential"
        elif total_predicted > 100:
            impact_class = "Influential"
        elif total_predicted > 10:
            impact_class = "Moderate Impact"
        else:
            impact_class = "Limited Impact"

        # Calculate percentile rank
        all_citations = [p.citations for p in self.papers.values()]
        percentile = stats.percentileofscore(all_citations, paper.citations)

        return {
            'paper_id': paper.paper_id,
            'current_citations': paper.citations,
            'predicted_citations': predicted_citations,
            'confidence_intervals': confidence_intervals,
            'total_predicted': total_predicted,
            'impact_class': impact_class,
            'percentile_rank': percentile,
            'fitness_parameter': fitness,
            'breakthrough_probability': self._calculate_breakthrough_probability(paper)
        }

    def _calculate_breakthrough_probability(self, paper: Paper) -> float:
        """
        Calculate probability that paper represents a breakthrough.

        Based on Uzzi et al. (2013) - Atypical combinations of prior work.
        """
        if not paper.references:
            return 0.0

        # Calculate novelty: unusual combinations of references
        ref_pairs = list(itertools.combinations(paper.references, 2))
        if not ref_pairs:
            return 0.0

        co_citation_scores = []
        for ref1, ref2 in ref_pairs[:50]:  # Limit for efficiency
            # How often are these papers cited together?
            co_citations = 0
            for other_paper in self.papers.values():
                if ref1 in other_paper.references and ref2 in other_paper.references:
                    co_citations += 1

            # Lower co-citation = more novel combination
            co_citation_scores.append(co_citations)

        # Breakthrough papers have some very novel and some conventional combinations
        if co_citation_scores:
            novelty = 1 / (1 + np.mean(co_citation_scores))
            conventionality = np.percentile(co_citation_scores, 90) / max(max(co_citation_scores), 1)

            # Breakthrough probability increases with high novelty AND some conventionality
            breakthrough_prob = novelty * 0.6 + conventionality * 0.4

            # Adjust for early citations (breakthroughs often recognized early)
            if paper.age() <= 3 and paper.citations > 10:
                breakthrough_prob *= 1.5

            return min(1.0, breakthrough_prob)

        return 0.0

    def detect_emerging_fields(self, min_papers: int = 10, growth_threshold: float = 0.5) -> List[ResearchTrend]:
        """
        Identify emerging research fields using keyword and citation patterns.

        Based on Klavans & Boyack (2017) research front detection.
        """
        # Extract keywords from all papers by year
        keywords_by_year = defaultdict(lambda: defaultdict(int))

        for paper in self.papers.values():
            for keyword in paper.keywords:
                keywords_by_year[paper.year][keyword.lower()] += 1

        # Calculate growth rates
        emerging_trends = []
        current_year = max(paper.year for paper in self.papers.values())

        all_keywords = set()
        for year_keywords in keywords_by_year.values():
            all_keywords.update(year_keywords.keys())

        for keyword in all_keywords:
            yearly_counts = []
            years = sorted(keywords_by_year.keys())

            for year in years[-5:]:  # Last 5 years
                yearly_counts.append(keywords_by_year[year].get(keyword, 0))

            if sum(yearly_counts) < min_papers:
                continue

            # Calculate growth rate using linear regression
            if len(yearly_counts) > 2:
                X = np.arange(len(yearly_counts)).reshape(-1, 1)
                y = np.array(yearly_counts)

                if y[-1] > 0:  # Must have recent activity
                    model = LinearRegression().fit(X, y)
                    growth_rate = model.coef_[0] / max(np.mean(y), 1)

                    if growth_rate > growth_threshold:
                        # Find papers with this keyword
                        keyword_papers = [
                            p.paper_id for p in self.papers.values()
                            if keyword in [k.lower() for k in p.keywords]
                        ]

                        # Predict peak year using logistic growth model
                        predicted_peak = current_year + int(10 / (1 + growth_rate))

                        trend = ResearchTrend(
                            trend_id=f"trend_{keyword}",
                            keywords=[keyword],
                            papers=keyword_papers,
                            start_year=years[0],
                            growth_rate=growth_rate,
                            predicted_peak_year=predicted_peak,
                            field=self._infer_field(keyword_papers),
                            is_emerging=True,
                            breakthrough_probability=np.mean([
                                self._calculate_breakthrough_probability(self.papers[pid])
                                for pid in keyword_papers[:10]
                            ])
                        )
                        emerging_trends.append(trend)

        # Cluster similar trends
        merged_trends = self._merge_similar_trends(emerging_trends)
        self.trends = merged_trends

        logger.info(f"Detected {len(merged_trends)} emerging research fields")
        return merged_trends

    def _merge_similar_trends(self, trends: List[ResearchTrend]) -> List[ResearchTrend]:
        """Merge trends with overlapping papers."""
        if not trends:
            return []

        # Simple merging based on paper overlap
        merged = []
        used = set()

        for i, trend1 in enumerate(trends):
            if i in used:
                continue

            merged_keywords = trend1.keywords.copy()
            merged_papers = set(trend1.papers)

            for j, trend2 in enumerate(trends[i+1:], i+1):
                if j in used:
                    continue

                # Check overlap
                overlap = len(set(trend1.papers) & set(trend2.papers))
                if overlap > len(trend1.papers) * 0.3:
                    merged_keywords.extend(trend2.keywords)
                    merged_papers.update(trend2.papers)
                    used.add(j)

            # Create merged trend
            merged_trend = ResearchTrend(
                trend_id=f"merged_trend_{i}",
                keywords=list(set(merged_keywords)),
                papers=list(merged_papers),
                start_year=trend1.start_year,
                growth_rate=trend1.growth_rate,
                predicted_peak_year=trend1.predicted_peak_year,
                field=trend1.field,
                is_emerging=True,
                breakthrough_probability=trend1.breakthrough_probability
            )
            merged.append(merged_trend)

        return merged

    def _infer_field(self, paper_ids: List[str]) -> ResearchField:
        """Infer research field from paper IDs."""
        fields = []
        for pid in paper_ids[:20]:  # Sample for efficiency
            if pid in self.papers:
                fields.append(self.papers[pid].field)

        if fields:
            most_common = Counter(fields).most_common(1)[0][0]
            return most_common

        return ResearchField.INTERDISCIPLINARY

    def analyze_gender_bias(self) -> Dict[str, Any]:
        """
        Analyze gender bias in citations and collaborations.

        Based on Larivière et al. (2013) and Dworkin et al. (2020).
        """
        results = {
            'citation_bias': {},
            'collaboration_patterns': {},
            'first_author_ratio': {},
            'last_author_ratio': {},
            'recommendations': []
        }

        # Separate authors by gender
        male_authors = [a for a in self.authors.values() if a.gender == 'male']
        female_authors = [a for a in self.authors.values() if a.gender == 'female']

        if not male_authors or not female_authors:
            results['error'] = "Insufficient gender data"
            return results

        # Citation bias analysis
        male_citations = [a.total_citations for a in male_authors]
        female_citations = [a.total_citations for a in female_authors]

        # Mann-Whitney U test for citation differences
        u_stat, p_value = stats.mannwhitneyu(male_citations, female_citations)

        results['citation_bias'] = {
            'mean_citations_male': np.mean(male_citations),
            'mean_citations_female': np.mean(female_citations),
            'median_citations_male': np.median(male_citations),
            'median_citations_female': np.median(female_citations),
            'statistical_test': 'Mann-Whitney U',
            'p_value': p_value,
            'significant_bias': p_value < 0.05
        }

        # Collaboration patterns
        male_ids = {a.author_id for a in male_authors}
        female_ids = {a.author_id for a in female_authors}

        same_gender_collab = 0
        mixed_gender_collab = 0

        if self.collaboration_network:
            for edge in self.collaboration_network.edges:
                author1, author2, _ = edge
                if (author1 in male_ids and author2 in male_ids) or \
                   (author1 in female_ids and author2 in female_ids):
                    same_gender_collab += 1
                else:
                    mixed_gender_collab += 1

        results['collaboration_patterns'] = {
            'same_gender_collaborations': same_gender_collab,
            'mixed_gender_collaborations': mixed_gender_collab,
            'homophily_index': same_gender_collab / max(same_gender_collab + mixed_gender_collab, 1)
        }

        # First and last author analysis (proxy for seniority)
        first_author_male = 0
        first_author_female = 0
        last_author_male = 0
        last_author_female = 0

        for paper in self.papers.values():
            if paper.authors:
                first = paper.authors[0]
                last = paper.authors[-1] if len(paper.authors) > 1 else None

                # Check first author gender
                for author in self.authors.values():
                    if author.name == first:
                        if author.gender == 'male':
                            first_author_male += 1
                        elif author.gender == 'female':
                            first_author_female += 1

                    if last and author.name == last:
                        if author.gender == 'male':
                            last_author_male += 1
                        elif author.gender == 'female':
                            last_author_female += 1

        results['first_author_ratio'] = {
            'male': first_author_male,
            'female': first_author_female,
            'ratio': first_author_female / max(first_author_male + first_author_female, 1)
        }

        results['last_author_ratio'] = {
            'male': last_author_male,
            'female': last_author_female,
            'ratio': last_author_female / max(last_author_male + last_author_female, 1)
        }

        # Generate recommendations
        if results['citation_bias']['significant_bias']:
            results['recommendations'].append(
                "Significant citation bias detected. Consider implementing blind review processes."
            )

        if results['collaboration_patterns']['homophily_index'] > 0.7:
            results['recommendations'].append(
                "High gender homophily in collaborations. Encourage diverse research teams."
            )

        if results['last_author_ratio']['ratio'] < 0.3:
            results['recommendations'].append(
                "Low female representation in senior (last) author positions. Support mentorship programs."
            )

        return results

    def predict_journal_quality(self, journal: Journal) -> Dict[str, Any]:
        """
        Predict journal quality and detect predatory journals.

        Based on Bohannon (2013) and Grudniewicz et al. (2019).
        """
        quality_score = 0.0
        red_flags = []
        green_flags = []

        # Check impact factor
        if journal.impact_factor > 2.0:
            quality_score += 0.2
            green_flags.append("Reasonable impact factor")
        elif journal.impact_factor < 0.5:
            red_flags.append("Very low impact factor")
            quality_score -= 0.1

        # Check review time
        if journal.review_time_days < 14:
            red_flags.append("Suspiciously fast review time")
            quality_score -= 0.3
        elif journal.review_time_days > 30 and journal.review_time_days < 180:
            green_flags.append("Reasonable review timeline")
            quality_score += 0.1

        # Check acceptance rate
        if journal.acceptance_rate > 0.9:
            red_flags.append("Very high acceptance rate")
            quality_score -= 0.3
        elif journal.acceptance_rate < 0.5:
            green_flags.append("Selective acceptance rate")
            quality_score += 0.1

        # Check publication fees for OA journals
        if journal.open_access:
            if journal.publication_fee > 5000:
                red_flags.append("Excessive publication fees")
                quality_score -= 0.2
            elif journal.publication_fee == 0:
                green_flags.append("No author charges (diamond OA)")
                quality_score += 0.1

        # Check founding year
        if datetime.now().year - journal.founded_year < 3:
            red_flags.append("Very new journal")
            quality_score -= 0.1
        elif datetime.now().year - journal.founded_year > 20:
            green_flags.append("Established journal")
            quality_score += 0.2

        # Normalize quality score
        quality_score = max(0, min(1, (quality_score + 1) / 2))

        # Predict predatory probability
        predatory_probability = 0.0

        if len(red_flags) >= 3:
            predatory_probability = 0.8
        elif len(red_flags) >= 2:
            predatory_probability = 0.5
        elif len(red_flags) >= 1:
            predatory_probability = 0.3

        if len(green_flags) >= 3:
            predatory_probability *= 0.3

        return {
            'journal_id': journal.journal_id,
            'journal_name': journal.name,
            'quality_score': quality_score,
            'predatory_probability': predatory_probability,
            'is_likely_predatory': predatory_probability > 0.5,
            'red_flags': red_flags,
            'green_flags': green_flags,
            'recommendation': self._journal_recommendation(quality_score, predatory_probability)
        }

    def _journal_recommendation(self, quality_score: float, predatory_prob: float) -> str:
        """Generate recommendation for journal selection."""
        if predatory_prob > 0.7:
            return "AVOID - High risk of predatory journal"
        elif predatory_prob > 0.4:
            return "CAUTION - Possible predatory journal, verify independently"
        elif quality_score > 0.7:
            return "RECOMMENDED - High quality journal"
        elif quality_score > 0.5:
            return "ACCEPTABLE - Reasonable quality journal"
        else:
            return "MARGINAL - Consider alternatives"

    def analyze_funding_roi(self, funding_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyze return on investment for research funding.

        funding_data: {paper_id: funding_amount}
        """
        roi_results = []

        for paper_id, funding in funding_data.items():
            if paper_id not in self.papers:
                continue

            paper = self.papers[paper_id]

            # Calculate various impact metrics
            citations_per_dollar = paper.citations / max(funding, 1)
            citations_per_year_dollar = paper.citations_per_year() / max(funding, 1)

            # Calculate derivative impacts (papers citing this work)
            derivative_impact = 0
            if self.citation_network and self.citation_network.has_node(paper_id):
                citing_papers = list(self.citation_network.predecessors(paper_id))
                derivative_impact = sum(
                    self.papers[pid].citations for pid in citing_papers
                    if pid in self.papers
                )

            # Economic impact estimate (very simplified)
            # Assume each citation represents $10K of follow-up research
            economic_impact = (paper.citations + derivative_impact * 0.1) * 10000

            roi = (economic_impact - funding) / max(funding, 1)

            roi_results.append({
                'paper_id': paper_id,
                'funding': funding,
                'citations': paper.citations,
                'citations_per_dollar': citations_per_dollar,
                'derivative_impact': derivative_impact,
                'economic_impact': economic_impact,
                'roi': roi,
                'roi_category': self._categorize_roi(roi)
            })

        # Aggregate statistics
        if roi_results:
            rois = [r['roi'] for r in roi_results]
            return {
                'individual_projects': roi_results,
                'aggregate_stats': {
                    'mean_roi': np.mean(rois),
                    'median_roi': np.median(rois),
                    'std_roi': np.std(rois),
                    'total_funding': sum(funding_data.values()),
                    'total_economic_impact': sum(r['economic_impact'] for r in roi_results),
                    'high_roi_projects': len([r for r in roi_results if r['roi'] > 10]),
                    'negative_roi_projects': len([r for r in roi_results if r['roi'] < 0])
                },
                'recommendations': self._funding_recommendations(roi_results)
            }

        return {'error': 'No valid funding data found'}

    def _categorize_roi(self, roi: float) -> str:
        """Categorize ROI levels."""
        if roi > 100:
            return "Exceptional"
        elif roi > 10:
            return "Excellent"
        elif roi > 1:
            return "Good"
        elif roi > 0:
            return "Positive"
        else:
            return "Negative"

    def _funding_recommendations(self, roi_results: List[Dict]) -> List[str]:
        """Generate funding recommendations based on ROI analysis."""
        recommendations = []

        # Analyze patterns in high-ROI projects
        high_roi = [r for r in roi_results if r['roi'] > 10]
        if high_roi:
            avg_funding_high = np.mean([r['funding'] for r in high_roi])
            recommendations.append(
                f"High-ROI projects average funding: ${avg_funding_high:,.0f}"
            )

        # Check for diminishing returns
        if len(roi_results) > 10:
            fundings = [r['funding'] for r in roi_results]
            rois = [r['roi'] for r in roi_results]
            corr, p_value = stats.spearmanr(fundings, rois)

            if corr < -0.3 and p_value < 0.05:
                recommendations.append(
                    "Evidence of diminishing returns with larger grants"
                )

        return recommendations

    def predict_retraction_risk(self, paper: Paper) -> Dict[str, Any]:
        """
        Predict probability of paper retraction.

        Based on patterns from Retraction Watch database analysis.
        """
        risk_score = 0.0
        risk_factors = []

        # Check for extreme values (too good to be true)
        if paper.citations / max(paper.age(), 1) > 100:
            risk_factors.append("Unusually high citation rate")
            risk_score += 0.2

        # Check author history (simplified - would need retraction history)
        for author_name in paper.authors:
            # In real implementation, check if author has previous retractions
            pass

        # Check statistical patterns in abstract (would need NLP)
        if paper.abstract:
            # Look for: very high effect sizes, perfect p-values, etc.
            if 'p < 0.001' in paper.abstract and 'n =' in paper.abstract:
                # Extract sample size
                import re
                n_match = re.search(r'n = (\d+)', paper.abstract)
                if n_match:
                    n = int(n_match.group(1))
                    if n < 20:
                        risk_factors.append("Very small sample size with strong significance")
                        risk_score += 0.3

        # Check journal quality (predatory journals have higher retraction rates)
        for journal in self.journals.values():
            if journal.name == paper.journal:
                if journal.is_predatory:
                    risk_factors.append("Published in potentially predatory journal")
                    risk_score += 0.4
                break

        # Check for image duplication (would need image analysis)
        # Check for plagiarism (would need text comparison)

        # Normalize risk score
        risk_score = min(1.0, risk_score)

        return {
            'paper_id': paper.paper_id,
            'retraction_risk_score': risk_score,
            'risk_level': self._categorize_risk(risk_score),
            'risk_factors': risk_factors,
            'recommendation': self._retraction_recommendation(risk_score)
        }

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize retraction risk levels."""
        if risk_score > 0.7:
            return "High Risk"
        elif risk_score > 0.4:
            return "Moderate Risk"
        elif risk_score > 0.2:
            return "Low Risk"
        else:
            return "Minimal Risk"

    def _retraction_recommendation(self, risk_score: float) -> str:
        """Generate recommendation based on retraction risk."""
        if risk_score > 0.7:
            return "Requires careful scrutiny and independent verification"
        elif risk_score > 0.4:
            return "Review methodology and data carefully"
        elif risk_score > 0.2:
            return "Standard peer review recommended"
        else:
            return "Low concern for integrity issues"

    def analyze_research_diversity(self) -> Dict[str, Any]:
        """Analyze diversity in research topics, methods, and teams."""
        diversity_metrics = {
            'topical_diversity': self._calculate_topical_diversity(),
            'methodological_diversity': self._calculate_methodological_diversity(),
            'geographic_diversity': self._calculate_geographic_diversity(),
            'temporal_diversity': self._calculate_temporal_diversity(),
            'recommendations': []
        }

        # Generate recommendations based on diversity scores
        if diversity_metrics['topical_diversity'] < 0.3:
            diversity_metrics['recommendations'].append(
                "Low topical diversity - consider exploring adjacent fields"
            )

        if diversity_metrics['geographic_diversity'] < 0.2:
            diversity_metrics['recommendations'].append(
                "Limited geographic diversity - seek international collaborations"
            )

        return diversity_metrics

    def _calculate_topical_diversity(self) -> float:
        """Calculate diversity of research topics using entropy."""
        field_counts = Counter(p.field for p in self.papers.values())
        total = sum(field_counts.values())

        if total == 0:
            return 0.0

        # Shannon entropy
        entropy = 0.0
        for count in field_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)

        # Normalize by maximum entropy
        max_entropy = np.log(len(ResearchField))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_methodological_diversity(self) -> float:
        """Calculate diversity in research methods (simplified)."""
        # In real implementation, would analyze methods sections
        keyword_methods = ['experiment', 'simulation', 'theory', 'review', 'meta-analysis', 'survey']
        method_counts = defaultdict(int)

        for paper in self.papers.values():
            for keyword in paper.keywords:
                for method in keyword_methods:
                    if method in keyword.lower():
                        method_counts[method] += 1

        if not method_counts:
            return 0.0

        # Calculate entropy
        total = sum(method_counts.values())
        entropy = 0.0
        for count in method_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)

        max_entropy = np.log(len(keyword_methods))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _calculate_geographic_diversity(self) -> float:
        """Calculate geographic diversity of authors."""
        country_counts = Counter(a.country for a in self.authors.values() if a.country)

        if not country_counts:
            return 0.0

        # Simpson's diversity index
        total = sum(country_counts.values())
        diversity = 1.0

        for count in country_counts.values():
            p = count / total
            diversity -= p * p

        return diversity

    def _calculate_temporal_diversity(self) -> float:
        """Calculate temporal spread of publications."""
        years = [p.year for p in self.papers.values()]

        if not years:
            return 0.0

        # Coefficient of variation
        cv = np.std(years) / np.mean(years) if np.mean(years) > 0 else 0.0

        # Normalize to 0-1 scale
        return min(1.0, cv)

    def generate_science_of_science_report(self) -> str:
        """Generate comprehensive meta-science analysis report."""
        report = "# Meta-Science Analytics Report\n\n"

        report += "## Research Ecosystem Overview\n"
        report += f"- **Total Papers**: {len(self.papers)}\n"
        report += f"- **Total Authors**: {len(self.authors)}\n"
        report += f"- **Total Journals**: {len(self.journals)}\n\n"

        # Citation statistics
        if self.papers:
            citations = [p.citations for p in self.papers.values()]
            report += "## Citation Analysis\n"
            report += f"- **Mean Citations**: {np.mean(citations):.1f}\n"
            report += f"- **Median Citations**: {np.median(citations):.1f}\n"
            report += f"- **Max Citations**: {max(citations)}\n"
            report += f"- **Papers with 0 citations**: {len([c for c in citations if c == 0])}\n\n"

        # Collaboration network
        if self.collaboration_network:
            report += "## Collaboration Network\n"
            centrality = self.collaboration_network.centrality_scores()
            top_central = sorted(centrality['degree'].items(), key=lambda x: x[1], reverse=True)[:5]

            report += "### Most Connected Researchers\n"
            for author_id, score in top_central:
                if author_id in self.authors:
                    report += f"- {self.authors[author_id].name}: {score:.3f}\n"
            report += "\n"

        # Emerging fields
        if self.trends:
            report += "## Emerging Research Fields\n"
            for trend in self.trends[:5]:
                report += f"\n### {', '.join(trend.keywords[:3])}\n"
                report += f"- **Growth Rate**: {trend.growth_rate:.1%}\n"
                report += f"- **Papers**: {len(trend.papers)}\n"
                report += f"- **Predicted Peak**: {trend.predicted_peak_year}\n"
                report += f"- **Breakthrough Probability**: {trend.breakthrough_probability:.1%}\n"

        # Gender analysis
        gender_analysis = self.analyze_gender_bias()
        if 'citation_bias' in gender_analysis:
            report += "\n## Gender Analysis\n"
            report += f"- **Mean Citations (Male)**: {gender_analysis['citation_bias'].get('mean_citations_male', 0):.1f}\n"
            report += f"- **Mean Citations (Female)**: {gender_analysis['citation_bias'].get('mean_citations_female', 0):.1f}\n"
            if gender_analysis['citation_bias'].get('significant_bias'):
                report += "- ⚠️ **Significant gender bias detected**\n"

        # Research diversity
        diversity = self.analyze_research_diversity()
        report += "\n## Research Diversity\n"
        report += f"- **Topical Diversity**: {diversity['topical_diversity']:.2f}\n"
        report += f"- **Geographic Diversity**: {diversity['geographic_diversity']:.2f}\n"
        report += f"- **Methodological Diversity**: {diversity['methodological_diversity']:.2f}\n"

        if diversity['recommendations']:
            report += "\n### Recommendations\n"
            for rec in diversity['recommendations']:
                report += f"- {rec}\n"

        return report