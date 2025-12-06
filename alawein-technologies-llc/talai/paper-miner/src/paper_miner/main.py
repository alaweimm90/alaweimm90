#!/usr/bin/env python3
"""
PaperMiner - Bulk Research Paper Analysis

Analyze 100+ papers simultaneously to extract:
- Common methodologies
- Trending topics
- Research gaps
- Citation networks
- Temporal trends

Usage:
    python miner.py analyze --input papers.json --output insights.json
    python miner.py trends --input papers.json --field "machine-learning"
    python miner.py gaps --input papers.json --output gaps_report.txt
    python miner.py citations --input papers.json --min-citations 100
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set


@dataclass
class Paper:
    """Research paper metadata"""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    venue: str
    abstract: str
    citations: int = 0
    references: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    field: str = ""


@dataclass
class ResearchGap:
    """Identified research gap"""
    gap_type: str  # "methodological" | "empirical" | "theoretical" | "application"
    description: str
    evidence: List[str]  # Supporting quotes/observations
    frequency: int  # How many papers mention related issues
    impact_score: float  # Potential impact of addressing gap


@dataclass
class Trend:
    """Research trend"""
    topic: str
    momentum: float  # Growth rate
    papers_per_year: Dict[int, int]
    key_papers: List[str]
    emerging: bool  # True if recent surge


@dataclass
class MiningResult:
    """Complete analysis result"""
    total_papers: int
    date_range: tuple[int, int]
    top_authors: List[tuple[str, int]]  # (author, paper_count)
    top_venues: List[tuple[str, int]]
    common_methods: List[tuple[str, int]]  # (method, frequency)
    trending_topics: List[Trend]
    research_gaps: List[ResearchGap]
    citation_network: Dict[str, List[str]]  # paper_id -> cited_by
    temporal_trends: Dict[int, int]  # year -> paper_count
    keyword_evolution: Dict[int, List[str]]  # year -> top_keywords
    timestamp: str


class PaperMiner:
    """Extract insights from bulk papers"""

    def __init__(self):
        # Common research methods to detect
        self.method_patterns = [
            r'\b(deep learning|neural network|cnn|rnn|lstm|transformer|attention)\b',
            r'\b(random forest|svm|decision tree|logistic regression|linear regression)\b',
            r'\b(reinforcement learning|q-learning|policy gradient|actor-critic)\b',
            r'\b(supervised learning|unsupervised learning|semi-supervised)\b',
            r'\b(cross-validation|train-test split|bootstrapping)\b',
            r'\b(gradient descent|sgd|adam|backpropagation)\b',
            r'\b(clustering|k-means|dbscan|hierarchical)\b',
            r'\b(dimensionality reduction|pca|t-sne|umap)\b',
            r'\b(ensemble method|bagging|boosting|stacking)\b',
            r'\b(transfer learning|fine-tuning|pre-training)\b',
        ]

        # Gap indicators
        self.gap_keywords = {
            'methodological': ['limitation', 'challenge', 'difficulty', 'problem', 'issue'],
            'empirical': ['lack of data', 'insufficient', 'need for', 'require more'],
            'theoretical': ['not well understood', 'unclear', 'unexplained', 'theoretical gap'],
            'application': ['real-world', 'practical', 'deployment', 'production'],
        }

    def analyze(self, papers: List[Paper]) -> MiningResult:
        """Perform comprehensive analysis"""

        if not papers:
            raise ValueError("No papers to analyze")

        # Extract basic statistics
        total_papers = len(papers)
        years = [p.year for p in papers]
        date_range = (min(years), max(years))

        # Analyze authors
        top_authors = self._analyze_authors(papers)

        # Analyze venues
        top_venues = self._analyze_venues(papers)

        # Extract methods
        common_methods = self._extract_methods(papers)

        # Identify trends
        trending_topics = self._identify_trends(papers)

        # Find research gaps
        research_gaps = self._find_gaps(papers)

        # Build citation network
        citation_network = self._build_citation_network(papers)

        # Temporal trends
        temporal_trends = self._analyze_temporal(papers)

        # Keyword evolution
        keyword_evolution = self._analyze_keyword_evolution(papers)

        result = MiningResult(
            total_papers=total_papers,
            date_range=date_range,
            top_authors=top_authors[:20],
            top_venues=top_venues[:15],
            common_methods=common_methods[:20],
            trending_topics=trending_topics[:10],
            research_gaps=research_gaps[:15],
            citation_network=citation_network,
            temporal_trends=temporal_trends,
            keyword_evolution=keyword_evolution,
            timestamp=datetime.now().isoformat()
        )

        return result

    def _analyze_authors(self, papers: List[Paper]) -> List[tuple[str, int]]:
        """Count papers per author"""
        author_counts = Counter()

        for paper in papers:
            for author in paper.authors:
                author_counts[author] += 1

        return author_counts.most_common()

    def _analyze_venues(self, papers: List[Paper]) -> List[tuple[str, int]]:
        """Count papers per venue"""
        venue_counts = Counter(p.venue for p in papers)
        return venue_counts.most_common()

    def _extract_methods(self, papers: List[Paper]) -> List[tuple[str, int]]:
        """Extract common research methods"""
        method_counts = Counter()

        for paper in papers:
            text = (paper.title + " " + paper.abstract).lower()

            for pattern in self.method_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Normalize method name
                    method = match.lower().strip()
                    method_counts[method] += 1

        return method_counts.most_common()

    def _identify_trends(self, papers: List[Paper]) -> List[Trend]:
        """Identify trending research topics"""
        # Group papers by year and extract keywords
        year_keywords = defaultdict(list)

        for paper in papers:
            keywords = self._extract_keywords(paper.title + " " + paper.abstract)
            year_keywords[paper.year].extend(keywords)

        # Calculate keyword growth rates
        keyword_trends = defaultdict(lambda: defaultdict(int))

        for year, keywords in year_keywords.items():
            keyword_counts = Counter(keywords)
            for keyword, count in keyword_counts.items():
                keyword_trends[keyword][year] = count

        # Calculate momentum for each keyword
        trends = []

        for keyword, year_counts in keyword_trends.items():
            if len(year_counts) < 2:
                continue

            years = sorted(year_counts.keys())
            if len(years) < 2:
                continue

            # Calculate growth rate (simple linear)
            early_count = sum(year_counts[y] for y in years[:len(years)//2])
            late_count = sum(year_counts[y] for y in years[len(years)//2:])

            if early_count > 0:
                momentum = (late_count - early_count) / early_count
            else:
                momentum = late_count

            # Check if emerging (recent surge)
            recent_years = years[-2:]
            recent_count = sum(year_counts[y] for y in recent_years)
            emerging = recent_count > early_count * 1.5

            # Find key papers
            key_papers = [
                p.title for p in papers
                if keyword in (p.title + " " + p.abstract).lower()
            ][:3]

            trend = Trend(
                topic=keyword,
                momentum=round(momentum, 2),
                papers_per_year=dict(year_counts),
                key_papers=key_papers,
                emerging=emerging
            )

            trends.append(trend)

        # Sort by momentum
        trends.sort(key=lambda t: t.momentum, reverse=True)

        return trends

    def _find_gaps(self, papers: List[Paper]) -> List[ResearchGap]:
        """Identify research gaps"""
        gaps = []

        for gap_type, keywords in self.gap_keywords.items():
            evidence = []
            frequency = 0

            for paper in papers:
                text = (paper.title + " " + paper.abstract).lower()

                # Look for gap indicators
                for keyword in keywords:
                    if keyword in text:
                        frequency += 1

                        # Extract surrounding context
                        idx = text.find(keyword)
                        if idx != -1:
                            context = text[max(0, idx-50):idx+100]
                            evidence.append(f"{paper.title[:40]}...: ...{context}...")

            if frequency >= 3:  # Minimum threshold
                # Calculate impact score
                impact_score = min(10.0, frequency / len(papers) * 100)

                gap = ResearchGap(
                    gap_type=gap_type,
                    description=f"Multiple papers indicate {gap_type} gaps",
                    evidence=evidence[:5],  # Top 5 examples
                    frequency=frequency,
                    impact_score=round(impact_score, 1)
                )

                gaps.append(gap)

        # Sort by impact
        gaps.sort(key=lambda g: g.impact_score, reverse=True)

        return gaps

    def _build_citation_network(self, papers: List[Paper]) -> Dict[str, List[str]]:
        """Build citation graph"""
        network = defaultdict(list)

        for paper in papers:
            for ref in paper.references:
                # Find if reference is in our dataset
                for other in papers:
                    if ref.lower() in other.title.lower():
                        network[other.paper_id].append(paper.paper_id)

        return dict(network)

    def _analyze_temporal(self, papers: List[Paper]) -> Dict[int, int]:
        """Analyze publication trends over time"""
        year_counts = Counter(p.year for p in papers)
        return dict(year_counts)

    def _analyze_keyword_evolution(self, papers: List[Paper]) -> Dict[int, List[str]]:
        """Track keyword evolution over time"""
        year_keywords = defaultdict(list)

        for paper in papers:
            keywords = self._extract_keywords(paper.title + " " + paper.abstract)
            year_keywords[paper.year].extend(keywords)

        # Get top keywords per year
        evolution = {}
        for year, keywords in year_keywords.items():
            top_keywords = [kw for kw, _ in Counter(keywords).most_common(10)]
            evolution[year] = top_keywords

        return evolution

    def _extract_keywords(self, text: str, min_freq: int = 1) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (in production would use TF-IDF, NER, etc.)
        text = text.lower()

        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                     'this', 'that', 'these', 'those', 'we', 'our', 'paper', 'study'}

        # Extract words
        words = re.findall(r'\b[a-z]+\b', text)
        keywords = [w for w in words if w not in stopwords and len(w) > 3]

        # Extract bigrams
        bigrams = [f"{words[i]} {words[i+1]}"
                   for i in range(len(words)-1)
                   if words[i] not in stopwords and words[i+1] not in stopwords]

        return keywords + bigrams


def print_results(result: MiningResult):
    """Print formatted mining results"""
    print(f"\n{'='*70}")
    print(f"PAPER MINING RESULTS")
    print(f"{'='*70}\n")

    print(f"Total papers analyzed: {result.total_papers}")
    print(f"Date range: {result.date_range[0]}-{result.date_range[1]}")

    print(f"\n{'='*70}")
    print(f"TOP AUTHORS")
    print(f"{'='*70}")
    for author, count in result.top_authors[:10]:
        print(f"  {author:30s} {count:3d} papers")

    print(f"\n{'='*70}")
    print(f"TOP VENUES")
    print(f"{'='*70}")
    for venue, count in result.top_venues[:10]:
        print(f"  {venue:30s} {count:3d} papers")

    print(f"\n{'='*70}")
    print(f"COMMON METHODS")
    print(f"{'='*70}")
    for method, count in result.common_methods[:15]:
        bar = '#' * int(count / result.total_papers * 50)
        print(f"  {method:30s} [{bar:20s}] {count:3d}")

    print(f"\n{'='*70}")
    print(f"TRENDING TOPICS")
    print(f"{'='*70}")
    for trend in result.trending_topics[:10]:
        status = "EMERGING" if trend.emerging else "growing"
        print(f"\n  {trend.topic} ({status})")
        print(f"    Momentum: {trend.momentum:+.2f}")
        print(f"    Papers: {sum(trend.papers_per_year.values())}")
        if trend.key_papers:
            print(f"    Key paper: {trend.key_papers[0][:50]}...")

    print(f"\n{'='*70}")
    print(f"RESEARCH GAPS")
    print(f"{'='*70}")
    for gap in result.research_gaps[:5]:
        print(f"\n  {gap.gap_type.upper()} GAP")
        print(f"    Frequency: {gap.frequency} papers")
        print(f"    Impact score: {gap.impact_score}/10")
        if gap.evidence:
            print(f"    Example: {gap.evidence[0][:80]}...")

    print(f"\n{'='*70}")
    print(f"TEMPORAL TRENDS")
    print(f"{'='*70}")
    for year in sorted(result.temporal_trends.keys())[-10:]:
        count = result.temporal_trends[year]
        bar = '#' * int(count / max(result.temporal_trends.values()) * 40)
        print(f"  {year}: [{bar:40s}] {count:3d} papers")


def main():
    parser = argparse.ArgumentParser(
        description="PaperMiner - Bulk Research Paper Analysis"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Comprehensive analysis')
    analyze_parser.add_argument('--input', required=True, help='Papers JSON file (list of papers)')
    analyze_parser.add_argument('--output', help='Output JSON file')

    # Trends command
    trends_parser = subparsers.add_parser('trends', help='Extract trending topics')
    trends_parser.add_argument('--input', required=True, help='Papers JSON file')
    trends_parser.add_argument('--field', help='Filter by field')

    # Gaps command
    gaps_parser = subparsers.add_parser('gaps', help='Find research gaps')
    gaps_parser.add_argument('--input', required=True, help='Papers JSON file')
    gaps_parser.add_argument('--output', help='Output report file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Load papers
    with open(args.input, 'r') as f:
        papers_data = json.load(f)

    papers = [Paper(**p) for p in papers_data]

    # Filter if needed
    if hasattr(args, 'field') and args.field:
        papers = [p for p in papers if p.field.lower() == args.field.lower()]

    miner = PaperMiner()

    if args.command == 'analyze':
        result = miner.analyze(papers)
        print_results(result)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(result), f, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'trends':
        result = miner.analyze(papers)

        print(f"\nTrending topics in {args.field or 'all fields'}:")
        print("=" * 70)

        for trend in result.trending_topics[:15]:
            status = "EMERGING" if trend.emerging else "growing"
            print(f"\n{trend.topic} ({status})")
            print(f"  Momentum: {trend.momentum:+.2f}")
            print(f"  Total papers: {sum(trend.papers_per_year.values())}")

    elif args.command == 'gaps':
        result = miner.analyze(papers)

        output = []
        output.append("RESEARCH GAPS REPORT")
        output.append("=" * 70)
        output.append(f"\nAnalyzed {result.total_papers} papers\n")

        for gap in result.research_gaps:
            output.append(f"\n{gap.gap_type.upper()} GAP")
            output.append(f"Impact: {gap.impact_score}/10")
            output.append(f"Frequency: {gap.frequency} papers")
            output.append("\nEvidence:")
            for evidence in gap.evidence[:3]:
                output.append(f"  - {evidence[:100]}...")
            output.append("")

        report = "\n".join(output)
        print(report)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
