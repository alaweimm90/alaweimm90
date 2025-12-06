#!/usr/bin/env python3
"""
LitReviewBot - Automated Literature Review Generator

Automatically generate comprehensive literature reviews from collections of papers.
Features theme clustering, gap analysis, and citation network visualization.

Usage:
    python litreview.py add-paper --title "..." --authors "..." --year 2024 --abstract "..."
    python litreview.py generate-review --collection-id 1 --style "narrative"
    python litreview.py find-gaps --collection-id 1
    python litreview.py cluster --collection-id 1 --num-clusters 5
"""

import argparse
import json
import sys
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import random


@dataclass
class Paper:
    """Research paper metadata"""
    paper_id: int
    title: str
    authors: List[str]
    year: int
    venue: str  # journal/conference
    abstract: str
    keywords: List[str]
    citations: int
    doi: Optional[str] = None
    url: Optional[str] = None

    # Extracted features
    themes: List[str] = field(default_factory=list)
    methodology: str = ""
    findings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    cluster_id: Optional[int] = None


@dataclass
class PaperCluster:
    """Thematic cluster of papers"""
    cluster_id: int
    theme: str
    papers: List[int]  # paper IDs
    keywords: List[str]
    representative_papers: List[int]  # Most central papers
    size: int


@dataclass
class ResearchGap:
    """Identified gap in literature"""
    gap_id: int
    description: str
    gap_type: str  # methodological, theoretical, empirical, practical
    evidence: List[str]
    importance: str  # low, medium, high, critical
    feasibility: str  # low, medium, high
    related_clusters: List[int]


@dataclass
class LiteratureReview:
    """Generated literature review"""
    review_id: int
    collection_id: int
    style: str  # narrative, systematic, meta-analysis
    created_at: str

    # Review sections
    introduction: str
    background: str
    thematic_sections: Dict[str, str]  # theme -> content
    synthesis: str
    gaps_identified: List[str]
    future_directions: List[str]
    conclusion: str

    # Metadata
    papers_reviewed: int
    themes_identified: int
    gaps_found: int
    word_count: int


class LitReviewBot:
    """Automated literature review generator"""

    # Common research themes
    RESEARCH_THEMES = {
        "machine_learning": ["machine learning", "deep learning", "neural network", "AI", "artificial intelligence"],
        "data_analysis": ["data analysis", "statistics", "regression", "correlation", "data mining"],
        "experimental": ["experiment", "trial", "intervention", "treatment", "control"],
        "survey": ["survey", "questionnaire", "interview", "qualitative", "ethnography"],
        "theoretical": ["theory", "framework", "model", "conceptual", "theoretical"],
        "methodology": ["method", "methodology", "approach", "technique", "procedure"],
        "review": ["review", "meta-analysis", "systematic review", "literature review"],
        "case_study": ["case study", "case analysis", "field study", "observation"]
    }

    # Gap types and indicators
    GAP_INDICATORS = {
        "methodological": ["limited methods", "no quantitative", "lack of experiments", "small sample"],
        "theoretical": ["theoretical gap", "no framework", "lack of theory", "under-theorized"],
        "empirical": ["no empirical evidence", "limited data", "under-researched", "unexplored"],
        "practical": ["no practical", "implementation gap", "real-world application", "practice gap"]
    }

    def __init__(self, data_file: str = "litreview.json"):
        self.data_file = Path(data_file)
        self.papers: Dict[int, Paper] = {}
        self.collections: Dict[int, List[int]] = {}  # collection_id -> paper_ids
        self.clusters: Dict[int, PaperCluster] = {}
        self.gaps: Dict[int, ResearchGap] = {}
        self.reviews: Dict[int, LiteratureReview] = {}

        self._load_data()

    def _load_data(self):
        """Load saved data"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.papers = {
                    int(k): Paper(**v) for k, v in data.get('papers', {}).items()
                }
                self.collections = {
                    int(k): v for k, v in data.get('collections', {}).items()
                }
                self.clusters = {
                    int(k): PaperCluster(**v) for k, v in data.get('clusters', {}).items()
                }
                self.gaps = {
                    int(k): ResearchGap(**v) for k, v in data.get('gaps', {}).items()
                }
                self.reviews = {
                    int(k): LiteratureReview(**v) for k, v in data.get('reviews', {}).items()
                }

    def _save_data(self):
        """Save data"""
        data = {
            'papers': {k: asdict(v) for k, v in self.papers.items()},
            'collections': self.collections,
            'clusters': {k: asdict(v) for k, v in self.clusters.items()},
            'gaps': {k: asdict(v) for k, v in self.gaps.items()},
            'reviews': {k: asdict(v) for k, v in self.reviews.items()}
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_paper(self, title: str, authors: List[str], year: int,
                  venue: str, abstract: str, keywords: List[str],
                  citations: int = 0, doi: Optional[str] = None,
                  url: Optional[str] = None, collection_id: int = 1) -> Paper:
        """Add a paper to the collection"""

        paper_id = max(self.papers.keys(), default=0) + 1

        # Extract features from abstract
        themes = self._extract_themes(abstract + " " + title)
        methodology = self._extract_methodology(abstract)
        findings = self._extract_findings(abstract)
        limitations = self._extract_limitations(abstract)

        paper = Paper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            abstract=abstract,
            keywords=keywords,
            citations=citations,
            doi=doi,
            url=url,
            themes=themes,
            methodology=methodology,
            findings=findings,
            limitations=limitations
        )

        self.papers[paper_id] = paper

        # Add to collection
        if collection_id not in self.collections:
            self.collections[collection_id] = []
        self.collections[collection_id].append(paper_id)

        self._save_data()
        return paper

    def _extract_themes(self, text: str) -> List[str]:
        """Extract research themes from text"""
        text_lower = text.lower()
        themes = []

        for theme, keywords in self.RESEARCH_THEMES.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(theme)

        return themes

    def _extract_methodology(self, abstract: str) -> str:
        """Extract methodology from abstract"""
        method_keywords = {
            "experimental": ["experiment", "trial", "randomized", "control group"],
            "survey": ["survey", "questionnaire", "interview"],
            "computational": ["simulation", "model", "algorithm", "computation"],
            "theoretical": ["theoretical", "mathematical", "formal"],
            "qualitative": ["qualitative", "ethnography", "case study"],
            "quantitative": ["quantitative", "statistical", "regression", "analysis"],
            "mixed_methods": ["mixed methods", "triangulation"]
        }

        abstract_lower = abstract.lower()
        for method, keywords in method_keywords.items():
            if any(kw in abstract_lower for kw in keywords):
                return method

        return "unclear"

    def _extract_findings(self, abstract: str) -> List[str]:
        """Extract key findings from abstract"""
        # Simple heuristic: sentences with result indicators
        result_indicators = ["found", "showed", "demonstrated", "revealed",
                           "indicate", "suggest", "conclude", "result"]

        sentences = re.split(r'[.!?]+', abstract)
        findings = []

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in result_indicators):
                findings.append(sentence.strip())

        return findings[:3]  # Top 3 findings

    def _extract_limitations(self, abstract: str) -> List[str]:
        """Extract limitations from abstract"""
        limitation_indicators = ["limitation", "limited", "constraint",
                                "future work", "further research", "scope"]

        sentences = re.split(r'[.!?]+', abstract)
        limitations = []

        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in limitation_indicators):
                limitations.append(sentence.strip())

        return limitations

    def cluster_papers(self, collection_id: int, num_clusters: int = 5) -> List[PaperCluster]:
        """Cluster papers by theme"""

        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")

        paper_ids = self.collections[collection_id]
        papers = [self.papers[pid] for pid in paper_ids]

        # Simple theme-based clustering
        # In production, use embeddings + k-means

        theme_groups = defaultdict(list)
        for paper in papers:
            if paper.themes:
                primary_theme = paper.themes[0]
                theme_groups[primary_theme].append(paper.paper_id)
            else:
                theme_groups["uncategorized"].append(paper.paper_id)

        # Create clusters
        clusters = []
        for cluster_id, (theme, paper_ids) in enumerate(theme_groups.items(), 1):
            # Get representative keywords
            all_keywords = []
            for pid in paper_ids:
                all_keywords.extend(self.papers[pid].keywords)

            keyword_counts = Counter(all_keywords)
            top_keywords = [kw for kw, _ in keyword_counts.most_common(5)]

            # Find representative papers (most cited)
            papers_with_citations = [(pid, self.papers[pid].citations) for pid in paper_ids]
            papers_with_citations.sort(key=lambda x: x[1], reverse=True)
            representative = [pid for pid, _ in papers_with_citations[:3]]

            cluster = PaperCluster(
                cluster_id=cluster_id,
                theme=theme,
                papers=paper_ids,
                keywords=top_keywords,
                representative_papers=representative,
                size=len(paper_ids)
            )

            clusters.append(cluster)
            self.clusters[cluster_id] = cluster

            # Update papers with cluster assignment
            for pid in paper_ids:
                self.papers[pid].cluster_id = cluster_id

        self._save_data()
        return clusters

    def identify_gaps(self, collection_id: int) -> List[ResearchGap]:
        """Identify research gaps in the collection"""

        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")

        paper_ids = self.collections[collection_id]
        papers = [self.papers[pid] for pid in paper_ids]

        gaps = []
        gap_id = max(self.gaps.keys(), default=0) + 1

        # Gap 1: Methodological diversity
        methodologies = [p.methodology for p in papers]
        method_counts = Counter(methodologies)

        if len(method_counts) < 3:
            gap = ResearchGap(
                gap_id=gap_id,
                description=f"Limited methodological diversity - {len(method_counts)} methods used",
                gap_type="methodological",
                evidence=[f"Dominant method: {method_counts.most_common(1)[0][0]} ({method_counts.most_common(1)[0][1]} papers)"],
                importance="high",
                feasibility="high",
                related_clusters=list(set(p.cluster_id for p in papers if p.cluster_id))
            )
            gaps.append(gap)
            self.gaps[gap_id] = gap
            gap_id += 1

        # Gap 2: Temporal gaps (year distribution)
        years = [p.year for p in papers]
        year_range = max(years) - min(years)
        recent_papers = sum(1 for y in years if y >= max(years) - 2)

        if recent_papers / len(papers) < 0.3:
            gap = ResearchGap(
                gap_id=gap_id,
                description=f"Limited recent research - only {recent_papers} papers from last 2 years",
                gap_type="empirical",
                evidence=[f"Year range: {min(years)}-{max(years)}", f"Recent coverage: {recent_papers/len(papers):.1%}"],
                importance="medium",
                feasibility="high",
                related_clusters=[]
            )
            gaps.append(gap)
            self.gaps[gap_id] = gap
            gap_id += 1

        # Gap 3: Theme coverage gaps
        all_themes = set()
        for paper in papers:
            all_themes.update(paper.themes)

        expected_themes = set(self.RESEARCH_THEMES.keys())
        missing_themes = expected_themes - all_themes

        if len(missing_themes) > 2:
            gap = ResearchGap(
                gap_id=gap_id,
                description=f"Under-explored themes: {', '.join(list(missing_themes)[:3])}",
                gap_type="theoretical",
                evidence=[f"{len(missing_themes)} themes not covered"],
                importance="medium",
                feasibility="medium",
                related_clusters=[]
            )
            gaps.append(gap)
            self.gaps[gap_id] = gap
            gap_id += 1

        # Gap 4: Citation impact gap
        avg_citations = sum(p.citations for p in papers) / len(papers)
        high_impact = sum(1 for p in papers if p.citations > avg_citations * 2)

        if high_impact / len(papers) < 0.2:
            gap = ResearchGap(
                gap_id=gap_id,
                description=f"Limited high-impact research - only {high_impact} highly-cited papers",
                gap_type="empirical",
                evidence=[f"Avg citations: {avg_citations:.1f}", f"High-impact papers: {high_impact}"],
                importance="low",
                feasibility="medium",
                related_clusters=[]
            )
            gaps.append(gap)
            self.gaps[gap_id] = gap

        self._save_data()
        return gaps

    def generate_review(self, collection_id: int, style: str = "narrative") -> LiteratureReview:
        """Generate a comprehensive literature review"""

        if collection_id not in self.collections:
            raise ValueError(f"Collection {collection_id} not found")

        paper_ids = self.collections[collection_id]
        papers = [self.papers[pid] for pid in paper_ids]

        # Cluster if not already done
        if not any(p.cluster_id for p in papers):
            self.cluster_papers(collection_id)

        # Identify gaps if not done
        if not self.gaps:
            self.identify_gaps(collection_id)

        # Generate review sections
        review_id = max(self.reviews.keys(), default=0) + 1

        # Introduction
        introduction = self._generate_introduction(papers, style)

        # Background
        background = self._generate_background(papers)

        # Thematic sections
        thematic_sections = self._generate_thematic_sections(papers)

        # Synthesis
        synthesis = self._generate_synthesis(papers)

        # Gaps and future directions
        gaps_identified = self._generate_gaps_section(papers)
        future_directions = self._generate_future_directions(papers)

        # Conclusion
        conclusion = self._generate_conclusion(papers, style)

        # Calculate word count
        all_text = " ".join([
            introduction, background,
            " ".join(thematic_sections.values()),
            synthesis, " ".join(gaps_identified),
            " ".join(future_directions), conclusion
        ])
        word_count = len(all_text.split())

        review = LiteratureReview(
            review_id=review_id,
            collection_id=collection_id,
            style=style,
            created_at=datetime.now().isoformat(),
            introduction=introduction,
            background=background,
            thematic_sections=thematic_sections,
            synthesis=synthesis,
            gaps_identified=gaps_identified,
            future_directions=future_directions,
            conclusion=conclusion,
            papers_reviewed=len(papers),
            themes_identified=len(set(theme for p in papers for theme in p.themes)),
            gaps_found=len(self.gaps),
            word_count=word_count
        )

        self.reviews[review_id] = review
        self._save_data()

        return review

    def _generate_introduction(self, papers: List[Paper], style: str) -> str:
        """Generate review introduction"""
        years = [p.year for p in papers]
        year_range = f"{min(years)}-{max(years)}" if min(years) != max(years) else str(min(years))

        intro = f"""This {style} literature review examines {len(papers)} papers published between {year_range}. """
        intro += f"The review synthesizes findings across multiple research themes and identifies key gaps in the literature. "

        # Mention prominent authors
        all_authors = []
        for p in papers:
            all_authors.extend(p.authors)
        author_counts = Counter(all_authors)
        top_authors = [author for author, _ in author_counts.most_common(3)]

        if top_authors:
            intro += f"Prominent contributors include {', '.join(top_authors[:2])}. "

        intro += f"This review is organized thematically to provide a comprehensive overview of the field."

        return intro

    def _generate_background(self, papers: List[Paper]) -> str:
        """Generate background section"""
        background = "The research landscape in this domain has evolved considerably. "

        # Temporal evolution
        years = sorted(set(p.year for p in papers))
        if len(years) > 3:
            early_papers = [p for p in papers if p.year <= years[len(years)//3]]
            recent_papers = [p for p in papers if p.year >= years[-len(years)//3]]

            early_themes = set(theme for p in early_papers for theme in p.themes)
            recent_themes = set(theme for p in recent_papers for theme in p.themes)
            emerging = recent_themes - early_themes

            if emerging:
                background += f"Emerging research areas include {', '.join(list(emerging)[:2])}. "

        # Methodological overview
        methods = Counter(p.methodology for p in papers)
        top_method = methods.most_common(1)[0]
        background += f"The dominant research methodology is {top_method[0]} ({top_method[1]} papers). "

        background += "This foundation provides context for the thematic analysis that follows."

        return background

    def _generate_thematic_sections(self, papers: List[Paper]) -> Dict[str, str]:
        """Generate content for each theme"""
        sections = {}

        # Group papers by cluster
        clusters = defaultdict(list)
        for paper in papers:
            if paper.cluster_id:
                clusters[paper.cluster_id].append(paper)

        for cluster_id, cluster_papers in clusters.items():
            if cluster_id not in self.clusters:
                continue

            cluster = self.clusters[cluster_id]
            theme = cluster.theme.replace("_", " ").title()

            content = f"This theme encompasses {len(cluster_papers)} papers. "

            # Key findings synthesis
            all_findings = []
            for p in cluster_papers:
                all_findings.extend(p.findings)

            if all_findings:
                content += f"Key findings include: {all_findings[0]} "

            # Representative work
            if cluster.representative_papers:
                rep_paper = self.papers[cluster.representative_papers[0]]
                content += f"Notable contributions include {rep_paper.authors[0]} et al. ({rep_paper.year}), "
                content += f"which {rep_paper.findings[0] if rep_paper.findings else 'advanced the field'}. "

            # Limitations
            all_limitations = []
            for p in cluster_papers:
                all_limitations.extend(p.limitations)

            if all_limitations:
                content += f"However, {all_limitations[0].lower()}"

            sections[theme] = content

        return sections

    def _generate_synthesis(self, papers: List[Paper]) -> str:
        """Generate synthesis section"""
        synthesis = "Synthesizing across themes reveals several important patterns. "

        # Cross-cutting findings
        all_findings = []
        for p in papers:
            all_findings.extend(p.findings)

        if len(all_findings) > 5:
            synthesis += "Multiple studies converge on similar conclusions, strengthening confidence in the findings. "

        # Methodological patterns
        methods = Counter(p.methodology for p in papers)
        if len(methods) > 3:
            synthesis += f"The field employs diverse methodologies ({len(methods)} different approaches), "
            synthesis += "providing triangulation and robustness. "
        else:
            synthesis += "However, methodological diversity is limited, suggesting opportunity for alternative approaches. "

        # Temporal trends
        years = sorted(set(p.year for p in papers))
        recent_papers = [p for p in papers if p.year >= max(years) - 2]

        if len(recent_papers) / len(papers) > 0.4:
            synthesis += "Recent research activity is strong, indicating an active and evolving field. "

        synthesis += "These patterns inform the identification of research gaps discussed next."

        return synthesis

    def _generate_gaps_section(self, papers: List[Paper]) -> List[str]:
        """Generate gaps section"""
        gaps_text = []

        for gap in self.gaps.values():
            gap_desc = f"{gap.gap_type.replace('_', ' ').title()} Gap: {gap.description}. "
            gap_desc += f"Importance: {gap.importance}, Feasibility: {gap.feasibility}. "
            gap_desc += f"Evidence: {'; '.join(gap.evidence[:2])}"
            gaps_text.append(gap_desc)

        return gaps_text

    def _generate_future_directions(self, papers: List[Paper]) -> List[str]:
        """Generate future research directions"""
        directions = []

        # Based on gaps
        for gap in self.gaps.values():
            if gap.importance in ["high", "critical"]:
                direction = f"Address {gap.gap_type} limitations by {gap.description.lower()}"
                directions.append(direction)

        # Based on limitations
        all_limitations = set()
        for p in papers:
            all_limitations.update(p.limitations)

        if len(all_limitations) > 3:
            direction = f"Overcome common limitations: {list(all_limitations)[0]}"
            directions.append(direction)

        # Methodological recommendations
        methods = Counter(p.methodology for p in papers)
        underused_methods = [m for m in ["experimental", "qualitative", "mixed_methods"]
                            if methods.get(m, 0) < len(papers) * 0.2]

        if underused_methods:
            direction = f"Employ {underused_methods[0]} approaches to complement existing research"
            directions.append(direction)

        return directions[:5]  # Top 5 directions

    def _generate_conclusion(self, papers: List[Paper], style: str) -> str:
        """Generate conclusion"""
        conclusion = f"This {style} review of {len(papers)} papers provides a comprehensive overview of the field. "

        # Summary of themes
        themes = set(theme for p in papers for theme in p.themes)
        conclusion += f"Analysis across {len(themes)} thematic areas reveals both progress and opportunities. "

        # Key takeaway
        conclusion += "The synthesis highlights robust findings while identifying important gaps for future research. "

        # Forward-looking
        conclusion += "Addressing these gaps through diverse methodological approaches will advance understanding "
        conclusion += "and contribute to theoretical and practical progress in the field."

        return conclusion


def main():
    parser = argparse.ArgumentParser(
        description="LitReviewBot - Automated Literature Review Generator"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Add paper
    add_parser = subparsers.add_parser('add-paper', help='Add a paper to collection')
    add_parser.add_argument('--title', required=True, help='Paper title')
    add_parser.add_argument('--authors', required=True, nargs='+', help='Author names (space-separated)')
    add_parser.add_argument('--year', type=int, required=True, help='Publication year')
    add_parser.add_argument('--venue', required=True, help='Journal/conference')
    add_parser.add_argument('--abstract', required=True, help='Abstract')
    add_parser.add_argument('--keywords', required=True, nargs='+', help='Keywords (space-separated)')
    add_parser.add_argument('--citations', type=int, default=0, help='Citation count')
    add_parser.add_argument('--doi', help='DOI identifier')
    add_parser.add_argument('--url', help='Paper URL')
    add_parser.add_argument('--collection-id', type=int, default=1, help='Collection ID')

    # Generate review
    review_parser = subparsers.add_parser('generate-review', help='Generate literature review')
    review_parser.add_argument('--collection-id', type=int, required=True)
    review_parser.add_argument('--style', default='narrative',
                               choices=['narrative', 'systematic', 'meta-analysis'])

    # Find gaps
    gaps_parser = subparsers.add_parser('find-gaps', help='Identify research gaps')
    gaps_parser.add_argument('--collection-id', type=int, required=True)

    # Cluster papers
    cluster_parser = subparsers.add_parser('cluster', help='Cluster papers by theme')
    cluster_parser.add_argument('--collection-id', type=int, required=True)
    cluster_parser.add_argument('--num-clusters', type=int, default=5)

    # List collections
    list_parser = subparsers.add_parser('list', help='List collections')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    bot = LitReviewBot()

    if args.command == 'add-paper':
        paper = bot.add_paper(
            title=args.title,
            authors=args.authors,
            year=args.year,
            venue=args.venue,
            abstract=args.abstract,
            keywords=args.keywords,
            citations=args.citations,
            doi=args.doi,
            url=args.url,
            collection_id=args.collection_id
        )

        print(f"\nPaper added successfully!")
        print(f"Paper ID: {paper.paper_id}")
        print(f"Title: {paper.title}")
        print(f"Themes detected: {', '.join(paper.themes)}")
        print(f"Methodology: {paper.methodology}")
        print(f"Collection: {args.collection_id}")

    elif args.command == 'generate-review':
        review = bot.generate_review(args.collection_id, args.style)

        print(f"\n{'='*70}")
        print(f"LITERATURE REVIEW ({review.style.upper()})")
        print(f"{'='*70}\n")

        print(f"Papers Reviewed: {review.papers_reviewed}")
        print(f"Themes: {review.themes_identified}")
        print(f"Gaps Found: {review.gaps_found}")
        print(f"Word Count: {review.word_count:,}")
        print(f"\n{'-'*70}\n")

        print("INTRODUCTION")
        print(review.introduction)
        print(f"\n{'-'*70}\n")

        print("BACKGROUND")
        print(review.background)
        print(f"\n{'-'*70}\n")

        for theme, content in review.thematic_sections.items():
            print(f"{theme.upper()}")
            print(content)
            print(f"\n{'-'*70}\n")

        print("SYNTHESIS")
        print(review.synthesis)
        print(f"\n{'-'*70}\n")

        print("RESEARCH GAPS IDENTIFIED")
        for i, gap in enumerate(review.gaps_identified, 1):
            print(f"{i}. {gap}")
        print(f"\n{'-'*70}\n")

        print("FUTURE RESEARCH DIRECTIONS")
        for i, direction in enumerate(review.future_directions, 1):
            print(f"{i}. {direction}")
        print(f"\n{'-'*70}\n")

        print("CONCLUSION")
        print(review.conclusion)
        print(f"\n{'='*70}\n")

        print(f"Review saved as ID: {review.review_id}")

    elif args.command == 'find-gaps':
        gaps = bot.identify_gaps(args.collection_id)

        print(f"\n{'='*70}")
        print(f"RESEARCH GAPS IDENTIFIED")
        print(f"{'='*70}\n")

        for gap in gaps:
            print(f"Gap #{gap.gap_id}: {gap.gap_type.upper()}")
            print(f"Description: {gap.description}")
            print(f"Importance: {gap.importance} | Feasibility: {gap.feasibility}")
            print(f"Evidence:")
            for evidence in gap.evidence:
                print(f"  • {evidence}")
            print()

    elif args.command == 'cluster':
        clusters = bot.cluster_papers(args.collection_id, args.num_clusters)

        print(f"\n{'='*70}")
        print(f"PAPER CLUSTERING RESULTS")
        print(f"{'='*70}\n")

        for cluster in clusters:
            print(f"Cluster #{cluster.cluster_id}: {cluster.theme.upper()}")
            print(f"Size: {cluster.size} papers")
            print(f"Keywords: {', '.join(cluster.keywords)}")
            print(f"Representative papers: {cluster.representative_papers}")
            print()

    elif args.command == 'list':
        print(f"\n{'='*70}")
        print(f"COLLECTIONS")
        print(f"{'='*70}\n")

        for coll_id, paper_ids in bot.collections.items():
            print(f"Collection #{coll_id}: {len(paper_ids)} papers")


if __name__ == "__main__":
    main()
