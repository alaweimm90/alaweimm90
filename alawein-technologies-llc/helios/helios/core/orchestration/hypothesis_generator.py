"""
Hypothesis Generator

Generates research hypotheses from topics using literature search and LLM analysis.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from self_refutation import Hypothesis, HypothesisDomain


class HypothesisCandidate:
    """A candidate hypothesis with metadata"""

    def __init__(
        self,
        hypothesis: Hypothesis,
        novelty_score: float,
        feasibility_score: float,
        justification: str,
        related_papers: List[Dict],
    ):
        self.hypothesis = hypothesis
        self.novelty_score = novelty_score
        self.feasibility_score = feasibility_score
        self.justification = justification
        self.related_papers = related_papers

    @property
    def combined_score(self) -> float:
        """Combined score (novelty + feasibility)"""
        return (self.novelty_score * 0.6) + (self.feasibility_score * 0.4)


class HypothesisGenerator:
    """
    Generate research hypotheses from topics

    Process:
    1. Query literature (Semantic Scholar)
    2. Identify gaps using LLM
    3. Generate hypothesis candidates
    4. Score novelty and feasibility
    5. Rank and return top candidates
    """

    def __init__(self, orchestrator=None, cache_ttl_seconds: int = 900):
        """
        Initialize generator

        Args:
            orchestrator: AI Orchestrator for LLM calls
            cache_ttl_seconds: Seconds to cache literature lookups to avoid rate limits
        """
        self.orchestrator = orchestrator
        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1"
        self._cache_ttl_seconds = cache_ttl_seconds
        self._literature_cache: Dict[Tuple[str, int], Tuple[datetime, List[Dict]]] = {}

    async def generate_hypotheses(
        self,
        topic: str,
        domain: str = "optimization",
        num_candidates: int = 5,
        search_limit: int = 20,
    ) -> List[HypothesisCandidate]:
        """
        Generate hypotheses for a research topic

        Args:
            topic: Research topic (e.g., "QAP solving with reinforcement learning")
            domain: Scientific domain
            num_candidates: Number of hypotheses to generate
            search_limit: Number of papers to search

        Returns:
            List of ranked HypothesisCandidate objects
        """
        # 1. Literature search
        papers = await self._search_literature(topic, limit=search_limit)

        # 2. Analyze literature and identify gaps
        gaps = await self._identify_gaps(topic, papers)

        # 3. Generate hypothesis candidates
        candidates = await self._generate_candidates(
            topic, domain, gaps, papers, num_candidates
        )

        # 4. Score candidates
        scored_candidates = await self._score_candidates(candidates, papers)

        # 5. Rank and return top N
        scored_candidates.sort(key=lambda x: x.combined_score, reverse=True)

        return scored_candidates[:num_candidates]

    async def _search_literature(self, query: str, limit: int = 20) -> List[Dict]:
        """Search Semantic Scholar for relevant papers (cached + non-blocking)"""
        cache_key = (query.lower(), limit)
        cached = self._literature_cache.get(cache_key)

        if cached:
            cached_at, data = cached
            if datetime.now() - cached_at < timedelta(seconds=self._cache_ttl_seconds):
                return data

        data = await asyncio.to_thread(self._fetch_literature, query, limit)

        if data:
            self._literature_cache[cache_key] = (datetime.now(), data)

        return data or []

    def _fetch_literature(self, query: str, limit: int) -> List[Dict]:
        """Blocking HTTP call for literature search (runs in a worker thread)"""
        try:
            search_url = f"{self.semantic_scholar_api}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,citationCount,authors,venue",
            }

            response = requests.get(search_url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
        except Exception as e:
            print(f"Literature search failed: {e}")
        return []

    async def _identify_gaps(self, topic: str, papers: List[Dict]) -> List[str]:
        """Identify research gaps using LLM analysis"""
        if not self.orchestrator:
            return self._heuristic_gaps(topic, papers)

        from atlas_orchestrator import Task, TaskType

        # Summarize papers
        papers_summary = "\n".join(
            [
                f"- {p.get('title', 'Unknown')} ({p.get('year', 'N/A')}): {p.get('abstract', 'No abstract')[:200]}..."
                for p in papers[:10]
            ]
        )

        prompt = f"""Analyze this research area and identify 5 key research gaps:

Topic: {topic}

Recent papers:
{papers_summary}

Identify gaps where:
1. Methods haven't been tried yet
2. Combinations are unexplored
3. Assumptions can be relaxed
4. Scalability is unclear
5. New domains can be tested

List 5 specific, actionable research gaps.
Format: One gap per line, starting with "•"
"""

        task = Task(prompt=prompt, task_type=TaskType.RESEARCH, max_tokens=600)

        try:
            result = await self.orchestrator.execute(task)

            if result.success:
                gaps = []
                for line in result.content.strip().split("\n"):
                    line = line.strip()
                    if line and (line.startswith("•") or line.startswith("-")):
                        gap = line.lstrip("•-").strip()
                        if len(gap) > 15:
                            gaps.append(gap)
                return gaps[:5]
        except Exception:
            pass

        return self._heuristic_gaps(topic, papers)

    def _heuristic_gaps(self, topic: str, papers: List[Dict]) -> List[str]:
        """Heuristic gap identification"""
        gaps = [
            f"Novel application of existing methods to {topic}",
            f"Combining multiple approaches for {topic}",
            f"Scalability improvements for {topic}",
            f"Theoretical analysis of {topic}",
            f"Experimental validation on new domains for {topic}",
        ]
        return gaps

    async def _generate_candidates(
        self,
        topic: str,
        domain: str,
        gaps: List[str],
        papers: List[Dict],
        num_candidates: int,
    ) -> List[HypothesisCandidate]:
        """Generate hypothesis candidates from gaps"""
        if not self.orchestrator:
            return self._heuristic_candidates(topic, domain, gaps, num_candidates)

        from atlas_orchestrator import Task, TaskType

        gaps_text = "\n".join([f"{i+1}. {gap}" for i, gap in enumerate(gaps)])

        prompt = f"""Generate {num_candidates} research hypotheses for this topic:

Topic: {topic}
Domain: {domain}

Identified research gaps:
{gaps_text}

For each hypothesis, provide:
- Hypothesis claim (1-2 sentences, testable, specific)
- Brief justification (why this addresses a gap)
- Expected contribution

Format each as:
HYPOTHESIS {i}:
Claim: [specific testable claim]
Justification: [why this matters]
Contribution: [expected impact]

Generate {num_candidates} hypotheses addressing different gaps.
"""

        task = Task(prompt=prompt, task_type=TaskType.RESEARCH, max_tokens=1000)

        try:
            result = await self.orchestrator.execute(task)

            if result.success:
                return self._parse_hypotheses(result.content, domain, gaps, papers)
        except Exception:
            pass

        return self._heuristic_candidates(topic, domain, gaps, num_candidates)

    def _parse_hypotheses(
        self, content: str, domain: str, gaps: List[str], papers: List[Dict]
    ) -> List[HypothesisCandidate]:
        """Parse LLM output into hypothesis candidates"""
        candidates = []

        # Split by HYPOTHESIS markers
        sections = content.split("HYPOTHESIS")[1:]  # Skip first empty

        for section in sections:
            try:
                lines = section.strip().split("\n")

                claim = None
                justification = ""

                for line in lines:
                    line = line.strip()
                    if line.startswith("Claim:"):
                        claim = line.replace("Claim:", "").strip()
                    elif line.startswith("Justification:"):
                        justification = line.replace("Justification:", "").strip()

                if claim and len(claim) > 20:
                    # Create hypothesis
                    try:
                        hyp_domain = HypothesisDomain(domain)
                    except ValueError:
                        hyp_domain = HypothesisDomain.GENERAL

                    hypothesis = Hypothesis(
                        claim=claim,
                        domain=hyp_domain,
                        context=f"Generated from research gaps analysis",
                    )

                    candidate = HypothesisCandidate(
                        hypothesis=hypothesis,
                        novelty_score=0.7,  # Default, will be scored later
                        feasibility_score=0.6,
                        justification=justification,
                        related_papers=papers[:5],
                    )

                    candidates.append(candidate)
            except Exception:
                continue

        return candidates

    def _heuristic_candidates(
        self, topic: str, domain: str, gaps: List[str], num_candidates: int
    ) -> List[HypothesisCandidate]:
        """Heuristic candidate generation fallback"""
        candidates = []

        for i, gap in enumerate(gaps[:num_candidates]):
            claim = f"{topic}: {gap}"

            try:
                hyp_domain = HypothesisDomain(domain)
            except ValueError:
                hyp_domain = HypothesisDomain.GENERAL

            hypothesis = Hypothesis(claim=claim, domain=hyp_domain)

            candidate = HypothesisCandidate(
                hypothesis=hypothesis,
                novelty_score=0.6,
                feasibility_score=0.5,
                justification=f"Addresses research gap: {gap}",
                related_papers=[],
            )

            candidates.append(candidate)

        return candidates

    async def _score_candidates(
        self, candidates: List[HypothesisCandidate], papers: List[Dict]
    ) -> List[HypothesisCandidate]:
        """Score candidates for novelty and feasibility"""
        for candidate in candidates:
            # Novelty: Compare against existing papers
            candidate.novelty_score = self._score_novelty(candidate, papers)

            # Feasibility: Assess computational feasibility
            candidate.feasibility_score = self._score_feasibility(candidate)

        return candidates

    def _score_novelty(
        self, candidate: HypothesisCandidate, papers: List[Dict]
    ) -> float:
        """Score novelty (0-1)"""
        # Simple keyword overlap check
        claim_words = set(candidate.hypothesis.claim.lower().split())

        if not papers:
            return 0.8  # Assume novel if no papers found

        overlaps = []
        for paper in papers[:10]:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            paper_text = f"{title} {abstract}"
            paper_words = set(paper_text.split())

            # Jaccard similarity
            intersection = len(claim_words & paper_words)
            union = len(claim_words | paper_words)
            similarity = intersection / union if union > 0 else 0

            overlaps.append(similarity)

        # Lower overlap = higher novelty
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
        novelty = 1.0 - (avg_overlap * 2)  # Scale

        return max(0.0, min(1.0, novelty))

    def _score_feasibility(self, candidate: HypothesisCandidate) -> float:
        """Score computational feasibility (0-1)"""
        claim = candidate.hypothesis.claim.lower()

        # Negative indicators
        if any(
            term in claim
            for term in ["impossible", "violates", "requires physical lab"]
        ):
            return 0.2

        # Positive indicators
        score = 0.5

        if any(term in claim for term in ["algorithm", "method", "technique"]):
            score += 0.2

        if any(term in claim for term in ["improves", "optimizes", "enhances"]):
            score += 0.1

        if any(term in claim for term in ["computational", "simulation", "model"]):
            score += 0.2

        return min(1.0, score)
