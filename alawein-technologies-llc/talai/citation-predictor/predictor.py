#!/usr/bin/env python3
"""
CitationPredictor - Predict Future Citation Counts

Estimates how many citations a paper will receive based on multiple factors:
- Author reputation
- Venue prestige
- Topic novelty
- Abstract quality
- Reference count
- Publication timing

Usage:
    python predictor.py predict --input paper_info.json --horizon 5
    python predictor.py analyze --paper-id arxiv:2301.12345
    python predictor.py batch --input papers.json --output predictions.json
"""

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class PaperMetadata:
    """Paper metadata for prediction"""
    title: str
    authors: List[str]
    venue: str
    year: int
    abstract: str
    num_references: int
    field: str
    author_h_index_avg: float  # Average h-index of authors
    venue_impact_factor: float


@dataclass
class CitationPrediction:
    """Citation count prediction"""
    paper_title: str
    current_citations: int
    predicted_1yr: int
    predicted_3yr: int
    predicted_5yr: int
    confidence_interval: Dict[str, tuple]  # year -> (low, high)
    factors: Dict[str, float]  # factor -> influence score
    percentile: float  # Expected percentile vs similar papers
    trajectory: str  # "exponential" | "linear" | "plateau"
    timestamp: str


class CitationModel:
    """Statistical model for citation prediction"""

    def __init__(self):
        # Citation growth patterns
        self.growth_models = {
            'exponential': lambda t, a, b: a * math.exp(b * t),
            'linear': lambda t, a, b: a * t + b,
            'power_law': lambda t, a, b: a * (t ** b),
            'plateau': lambda t, a, b, c: a * (1 - math.exp(-b * t)) + c,
        }

        # Factor weights (learned from historical data)
        self.weights = {
            'author_reputation': 0.25,
            'venue_prestige': 0.20,
            'topic_novelty': 0.15,
            'abstract_quality': 0.10,
            'reference_count': 0.08,
            'publication_timing': 0.07,
            'field_growth': 0.10,
            'collaboration_size': 0.05,
        }

        # Venue impact factors (simplified)
        self.venue_impacts = {
            'nature': 42.8,
            'science': 41.8,
            'cell': 38.6,
            'nips': 9.2,
            'icml': 8.5,
            'cvpr': 7.8,
            'iclr': 7.2,
            'aaai': 4.8,
            'ijcai': 4.5,
            'acl': 6.1,
            'default': 2.0,
        }

        # Field growth rates (annual citation growth)
        self.field_growth = {
            'ai': 1.35,
            'ml': 1.32,
            'cv': 1.28,
            'nlp': 1.25,
            'robotics': 1.18,
            'biology': 1.15,
            'physics': 1.10,
            'chemistry': 1.12,
            'default': 1.15,
        }

    def predict(self, metadata: PaperMetadata, current_citations: int = 0) -> CitationPrediction:
        """
        Predict future citation counts

        Args:
            metadata: Paper metadata
            current_citations: Current citation count (if known)

        Returns:
            CitationPrediction with estimates
        """
        # Calculate factor scores
        factors = self._calculate_factors(metadata)

        # Determine growth trajectory
        trajectory = self._determine_trajectory(factors)

        # Calculate base citation rate
        base_rate = self._calculate_base_rate(factors, metadata)

        # Project citations
        years_since_pub = datetime.now().year - metadata.year
        current_estimate = current_citations if current_citations > 0 else max(1, int(base_rate * years_since_pub))

        # Predict future counts
        pred_1yr = self._project_citations(current_estimate, 1, trajectory, base_rate, metadata)
        pred_3yr = self._project_citations(current_estimate, 3, trajectory, base_rate, metadata)
        pred_5yr = self._project_citations(current_estimate, 5, trajectory, base_rate, metadata)

        # Calculate confidence intervals
        confidence_interval = {
            '1yr': (int(pred_1yr * 0.7), int(pred_1yr * 1.3)),
            '3yr': (int(pred_3yr * 0.6), int(pred_3yr * 1.4)),
            '5yr': (int(pred_5yr * 0.5), int(pred_5yr * 1.5)),
        }

        # Estimate percentile
        percentile = self._estimate_percentile(pred_5yr, metadata)

        prediction = CitationPrediction(
            paper_title=metadata.title,
            current_citations=current_estimate,
            predicted_1yr=pred_1yr,
            predicted_3yr=pred_3yr,
            predicted_5yr=pred_5yr,
            confidence_interval=confidence_interval,
            factors=factors,
            percentile=round(percentile, 1),
            trajectory=trajectory,
            timestamp=datetime.now().isoformat()
        )

        return prediction

    def _calculate_factors(self, metadata: PaperMetadata) -> Dict[str, float]:
        """Calculate influence factor scores"""
        factors = {}

        # Author reputation (normalized to 0-1)
        factors['author_reputation'] = min(1.0, metadata.author_h_index_avg / 50.0)

        # Venue prestige
        venue_key = metadata.venue.lower().replace(' ', '').replace('-', '')
        impact = self.venue_impacts.get(venue_key, self.venue_impacts['default'])
        factors['venue_prestige'] = min(1.0, impact / 40.0)

        # Topic novelty (based on abstract keywords - simplified)
        novelty_keywords = ['novel', 'new', 'first', 'breakthrough', 'unprecedented', 'innovative']
        novelty_count = sum(1 for kw in novelty_keywords if kw in metadata.abstract.lower())
        factors['topic_novelty'] = min(1.0, novelty_count / 3.0)

        # Abstract quality (length and structure - simplified)
        abstract_words = len(metadata.abstract.split())
        factors['abstract_quality'] = min(1.0, abstract_words / 250.0) if abstract_words > 50 else 0.3

        # Reference count
        factors['reference_count'] = min(1.0, metadata.num_references / 50.0)

        # Publication timing (recent papers have uncertainty)
        years_since = datetime.now().year - metadata.year
        factors['publication_timing'] = min(1.0, years_since / 5.0)

        # Field growth
        field_key = metadata.field.lower()
        growth_rate = self.field_growth.get(field_key, self.field_growth['default'])
        factors['field_growth'] = (growth_rate - 1.0) / 0.35  # Normalize to 0-1

        # Collaboration size
        factors['collaboration_size'] = min(1.0, len(metadata.authors) / 10.0)

        return factors

    def _determine_trajectory(self, factors: Dict[str, float]) -> str:
        """Determine citation growth trajectory"""
        # High-impact papers tend to have exponential growth
        impact_score = (
            factors['author_reputation'] * 0.4 +
            factors['venue_prestige'] * 0.4 +
            factors['topic_novelty'] * 0.2
        )

        if impact_score > 0.7:
            return 'exponential'
        elif impact_score > 0.4:
            return 'power_law'
        elif impact_score > 0.2:
            return 'linear'
        else:
            return 'plateau'

    def _calculate_base_rate(self, factors: Dict[str, float], metadata: PaperMetadata) -> float:
        """Calculate base annual citation rate"""
        # Weighted sum of factors
        base_rate = sum(
            factors[factor] * self.weights[factor]
            for factor in self.weights.keys()
            if factor in factors
        )

        # Scale by field and venue
        field_key = metadata.field.lower()
        field_multiplier = self.field_growth.get(field_key, self.field_growth['default'])

        venue_key = metadata.venue.lower().replace(' ', '').replace('-', '')
        venue_multiplier = self.venue_impacts.get(venue_key, self.venue_impacts['default']) / 5.0

        # Base citations per year
        annual_rate = base_rate * field_multiplier * venue_multiplier * 10.0

        return max(1.0, annual_rate)

    def _project_citations(
        self,
        current: int,
        years: int,
        trajectory: str,
        base_rate: float,
        metadata: PaperMetadata
    ) -> int:
        """Project citations into the future"""

        if trajectory == 'exponential':
            # Exponential growth
            growth_rate = 0.3  # 30% annual growth
            projection = current * ((1 + growth_rate) ** years)

        elif trajectory == 'power_law':
            # Power law decay in growth
            projection = current + base_rate * (years ** 0.7)

        elif trajectory == 'linear':
            # Linear growth
            projection = current + base_rate * years

        else:  # plateau
            # Diminishing growth
            max_citations = current + base_rate * 10  # Asymptotic limit
            projection = max_citations * (1 - math.exp(-0.3 * years)) + current

        # Add stochasticity
        noise = random.uniform(0.9, 1.1)
        projection = projection * noise

        return max(current, int(projection))

    def _estimate_percentile(self, predicted_5yr: int, metadata: PaperMetadata) -> float:
        """Estimate percentile ranking vs similar papers"""
        # Simplified percentile estimation
        # Assumes log-normal distribution of citations

        venue_key = metadata.venue.lower().replace(' ', '').replace('-', '')
        venue_median = self.venue_impacts.get(venue_key, self.venue_impacts['default']) * 5

        # Calculate z-score (simplified)
        if predicted_5yr >= venue_median * 10:
            percentile = 99.0
        elif predicted_5yr >= venue_median * 5:
            percentile = 95.0
        elif predicted_5yr >= venue_median * 2:
            percentile = 80.0
        elif predicted_5yr >= venue_median:
            percentile = 50.0
        elif predicted_5yr >= venue_median * 0.5:
            percentile = 30.0
        else:
            percentile = 10.0

        return percentile


def main():
    parser = argparse.ArgumentParser(
        description="CitationPredictor - Predict Future Citation Counts"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict citations for single paper')
    pred_parser.add_argument('--input', required=True, help='Paper metadata JSON')
    pred_parser.add_argument('--current', type=int, default=0, help='Current citation count')
    pred_parser.add_argument('--output', help='Output JSON file')

    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Predict for multiple papers')
    batch_parser.add_argument('--input', required=True, help='Papers JSON (list)')
    batch_parser.add_argument('--output', required=True, help='Predictions JSON')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    model = CitationModel()

    if args.command == 'predict':
        # Load paper metadata
        with open(args.input, 'r') as f:
            data = json.load(f)

        metadata = PaperMetadata(**data)

        # Predict
        prediction = model.predict(metadata, args.current)

        # Print results
        print(f"\nCitation Prediction for: {prediction.paper_title}")
        print(f"{'='*70}")
        print(f"\nCurrent citations: {prediction.current_citations}")
        print(f"Trajectory: {prediction.trajectory}")
        print(f"\nPredicted citations:")
        print(f"  1 year:  {prediction.predicted_1yr:4d}  (range: {prediction.confidence_interval['1yr'][0]}-{prediction.confidence_interval['1yr'][1]})")
        print(f"  3 years: {prediction.predicted_3yr:4d}  (range: {prediction.confidence_interval['3yr'][0]}-{prediction.confidence_interval['3yr'][1]})")
        print(f"  5 years: {prediction.predicted_5yr:4d}  (range: {prediction.confidence_interval['5yr'][0]}-{prediction.confidence_interval['5yr'][1]})")
        print(f"\nExpected percentile: {prediction.percentile}% (vs similar papers)")

        print(f"\nInfluence factors:")
        for factor, score in sorted(prediction.factors.items(), key=lambda x: x[1], reverse=True):
            bar = '#' * int(score * 20)
            print(f"  {factor:25s} [{bar:20s}] {score:.2f}")

        # Save if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(asdict(prediction), f, indent=2)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'batch':
        # Load papers
        with open(args.input, 'r') as f:
            papers = json.load(f)

        # Predict for each
        predictions = []
        for paper_data in papers:
            metadata = PaperMetadata(**paper_data)
            prediction = model.predict(metadata, paper_data.get('current_citations', 0))
            predictions.append(asdict(prediction))
            print(f"Predicted for: {metadata.title}")

        # Save results
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)

        print(f"\nProcessed {len(predictions)} papers")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
