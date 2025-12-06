#!/usr/bin/env python3
"""
ResearchPricer - Grant ROI Calculator

Predict research ROI before applying for grants.
Estimate publications, citations, career impact, and economic returns.
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

@dataclass
class ResearchProposal:
    """Research grant proposal for ROI calculation"""
    proposal_id: int
    title: str
    domain: str
    funding_amount: float
    duration_months: int
    team_size: int
    institution: str
    pi_name: str
    pi_h_index: int
    novelty_score: float  # 0-1, how novel is the research
    feasibility_score: float  # 0-1, how feasible
    impact_potential: str  # "low" | "medium" | "high" | "transformative"
    methodology: str
    prior_publications: int
    created_at: str

@dataclass
class ROIPrediction:
    """Predicted ROI for research proposal"""
    proposal_id: int

    # Publication metrics
    expected_publications: float
    journal_quality_avg: float  # Average impact factor
    conf_presentations: int

    # Citation metrics
    citations_1yr: float
    citations_3yr: float
    citations_5yr: float
    h_index_increase: float

    # Career impact
    tenure_probability: float
    promotion_probability: float
    future_grants_expected: float
    collaborations_formed: int

    # Economic impact
    total_roi_percent: float  # Total return / funding amount
    patent_value: float
    commercialization_potential: float
    industry_partnerships: int

    # Time metrics
    time_to_first_pub: int  # months
    time_to_breakthrough: Optional[int]  # months or None

    # Risk assessment
    failure_probability: float
    risk_level: str  # "low" | "medium" | "high"

    # Overall scores
    academic_value_score: float  # 0-100
    commercial_value_score: float  # 0-100
    career_value_score: float  # 0-100
    overall_recommendation: str  # "strongly_recommend" | "recommend" | "neutral" | "not_recommend"

    explanation: str
    comparable_grants: List[str]
    confidence_interval: Dict[str, float]  # low/high estimates

class ResearchPricer:
    """Calculate ROI for research grant proposals"""

    def __init__(self, data_file: str = "pricer.json"):
        self.data_file = Path(data_file)
        self.proposals: Dict[int, ResearchProposal] = {}
        self.predictions: Dict[int, ROIPrediction] = {}
        self._load_data()

    def _load_data(self):
        """Load proposals and predictions from JSON"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.proposals = {
                    int(k): ResearchProposal(**v)
                    for k, v in data.get('proposals', {}).items()
                }
                self.predictions = {
                    int(k): ROIPrediction(**v)
                    for k, v in data.get('predictions', {}).items()
                }

    def _save_data(self):
        """Save proposals and predictions to JSON"""
        data = {
            'proposals': {k: asdict(v) for k, v in self.proposals.items()},
            'predictions': {k: asdict(v) for k, v in self.predictions.items()}
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def submit_proposal(
        self,
        title: str,
        domain: str,
        funding_amount: float,
        duration_months: int,
        team_size: int,
        institution: str,
        pi_name: str,
        pi_h_index: int,
        novelty_score: float,
        feasibility_score: float,
        impact_potential: str,
        methodology: str,
        prior_publications: int
    ) -> ResearchProposal:
        """Submit a research proposal for ROI calculation"""
        proposal_id = len(self.proposals) + 1

        proposal = ResearchProposal(
            proposal_id=proposal_id,
            title=title,
            domain=domain,
            funding_amount=funding_amount,
            duration_months=duration_months,
            team_size=team_size,
            institution=institution,
            pi_name=pi_name,
            pi_h_index=pi_h_index,
            novelty_score=novelty_score,
            feasibility_score=feasibility_score,
            impact_potential=impact_potential,
            methodology=methodology,
            prior_publications=prior_publications,
            created_at=datetime.now().isoformat()
        )

        self.proposals[proposal_id] = proposal
        self._save_data()

        return proposal

    def calculate_roi(self, proposal_id: int) -> ROIPrediction:
        """Calculate predicted ROI for a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal {proposal_id} not found")

        proposal = self.proposals[proposal_id]

        # Base multipliers
        novelty_mult = 1 + (proposal.novelty_score * 0.5)
        feasibility_mult = proposal.feasibility_score

        impact_multipliers = {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "transformative": 4.0
        }
        impact_mult = impact_multipliers.get(proposal.impact_potential, 1.0)

        # PI experience factor
        pi_factor = min(2.0, 1 + (proposal.pi_h_index / 50))

        # Team size factor
        team_factor = min(1.5, 0.8 + (proposal.team_size * 0.1))

        # Duration factor
        duration_factor = (proposal.duration_months / 12) ** 0.7

        # Publication predictions
        base_pubs = (proposal.funding_amount / 50000) * duration_factor
        expected_pubs = base_pubs * novelty_mult * feasibility_mult * pi_factor * team_factor
        expected_pubs = max(1.0, min(expected_pubs, 20.0))

        journal_quality = 5.0 + (novelty_mult * 3) + (pi_factor * 2)
        journal_quality = min(journal_quality, 15.0)

        conf_presentations = int(expected_pubs * 1.5 + random.uniform(1, 3))

        # Citation predictions
        citations_per_paper = journal_quality * 4 * impact_mult
        citations_1yr = expected_pubs * citations_per_paper * 0.3
        citations_3yr = expected_pubs * citations_per_paper * 0.8
        citations_5yr = expected_pubs * citations_per_paper * 1.2

        h_index_increase = min(10.0, expected_pubs * 0.4 * impact_mult)

        # Career impact
        tenure_base = 0.4 if proposal.prior_publications > 5 else 0.2
        tenure_prob = min(0.95, tenure_base + (h_index_increase * 0.05) + (novelty_mult * 0.1))

        promotion_prob = min(0.90, 0.3 + (h_index_increase * 0.04) + (impact_mult * 0.1))

        future_grants = (proposal.funding_amount / 100000) * pi_factor * impact_mult * 1.5
        future_grants = max(1.0, min(future_grants, 10.0))

        collaborations = int(team_factor * 3 + random.uniform(1, 5))

        # Economic impact
        patent_probability = 0.3 if proposal.impact_potential in ["high", "transformative"] else 0.1
        patent_value = 0
        if random.random() < patent_probability:
            patent_value = random.uniform(50000, 500000) * impact_mult

        commercialization_potential = 0
        if proposal.impact_potential in ["high", "transformative"]:
            commercialization_potential = proposal.funding_amount * random.uniform(0.5, 3.0) * impact_mult

        industry_partnerships = 0
        if proposal.impact_potential in ["high", "transformative"]:
            industry_partnerships = random.randint(1, 4)

        # Total economic ROI
        publication_value = expected_pubs * 20000  # $20k value per publication
        citation_value = citations_5yr * 1000  # $1k per citation
        career_value = (tenure_prob * 200000) + (promotion_prob * 100000)
        future_grant_value = future_grants * 100000

        total_value = (
            publication_value +
            citation_value +
            patent_value +
            commercialization_potential +
            career_value +
            future_grant_value
        )

        total_roi_percent = ((total_value - proposal.funding_amount) / proposal.funding_amount) * 100

        # Time metrics
        time_to_first_pub = max(6, int(12 / feasibility_mult))

        time_to_breakthrough = None
        if proposal.impact_potential in ["high", "transformative"]:
            if random.random() < 0.4:
                time_to_breakthrough = int(proposal.duration_months * random.uniform(0.5, 0.8))

        # Risk assessment
        failure_prob = 1 - (feasibility_mult * 0.7 + novelty_mult * 0.3)
        failure_prob = max(0.05, min(failure_prob, 0.8))

        if failure_prob < 0.2:
            risk_level = "low"
        elif failure_prob < 0.5:
            risk_level = "medium"
        else:
            risk_level = "high"

        # Overall scores
        academic_value = min(100, (expected_pubs * 5) + (citations_5yr / 10) + (h_index_increase * 5))
        commercial_value = min(100, (commercialization_potential / 10000) + (patent_value / 5000))
        career_value = min(100, (tenure_prob * 40) + (promotion_prob * 30) + (future_grants * 5))

        # Recommendation
        overall_score = (academic_value + commercial_value + career_value) / 3
        if overall_score > 70 and risk_level != "high":
            recommendation = "strongly_recommend"
        elif overall_score > 50:
            recommendation = "recommend"
        elif overall_score > 30:
            recommendation = "neutral"
        else:
            recommendation = "not_recommend"

        # Explanation
        explanation = self._generate_explanation(
            proposal, expected_pubs, citations_5yr, total_roi_percent,
            risk_level, academic_value, commercial_value, career_value
        )

        # Comparable grants (mock data)
        comparable = [
            f"Similar {proposal.domain} grant at {random.choice(['MIT', 'Stanford', 'Berkeley', 'Harvard'])}",
            f"Comparable {proposal.impact_potential} impact project",
            f"Related {proposal.methodology} methodology study"
        ]

        # Confidence intervals
        confidence = {
            "publications_low": expected_pubs * 0.7,
            "publications_high": expected_pubs * 1.3,
            "citations_low": citations_5yr * 0.5,
            "citations_high": citations_5yr * 1.5,
            "roi_low": total_roi_percent * 0.6,
            "roi_high": total_roi_percent * 1.4
        }

        prediction = ROIPrediction(
            proposal_id=proposal_id,
            expected_publications=round(expected_pubs, 2),
            journal_quality_avg=round(journal_quality, 2),
            conf_presentations=conf_presentations,
            citations_1yr=round(citations_1yr, 1),
            citations_3yr=round(citations_3yr, 1),
            citations_5yr=round(citations_5yr, 1),
            h_index_increase=round(h_index_increase, 2),
            tenure_probability=round(tenure_prob, 3),
            promotion_probability=round(promotion_prob, 3),
            future_grants_expected=round(future_grants, 2),
            collaborations_formed=collaborations,
            total_roi_percent=round(total_roi_percent, 1),
            patent_value=round(patent_value, 2),
            commercialization_potential=round(commercialization_potential, 2),
            industry_partnerships=industry_partnerships,
            time_to_first_pub=time_to_first_pub,
            time_to_breakthrough=time_to_breakthrough,
            failure_probability=round(failure_prob, 3),
            risk_level=risk_level,
            academic_value_score=round(academic_value, 1),
            commercial_value_score=round(commercial_value, 1),
            career_value_score=round(career_value, 1),
            overall_recommendation=recommendation,
            explanation=explanation,
            comparable_grants=comparable,
            confidence_interval=confidence
        )

        self.predictions[proposal_id] = prediction
        self._save_data()

        return prediction

    def _generate_explanation(
        self, proposal, expected_pubs, citations, roi, risk,
        academic, commercial, career
    ) -> str:
        """Generate human-readable explanation"""
        parts = []

        parts.append(f"This {proposal.impact_potential}-impact {proposal.domain} project ")
        parts.append(f"with ${proposal.funding_amount:,.0f} funding over {proposal.duration_months} months ")
        parts.append(f"is expected to produce {expected_pubs:.1f} publications ")
        parts.append(f"and {citations:.0f} citations within 5 years. ")

        parts.append(f"\n\nThe PI's h-index of {proposal.pi_h_index} and {proposal.prior_publications} prior publications ")
        parts.append(f"suggest {'strong' if proposal.pi_h_index > 20 else 'moderate'} research capability. ")

        parts.append(f"\n\nWith a novelty score of {proposal.novelty_score:.2f} and feasibility of {proposal.feasibility_score:.2f}, ")
        parts.append(f"this project carries {risk} risk. ")

        parts.append(f"\n\nExpected ROI: {roi:.1f}% with academic value {academic:.0f}/100, ")
        parts.append(f"commercial value {commercial:.0f}/100, and career value {career:.0f}/100.")

        return ''.join(parts)

    def compare_proposals(self, proposal_ids: List[int]) -> List[Dict]:
        """Compare multiple proposals side-by-side"""
        if not all(pid in self.predictions for pid in proposal_ids):
            missing = [pid for pid in proposal_ids if pid not in self.predictions]
            raise ValueError(f"Predictions not found for proposals: {missing}")

        comparisons = []
        for pid in proposal_ids:
            proposal = self.proposals[pid]
            prediction = self.predictions[pid]

            comparisons.append({
                'proposal_id': pid,
                'title': proposal.title,
                'funding': proposal.funding_amount,
                'roi_percent': prediction.total_roi_percent,
                'publications': prediction.expected_publications,
                'citations_5yr': prediction.citations_5yr,
                'academic_value': prediction.academic_value_score,
                'commercial_value': prediction.commercial_value_score,
                'career_value': prediction.career_value_score,
                'risk_level': prediction.risk_level,
                'recommendation': prediction.overall_recommendation
            })

        # Sort by ROI
        comparisons.sort(key=lambda x: x['roi_percent'], reverse=True)
        return comparisons

    def get_domain_statistics(self, domain: str) -> Dict:
        """Get aggregate statistics for a domain"""
        domain_predictions = [
            pred for pid, pred in self.predictions.items()
            if self.proposals[pid].domain == domain
        ]

        if not domain_predictions:
            return {"error": f"No predictions found for domain: {domain}"}

        return {
            'domain': domain,
            'total_proposals': len(domain_predictions),
            'avg_roi': sum(p.total_roi_percent for p in domain_predictions) / len(domain_predictions),
            'avg_publications': sum(p.expected_publications for p in domain_predictions) / len(domain_predictions),
            'avg_citations_5yr': sum(p.citations_5yr for p in domain_predictions) / len(domain_predictions),
            'avg_academic_value': sum(p.academic_value_score for p in domain_predictions) / len(domain_predictions),
            'avg_commercial_value': sum(p.commercial_value_score for p in domain_predictions) / len(domain_predictions),
            'high_risk_percent': sum(1 for p in domain_predictions if p.risk_level == "high") / len(domain_predictions) * 100,
            'strongly_recommended': sum(1 for p in domain_predictions if p.overall_recommendation == "strongly_recommend")
        }

def main():
    parser = argparse.ArgumentParser(description="ResearchPricer - Grant ROI Calculator")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Submit proposal
    submit_parser = subparsers.add_parser('submit', help='Submit a research proposal')
    submit_parser.add_argument('--title', required=True, help='Proposal title')
    submit_parser.add_argument('--domain', required=True, help='Research domain')
    submit_parser.add_argument('--funding', type=float, required=True, help='Funding amount ($)')
    submit_parser.add_argument('--duration', type=int, required=True, help='Duration (months)')
    submit_parser.add_argument('--team-size', type=int, required=True, help='Team size')
    submit_parser.add_argument('--institution', required=True, help='Institution')
    submit_parser.add_argument('--pi-name', required=True, help='PI name')
    submit_parser.add_argument('--pi-h-index', type=int, required=True, help='PI h-index')
    submit_parser.add_argument('--novelty', type=float, required=True, help='Novelty score (0-1)')
    submit_parser.add_argument('--feasibility', type=float, required=True, help='Feasibility score (0-1)')
    submit_parser.add_argument('--impact', required=True, choices=['low', 'medium', 'high', 'transformative'])
    submit_parser.add_argument('--methodology', required=True, help='Research methodology')
    submit_parser.add_argument('--prior-pubs', type=int, required=True, help='PI prior publications')

    # Calculate ROI
    calc_parser = subparsers.add_parser('calculate', help='Calculate ROI for a proposal')
    calc_parser.add_argument('--proposal-id', type=int, required=True, help='Proposal ID')

    # Compare proposals
    compare_parser = subparsers.add_parser('compare', help='Compare multiple proposals')
    compare_parser.add_argument('--proposal-ids', required=True, help='Comma-separated proposal IDs')

    # Domain statistics
    stats_parser = subparsers.add_parser('domain-stats', help='Get domain statistics')
    stats_parser.add_argument('--domain', required=True, help='Research domain')

    args = parser.parse_args()
    pricer = ResearchPricer()

    if args.command == 'submit':
        proposal = pricer.submit_proposal(
            title=args.title,
            domain=args.domain,
            funding_amount=args.funding,
            duration_months=args.duration,
            team_size=args.team_size,
            institution=args.institution,
            pi_name=args.pi_name,
            pi_h_index=args.pi_h_index,
            novelty_score=args.novelty,
            feasibility_score=args.feasibility,
            impact_potential=args.impact,
            methodology=args.methodology,
            prior_publications=args.prior_pubs
        )

        print("\nProposal submitted successfully!")
        print(f"Proposal ID: {proposal.proposal_id}")
        print(f"Title: {proposal.title}")
        print(f"Funding: ${proposal.funding_amount:,.2f}")
        print(f"Duration: {proposal.duration_months} months")
        print(f"\nNext: Run 'python pricer.py calculate --proposal-id {proposal.proposal_id}' to get ROI prediction")

    elif args.command == 'calculate':
        prediction = pricer.calculate_roi(args.proposal_id)
        proposal = pricer.proposals[args.proposal_id]

        print("\n" + "="*70)
        print(f"ROI PREDICTION - {proposal.title}")
        print("="*70)

        print(f"\nFunding: ${proposal.funding_amount:,.2f} over {proposal.duration_months} months")
        print(f"PI: {proposal.pi_name} (h-index: {proposal.pi_h_index})")
        print(f"Domain: {proposal.domain} | Impact: {proposal.impact_potential}")

        print("\n📚 PUBLICATION METRICS")
        print(f"  Expected publications: {prediction.expected_publications}")
        print(f"  Average journal IF: {prediction.journal_quality_avg}")
        print(f"  Conference presentations: {prediction.conf_presentations}")
        print(f"  Time to first publication: {prediction.time_to_first_pub} months")
        if prediction.time_to_breakthrough:
            print(f"  Time to breakthrough: {prediction.time_to_breakthrough} months")

        print("\n📊 CITATION METRICS")
        print(f"  1-year citations: {prediction.citations_1yr:.0f}")
        print(f"  3-year citations: {prediction.citations_3yr:.0f}")
        print(f"  5-year citations: {prediction.citations_5yr:.0f}")
        print(f"  H-index increase: +{prediction.h_index_increase}")

        print("\n🎓 CAREER IMPACT")
        print(f"  Tenure probability: {prediction.tenure_probability*100:.1f}%")
        print(f"  Promotion probability: {prediction.promotion_probability*100:.1f}%")
        print(f"  Future grants expected: {prediction.future_grants_expected}")
        print(f"  Collaborations formed: {prediction.collaborations_formed}")

        print("\n💰 ECONOMIC IMPACT")
        print(f"  Total ROI: {prediction.total_roi_percent:+.1f}%")
        if prediction.patent_value > 0:
            print(f"  Patent value: ${prediction.patent_value:,.2f}")
        if prediction.commercialization_potential > 0:
            print(f"  Commercialization potential: ${prediction.commercialization_potential:,.2f}")
        if prediction.industry_partnerships > 0:
            print(f"  Industry partnerships: {prediction.industry_partnerships}")

        print("\n⚠️  RISK ASSESSMENT")
        print(f"  Failure probability: {prediction.failure_probability*100:.1f}%")
        print(f"  Risk level: {prediction.risk_level.upper()}")

        print("\n📈 OVERALL SCORES")
        print(f"  Academic value: {prediction.academic_value_score:.1f}/100")
        print(f"  Commercial value: {prediction.commercial_value_score:.1f}/100")
        print(f"  Career value: {prediction.career_value_score:.1f}/100")

        print(f"\n🎯 RECOMMENDATION: {prediction.overall_recommendation.upper().replace('_', ' ')}")

        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        print(prediction.explanation)

        print("\n📋 COMPARABLE GRANTS:")
        for comp in prediction.comparable_grants:
            print(f"  • {comp}")

        print("\n📊 CONFIDENCE INTERVALS:")
        print(f"  Publications: {prediction.confidence_interval['publications_low']:.1f} - {prediction.confidence_interval['publications_high']:.1f}")
        print(f"  Citations (5yr): {prediction.confidence_interval['citations_low']:.0f} - {prediction.confidence_interval['citations_high']:.0f}")
        print(f"  ROI: {prediction.confidence_interval['roi_low']:.1f}% - {prediction.confidence_interval['roi_high']:.1f}%")

    elif args.command == 'compare':
        proposal_ids = [int(x.strip()) for x in args.proposal_ids.split(',')]
        comparisons = pricer.compare_proposals(proposal_ids)

        print("\n" + "="*70)
        print("PROPOSAL COMPARISON (sorted by ROI)")
        print("="*70)

        for i, comp in enumerate(comparisons, 1):
            print(f"\n#{i}. {comp['title']}")
            print(f"    Funding: ${comp['funding']:,.0f}")
            print(f"    ROI: {comp['roi_percent']:+.1f}%")
            print(f"    Publications: {comp['publications']}")
            print(f"    Citations (5yr): {comp['citations_5yr']:.0f}")
            print(f"    Academic: {comp['academic_value']:.0f} | Commercial: {comp['commercial_value']:.0f} | Career: {comp['career_value']:.0f}")
            print(f"    Risk: {comp['risk_level']} | Recommendation: {comp['recommendation']}")

    elif args.command == 'domain-stats':
        stats = pricer.get_domain_statistics(args.domain)

        if 'error' in stats:
            print(f"\nError: {stats['error']}")
        else:
            print("\n" + "="*70)
            print(f"DOMAIN STATISTICS - {stats['domain'].upper()}")
            print("="*70)

            print(f"\nTotal proposals analyzed: {stats['total_proposals']}")
            print(f"Average ROI: {stats['avg_roi']:.1f}%")
            print(f"Average publications: {stats['avg_publications']:.2f}")
            print(f"Average citations (5yr): {stats['avg_citations_5yr']:.1f}")
            print(f"Average academic value: {stats['avg_academic_value']:.1f}/100")
            print(f"Average commercial value: {stats['avg_commercial_value']:.1f}/100")
            print(f"High-risk proposals: {stats['high_risk_percent']:.1f}%")
            print(f"Strongly recommended: {stats['strongly_recommended']}")

if __name__ == '__main__':
    main()
