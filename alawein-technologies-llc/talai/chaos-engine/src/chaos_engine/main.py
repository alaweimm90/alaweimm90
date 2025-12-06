#!/usr/bin/env python3
"""
ChaosEngine - Force Random Domain Collisions

Generate novel ideas by randomly combining concepts from different domains.
"What if biology used blockchain?" "What if quantum physics solved traffic?"
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Domain knowledge bases
DOMAINS = {
    'biology': [
        'evolution', 'DNA sequencing', 'cellular respiration', 'photosynthesis',
        'immune system', 'protein folding', 'genetic algorithms', 'ecosystems',
        'neural networks', 'symbiosis', 'homeostasis', 'gene expression'
    ],
    'physics': [
        'quantum mechanics', 'relativity', 'thermodynamics', 'wave-particle duality',
        'string theory', 'superconductivity', 'particle physics', 'gravitational waves',
        'dark matter', 'nuclear fusion', 'quantum entanglement', 'conservation laws'
    ],
    'computer_science': [
        'machine learning', 'blockchain', 'distributed systems', 'encryption',
        'neural networks', 'algorithms', 'cloud computing', 'edge computing',
        'version control', 'garbage collection', 'consensus protocols', 'hash functions'
    ],
    'economics': [
        'market equilibrium', 'game theory', 'auction theory', 'incentive design',
        'price discovery', 'resource allocation', 'nash equilibrium', 'behavioral economics',
        'network effects', 'supply and demand', 'inflation', 'arbitrage'
    ],
    'psychology': [
        'cognitive biases', 'memory formation', 'reward systems', 'attention mechanisms',
        'social dynamics', 'habit formation', 'decision making', 'perception',
        'motivation theory', 'learning models', 'behavioral conditioning', 'priming'
    ],
    'mathematics': [
        'graph theory', 'topology', 'probability', 'optimization',
        'differential equations', 'group theory', 'number theory', 'fractals',
        'chaos theory', 'statistical inference', 'linear algebra', 'calculus'
    ],
    'engineering': [
        'feedback control', 'signal processing', 'materials science', 'robotics',
        'energy efficiency', 'structural optimization', 'manufacturing', 'automation',
        'sensor networks', 'actuators', 'modular design', 'redundancy'
    ],
    'medicine': [
        'immune response', 'drug delivery', 'diagnostics', 'personalized treatment',
        'clinical trials', 'biomarkers', 'regenerative medicine', 'vaccines',
        'antibiotics', 'surgery', 'preventive care', 'telemedicine'
    ],
    'social_science': [
        'social networks', 'cultural evolution', 'collective behavior', 'institutions',
        'power dynamics', 'information cascades', 'social norms', 'coordination games',
        'trust mechanisms', 'group identity', 'social learning', 'cooperation'
    ],
    'art': [
        'composition', 'color theory', 'perspective', 'abstraction',
        'minimalism', 'generative art', 'visual hierarchy', 'contrast',
        'storytelling', 'improvisation', 'remix culture', 'aesthetics'
    ]
}

PROBLEMS = {
    'biology': [
        'protein misfolding diseases', 'drug resistance', 'ecosystem collapse',
        'genetic disorders', 'cancer metastasis', 'aging', 'biodiversity loss'
    ],
    'physics': [
        'quantum decoherence', 'dark energy mystery', 'room temperature superconductors',
        'nuclear waste', 'fusion energy containment', 'measurement precision'
    ],
    'computer_science': [
        'scalability limits', 'security vulnerabilities', 'data privacy',
        'algorithmic bias', 'latency', 'energy consumption', 'complexity explosion'
    ],
    'economics': [
        'market manipulation', 'wealth inequality', 'externalities',
        'coordination failures', 'information asymmetry', 'bubble formation'
    ],
    'psychology': [
        'addiction', 'depression', 'anxiety', 'PTSD', 'cognitive decline',
        'attention deficits', 'motivation loss', 'social isolation'
    ],
    'engineering': [
        'material fatigue', 'thermal management', 'vibration damping',
        'noise reduction', 'efficiency limits', 'manufacturing defects'
    ],
    'medicine': [
        'antibiotic resistance', 'cancer recurrence', 'organ transplant rejection',
        'chronic pain', 'autoimmune diseases', 'rare diseases', 'global pandemics'
    ],
    'social_science': [
        'misinformation spread', 'polarization', 'collective action problems',
        'institutional corruption', 'social fragmentation', 'trust erosion'
    ],
    'environment': [
        'climate change', 'pollution', 'resource depletion', 'species extinction',
        'ocean acidification', 'deforestation', 'water scarcity'
    ],
    'urban': [
        'traffic congestion', 'urban sprawl', 'housing affordability',
        'infrastructure decay', 'waste management', 'noise pollution'
    ]
}

@dataclass
class Collision:
    """A domain collision"""
    collision_id: int
    source_domain: str
    source_concept: str
    target_domain: str
    target_problem: str
    novelty_score: float
    feasibility_score: float
    impact_score: float
    overall_score: float
    idea_statement: str
    mechanism: str
    applications: List[str]
    challenges: List[str]
    next_steps: List[str]
    comparable_ideas: List[str]
    created_at: str

@dataclass
class CollisionSession:
    """A chaos session generating multiple collisions"""
    session_id: int
    collisions_generated: int
    domains_used: List[str]
    best_collision_id: int
    avg_novelty: float
    avg_feasibility: float
    timestamp: str

class ChaosEngine:
    """Generate ideas through random domain collisions"""

    def __init__(self, data_file: str = "chaos.json"):
        self.data_file = Path(data_file)
        self.collisions: Dict[int, Collision] = {}
        self.sessions: Dict[int, CollisionSession] = {}
        self._load_data()

    def _load_data(self):
        """Load collisions from JSON"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.collisions = {
                    int(k): Collision(**v)
                    for k, v in data.get('collisions', {}).items()
                }
                self.sessions = {
                    int(k): CollisionSession(**v)
                    for k, v in data.get('sessions', {}).items()
                }

    def _save_data(self):
        """Save collisions to JSON"""
        data = {
            'collisions': {k: asdict(v) for k, v in self.collisions.items()},
            'sessions': {k: asdict(v) for k, v in self.sessions.items()}
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def collide(
        self,
        source_domain: Optional[str] = None,
        target_domain: Optional[str] = None
    ) -> Collision:
        """Generate a single random collision"""

        # Random selection if not specified
        if source_domain is None:
            source_domain = random.choice(list(DOMAINS.keys()))
        if target_domain is None:
            # Ensure different domain
            available = [d for d in DOMAINS.keys() if d != source_domain]
            if not available:
                available = list(DOMAINS.keys())
            target_domain = random.choice(available)

        # Get random concept and problem
        source_concept = random.choice(DOMAINS[source_domain])

        # Target problem from target domain or related problems
        if target_domain in PROBLEMS:
            target_problem = random.choice(PROBLEMS[target_domain])
        else:
            # Use a related domain
            target_problem = random.choice(PROBLEMS.get('environment', ['general challenge']))

        # Generate collision
        collision_id = len(self.collisions) + 1

        # Create idea statement
        idea = self._generate_idea(source_domain, source_concept, target_domain, target_problem)
        mechanism = self._generate_mechanism(source_concept, target_problem)
        applications = self._generate_applications(source_concept, target_domain, target_problem)
        challenges = self._identify_challenges(source_domain, target_domain, source_concept)
        next_steps = self._suggest_next_steps(source_concept, target_problem)
        comparable = self._find_comparable(source_domain, target_domain)

        # Score the collision
        novelty = self._score_novelty(source_domain, target_domain, source_concept)
        feasibility = self._score_feasibility(source_domain, target_domain)
        impact = self._score_impact(target_problem, source_concept)
        overall = (novelty * 0.4 + feasibility * 0.3 + impact * 0.3)

        collision = Collision(
            collision_id=collision_id,
            source_domain=source_domain,
            source_concept=source_concept,
            target_domain=target_domain,
            target_problem=target_problem,
            novelty_score=novelty,
            feasibility_score=feasibility,
            impact_score=impact,
            overall_score=overall,
            idea_statement=idea,
            mechanism=mechanism,
            applications=applications,
            challenges=challenges,
            next_steps=next_steps,
            comparable_ideas=comparable,
            created_at=datetime.now().isoformat()
        )

        self.collisions[collision_id] = collision
        self._save_data()

        return collision

    def stampede(self, count: int = 10, source: Optional[str] = None) -> CollisionSession:
        """Generate multiple collisions rapidly"""
        session_id = len(self.sessions) + 1
        collision_ids = []
        domains_used = set()

        for _ in range(count):
            collision = self.collide(source_domain=source)
            collision_ids.append(collision.collision_id)
            domains_used.add(collision.source_domain)
            domains_used.add(collision.target_domain)

        # Find best collision
        best = max(collision_ids, key=lambda cid: self.collisions[cid].overall_score)

        # Calculate averages
        collisions = [self.collisions[cid] for cid in collision_ids]
        avg_novelty = sum(c.novelty_score for c in collisions) / len(collisions)
        avg_feasibility = sum(c.feasibility_score for c in collisions) / len(collisions)

        session = CollisionSession(
            session_id=session_id,
            collisions_generated=count,
            domains_used=sorted(list(domains_used)),
            best_collision_id=best,
            avg_novelty=avg_novelty,
            avg_feasibility=avg_feasibility,
            timestamp=datetime.now().isoformat()
        )

        self.sessions[session_id] = session
        self._save_data()

        return session, collision_ids

    def _generate_idea(self, source_domain: str, source_concept: str,
                      target_domain: str, target_problem: str) -> str:
        """Generate idea statement"""
        templates = [
            f"Apply {source_concept} from {source_domain} to solve {target_problem} in {target_domain}",
            f"Use {source_domain}'s {source_concept} approach to address {target_problem}",
            f"What if we treated {target_problem} using principles from {source_concept}?",
            f"Reframe {target_problem} as a {source_concept} problem",
            f"Import {source_concept} methodology to tackle {target_problem}"
        ]
        return random.choice(templates)

    def _generate_mechanism(self, source_concept: str, target_problem: str) -> str:
        """Generate mechanism explanation"""
        return (
            f"{source_concept.title()} provides a framework for understanding "
            f"{target_problem}. By mapping the problem structure to {source_concept} principles, "
            f"we can leverage existing solutions and insights from that domain."
        )

    def _generate_applications(self, source_concept: str, target_domain: str,
                              target_problem: str) -> List[str]:
        """Generate potential applications"""
        return [
            f"Direct application of {source_concept} to {target_problem}",
            f"Hybrid approach combining {source_concept} with existing {target_domain} methods",
            f"Use {source_concept} for diagnosis/analysis of {target_problem}",
            f"Metaphorical insight: {source_concept} as lens for {target_problem}"
        ]

    def _identify_challenges(self, source_domain: str, target_domain: str,
                           source_concept: str) -> List[str]:
        """Identify implementation challenges"""
        return [
            f"Conceptual gap between {source_domain} and {target_domain}",
            f"Adapting {source_concept} to different constraints in {target_domain}",
            "Requires interdisciplinary expertise",
            "May face institutional/cultural resistance",
            "Validation and testing in new context"
        ]

    def _suggest_next_steps(self, source_concept: str, target_problem: str) -> List[str]:
        """Suggest next steps"""
        return [
            f"Literature review: Has {source_concept} been applied to similar problems?",
            f"Proof of concept: Small-scale test of {source_concept} on {target_problem}",
            "Build interdisciplinary team",
            "Identify analogous successes in other domains",
            "Develop theoretical framework for transfer"
        ]

    def _find_comparable(self, source_domain: str, target_domain: str) -> List[str]:
        """Find comparable cross-domain ideas"""
        examples = [
            "Ant colony optimization → network routing",
            "Evolutionary algorithms → engineering design",
            "Immune system → computer security",
            "Market mechanisms → resource allocation in computing",
            "Neural networks → everywhere"
        ]
        return random.sample(examples, min(3, len(examples)))

    def _score_novelty(self, source_domain: str, target_domain: str,
                      source_concept: str) -> float:
        """Score novelty of collision"""
        # More distant domains = higher novelty
        distance_scores = {
            ('physics', 'art'): 0.9,
            ('mathematics', 'psychology'): 0.85,
            ('biology', 'economics'): 0.8,
            ('computer_science', 'medicine'): 0.6,
        }

        pair = tuple(sorted([source_domain, target_domain]))
        base_novelty = distance_scores.get(pair, 0.7)

        # Add randomness
        novelty = base_novelty + random.uniform(-0.1, 0.1)
        return min(1.0, max(0.0, novelty))

    def _score_feasibility(self, source_domain: str, target_domain: str) -> float:
        """Score feasibility of implementation"""
        # Closer domains = higher feasibility
        feasibility_map = {
            ('biology', 'medicine'): 0.8,
            ('computer_science', 'engineering'): 0.75,
            ('mathematics', 'physics'): 0.85,
            ('psychology', 'social_science'): 0.7
        }

        pair = tuple(sorted([source_domain, target_domain]))
        base_feasibility = feasibility_map.get(pair, 0.5)

        feasibility = base_feasibility + random.uniform(-0.15, 0.15)
        return min(1.0, max(0.2, feasibility))

    def _score_impact(self, target_problem: str, source_concept: str) -> float:
        """Score potential impact"""
        high_impact_problems = [
            'climate change', 'cancer', 'antibiotic resistance',
            'wealth inequality', 'misinformation'
        ]

        base_impact = 0.8 if any(p in target_problem for p in high_impact_problems) else 0.6
        impact = base_impact + random.uniform(-0.1, 0.2)
        return min(1.0, max(0.3, impact))

    def get_top_collisions(self, limit: int = 10, min_score: float = 0.0) -> List[Collision]:
        """Get top-scoring collisions"""
        filtered = [c for c in self.collisions.values() if c.overall_score >= min_score]
        sorted_collisions = sorted(filtered, key=lambda c: c.overall_score, reverse=True)
        return sorted_collisions[:limit]

    def get_by_domain(self, domain: str, role: str = 'source') -> List[Collision]:
        """Get collisions by domain (source or target)"""
        if role == 'source':
            return [c for c in self.collisions.values() if c.source_domain == domain]
        else:
            return [c for c in self.collisions.values() if c.target_domain == domain]

def main():
    parser = argparse.ArgumentParser(description="ChaosEngine - Random Domain Collisions")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Single collision
    collide_parser = subparsers.add_parser('collide', help='Generate single collision')
    collide_parser.add_argument('--source', help='Source domain (optional)')
    collide_parser.add_argument('--target', help='Target domain (optional)')

    # Stampede
    stampede_parser = subparsers.add_parser('stampede', help='Generate multiple collisions')
    stampede_parser.add_argument('--count', type=int, default=10, help='Number of collisions')
    stampede_parser.add_argument('--source', help='Fix source domain (optional)')

    # Top collisions
    top_parser = subparsers.add_parser('top', help='Show top collisions')
    top_parser.add_argument('--limit', type=int, default=10, help='Number to show')
    top_parser.add_argument('--min-score', type=float, default=0.0, help='Minimum score')

    # Domain focus
    domain_parser = subparsers.add_parser('domain', help='Collisions for specific domain')
    domain_parser.add_argument('--domain', required=True, help='Domain name')
    domain_parser.add_argument('--role', choices=['source', 'target'], default='source')

    args = parser.parse_args()
    engine = ChaosEngine()

    if args.command == 'collide':
        collision = engine.collide(source_domain=args.source, target_domain=args.target)

        print("\n" + "="*70)
        print("DOMAIN COLLISION")
        print("="*70)

        print(f"\n🔀 {collision.source_domain.upper()} × {collision.target_domain.upper()}")
        print(f"   {collision.source_concept} → {collision.target_problem}")

        print(f"\n💡 IDEA")
        print(f"   {collision.idea_statement}")

        print(f"\n🔧 MECHANISM")
        print(f"   {collision.mechanism}")

        print(f"\n📊 SCORES")
        print(f"   Novelty:     {collision.novelty_score:.2f}/1.00")
        print(f"   Feasibility: {collision.feasibility_score:.2f}/1.00")
        print(f"   Impact:      {collision.impact_score:.2f}/1.00")
        print(f"   OVERALL:     {collision.overall_score:.2f}/1.00")

        print(f"\n🎯 APPLICATIONS")
        for app in collision.applications:
            print(f"   • {app}")

        print(f"\n⚠️  CHALLENGES")
        for challenge in collision.challenges:
            print(f"   • {challenge}")

        print(f"\n📝 NEXT STEPS")
        for step in collision.next_steps:
            print(f"   • {step}")

        print(f"\n🔍 COMPARABLE IDEAS")
        for comp in collision.comparable_ideas:
            print(f"   • {comp}")

        print(f"\nCollision ID: {collision.collision_id}")

    elif args.command == 'stampede':
        session, collision_ids = engine.stampede(count=args.count, source=args.source)

        print("\n" + "="*70)
        print(f"CHAOS STAMPEDE - {args.count} COLLISIONS")
        print("="*70)

        print(f"\nSession ID: {session.session_id}")
        print(f"Domains explored: {', '.join(session.domains_used)}")
        print(f"Average novelty: {session.avg_novelty:.2f}")
        print(f"Average feasibility: {session.avg_feasibility:.2f}")

        print(f"\n🏆 TOP 5 COLLISIONS:")
        top_ids = sorted(collision_ids,
                        key=lambda cid: engine.collisions[cid].overall_score,
                        reverse=True)[:5]

        for i, cid in enumerate(top_ids, 1):
            c = engine.collisions[cid]
            print(f"\n{i}. [{c.overall_score:.2f}] {c.source_domain} → {c.target_domain}")
            print(f"   {c.source_concept} solving {c.target_problem}")
            print(f"   {c.idea_statement}")

        print(f"\n💾 All {args.count} collisions saved. Use 'python chaos.py top' to see more.")

    elif args.command == 'top':
        collisions = engine.get_top_collisions(limit=args.limit, min_score=args.min_score)

        print("\n" + "="*70)
        print(f"TOP {len(collisions)} COLLISIONS")
        print("="*70)

        for i, c in enumerate(collisions, 1):
            print(f"\n{i}. [{c.overall_score:.2f}] {c.source_domain} × {c.target_domain}")
            print(f"   {c.source_concept} → {c.target_problem}")
            print(f"   💡 {c.idea_statement}")
            print(f"   📊 N:{c.novelty_score:.2f} F:{c.feasibility_score:.2f} I:{c.impact_score:.2f}")

    elif args.command == 'domain':
        collisions = engine.get_by_domain(args.domain, role=args.role)

        print("\n" + "="*70)
        print(f"COLLISIONS - {args.domain.upper()} ({args.role})")
        print("="*70)

        if not collisions:
            print(f"\nNo collisions found with {args.domain} as {args.role}")
        else:
            for i, c in enumerate(collisions, 1):
                print(f"\n{i}. [{c.overall_score:.2f}] {c.source_domain} → {c.target_domain}")
                print(f"   {c.idea_statement}")

if __name__ == '__main__':
    main()
