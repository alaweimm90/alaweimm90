#!/usr/bin/env python3
"""
GhostResearcher - Resurrect Dead Scientists

"What would Einstein say about quantum computing?"
"How would Feynman approach climate modeling?"
"What would Ada Lovelace think of modern AI?"

Simulate perspectives of historical scientific geniuses on modern problems.
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# Historical scientists database
SCIENTISTS = {
    'einstein': {
        'name': 'Albert Einstein',
        'life': '1879-1955',
        'field': 'Physics',
        'known_for': [
            'General and special relativity',
            'Photoelectric effect',
            'E=mc²',
            'Thought experiments',
            'Unified field theory attempts'
        ],
        'personality': [
            'Preferred thought experiments over complex math',
            'Skeptical of quantum mechanics ("God does not play dice")',
            'Valued simplicity and elegance',
            'Independent thinker, questioned authority',
            'Used visualization and intuition'
        ],
        'quotes': [
            'Imagination is more important than knowledge',
            'Make things as simple as possible, but not simpler',
            'The most incomprehensible thing about the world is that it is comprehensible'
        ]
    },
    'feynman': {
        'name': 'Richard Feynman',
        'life': '1918-1988',
        'field': 'Physics',
        'known_for': [
            'Quantum electrodynamics',
            'Feynman diagrams',
            'Path integral formulation',
            'Challenger disaster investigation',
            'Pedagogy and explanation'
        ],
        'personality': [
            'Focused on first principles',
            'Explained complex ideas simply',
            'Practical and hands-on',
            'Skeptical of mathematical formalism without physical intuition',
            'Playful and iconoclastic'
        ],
        'quotes': [
            'What I cannot create, I do not understand',
            'The first principle is that you must not fool yourself',
            'Everything that living things do can be understood in terms of atoms'
        ]
    },
    'curie': {
        'name': 'Marie Curie',
        'life': '1867-1934',
        'field': 'Physics/Chemistry',
        'known_for': [
            'Discovery of radium and polonium',
            'Theory of radioactivity',
            'First woman Nobel Prize',
            'Mobile X-ray units in WWI',
            'Persistence despite discrimination'
        ],
        'personality': [
            'Methodical and meticulous',
            'Persevered through hardship',
            'Believed in practical applications',
            'Collaborative (worked with husband Pierre)',
            'Dedicated to advancing science and helping humanity'
        ],
        'quotes': [
            'Nothing in life is to be feared, it is only to be understood',
            'I was taught that the way of progress is neither swift nor easy',
            'One never notices what has been done; one can only see what remains to be done'
        ]
    },
    'darwin': {
        'name': 'Charles Darwin',
        'life': '1809-1882',
        'field': 'Biology',
        'known_for': [
            'Theory of evolution',
            'Natural selection',
            'Origin of Species',
            'Observations on HMS Beagle',
            'Gradualism and variation'
        ],
        'personality': [
            'Patient observer',
            'Meticulous data collector',
            'Synthesized diverse observations',
            'Willing to wait decades before publishing',
            'Aware of societal implications'
        ],
        'quotes': [
            'It is not the strongest species that survive, but the most adaptable',
            'In the long history of humankind, those who learned to collaborate most effectively have prevailed',
            'Ignorance more frequently begets confidence than does knowledge'
        ]
    },
    'turing': {
        'name': 'Alan Turing',
        'life': '1912-1954',
        'field': 'Mathematics/Computer Science',
        'known_for': [
            'Turing machine',
            'Breaking Enigma code',
            'Turing test',
            'Foundations of computer science',
            'Mathematical biology work'
        ],
        'personality': [
            'Abstract thinker',
            'Interested in foundations',
            'Bridged theory and practice',
            'Asked fundamental questions about computation and intelligence',
            'Unconventional and eccentric'
        ],
        'quotes': [
            'We can only see a short distance ahead, but we can see plenty there that needs to be done',
            'A computer would deserve to be called intelligent if it could deceive a human',
            'Mathematical reasoning may be regarded as the exercise of a combination of intuition and ingenuity'
        ]
    },
    'lovelace': {
        'name': 'Ada Lovelace',
        'life': '1815-1852',
        'field': 'Mathematics/Computing',
        'known_for': [
            'First computer program',
            'Notes on Analytical Engine',
            'Vision of general-purpose computing',
            'Recognized computing beyond calculation',
            'Poetical science'
        ],
        'personality': [
            'Visionary and imaginative',
            'Combined poetry and mathematics',
            'Saw potential beyond immediate applications',
            'Collaborative with Babbage',
            'Challenged gender barriers'
        ],
        'quotes': [
            'The Analytical Engine weaves algebraic patterns just as the Jacquard loom weaves flowers and leaves',
            'We may say most aptly that the Analytical Engine weaves algebraical patterns',
            'Imagination is the discovering faculty, pre-eminently'
        ]
    },
    'newton': {
        'name': 'Isaac Newton',
        'life': '1642-1727',
        'field': 'Physics/Mathematics',
        'known_for': [
            'Laws of motion',
            'Universal gravitation',
            'Calculus (with Leibniz)',
            'Optics',
            'Principia Mathematica'
        ],
        'personality': [
            'Rigorous and systematic',
            'Sought universal laws',
            'Combined mathematics with physical principles',
            'Isolated and secretive',
            'Believed in mechanical universe'
        ],
        'quotes': [
            'If I have seen further, it is by standing on the shoulders of giants',
            'To every action there is always opposed an equal reaction',
            'Truth is ever to be found in simplicity'
        ]
    },
    'franklin': {
        'name': 'Rosalind Franklin',
        'life': '1920-1958',
        'field': 'Chemistry',
        'known_for': [
            'DNA X-ray crystallography (Photo 51)',
            'Virus structure research',
            'Precise experimental technique',
            'Contributions to DNA structure discovery'
        ],
        'personality': [
            'Meticulous experimentalist',
            'Demanded rigor and precision',
            'Cautious about speculation',
            'Underappreciated during lifetime',
            'Let data speak'
        ],
        'quotes': [
            'Science and everyday life cannot and should not be separated',
            'You look at science as something very elite, which only a few people can do',
            'In my view, all that is necessary for faith is the belief that by doing our best we shall succeed in our aims'
        ]
    }
}

@dataclass
class Consultation:
    """A consultation with a ghost researcher"""
    consultation_id: int
    scientist: str
    modern_problem: str
    domain: str

    # Perspective
    initial_reaction: str
    analogies_to_their_time: List[str]
    how_they_would_approach: str
    predicted_obstacles: List[str]

    # Analysis
    key_insights: List[str]
    experimental_suggestions: List[str]
    theoretical_framework: str

    # Style
    characteristic_quotes: List[str]
    thought_experiments: List[str]

    # Meta
    confidence_in_opinion: float
    limitations_of_perspective: List[str]
    created_at: str

class GhostResearcher:
    """Simulate perspectives of historical scientists"""

    def __init__(self, data_file: str = "ghost.json"):
        self.data_file = Path(data_file)
        self.consultations: Dict[int, Consultation] = {}
        self._load_data()

    def _load_data(self):
        """Load consultations from JSON"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.consultations = {
                    int(k): Consultation(**v)
                    for k, v in data.get('consultations', {}).items()
                }

    def _save_data(self):
        """Save consultations to JSON"""
        data = {
            'consultations': {k: asdict(v) for k, v in self.consultations.items()}
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def consult(self, scientist: str, modern_problem: str, domain: str) -> Consultation:
        """Consult a historical scientist about a modern problem"""
        if scientist not in SCIENTISTS:
            raise ValueError(f"Unknown scientist: {scientist}. Available: {', '.join(SCIENTISTS.keys())}")

        consultation_id = len(self.consultations) + 1
        profile = SCIENTISTS[scientist]

        # Generate perspective
        initial_reaction = self._generate_initial_reaction(profile, modern_problem)
        analogies = self._find_analogies(profile, modern_problem, domain)
        approach = self._generate_approach(profile, modern_problem, domain)
        obstacles = self._predict_obstacles(profile, modern_problem)

        # Analysis
        insights = self._generate_insights(profile, modern_problem, domain)
        experiments = self._suggest_experiments(profile, modern_problem, domain)
        framework = self._theoretical_framework(profile, modern_problem, domain)

        # Style
        quotes = self._generate_characteristic_quotes(profile, modern_problem)
        thought_experiments = self._create_thought_experiments(profile, modern_problem)

        # Meta
        confidence = self._estimate_confidence(profile, domain, modern_problem)
        limitations = self._identify_limitations(profile, modern_problem)

        consultation = Consultation(
            consultation_id=consultation_id,
            scientist=scientist,
            modern_problem=modern_problem,
            domain=domain,
            initial_reaction=initial_reaction,
            analogies_to_their_time=analogies,
            how_they_would_approach=approach,
            predicted_obstacles=obstacles,
            key_insights=insights,
            experimental_suggestions=experiments,
            theoretical_framework=framework,
            characteristic_quotes=quotes,
            thought_experiments=thought_experiments,
            confidence_in_opinion=confidence,
            limitations_of_perspective=limitations,
            created_at=datetime.now().isoformat()
        )

        self.consultations[consultation_id] = consultation
        self._save_data()

        return consultation

    def _generate_initial_reaction(self, profile: Dict, problem: str) -> str:
        """Generate initial reaction based on personality"""
        reactions = [
            f"Fascinating! This reminds me of when I was working on {profile['known_for'][0].lower()}.",
            f"The fundamental question here seems to be similar to what I explored with {profile['known_for'][1].lower()}.",
            f"From my perspective in {profile['field'].lower()}, I see this as a question about...",
            f"This problem would have seemed like science fiction in my time, but the principles are timeless."
        ]
        return random.choice(reactions)

    def _find_analogies(self, profile: Dict, problem: str, domain: str) -> List[str]:
        """Find analogies to their time period"""
        life_start = int(profile['life'].split('-')[0])
        era = self._determine_era(life_start)

        return [
            f"In my time, we faced {era['challenge']} - similar in that both require {era['approach']}",
            f"This is like when {era['technology']} was first developed - revolutionary but requiring new thinking",
            f"The debates around {era['controversy']} in my era parallel the questions raised here"
        ]

    def _determine_era(self, year: int) -> Dict[str, str]:
        """Determine historical era context"""
        if year < 1800:
            return {
                'challenge': 'understanding celestial mechanics',
                'approach': 'mathematical rigor and observation',
                'technology': 'the steam engine',
                'controversy': 'the nature of light'
            }
        elif year < 1900:
            return {
                'challenge': 'explaining chemical reactions',
                'approach': 'systematic experimentation',
                'technology': 'the telegraph',
                'controversy': 'evolution and natural selection'
            }
        elif year < 1950:
            return {
                'challenge': 'atomic structure and quantum effects',
                'approach': 'probabilistic thinking',
                'technology': 'the computer',
                'controversy': 'quantum mechanics interpretation'
            }
        else:
            return {
                'challenge': 'DNA structure and molecular biology',
                'approach': 'interdisciplinary collaboration',
                'technology': 'nuclear energy',
                'controversy': 'the nature of life'
            }

    def _generate_approach(self, profile: Dict, problem: str, domain: str) -> str:
        """Generate how they would approach the problem"""
        if 'thought experiments' in profile['personality'][0].lower():
            return f"I would start with a thought experiment: Imagine {problem} in the simplest possible case. Strip away all complexity. What remains?"

        elif 'first principles' in profile['personality'][0].lower():
            return f"We must return to first principles. What are the fundamental building blocks of {problem}? Can we derive everything from basic assumptions?"

        elif 'meticulous' in profile['personality'][0].lower():
            return f"Careful, systematic measurement is key. I would design experiments to isolate each variable in {problem} and measure precisely."

        elif 'observe' in profile['personality'][0].lower():
            return f"Begin with observation. Collect as many instances of {problem} as possible. Patterns will emerge from the data."

        else:
            return f"I would bring my experience in {profile['field']} to bear on {problem}, looking for mathematical structures and physical principles."

    def _predict_obstacles(self, profile: Dict, problem: str) -> List[str]:
        """Predict obstacles based on their historical challenges"""
        return [
            "Insufficient precision in measurements",
            "Resistance from established thinking",
            "Mathematical tools not yet developed",
            "Difficulty in experimental validation",
            "Unrecognized assumptions limiting progress"
        ]

    def _generate_insights(self, profile: Dict, problem: str, domain: str) -> List[str]:
        """Generate key insights from their perspective"""
        insights = []

        # Add domain-specific insight
        insights.append(f"The problem of {problem} is fundamentally about {self._identify_fundamental_nature(domain)}")

        # Add methodological insight
        insights.append(f"Approach this through {profile['field'].lower()} principles - look for {self._identify_principle(profile)}")

        # Add philosophical insight
        insights.append(f"Remember: {random.choice(profile['quotes'])}")

        return insights

    def _identify_fundamental_nature(self, domain: str) -> str:
        """Identify fundamental nature of problem"""
        natures = {
            'physics': 'energy and information flow',
            'biology': 'evolution and adaptation',
            'computer_science': 'computation and complexity',
            'medicine': 'maintaining homeostasis',
            'mathematics': 'structure and relationships'
        }
        return natures.get(domain, 'system dynamics and feedback')

    def _identify_principle(self, profile: Dict) -> str:
        """Identify key principle from their work"""
        if 'relativity' in str(profile['known_for']):
            return 'invariants and symmetries'
        elif 'evolution' in str(profile['known_for']):
            return 'variation and selection'
        elif 'quantum' in str(profile['known_for']):
            return 'discreteness and probability'
        elif 'computing' in str(profile['known_for']):
            return 'algorithms and state machines'
        else:
            return 'conservation laws and Librex'

    def _suggest_experiments(self, profile: Dict, problem: str, domain: str) -> List[str]:
        """Suggest experiments in their style"""
        return [
            f"Design a controlled experiment varying only one aspect of {problem}",
            f"Create a simplified model that captures the essence of {problem}",
            f"Look for edge cases where {problem} behaves unexpectedly",
            f"Measure the relevant quantities with maximum precision",
            f"Compare predictions from different theoretical frameworks"
        ]

    def _theoretical_framework(self, profile: Dict, problem: str, domain: str) -> str:
        """Propose theoretical framework"""
        return (
            f"I would propose a framework based on {profile['field'].lower()} principles. "
            f"The key variables are [to be identified through {profile['personality'][0].lower()}]. "
            f"The governing equations should be derivable from {self._identify_principle(profile)}. "
            f"This framework predicts that {problem} exhibits behavior similar to {profile['known_for'][0].lower()}."
        )

    def _generate_characteristic_quotes(self, profile: Dict, problem: str) -> List[str]:
        """Generate quotes in their style"""
        base_quotes = profile['quotes'][:2]

        # Add problem-specific quote
        if 'simple' in profile['quotes'][0].lower():
            custom = f"The solution to {problem} should be elegant in its simplicity"
        elif 'understand' in profile['quotes'][0].lower():
            custom = f"We must understand {problem} at the deepest level"
        else:
            custom = f"Let us approach {problem} with imagination and rigor"

        return base_quotes + [custom]

    def _create_thought_experiments(self, profile: Dict, problem: str) -> List[str]:
        """Create thought experiments"""
        return [
            f"Imagine {problem} in a universe with different physical constants. What changes?",
            f"Consider the limiting case where {problem} approaches infinity or zero",
            f"What is the simplest system that exhibits {problem}?",
            f"If an omniscient being observed {problem}, what would they see that we miss?"
        ]

    def _estimate_confidence(self, profile: Dict, domain: str, problem: str) -> float:
        """Estimate confidence in opinion"""
        # Higher confidence if domain matches
        if domain.lower() in profile['field'].lower():
            base = 0.8
        else:
            base = 0.5

        # Adjust for problem complexity
        base -= random.uniform(0, 0.2)

        return round(max(0.3, min(0.9, base)), 2)

    def _identify_limitations(self, profile: Dict, problem: str) -> List[str]:
        """Identify limitations of their perspective"""
        death_year = int(profile['life'].split('-')[1])
        decades_since = (2024 - death_year) // 10

        return [
            f"Lacks knowledge of developments in the {decades_since} decades since {death_year}",
            f"May not appreciate modern computational capabilities",
            f"Historical biases from {profile['life'].split('-')[0]}s era",
            f"Limited by {profile['field']} paradigm"
        ]

    def get_scientist_info(self, scientist: str) -> Dict:
        """Get information about a scientist"""
        if scientist not in SCIENTISTS:
            raise ValueError(f"Unknown scientist: {scientist}")
        return SCIENTISTS[scientist]

    def list_scientists(self) -> List[str]:
        """List available scientists"""
        return sorted(SCIENTISTS.keys())

def main():
    parser = argparse.ArgumentParser(description="GhostResearcher - Resurrect Dead Scientists")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Consult
    consult_parser = subparsers.add_parser('consult', help='Consult a historical scientist')
    consult_parser.add_argument('--scientist', required=True, help='Scientist name (e.g., einstein, feynman)')
    consult_parser.add_argument('--problem', required=True, help='Modern problem to discuss')
    consult_parser.add_argument('--domain', required=True, help='Problem domain')

    # Info
    info_parser = subparsers.add_parser('info', help='Get info about a scientist')
    info_parser.add_argument('--scientist', required=True, help='Scientist name')

    # List
    list_parser = subparsers.add_parser('list', help='List available scientists')

    args = parser.parse_args()
    ghost = GhostResearcher()

    if args.command == 'consult':
        consultation = ghost.consult(args.scientist, args.problem, args.domain)
        profile = SCIENTISTS[args.scientist]

        print("\n" + "="*70)
        print(f"CONSULTATION WITH {profile['name'].upper()}")
        print(f"{profile['life']} | {profile['field']}")
        print("="*70)

        print(f"\n📜 MODERN PROBLEM")
        print(f"   {consultation.modern_problem}")
        print(f"   Domain: {consultation.domain}")

        print(f"\n💭 INITIAL REACTION")
        print(f"   {consultation.initial_reaction}")

        print(f"\n🔗 ANALOGIES TO THEIR TIME")
        for analogy in consultation.analogies_to_their_time:
            print(f"   • {analogy}")

        print(f"\n🔬 HOW THEY WOULD APPROACH IT")
        print(f"   {consultation.how_they_would_approach}")

        print(f"\n💡 KEY INSIGHTS")
        for insight in consultation.key_insights:
            print(f"   • {insight}")

        print(f"\n🧪 EXPERIMENTAL SUGGESTIONS")
        for exp in consultation.experimental_suggestions:
            print(f"   • {exp}")

        print(f"\n📐 THEORETICAL FRAMEWORK")
        print(f"   {consultation.theoretical_framework}")

        print(f"\n💬 CHARACTERISTIC QUOTES")
        for quote in consultation.characteristic_quotes:
            print(f"   \"{quote}\"")

        print(f"\n🤔 THOUGHT EXPERIMENTS")
        for te in consultation.thought_experiments:
            print(f"   • {te}")

        print(f"\n⚠️  PREDICTED OBSTACLES")
        for obs in consultation.predicted_obstacles:
            print(f"   • {obs}")

        print(f"\n📊 CONFIDENCE: {consultation.confidence_in_opinion*100:.0f}%")

        print(f"\n⚡ LIMITATIONS OF PERSPECTIVE")
        for lim in consultation.limitations_of_perspective:
            print(f"   • {lim}")

        print(f"\nConsultation ID: {consultation.consultation_id}")

    elif args.command == 'info':
        info = ghost.get_scientist_info(args.scientist)

        print("\n" + "="*70)
        print(f"{info['name'].upper()}")
        print(f"{info['life']} | {info['field']}")
        print("="*70)

        print("\n🏆 KNOWN FOR:")
        for item in info['known_for']:
            print(f"   • {item}")

        print("\n👤 PERSONALITY:")
        for trait in info['personality']:
            print(f"   • {trait}")

        print("\n💬 FAMOUS QUOTES:")
        for quote in info['quotes']:
            print(f"   \"{quote}\"")

    elif args.command == 'list':
        scientists = ghost.list_scientists()

        print("\n" + "="*70)
        print("AVAILABLE SCIENTISTS")
        print("="*70)

        for sci in scientists:
            profile = SCIENTISTS[sci]
            print(f"\n• {sci}")
            print(f"  {profile['name']} ({profile['life']})")
            print(f"  {profile['field']} - {profile['known_for'][0]}")

if __name__ == '__main__':
    main()
