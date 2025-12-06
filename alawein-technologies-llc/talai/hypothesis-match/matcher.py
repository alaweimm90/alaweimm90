#!/usr/bin/env python3
"""
HypothesisMatch - Scientific Tinder for Researchers

Match researchers based on complementary skills, research interests, and hypothesis compatibility.
Swipe on hypotheses, find collaborators with matching or opposing views.

Usage:
    python matcher.py create-profile --name "Dr. Smith" --skills "ML,CV,NLP" --interests "medical-imaging"
    python matcher.py add-hypothesis --profile-id 1 --text "Deep learning can diagnose cancer better than radiologists"
    python matcher.py match --profile-id 1 --mode complementary
    python matcher.py swipe --profile-id 1 --hypothesis-id 42 --action SUPPORT
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ResearcherProfile:
    """Researcher profile"""
    profile_id: int
    name: str
    institution: str
    skills: List[str]  # e.g., ["ML", "CV", "statistics"]
    interests: List[str]  # Research topics
    h_index: int
    hypotheses: List[str]  # Research hypotheses they support
    anti_hypotheses: List[str]  # Hypotheses they oppose
    looking_for: str  # "complementary" | "similar" | "opposing"
    created_at: str


@dataclass
class Hypothesis:
    """Research hypothesis"""
    hypothesis_id: int
    text: str
    field: str
    author_id: int
    support_count: int = 0
    oppose_count: int = 0
    debate_count: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: str = ""


@dataclass
class Match:
    """Researcher match"""
    match_id: int
    researcher_1: int
    researcher_2: int
    compatibility_score: float  # 0-100
    match_type: str  # "complementary" | "similar" | "adversarial"
    shared_interests: List[str]
    hypothesis_alignment: Dict[str, str]  # hypothesis_id -> "agree" | "disagree"
    collaboration_potential: float  # 0-100
    reasons: List[str]
    timestamp: str


class MatchingEngine:
    """Core matching algorithm"""

    def __init__(self, profiles_file: str = "profiles.json", hypotheses_file: str = "hypotheses.json"):
        self.profiles_file = Path(profiles_file)
        self.hypotheses_file = Path(hypotheses_file)
        self.profiles: Dict[int, ResearcherProfile] = {}
        self.hypotheses: Dict[int, Hypothesis] = {}
        self.matches: List[Match] = []

        self._load_data()

    def _load_data(self):
        """Load profiles and hypotheses from files"""
        if self.profiles_file.exists():
            with open(self.profiles_file, 'r') as f:
                data = json.load(f)
                self.profiles = {
                    int(k): ResearcherProfile(**v) for k, v in data.items()
                }

        if self.hypotheses_file.exists():
            with open(self.hypotheses_file, 'r') as f:
                data = json.load(f)
                self.hypotheses = {
                    int(k): Hypothesis(**v) for k, v in data.items()
                }

    def _save_data(self):
        """Save profiles and hypotheses to files"""
        with open(self.profiles_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.profiles.items()}, f, indent=2)

        with open(self.hypotheses_file, 'w') as f:
            json.dump({k: asdict(v) for k, v in self.hypotheses.items()}, f, indent=2)

    def create_profile(
        self,
        name: str,
        institution: str,
        skills: List[str],
        interests: List[str],
        h_index: int,
        looking_for: str = "complementary"
    ) -> ResearcherProfile:
        """Create a new researcher profile"""
        profile_id = max(self.profiles.keys(), default=0) + 1

        profile = ResearcherProfile(
            profile_id=profile_id,
            name=name,
            institution=institution,
            skills=[s.strip().lower() for s in skills],
            interests=[i.strip().lower() for i in interests],
            h_index=h_index,
            hypotheses=[],
            anti_hypotheses=[],
            looking_for=looking_for,
            created_at=datetime.now().isoformat()
        )

        self.profiles[profile_id] = profile
        self._save_data()

        return profile

    def add_hypothesis(
        self,
        author_id: int,
        text: str,
        field: str,
        tags: List[str] = None
    ) -> Hypothesis:
        """Add a research hypothesis"""
        hypothesis_id = max(self.hypotheses.keys(), default=0) + 1

        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            text=text,
            field=field.lower(),
            author_id=author_id,
            tags=tags or [],
            created_at=datetime.now().isoformat()
        )

        self.hypotheses[hypothesis_id] = hypothesis

        # Add to author's supported hypotheses
        if author_id in self.profiles:
            self.profiles[author_id].hypotheses.append(text)

        self._save_data()

        return hypothesis

    def swipe(self, profile_id: int, hypothesis_id: int, action: str):
        """
        Swipe on a hypothesis

        Args:
            profile_id: Researcher ID
            hypothesis_id: Hypothesis ID
            action: "SUPPORT" | "OPPOSE" | "DEBATE"
        """
        if profile_id not in self.profiles or hypothesis_id not in self.hypotheses:
            raise ValueError("Invalid profile or hypothesis ID")

        hypothesis = self.hypotheses[hypothesis_id]
        profile = self.profiles[profile_id]

        if action == "SUPPORT":
            hypothesis.support_count += 1
            profile.hypotheses.append(hypothesis.text)
        elif action == "OPPOSE":
            hypothesis.oppose_count += 1
            profile.anti_hypotheses.append(hypothesis.text)
        elif action == "DEBATE":
            hypothesis.debate_count += 1

        self._save_data()

    def find_matches(self, profile_id: int, mode: str = "complementary", limit: int = 10) -> List[Match]:
        """
        Find potential collaborators

        Args:
            profile_id: Researcher to match
            mode: "complementary" | "similar" | "adversarial"
            limit: Maximum matches to return
        """
        if profile_id not in self.profiles:
            raise ValueError("Profile not found")

        source = self.profiles[profile_id]
        candidates = []

        for candidate_id, candidate in self.profiles.items():
            if candidate_id == profile_id:
                continue

            # Calculate match score
            score, match_type, reasons = self._calculate_match_score(
                source, candidate, mode
            )

            if score > 20:  # Minimum threshold
                # Analyze hypothesis alignment
                alignment = self._analyze_hypothesis_alignment(source, candidate)

                # Calculate collaboration potential
                collab_potential = self._calculate_collaboration_potential(
                    source, candidate, score, alignment
                )

                # Find shared interests
                shared = set(source.interests) & set(candidate.interests)

                match = Match(
                    match_id=len(self.matches) + len(candidates) + 1,
                    researcher_1=source.profile_id,
                    researcher_2=candidate.profile_id,
                    compatibility_score=round(score, 1),
                    match_type=match_type,
                    shared_interests=list(shared),
                    hypothesis_alignment=alignment,
                    collaboration_potential=round(collab_potential, 1),
                    reasons=reasons,
                    timestamp=datetime.now().isoformat()
                )

                candidates.append((score, match))

        # Sort by score and return top matches
        candidates.sort(reverse=True, key=lambda x: x[0])
        matches = [m for _, m in candidates[:limit]]

        return matches

    def _calculate_match_score(
        self,
        researcher: ResearcherProfile,
        candidate: ResearcherProfile,
        mode: str
    ) -> tuple[float, str, List[str]]:
        """Calculate compatibility score"""
        score = 0.0
        reasons = []
        match_type = mode

        # Skill complementarity
        skill_overlap = set(researcher.skills) & set(candidate.skills)
        skill_complement = (set(researcher.skills) | set(candidate.skills)) - skill_overlap

        if mode == "complementary":
            # Reward complementary skills
            score += len(skill_complement) * 8
            if skill_complement:
                reasons.append(f"Complementary skills: {', '.join(list(skill_complement)[:3])}")

            # Some overlap is good
            score += len(skill_overlap) * 3
            if skill_overlap:
                reasons.append(f"Shared expertise: {', '.join(list(skill_overlap)[:3])}")

        elif mode == "similar":
            # Reward skill overlap
            score += len(skill_overlap) * 10
            if skill_overlap:
                reasons.append(f"Similar skills: {', '.join(skill_overlap)}")

            # Penalize too much difference
            score -= len(skill_complement) * 2

        elif mode == "adversarial":
            # Reward opposing viewpoints
            match_type = "adversarial"

        # Research interest alignment
        interest_overlap = set(researcher.interests) & set(candidate.interests)
        score += len(interest_overlap) * 12

        if interest_overlap:
            reasons.append(f"Shared interests: {', '.join(interest_overlap)}")

        # H-index compatibility (prefer similar or complementary levels)
        h_diff = abs(researcher.h_index - candidate.h_index)
        if h_diff < 10:
            score += 10
            reasons.append(f"Similar research impact (h-index: {researcher.h_index} vs {candidate.h_index})")
        elif h_diff < 25:
            score += 5

        # Hypothesis alignment
        if mode == "adversarial":
            # Reward opposing hypotheses
            opposing_count = sum(
                1 for h in researcher.hypotheses
                if h in candidate.anti_hypotheses
            )
            score += opposing_count * 15
            if opposing_count > 0:
                reasons.append(f"{opposing_count} hypotheses with opposing views")

        else:
            # Reward agreement
            agreement_count = sum(
                1 for h in researcher.hypotheses
                if h in candidate.hypotheses
            )
            score += agreement_count * 10
            if agreement_count > 0:
                reasons.append(f"{agreement_count} shared hypotheses")

        return score, match_type, reasons

    def _analyze_hypothesis_alignment(
        self,
        researcher: ResearcherProfile,
        candidate: ResearcherProfile
    ) -> Dict[str, str]:
        """Analyze hypothesis agreement/disagreement"""
        alignment = {}

        for hyp in researcher.hypotheses:
            if hyp in candidate.hypotheses:
                alignment[hyp[:50]] = "agree"
            elif hyp in candidate.anti_hypotheses:
                alignment[hyp[:50]] = "disagree"

        return alignment

    def _calculate_collaboration_potential(
        self,
        researcher: ResearcherProfile,
        candidate: ResearcherProfile,
        compatibility_score: float,
        alignment: Dict[str, str]
    ) -> float:
        """Estimate likelihood of successful collaboration"""
        potential = compatibility_score * 0.6  # Base on compatibility

        # Bonus for some disagreement (productive debate)
        disagreements = sum(1 for v in alignment.values() if v == "disagree")
        if 1 <= disagreements <= 3:
            potential += 15

        # Bonus for complementary skills
        skill_union = len(set(researcher.skills) | set(candidate.skills))
        potential += skill_union * 2

        return min(100.0, potential)


def main():
    parser = argparse.ArgumentParser(
        description="HypothesisMatch - Scientific Tinder for Researchers"
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create profile
    profile_parser = subparsers.add_parser('create-profile', help='Create researcher profile')
    profile_parser.add_argument('--name', required=True, help='Researcher name')
    profile_parser.add_argument('--institution', default='Unknown', help='Institution')
    profile_parser.add_argument('--skills', required=True, help='Comma-separated skills')
    profile_parser.add_argument('--interests', required=True, help='Comma-separated interests')
    profile_parser.add_argument('--h-index', type=int, default=10, help='H-index')
    profile_parser.add_argument('--looking-for', choices=['complementary', 'similar', 'adversarial'], default='complementary')

    # Add hypothesis
    hyp_parser = subparsers.add_parser('add-hypothesis', help='Add research hypothesis')
    hyp_parser.add_argument('--profile-id', type=int, required=True, help='Author profile ID')
    hyp_parser.add_argument('--text', required=True, help='Hypothesis text')
    hyp_parser.add_argument('--field', required=True, help='Research field')
    hyp_parser.add_argument('--tags', help='Comma-separated tags')

    # Swipe
    swipe_parser = subparsers.add_parser('swipe', help='Swipe on hypothesis')
    swipe_parser.add_argument('--profile-id', type=int, required=True)
    swipe_parser.add_argument('--hypothesis-id', type=int, required=True)
    swipe_parser.add_argument('--action', choices=['SUPPORT', 'OPPOSE', 'DEBATE'], required=True)

    # Find matches
    match_parser = subparsers.add_parser('match', help='Find collaborators')
    match_parser.add_argument('--profile-id', type=int, required=True)
    match_parser.add_argument('--mode', choices=['complementary', 'similar', 'adversarial'], default='complementary')
    match_parser.add_argument('--limit', type=int, default=10)

    # List profiles
    list_parser = subparsers.add_parser('list', help='List all profiles')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = MatchingEngine()

    if args.command == 'create-profile':
        profile = engine.create_profile(
            name=args.name,
            institution=args.institution,
            skills=[s.strip() for s in args.skills.split(',')],
            interests=[i.strip() for i in args.interests.split(',')],
            h_index=args.h_index,
            looking_for=args.looking_for
        )

        print(f"\nCreated profile ID {profile.profile_id} for {profile.name}")
        print(f"Skills: {', '.join(profile.skills)}")
        print(f"Interests: {', '.join(profile.interests)}")
        print(f"Looking for: {profile.looking_for} collaborators")

    elif args.command == 'add-hypothesis':
        tags = [t.strip() for t in args.tags.split(',')] if args.tags else []

        hypothesis = engine.add_hypothesis(
            author_id=args.profile_id,
            text=args.text,
            field=args.field,
            tags=tags
        )

        print(f"\nCreated hypothesis ID {hypothesis.hypothesis_id}")
        print(f"Text: {hypothesis.text}")
        print(f"Field: {hypothesis.field}")

    elif args.command == 'swipe':
        engine.swipe(args.profile_id, args.hypothesis_id, args.action)
        print(f"\nSwiped {args.action} on hypothesis {args.hypothesis_id}")

    elif args.command == 'match':
        matches = engine.find_matches(args.profile_id, args.mode, args.limit)

        if not matches:
            print("\nNo matches found. Try creating more profiles!")
            sys.exit(0)

        researcher = engine.profiles[args.profile_id]

        print(f"\nMatches for {researcher.name} (looking for {args.mode} collaborators):")
        print("=" * 70)

        for i, match in enumerate(matches, 1):
            candidate = engine.profiles[match.researcher_2]

            print(f"\n{i}. {candidate.name} @ {candidate.institution}")
            print(f"   Compatibility: {match.compatibility_score:.1f}/100")
            print(f"   Collaboration Potential: {match.collaboration_potential:.1f}/100")
            print(f"   Match Type: {match.match_type}")

            if match.shared_interests:
                print(f"   Shared Interests: {', '.join(match.shared_interests)}")

            print(f"   Why matched:")
            for reason in match.reasons:
                print(f"     - {reason}")

            if match.hypothesis_alignment:
                print(f"   Hypothesis Alignment:")
                for hyp, stance in list(match.hypothesis_alignment.items())[:3]:
                    print(f"     {stance.upper()}: {hyp}...")

    elif args.command == 'list':
        print(f"\nAll Researcher Profiles:")
        print("=" * 70)

        for profile in engine.profiles.values():
            print(f"\nID {profile.profile_id}: {profile.name}")
            print(f"  Institution: {profile.institution}")
            print(f"  Skills: {', '.join(profile.skills)}")
            print(f"  Interests: {', '.join(profile.interests)}")
            print(f"  H-index: {profile.h_index}")
            print(f"  Hypotheses: {len(profile.hypotheses)} supported, {len(profile.anti_hypotheses)} opposed")


if __name__ == "__main__":
    main()
