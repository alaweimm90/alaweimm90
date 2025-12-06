#!/usr/bin/env python3
"""
Turingo - The Autonomous Problem-Solving Carnival
Main orchestrator for the AI research rodeo
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Import agents
from agents.executive.ringmaster import Ringmaster
from agents.executive.blueprint_boss import BlueprintBoss
from agents.executive.deal_maker import DealMaker
from agents.executive.ethics_enforcer import EthicsEnforcer

from agents.specialists.puzzle_prodigy import PuzzleProdigy
from agents.specialists.quantum_quokka import QuantumQuokka
from agents.specialists.ml_magician import MLMagician
from agents.specialists.analogy_alchemist import AnalogyAlchemist
from agents.specialists.proof_pirate import ProofPirate
from agents.specialists.verification_vigilante import VerificationVigilante
from agents.specialists.benchmark_bandit import BenchmarkBandit
from agents.specialists.code_cowboy import CodeCowboy
from agents.specialists.novelty_ninja import NoveltyNinja
from agents.specialists.skeptic_sorcerer import SkepticSorcerer

# Import workflows
from workflows.sop_01_selection import ProblemSelectionSOP
from workflows.sop_02_stampede import MultiParadigmStampedeSOP
from workflows.sop_03_validation import ValidationRodeoSOP
from workflows.sop_04_publication import PublicationParadeSOP
from workflows.sop_05_metalearning import MetaLearningSOP


@dataclass
class TuringoResult:
    """Result from Turingo rodeo"""
    problem_type: str
    instance_name: str
    objective_value: float
    solution: List[int]
    winning_paradigm: str  # quantum | ml | classical | hybrid
    improvement_over_sota: float  # percentage
    computation_time_seconds: float
    validation_status: str  # PASS | CONDITIONAL | FAIL
    novelty_score: float  # 0-100
    proof_verified: bool
    agents_used: List[str]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TuringoRodeo:
    """
    Main orchestrator for the Turingo research carnival

    Coordinates all 14 agents through 5 SOPs to solve optimization problems
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the carnival"""
        self.config = self._load_config(config_path)

        # Initialize knowledge base
        self.knowledge_base_path = Path("knowledge_base")
        self.knowledge_base_path.mkdir(exist_ok=True)

        # Initialize Executive Core (Always Active)
        self.ringmaster = Ringmaster()
        self.blueprint_boss = BlueprintBoss()
        self.deal_maker = DealMaker()
        self.ethics_enforcer = EthicsEnforcer()

        # Initialize Core Specialists
        self.puzzle_prodigy = PuzzleProdigy()
        self.quantum_quokka = QuantumQuokka()
        self.ml_magician = MLMagician()
        self.analogy_alchemist = AnalogyAlchemist()
        self.proof_pirate = ProofPirate()
        self.verification_vigilante = VerificationVigilante()
        self.benchmark_bandit = BenchmarkBandit()
        self.code_cowboy = CodeCowboy()
        self.novelty_ninja = NoveltyNinja()
        self.skeptic_sorcerer = SkepticSorcerer()

        # Initialize SOPs
        self.sop_selection = ProblemSelectionSOP(
            self.deal_maker, self.puzzle_prodigy, self.ringmaster
        )
        self.sop_stampede = MultiParadigmStampedeSOP(
            self.quantum_quokka, self.ml_magician, self.code_cowboy
        )
        self.sop_validation = ValidationRodeoSOP(
            self.benchmark_bandit, self.novelty_ninja, self.skeptic_sorcerer
        )
        self.sop_publication = PublicationParadeSOP()
        self.sop_metalearning = MetaLearningSOP()

        print("🎪 Turingo Carnival initialized!")
        print(f"   Executive Core: 4 agents active")
        print(f"   Core Specialists: 10 agents ready")
        print(f"   SOPs: 5 workflows loaded")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                return json.load(f)

        # Default config
        return {
            "paradigms": ["quantum", "ml", "classical"],
            "validation_level": "rigorous",
            "time_limit_hours": 2,
            "compute_budget_usd": 100,
            "novelty_threshold": 65,
            "require_proof": False
        }

    async def solve(
        self,
        problem_type: str,
        instance: str,
        paradigms: Optional[List[str]] = None,
        time_limit_hours: float = 2,
        validation_level: str = "rigorous"
    ) -> TuringoResult:
        """
        Solve a single optimization problem instance

        Args:
            problem_type: Type of problem (qap, tsp, vrp, etc.)
            instance: Problem instance name
            paradigms: Which approaches to use (quantum, ml, classical)
            time_limit_hours: Time limit
            validation_level: quick | standard | rigorous

        Returns:
            TuringoResult with solution and metadata
        """
        start_time = datetime.now()

        print(f"\n🎪 TURINGO RODEO: {problem_type.upper()} - {instance}")
        print(f"{'='*60}")

        # Step 1: Problem Analysis (Puzzle Prodigy)
        print("\n📚 Step 1: Problem Analysis...")
        problem_analysis = await self.puzzle_prodigy.analyze_problem(
            problem_type, instance
        )
        print(f"   Structure: {problem_analysis['structure']}")
        print(f"   Difficulty: {problem_analysis['difficulty']}/10")

        # Step 2: Ethics Check (Ethics Enforcer)
        print("\n⚖️  Step 2: Ethics Validation...")
        ethics_ok = await self.ethics_enforcer.validate(problem_type, instance)
        if not ethics_ok:
            raise ValueError("Problem failed ethics validation")
        print("   ✅ Ethics check passed")

        # Step 3: Architecture Design (Blueprint Boss)
        print("\n🏗️  Step 3: Algorithm Architecture...")
        architecture = await self.blueprint_boss.design_architecture(
            problem_analysis, paradigms or self.config["paradigms"]
        )
        print(f"   Paradigms: {', '.join(architecture['paradigms'])}")

        # Step 4: Multi-Paradigm Stampede (SOP-02)
        print(f"\n🏇 Step 4: Multi-Paradigm Stampede...")
        print(f"   Alpha Squad (Quantum) launching...")
        print(f"   Beta Brigade (ML) launching...")
        print(f"   Gamma Gang (Classical) launching...")

        candidate_solutions = await self.sop_stampede.execute(
            problem_type, instance, architecture, time_limit_hours
        )

        print(f"\n   🎯 {len(candidate_solutions)} candidate solutions generated")
        for i, sol in enumerate(candidate_solutions[:3], 1):
            print(f"   {i}. {sol['paradigm']}: obj={sol['objective']:.2f}")

        # Step 5: Select Best Solution
        best_solution = min(candidate_solutions, key=lambda x: x['objective'])
        print(f"\n🏆 Best solution: {best_solution['paradigm']}")
        print(f"   Objective: {best_solution['objective']:.2f}")

        # Step 6: Validation Rodeo (SOP-03)
        print(f"\n🔬 Step 6: Validation Rodeo...")
        validation_result = await self.sop_validation.execute(
            problem_type, instance, best_solution, validation_level
        )

        print(f"   Verdict: {validation_result['verdict']}")
        print(f"   Novelty: {validation_result['novelty_score']:.1f}/100")

        # Step 7: Proof Verification (if claimed optimal)
        proof_verified = False
        if best_solution.get('claims_optimal'):
            print(f"\n📜 Step 7: Proof Verification...")
            proof_verified = await self.verification_vigilante.verify_proof(
                problem_type, instance, best_solution
            )
            print(f"   Proof verified: {proof_verified}")

        # Compute metrics
        duration = (datetime.now() - start_time).total_seconds()

        # Get SOTA baseline for comparison
        sota_baseline = await self.benchmark_bandit.get_sota_baseline(
            problem_type, instance
        )

        improvement = 0.0
        if sota_baseline:
            improvement = ((sota_baseline - best_solution['objective']) /
                          sota_baseline * 100)

        # Create result
        result = TuringoResult(
            problem_type=problem_type,
            instance_name=instance,
            objective_value=best_solution['objective'],
            solution=best_solution['solution'],
            winning_paradigm=best_solution['paradigm'],
            improvement_over_sota=improvement,
            computation_time_seconds=duration,
            validation_status=validation_result['verdict'],
            novelty_score=validation_result['novelty_score'],
            proof_verified=proof_verified,
            agents_used=[
                "Puzzle Prodigy", "Ethics Enforcer", "Blueprint Boss",
                best_solution['paradigm'].title() + " Squad",
                "Benchmark Bandit", "Novelty Ninja", "Skeptic Sorcerer"
            ],
            timestamp=datetime.now().isoformat()
        )

        # Save result
        self._save_result(result)

        # Print summary
        self._print_summary(result)

        return result

    async def rodeo(
        self,
        mode: str = "autonomous",
        duration_weeks: int = 2,
        budget_usd: float = 1000
    ) -> List[TuringoResult]:
        """
        Run full autonomous rodeo

        Args:
            mode: autonomous | interactive
            duration_weeks: How long to run
            budget_usd: Compute budget

        Returns:
            List of TuringoResults
        """
        print(f"\n🎪 AUTONOMOUS RODEO MODE")
        print(f"{'='*60}")
        print(f"Duration: {duration_weeks} weeks")
        print(f"Budget: ${budget_usd:,.2f}")
        print(f"Mode: {mode}")

        results = []

        # Step 1: Problem Selection (SOP-01)
        print(f"\n📋 Step 1: Problem Selection (SOP-01)...")
        selected_problems = await self.sop_selection.execute(
            target_count=5, budget_usd=budget_usd
        )

        print(f"\n   Selected {len(selected_problems)} problems:")
        for i, prob in enumerate(selected_problems, 1):
            print(f"   {i}. {prob['type']}/{prob['instance']} (ROI: {prob['roi']:.1f})")

        # Step 2: Solve each problem
        for prob in selected_problems:
            print(f"\n{'='*60}")
            result = await self.solve(
                problem_type=prob['type'],
                instance=prob['instance'],
                paradigms=prob['recommended_paradigms']
            )
            results.append(result)

            # Check if we beat SOTA
            if result.improvement_over_sota > 0:
                print(f"\n🎉 NEW BEST SOLUTION FOUND!")
                print(f"   Improvement: +{result.improvement_over_sota:.1f}%")

                # Trigger publication workflow
                if mode == "autonomous":
                    await self.sop_publication.execute(result)

        # Step 3: Meta-Learning (SOP-05)
        print(f"\n🔄 Step 3: Meta-Learning...")
        await self.sop_metalearning.execute(results)

        # Summary
        print(f"\n{'='*60}")
        print(f"🎪 RODEO COMPLETE")
        print(f"{'='*60}")
        print(f"Problems solved: {len(results)}")
        print(f"New best solutions: {sum(1 for r in results if r.improvement_over_sota > 0)}")
        print(f"Average improvement: {sum(r.improvement_over_sota for r in results) / len(results):.1f}%")

        return results

    def _save_result(self, result: TuringoResult):
        """Save result to knowledge base"""
        output_dir = Path("results") / result.problem_type
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{result.instance_name}_{result.timestamp}.json"
        output_file.write_text(json.dumps(result.to_dict(), indent=2))

    def _print_summary(self, result: TuringoResult):
        """Print result summary"""
        print(f"\n{'='*60}")
        print(f"🎪 TURINGO RODEO COMPLETE")
        print(f"{'='*60}")
        print(f"Problem: {result.problem_type}/{result.instance_name}")
        print(f"Objective: {result.objective_value:.2f}")
        print(f"Paradigm: {result.winning_paradigm}")
        if result.improvement_over_sota > 0:
            print(f"Improvement: +{result.improvement_over_sota:.1f}% 🎉")
        print(f"Time: {result.computation_time_seconds:.1f}s")
        print(f"Validation: {result.validation_status}")
        print(f"Novelty: {result.novelty_score:.1f}/100")
        print(f"Agents: {', '.join(result.agents_used[:3])}...")
        print(f"{'='*60}\n")


def main():
    """CLI interface"""
    parser = argparse.ArgumentParser(
        description="Turingo - The Autonomous Problem-Solving Carnival"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve single problem")
    solve_parser.add_argument("--problem", required=True, help="Problem type (qap, tsp, etc.)")
    solve_parser.add_argument("--instance", required=True, help="Instance name")
    solve_parser.add_argument("--paradigms", nargs="+", default=["quantum", "ml", "classical"])
    solve_parser.add_argument("--time-limit", type=float, default=2.0, help="Hours")
    solve_parser.add_argument("--validation", default="rigorous", choices=["quick", "standard", "rigorous"])
    solve_parser.add_argument("--output", help="Output file")

    # Rodeo command
    rodeo_parser = subparsers.add_parser("rodeo", help="Run autonomous rodeo")
    rodeo_parser.add_argument("--mode", default="autonomous", choices=["autonomous", "interactive"])
    rodeo_parser.add_argument("--duration", type=int, default=2, help="Duration in weeks")
    rodeo_parser.add_argument("--budget", type=float, default=1000, help="Budget in USD")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize Turingo
    turingo = TuringoRodeo()

    if args.command == "solve":
        result = asyncio.run(turingo.solve(
            problem_type=args.problem,
            instance=args.instance,
            paradigms=args.paradigms,
            time_limit_hours=args.time_limit,
            validation_level=args.validation
        ))

        if args.output:
            Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))

    elif args.command == "rodeo":
        results = asyncio.run(turingo.rodeo(
            mode=args.mode,
            duration_weeks=args.duration,
            budget_usd=args.budget
        ))


if __name__ == "__main__":
    main()
