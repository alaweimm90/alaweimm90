"""
Example 4: Logic Puzzle Solver using UARO

Demonstrates:
- Logic problem formulation
- ForwardChaining primitive
- BackwardChaining primitive
- Proof document generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import solve_with_uaro, explain_solution


class LogicPuzzle:
    """
    Logic puzzle: Determine weather from observations

    Facts:
    - The sky is blue
    - The grass is green

    Rules:
    - If sky is blue → it's daytime
    - If it's daytime → sun is out
    - If sun is out → temperature is warm
    - If grass is green → it rained recently
    - If it rained recently and sun is out → there might be a rainbow

    Goal: Prove "there might be a rainbow"
    """

    def __init__(self):
        # Initial facts (observations)
        self.facts = {
            "sky_is_blue",
            "grass_is_green"
        }

        # Inference rules (premise → conclusion)
        self.rules = [
            ("sky_is_blue", "daytime"),
            ("daytime", "sun_is_out"),
            ("sun_is_out", "temperature_is_warm"),
            ("grass_is_green", "rained_recently"),
            (["sun_is_out", "rained_recently"], "rainbow_possible"),
        ]

        # Goal to prove
        self.goal = "rainbow_possible"


class WeatherReasoningProblem(LogicPuzzle):
    """Extended logic puzzle with more complex reasoning"""

    def __init__(self):
        super().__init__()

        # Add more facts
        self.facts.update({
            "birds_singing",
            "flowers_blooming"
        })

        # Add more rules
        self.rules.extend([
            ("temperature_is_warm", "comfortable_outside"),
            ("birds_singing", "spring_season"),
            ("flowers_blooming", "spring_season"),
            (["comfortable_outside", "spring_season"], "perfect_picnic_weather"),
        ])


def print_knowledge_base(puzzle):
    """Print knowledge base"""
    print("\nKnowledge Base:")
    print("-" * 60)

    print("\nInitial Facts:")
    for fact in sorted(puzzle.facts):
        print(f"  - {fact.replace('_', ' ').title()}")

    print("\nInference Rules:")
    for i, rule in enumerate(puzzle.rules, 1):
        if isinstance(rule[0], list):
            premises = " AND ".join(p.replace("_", " ").title() for p in rule[0])
            conclusion = rule[1].replace("_", " ").title()
            print(f"  {i}. If {premises} → {conclusion}")
        else:
            premise = rule[0].replace("_", " ").title()
            conclusion = rule[1].replace("_", " ").title()
            print(f"  {i}. If {premise} → {conclusion}")

    print()


def main():
    """Solve logic puzzle with UARO"""

    puzzle = LogicPuzzle()

    print("=" * 60)
    print("UARO Example 4: Logic Puzzle Solver")
    print("=" * 60)

    print_knowledge_base(puzzle)

    print(f"Goal: Prove '{puzzle.goal.replace('_', ' ').title()}'")
    print()

    print("Solving with UARO...")
    print()

    # Solve with UARO
    result = solve_with_uaro(puzzle, max_iterations=100)

    # Print results
    print("=" * 60)
    print("Solution Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration_seconds:.3f} seconds")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Primitives used: {', '.join(result.primitives_used)}")
    print()

    # Show derived facts
    if isinstance(result.solution, set):
        derived = result.solution - puzzle.facts
        print("Derived Facts:")
        for fact in sorted(derived):
            print(f"  - {fact.replace('_', ' ').title()}")
        print()

        if puzzle.goal in result.solution:
            print(f"✓ Successfully proved: {puzzle.goal.replace('_', ' ').title()}")
        else:
            print(f"✗ Could not prove: {puzzle.goal.replace('_', ' ').title()}")
        print()

    # Show reasoning trace
    print("Reasoning Trace:")
    print("-" * 60)
    for step in result.reasoning_trace[:8]:
        print(f"Step {step.iteration}: {step.primitive_name}")
        print(f"  Success: {step.success}")
        print(f"  Confidence: {step.confidence:.2%}")
        print(f"  Reasoning: {step.reasoning}")
        print()

    if len(result.reasoning_trace) > 8:
        print(f"... ({len(result.reasoning_trace) - 8} more steps)")
        print()

    # Generate proof document
    proof_md = explain_solution(result, format="markdown")

    # Save proof
    output_file = Path(__file__).parent / "proofs" / "logic_puzzle_proof.md"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(proof_md)

    print(f"Proof document saved to: {output_file}")
    print()

    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- UARO recognized this as a logic problem")
    print("- Applied forward chaining to derive new facts")
    print("- Applied backward chaining to prove goal")
    print("- Generated complete proof chain")
    print()

    # Now solve more complex puzzle
    print("\n")
    print("=" * 60)
    print("Extended Weather Reasoning")
    print("=" * 60)

    extended_puzzle = WeatherReasoningProblem()
    print_knowledge_base(extended_puzzle)

    print("Solving extended puzzle...")
    result2 = solve_with_uaro(extended_puzzle, max_iterations=150)

    print(f"\nSuccess: {result2.success}")
    print(f"Iterations: {result2.iterations}")
    print(f"Duration: {result2.duration_seconds:.3f} seconds")
    print(f"Primitives used: {', '.join(result2.primitives_used)}")

    if isinstance(result2.solution, set):
        all_derived = result2.solution - extended_puzzle.facts
        print(f"\nTotal derived facts: {len(all_derived)}")
        print("\nAll Conclusions:")
        for fact in sorted(all_derived):
            print(f"  - {fact.replace('_', ' ').title()}")
    print()


if __name__ == "__main__":
    main()
