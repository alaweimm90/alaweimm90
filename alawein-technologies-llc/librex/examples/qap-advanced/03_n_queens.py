"""
Example 3: N-Queens Puzzle using UARO

Demonstrates:
- Constraint satisfaction formulation
- BacktrackingSearch primitive
- Constraint propagation
- Proof document generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import solve_with_uaro, explain_solution


class NQueensProblem:
    """N-Queens puzzle as CSP"""

    def __init__(self, n=8):
        """
        Initialize N-Queens problem

        Args:
            n: Board size (n x n)
        """
        self.n = n

        # Variables: one per row (which column to place queen)
        self.variables = list(range(n))

        # Domains: which columns available for each row
        self.domains = {row: list(range(n)) for row in self.variables}

        # Constraints: no two queens attack each other
        self.constraints = [NQueensConstraint(n)]

    def is_complete(self, assignment):
        """Check if all queens placed"""
        return len(assignment) == self.n


class NQueensConstraint:
    """N-Queens constraint: no attacks"""

    def __init__(self, n):
        self.n = n

    def is_satisfied(self, assignment):
        """Check if current assignment is valid"""
        for row1, col1 in assignment.items():
            for row2, col2 in assignment.items():
                if row1 >= row2:
                    continue  # Only check each pair once

                # Same column
                if col1 == col2:
                    return False

                # Same diagonal
                if abs(row1 - row2) == abs(col1 - col2):
                    return False

        return True


def visualize_board(n, assignment):
    """Visualize N-Queens board"""
    print("\nBoard:")
    print("-" * (2 * n + 1))

    for row in range(n):
        line = []
        for col in range(n):
            if row in assignment and assignment[row] == col:
                line.append("Q")
            else:
                line.append(".")
        print(" ".join(line))

    print("-" * (2 * n + 1))
    print()


def main():
    """Solve N-Queens with UARO"""

    n = 8  # 8x8 board

    problem = NQueensProblem(n)

    print("=" * 60)
    print(f"UARO Example 3: {n}-Queens Puzzle")
    print("=" * 60)
    print()
    print(f"Board size: {n}x{n}")
    print(f"Goal: Place {n} queens so none attack each other")
    print()

    print("Solving with UARO...")
    print()

    # Solve with UARO
    result = solve_with_uaro(problem, max_iterations=500)

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

    # Visualize solution if found
    if result.success and isinstance(result.solution, dict):
        visualize_board(n, result.solution)
        print("Queen positions (row: column):")
        for row in sorted(result.solution.keys()):
            print(f"  Row {row}: Column {result.solution[row]}")
        print()

    # Show reasoning trace
    print("Reasoning Trace (first 5 steps):")
    print("-" * 60)
    for step in result.reasoning_trace[:5]:
        print(f"Step {step.iteration}: {step.primitive_name}")
        print(f"  Success: {step.success}")
        print(f"  Confidence: {step.confidence:.2%}")
        print(f"  Reasoning: {step.reasoning}")
        print()

    if len(result.reasoning_trace) > 5:
        print(f"... ({len(result.reasoning_trace) - 5} more steps)")
        print()

    # Generate proof document
    proof_md = explain_solution(result, format="markdown")

    # Save proof
    output_file = Path(__file__).parent / "proofs" / "n_queens_proof.md"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(proof_md)

    print(f"Proof document saved to: {output_file}")
    print()

    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- UARO recognized this as a constraint satisfaction problem")
    print("- Applied backtracking search automatically")
    print("- Pruned invalid assignments early")
    print("- Generated complete proof of solution validity")
    print()


if __name__ == "__main__":
    main()
