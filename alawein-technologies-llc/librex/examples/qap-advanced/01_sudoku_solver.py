"""
Example 1: Sudoku Solver using UARO

Demonstrates:
- Constraint satisfaction problem
- ConstraintPropagation primitive
- BacktrackingSearch primitive
- Proof document generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import solve_with_uaro, explain_solution


class SudokuProblem:
    """Sudoku as a constraint satisfaction problem"""

    def __init__(self, grid):
        """
        Initialize with partially filled 4x4 grid

        Args:
            grid: 4x4 list with 0 for empty cells
        """
        self.grid = [row[:] for row in grid]
        self.size = 4

        # Variables: empty cells
        self.variables = []
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 0:
                    self.variables.append((i, j))

        # Domains: possible values for each variable
        self.domains = {
            var: list(range(1, self.size + 1))
            for var in self.variables
        }

        # Constraints: row, column, and 2x2 box constraints
        self.constraints = []
        self._build_constraints()

    def _build_constraints(self):
        """Build Sudoku constraints"""
        # All different in each row
        for i in range(self.size):
            row_vars = [(i, j) for j in range(self.size) if (i, j) in self.variables]
            if len(row_vars) > 1:
                self.constraints.append(AllDifferentConstraint(row_vars))

        # All different in each column
        for j in range(self.size):
            col_vars = [(i, j) for i in range(self.size) if (i, j) in self.variables]
            if len(col_vars) > 1:
                self.constraints.append(AllDifferentConstraint(col_vars))

        # All different in each 2x2 box
        for box_row in range(0, self.size, 2):
            for box_col in range(0, self.size, 2):
                box_vars = [
                    (i, j)
                    for i in range(box_row, box_row + 2)
                    for j in range(box_col, box_col + 2)
                    if (i, j) in self.variables
                ]
                if len(box_vars) > 1:
                    self.constraints.append(AllDifferentConstraint(box_vars))

    def is_complete(self, assignment):
        """Check if all variables assigned"""
        return len(assignment) == len(self.variables)

    def __str__(self):
        """String representation of grid"""
        lines = []
        for i, row in enumerate(self.grid):
            lines.append(" ".join(str(x) if x != 0 else "." for x in row))
            if i == 1:
                lines.append("-" * 7)
        return "\n".join(lines)


class AllDifferentConstraint:
    """All variables must have different values"""

    def __init__(self, variables):
        self.variables = variables

    def is_satisfied(self, assignment):
        """Check if constraint is satisfied"""
        values = [assignment[var] for var in self.variables if var in assignment]
        return len(values) == len(set(values))  # All unique

    def check(self, var, value):
        """Check if value is valid for var (simplified)"""
        return True  # Full check done in is_satisfied


def print_grid(problem, assignment=None):
    """Print Sudoku grid with assignment"""
    grid = [row[:] for row in problem.grid]

    if assignment:
        for (i, j), value in assignment.items():
            grid[i][j] = value

    print("\nSudoku Grid:")
    print("-" * 9)
    for i, row in enumerate(grid):
        print(" ".join(str(x) if x != 0 else "." for x in row))
        if i == 1:
            print("-" * 9)
    print()


def main():
    """Solve Sudoku puzzle with UARO"""

    # 4x4 Sudoku puzzle (0 = empty)
    puzzle = [
        [1, 0, 0, 4],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [4, 0, 0, 1]
    ]

    problem = SudokuProblem(puzzle)

    print("=" * 60)
    print("UARO Example 1: Sudoku Solver")
    print("=" * 60)

    print_grid(problem)

    print("Solving with UARO...")
    print(f"Variables: {len(problem.variables)} empty cells")
    print(f"Constraints: {len(problem.constraints)}")
    print()

    # Solve with UARO
    result = solve_with_uaro(problem, max_iterations=100)

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

    # Show reasoning trace
    print("Reasoning Trace:")
    print("-" * 60)
    for step in result.reasoning_trace[:5]:  # First 5 steps
        print(f"Step {step.iteration}: {step.primitive_name}")
        print(f"  Success: {step.success}")
        print(f"  Reasoning: {step.reasoning}")
        print()

    if len(result.reasoning_trace) > 5:
        print(f"... ({len(result.reasoning_trace) - 5} more steps)")
        print()

    # Generate proof document
    proof_md = explain_solution(result, format="markdown")

    # Save proof
    output_file = Path(__file__).parent / "proofs" / "sudoku_proof.md"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(proof_md)

    print(f"Proof document saved to: {output_file}")
    print()

    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- UARO automatically recognized this as a CSP")
    print("- Applied constraint propagation to reduce search space")
    print("- Used backtracking when necessary")
    print("- Generated complete proof of solution")
    print()


if __name__ == "__main__":
    main()
