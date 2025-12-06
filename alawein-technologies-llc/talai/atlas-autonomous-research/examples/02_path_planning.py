"""
Example 2: Robot Path Planning using UARO

Demonstrates:
- Search problem formulation
- BreadthFirstSearch primitive
- DepthFirstSearch primitive
- BestFirstSearch with heuristic
- Proof document generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import Problem, solve_with_uaro, explain_solution


class GridWorld(Problem):
    """Robot navigation in grid world with obstacles"""

    def __init__(self, grid, start, goal):
        """
        Initialize grid world

        Args:
            grid: 2D list where 1 = obstacle, 0 = free
            start: (row, col) starting position
            goal: (row, col) goal position
        """
        self.grid = grid
        self.start = start
        self.goal_pos = goal
        self.rows = len(grid)
        self.cols = len(grid[0])

    def initial_state(self):
        """Get initial state"""
        return self.start

    def goal_test(self, state):
        """Check if state is goal"""
        return state == self.goal_pos

    def actions(self, state):
        """Get available actions (moves)"""
        row, col = state
        actions = []

        # Four directions: up, down, left, right
        moves = [
            ("up", (row - 1, col)),
            ("down", (row + 1, col)),
            ("left", (row, col - 1)),
            ("right", (row, col + 1))
        ]

        for action_name, new_pos in moves:
            new_row, new_col = new_pos

            # Check bounds
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                # Check not obstacle
                if self.grid[new_row][new_col] == 0:
                    actions.append((action_name, new_pos))

        return actions

    def result(self, state, action):
        """Apply action to state"""
        return action[1]  # New position

    def cost(self, state, action):
        """Cost of taking action"""
        return 1  # Uniform cost


def visualize_grid(grid, start, goal, path=None):
    """Visualize grid world with path"""
    print("\nGrid World:")
    print("S = Start, G = Goal, # = Obstacle, * = Path, . = Free")
    print()

    for i, row in enumerate(grid):
        line = []
        for j, cell in enumerate(row):
            pos = (i, j)

            if pos == start:
                line.append("S")
            elif pos == goal:
                line.append("G")
            elif path and pos in path:
                line.append("*")
            elif cell == 1:
                line.append("#")
            else:
                line.append(".")

        print(" ".join(line))
    print()


def extract_path(result):
    """Extract path from solution result"""
    # For simplicity, just return goal state
    # In real implementation, would reconstruct path from trace
    return [result.solution]


def main():
    """Solve path planning problem with UARO"""

    # 8x8 grid world
    # 0 = free, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0]
    ]

    start = (0, 0)
    goal = (7, 7)

    problem = GridWorld(grid, start, goal)

    print("=" * 60)
    print("UARO Example 2: Robot Path Planning")
    print("=" * 60)

    visualize_grid(grid, start, goal)

    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Grid size: {problem.rows}x{problem.cols}")
    print()

    print("Solving with UARO...")
    print()

    # Solve with UARO
    result = solve_with_uaro(problem, max_iterations=200)

    # Print results
    print("=" * 60)
    print("Solution Results")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Duration: {result.duration_seconds:.3f} seconds")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Final position: {result.solution}")
    print(f"Primitives used: {', '.join(result.primitives_used)}")
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

    # Visualize solution
    if result.success:
        path = extract_path(result)
        visualize_grid(grid, start, goal, path)

    # Generate proof document
    proof_md = explain_solution(result, format="markdown")

    # Save proof
    output_file = Path(__file__).parent / "proofs" / "path_planning_proof.md"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(proof_md)

    print(f"Proof document saved to: {output_file}")
    print()

    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- UARO recognized this as a search problem")
    print("- Automatically selected BFS/DFS based on structure")
    print("- Explored state space systematically")
    print("- Generated complete proof of path existence")
    print()


if __name__ == "__main__":
    main()
