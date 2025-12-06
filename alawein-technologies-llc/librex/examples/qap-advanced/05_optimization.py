"""
Example 5: Optimization Problem using UARO

Demonstrates:
- Optimization problem formulation
- LocalSearch (hill climbing) primitive
- SimulatedAnnealing primitive
- Proof document generation
"""

import sys
from pathlib import Path
import math
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uaro import solve_with_uaro, explain_solution


class TSPState:
    """State for Traveling Salesman Problem"""

    def __init__(self, tour, distances):
        """
        Initialize TSP state

        Args:
            tour: List of city indices representing tour order
            distances: 2D distance matrix
        """
        self.tour = tour
        self.distances = distances

    def neighbors(self):
        """Generate neighbor states (2-opt swaps)"""
        neighbors = []

        for i in range(len(self.tour)):
            for j in range(i + 2, len(self.tour)):
                # Swap tour[i:j]
                new_tour = self.tour[:i] + self.tour[i:j][::-1] + self.tour[j:]
                neighbors.append(TSPState(new_tour, self.distances))

        return neighbors

    def random_neighbor(self):
        """Generate random neighbor (for simulated annealing)"""
        i = random.randint(0, len(self.tour) - 1)
        j = random.randint(0, len(self.tour) - 1)

        if i > j:
            i, j = j, i

        if j - i < 2:
            # Can't swap, return self
            return self

        new_tour = self.tour[:i] + self.tour[i:j][::-1] + self.tour[j:]
        return TSPState(new_tour, self.distances)

    def copy(self):
        """Create copy of state"""
        return TSPState(self.tour[:], self.distances)

    def __eq__(self, other):
        """Check equality"""
        return self.tour == other.tour

    def __str__(self):
        """String representation"""
        return f"Tour: {self.tour}"


def calculate_tour_length(tour, distances):
    """Calculate total tour length"""
    length = 0
    for i in range(len(tour)):
        from_city = tour[i]
        to_city = tour[(i + 1) % len(tour)]
        length += distances[from_city][to_city]
    return length


def create_distance_matrix(cities):
    """Create distance matrix from city coordinates"""
    n = len(cities)
    distances = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                distances[i][j] = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    return distances


def visualize_tour(cities, tour, title="Tour"):
    """ASCII visualization of tour"""
    print(f"\n{title}:")
    print("-" * 40)

    # Show tour order
    print("Tour order:", " → ".join(str(c) for c in tour))

    # Show city coordinates
    print("\nCity Coordinates:")
    for i, (x, y) in enumerate(cities):
        marker = " *" if i == tour[0] else ""
        print(f"  City {i}: ({x:3.1f}, {y:3.1f}){marker}")

    print()


def main():
    """Solve TSP with UARO"""

    # Define cities (x, y coordinates)
    cities = [
        (0, 0),    # City 0
        (1, 3),    # City 1
        (4, 3),    # City 2
        (6, 1),    # City 3
        (3, 0),    # City 4
        (5, 5),    # City 5
    ]

    # Create distance matrix
    distances = create_distance_matrix(cities)

    # Initial random tour
    initial_tour = list(range(len(cities)))
    random.shuffle(initial_tour)

    initial_state = TSPState(initial_tour, distances)

    print("=" * 60)
    print("UARO Example 5: Traveling Salesman Problem")
    print("=" * 60)

    print(f"\nNumber of cities: {len(cities)}")
    print(f"Possible tours: {math.factorial(len(cities) - 1) // 2}")

    visualize_tour(cities, initial_tour, "Initial Random Tour")

    initial_length = calculate_tour_length(initial_tour, distances)
    print(f"Initial tour length: {initial_length:.2f}")
    print()

    print("Solving with UARO...")
    print()

    # Define objective function (negative because we minimize)
    def objective(state):
        length = calculate_tour_length(state.tour, state.distances)
        return -length  # Negative for minimization

    # Create wrapper problem
    class TSPProblem:
        def __init__(self, initial_state, objective_fn):
            self.initial_state_value = initial_state
            self.objective = objective_fn

    problem = TSPProblem(initial_state, objective)

    # Solve with UARO
    result = solve_with_uaro(problem, max_iterations=300)

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

    # Show final tour
    if hasattr(result.solution, 'tour'):
        final_tour = result.solution.tour
        final_length = calculate_tour_length(final_tour, distances)

        visualize_tour(cities, final_tour, "Optimized Tour")

        print(f"Final tour length: {final_length:.2f}")
        print(f"Improvement: {initial_length - final_length:.2f} ({(initial_length - final_length) / initial_length * 100:.1f}%)")
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
    output_file = Path(__file__).parent / "proofs" / "tsp_proof.md"
    output_file.parent.mkdir(exist_ok=True)
    output_file.write_text(proof_md)

    print(f"Proof document saved to: {output_file}")
    print()

    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("- UARO recognized this as an optimization problem")
    print("- Applied local search (hill climbing) to find better solutions")
    print("- May have used simulated annealing to escape local optima")
    print("- Generated proof of solution quality")
    print()


if __name__ == "__main__":
    main()
