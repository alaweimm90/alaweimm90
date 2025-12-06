# UARO Example Gallery

This directory contains runnable examples demonstrating UARO solving diverse problems across different domains.

## Examples

### 1. Sudoku Solver (`01_sudoku_solver.py`)

**Problem Type**: Constraint Satisfaction Problem (CSP)

**Demonstrates**:
- CSP formulation with variables, domains, constraints
- ConstraintPropagation primitive
- BacktrackingSearch primitive
- Proof document generation

**Run**:
```bash
cd examples
python 01_sudoku_solver.py
```

**Key Learning**: UARO automatically recognizes constraint satisfaction structure and applies appropriate primitives (constraint propagation + backtracking).

---

### 2. Robot Path Planning (`02_path_planning.py`)

**Problem Type**: Search Problem

**Demonstrates**:
- Problem interface implementation (initial_state, goal_test, actions, result, cost)
- BreadthFirstSearch primitive
- DepthFirstSearch primitive
- State space exploration

**Run**:
```bash
cd examples
python 02_path_planning.py
```

**Key Learning**: UARO identifies search problem structure and systematically explores state space to find path from start to goal.

---

### 3. N-Queens Puzzle (`03_n_queens.py`)

**Problem Type**: Constraint Satisfaction Problem

**Demonstrates**:
- Classic backtracking problem
- Constraint formulation (no two queens attack)
- Early pruning of invalid assignments
- Solution visualization

**Run**:
```bash
cd examples
python 03_n_queens.py
```

**Key Learning**: UARO uses backtracking with constraint checking to efficiently solve combinatorial problems.

---

### 4. Logic Puzzle Solver (`04_logic_puzzle.py`)

**Problem Type**: Logic Problem

**Demonstrates**:
- Knowledge base with facts and rules
- ForwardChaining primitive (derive all consequences)
- BackwardChaining primitive (prove specific goal)
- Inference chain visualization

**Run**:
```bash
cd examples
python 04_logic_puzzle.py
```

**Key Learning**: UARO applies forward and backward chaining to derive new facts and prove goals from knowledge base.

---

### 5. Traveling Salesman Problem (`05_optimization.py`)

**Problem Type**: Optimization Problem

**Demonstrates**:
- LocalSearch (hill climbing) primitive
- SimulatedAnnealing primitive
- Neighbor generation
- Objective function optimization

**Run**:
```bash
cd examples
python 05_optimization.py
```

**Key Learning**: UARO uses local search and simulated annealing to optimize tour length, escaping local optima.

---

## Running All Examples

To run all examples sequentially:

```bash
cd examples
for file in 0*.py; do
    echo "Running $file..."
    python "$file"
    echo ""
done
```

## Proof Documents

Each example generates a proof document in `examples/proofs/` showing:
- Complete reasoning trace
- Primitives used at each step
- Confidence evolution
- Solution validation
- Known limitations

Proof documents are available in multiple formats:
- Markdown (`.md`)
- HTML (`.html`)
- LaTeX (`.tex`)
- JSON (`.json`)

## Understanding the Output

Each example shows:

1. **Problem Setup**: Initial state and goal
2. **UARO Solving**: Automatic primitive selection
3. **Solution Results**: Success, iterations, duration, confidence
4. **Reasoning Trace**: Step-by-step decision log
5. **Key Insights**: What UARO learned about the problem

## Extending Examples

To create your own UARO example:

1. **Define your problem** using one of:
   - `Problem` interface (for search problems)
   - CSP structure (variables, domains, constraints)
   - Logic structure (facts, rules)
   - Optimization structure (objective, neighbors)

2. **Solve with UARO**:
   ```python
   from uaro import solve_with_uaro
   result = solve_with_uaro(problem)
   ```

3. **Generate proof**:
   ```python
   from uaro import explain_solution
   proof = explain_solution(result, format="markdown")
   ```

## Problem Type Recognition

UARO automatically recognizes problem structure:

| Attributes | Recognized As | Primitives Used |
|------------|---------------|-----------------|
| `initial_state`, `goal_test`, `actions`, `result` | Search Problem | BFS, DFS, BestFirst, Beam |
| `variables`, `domains`, `constraints` | CSP | ConstraintPropagation, Backtracking |
| `facts`, `rules` | Logic Problem | ForwardChaining, BackwardChaining |
| `objective`, `neighbors` | Optimization | LocalSearch, SimulatedAnnealing |
| `split` | Decomposable | DivideAndConquer, Hierarchical |

## Performance Notes

- **Sudoku**: Typically solves in < 10 iterations
- **Path Planning**: Depends on grid size and obstacle density
- **N-Queens**: O(n!) worst case, but pruning helps significantly
- **Logic Puzzles**: Linear in number of rules
- **TSP**: Finds good solutions (not optimal) in reasonable time

## Next Steps

After exploring these examples:

1. **Try modifying parameters**: Change grid size, number of queens, city count
2. **Add your own problems**: Implement Problem interface for your domain
3. **Compare primitives**: See which primitives work best for your problem type
4. **Publish to marketplace**: Share your custom primitives with community

## References

- UARO Documentation: `../CYCLES_27-32_UARO_REPORT.md`
- Primitive API: `../src/uaro/reasoning_primitives.py`
- Solver API: `../src/uaro/universal_solver.py`
- Explainability API: `../src/uaro/explainability.py`
