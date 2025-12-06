"""
Advanced optimization methods for Librex

This module contains state-of-the-art metaheuristic algorithms
for solving complex optimization problems.

Methods:
- Ant Colony Optimization (ACO): Pheromone-based search for combinatorial problems
- Particle Swarm Optimization (PSO): Swarm intelligence for continuous/discrete optimization
- Variable Neighborhood Search (VNS): Multiple neighborhood structures exploration
- Iterated Local Search (ILS): Perturbation-based local search framework
- GRASP: Greedy Randomized Adaptive Search Procedure
"""

from Librex.methods.advanced.aco import ant_colony_optimize
from Librex.methods.advanced.grasp import grasp_optimize
from Librex.methods.advanced.ils import iterated_local_search_optimize
from Librex.methods.advanced.pso import particle_swarm_optimize
from Librex.methods.advanced.vns import variable_neighborhood_search_optimize

__all__ = [
    'ant_colony_optimize',
    'particle_swarm_optimize',
    'variable_neighborhood_search_optimize',
    'iterated_local_search_optimize',
    'grasp_optimize',
]