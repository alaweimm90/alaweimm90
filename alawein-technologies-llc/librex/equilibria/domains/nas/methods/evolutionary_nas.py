"""Evolutionary algorithms for Neural Architecture Search.

Implements population-based search methods including:
- Standard evolutionary NAS
- Regularized evolution (AmoebaNet)
- Multi-objective evolutionary NAS
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
import random
from dataclasses import dataclass
import warnings

from ..architecture import NASCell, MacroArchitecture, Operation, OperationType, Layer
from ..nas_problem import NASProblem, SearchSpace


@dataclass
class Individual:
    """Individual in the evolutionary population."""
    architecture: Union[NASCell, MacroArchitecture]
    encoding: np.ndarray
    fitness: float
    metrics: Dict[str, float]
    age: int = 0


class EvolutionaryNAS:
    """
    Evolutionary algorithm for NAS.

    Uses genetic operations (mutation, crossover) to evolve architectures.
    """

    def __init__(self,
                 problem: NASProblem,
                 population_size: int = 50,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.9,
                 elitism_ratio: float = 0.1):
        """
        Initialize evolutionary NAS.

        Args:
            problem: NAS problem to solve
            population_size: Size of population
            tournament_size: Size for tournament selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_ratio: Fraction of population to preserve
        """
        self.problem = problem
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_size = int(population_size * elitism_ratio)

    def initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []

        for _ in range(self.population_size):
            # Create architecture with random initialization
            if self.problem.search_space == SearchSpace.CELL:
                arch = self.problem.create_architecture()
                self._random_cell_architecture(arch)
            else:
                arch = self.problem.create_architecture()
                self._random_macro_architecture(arch)

            # Create individual
            encoding = self._encode_architecture(arch)
            metrics = self.problem.evaluate_architecture(arch, return_all_metrics=True)
            fitness = metrics.get('objective', 0)

            individual = Individual(
                architecture=arch,
                encoding=encoding,
                fitness=fitness,
                metrics=metrics
            )
            population.append(individual)

        return population

    def _random_cell_architecture(self, cell: NASCell):
        """Randomize cell architecture."""
        cell.edges = []

        # Add random edges
        for to_node in range(cell.n_inputs, cell.n_inputs + cell.n_nodes):
            # Each node gets 2 random input connections
            n_connections = min(2, to_node)
            from_nodes = np.random.choice(to_node, size=n_connections, replace=False)

            for from_node in from_nodes:
                op_type = np.random.choice(list(OperationType))
                channels = np.random.choice([32, 64, 128, 256])
                op = Operation(op_type=op_type, channels=channels)
                cell.add_edge(int(from_node), to_node, op)

    def _random_macro_architecture(self, arch: MacroArchitecture):
        """Randomize macro architecture."""
        arch.layers = []

        # Random depth
        depth = np.random.randint(3, min(15, arch.max_layers))

        # Random layers
        prev_channels = arch.input_channels
        for i in range(depth):
            if i == depth - 1:
                # Final layer
                layer = Layer('fc', arch.num_classes)
            else:
                layer_type = np.random.choice(['conv', 'residual', 'pool'])
                if layer_type == 'pool':
                    channels = prev_channels
                else:
                    channels = np.random.choice([32, 64, 128, 256, 512])

                kernel_size = np.random.choice([1, 3, 5]) if layer_type == 'conv' else None

                layer = Layer(
                    layer_type=layer_type,
                    channels=channels,
                    kernel_size=kernel_size,
                    stride=2 if layer_type == 'pool' else 1
                )
                prev_channels = channels

            arch.layers.append(layer)

        # Random skip connections
        n_skips = np.random.randint(0, min(3, depth // 2))
        for _ in range(n_skips):
            from_idx = np.random.randint(0, depth - 1)
            to_idx = np.random.randint(from_idx + 1, depth)
            arch.add_skip_connection(from_idx, to_idx)

    def _encode_architecture(self, arch: Union[NASCell, MacroArchitecture]) -> np.ndarray:
        """Encode architecture to vector."""
        if isinstance(arch, NASCell):
            return arch.to_encoding()
        else:
            return arch.to_flat_encoding()

    def _decode_architecture(self, encoding: np.ndarray) -> Union[NASCell, MacroArchitecture]:
        """Decode vector to architecture."""
        arch = self.problem.create_architecture()
        if isinstance(arch, NASCell):
            arch.from_encoding(encoding)
        else:
            arch.from_flat_encoding(encoding)
        return arch

    def tournament_selection(self, population: List[Individual]) -> Individual:
        """Select individual using tournament selection."""
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate:
            # No crossover, return copies
            return parent1, parent2

        # Uniform crossover on encodings
        encoding1 = parent1.encoding.copy()
        encoding2 = parent2.encoding.copy()

        mask = np.random.random(len(encoding1)) < 0.5
        temp = encoding1[mask].copy()
        encoding1[mask] = encoding2[mask]
        encoding2[mask] = temp

        # Create offspring
        arch1 = self._decode_architecture(encoding1)
        arch2 = self._decode_architecture(encoding2)

        # Evaluate
        metrics1 = self.problem.evaluate_architecture(arch1, return_all_metrics=True)
        metrics2 = self.problem.evaluate_architecture(arch2, return_all_metrics=True)

        offspring1 = Individual(
            architecture=arch1,
            encoding=encoding1,
            fitness=metrics1.get('objective', 0),
            metrics=metrics1
        )

        offspring2 = Individual(
            architecture=arch2,
            encoding=encoding2,
            fitness=metrics2.get('objective', 0),
            metrics=metrics2
        )

        return offspring1, offspring2

    def mutate(self, individual: Individual) -> Individual:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual

        encoding = individual.encoding.copy()

        if self.problem.search_space == SearchSpace.CELL:
            # Mutate cell encoding
            n_mutations = np.random.randint(1, 4)
            for _ in range(n_mutations):
                idx = np.random.randint(len(encoding))
                if idx % 2 == 0:
                    # Operation index
                    encoding[idx] = np.random.randint(0, len(OperationType))
                else:
                    # Channel size
                    encoding[idx] = np.random.choice([32, 64, 128, 256])

        else:  # MacroArchitecture
            # Mutate macro encoding
            mutation_type = np.random.choice(['layer', 'depth', 'skip'])

            if mutation_type == 'layer':
                # Change a layer type or parameters
                idx = np.random.randint(1, self.problem.macro_config['max_layers'])
                if idx * 3 < len(encoding):
                    encoding[idx * 3] = np.random.randint(0, 6)  # Layer type
                    encoding[idx * 3 + 1] = np.random.choice([32, 64, 128, 256, 512])  # Channels
                    encoding[idx * 3 + 2] = np.random.choice([0, 1, 3, 5])  # Kernel size

            elif mutation_type == 'depth':
                # Change depth
                current_depth = int(encoding[0])
                new_depth = np.clip(current_depth + np.random.randint(-2, 3), 3, 15)
                encoding[0] = new_depth

            else:  # Skip connection
                # Toggle a random skip connection
                skip_start = 1 + self.problem.macro_config['max_layers'] * 3
                if skip_start < len(encoding):
                    idx = np.random.randint(skip_start, len(encoding))
                    encoding[idx] = 1 - encoding[idx]  # Toggle

        # Create mutated individual
        arch = self._decode_architecture(encoding)
        metrics = self.problem.evaluate_architecture(arch, return_all_metrics=True)

        return Individual(
            architecture=arch,
            encoding=encoding,
            fitness=metrics.get('objective', 0),
            metrics=metrics
        )

    def evolve_generation(self, population: List[Individual]) -> List[Individual]:
        """Evolve one generation."""
        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        # Elitism: keep best individuals
        new_population = population[:self.elitism_size]

        # Generate offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = self.tournament_selection(population)
            parent2 = self.tournament_selection(population)

            # Crossover
            offspring1, offspring2 = self.crossover(parent1, parent2)

            # Mutation
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # Age increment
            offspring1.age = 0
            offspring2.age = 0

            new_population.extend([offspring1, offspring2])

        # Trim to population size
        new_population = new_population[:self.population_size]

        # Age existing individuals
        for ind in new_population:
            ind.age += 1

        return new_population

    def run(self, n_generations: int = 50) -> Dict[str, Any]:
        """
        Run evolutionary NAS.

        Args:
            n_generations: Number of generations to evolve

        Returns:
            Dictionary with results
        """
        # Initialize population
        population = self.initialize_population()
        best_individual = max(population, key=lambda x: x.fitness)

        history = {
            'best_fitness': [best_individual.fitness],
            'avg_fitness': [np.mean([ind.fitness for ind in population])],
            'best_metrics': [best_individual.metrics]
        }

        # Evolution loop
        for generation in range(n_generations):
            # Evolve
            population = self.evolve_generation(population)

            # Track best
            current_best = max(population, key=lambda x: x.fitness)
            if current_best.fitness > best_individual.fitness:
                best_individual = current_best

            # Update history
            history['best_fitness'].append(best_individual.fitness)
            history['avg_fitness'].append(np.mean([ind.fitness for ind in population]))
            history['best_metrics'].append(best_individual.metrics)

            # Progress report
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{n_generations}: "
                     f"Best fitness: {best_individual.fitness:.4f}, "
                     f"Avg fitness: {history['avg_fitness'][-1]:.4f}")

        return {
            'best_architecture': best_individual.architecture,
            'best_encoding': best_individual.encoding,
            'best_fitness': best_individual.fitness,
            'best_metrics': best_individual.metrics,
            'final_population': population,
            'history': history
        }


def evolutionary_nas(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run evolutionary NAS with configuration.

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    evo_nas = EvolutionaryNAS(
        problem=problem,
        population_size=config.get('population_size', 50),
        tournament_size=config.get('tournament_size', 5),
        mutation_rate=config.get('mutation_rate', 0.1),
        crossover_rate=config.get('crossover_rate', 0.9),
        elitism_ratio=config.get('elitism_ratio', 0.1)
    )

    return evo_nas.run(n_generations=config.get('n_generations', 50))


def regularized_evolution(problem: NASProblem, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Regularized evolution (AmoebaNet style).

    Uses aging to maintain diversity and prevent premature convergence.

    Args:
        problem: NAS problem to solve
        config: Configuration dictionary

    Returns:
        Results dictionary
    """
    population_size = config.get('population_size', 100)
    sample_size = config.get('sample_size', 25)
    n_iterations = config.get('n_iterations', 1000)
    max_age = config.get('max_age', 50)

    # Initialize population
    evo_nas = EvolutionaryNAS(problem=problem, population_size=population_size)
    population = evo_nas.initialize_population()

    best_individual = max(population, key=lambda x: x.fitness)
    history = []

    for iteration in range(n_iterations):
        # Sample subset for tournament
        sample = random.sample(population, sample_size)
        parent = max(sample, key=lambda x: x.fitness)

        # Create offspring through mutation only (no crossover in regularized evolution)
        offspring = evo_nas.mutate(parent)
        offspring.age = 0

        # Add to population
        population.append(offspring)

        # Remove oldest individual
        oldest_idx = np.argmax([ind.age for ind in population])
        population.pop(oldest_idx)

        # Age all individuals
        for ind in population:
            ind.age += 1

        # Track best
        if offspring.fitness > best_individual.fitness:
            best_individual = offspring

        # Record history
        if iteration % 100 == 0:
            history.append({
                'iteration': iteration,
                'best_fitness': best_individual.fitness,
                'population_diversity': len(set(ind.architecture.get_hash()
                                              if hasattr(ind.architecture, 'get_hash')
                                              else str(ind.encoding) for ind in population))
            })

    return {
        'best_architecture': best_individual.architecture,
        'best_fitness': best_individual.fitness,
        'best_metrics': best_individual.metrics,
        'history': history
    }