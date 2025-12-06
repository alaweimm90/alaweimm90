"""
HELIOS Domains: Research areas and problem types

HELIOS Domains enable autonomous research discovery in specific problem areas:

1. **Quantum** - Quantum computing and quantum algorithms
2. **Materials** - Materials science and crystal structure discovery
3. **Optimization** - Combinatorial optimization problems
4. **ML** - Machine learning and neural networks
5. **NAS** - Neural architecture search (specialized ML)
6. **Synthesis** - Drug discovery and molecular synthesis
7. **Graph** - Graph optimization and network problems

Each domain has:
- Problem definitions and benchmarks
- Solution methods (baselines + novel)
- HELIOS validation framework
- Meta-learning integration
- Publication-ready infrastructure
"""

# Initialize domain classes
QuantumDomain = None
MaterialsDomain = None
OptimizationDomain = None
MLDomain = None
NASDomain = None
SynthesisDomain = None
GraphDomain = None

try:
    from helios.domains.quantum import QuantumDomain
    from helios.domains.materials import MaterialsDomain
    from helios.domains.optimization import OptimizationDomain
    from helios.domains.ml import MLDomain
    from helios.domains.nas import NASDomain
    from helios.domains.synthesis import SynthesisDomain
    from helios.domains.graph import GraphDomain
except ImportError:
    pass

DOMAINS = {
    'quantum': QuantumDomain,
    'materials': MaterialsDomain,
    'optimization': OptimizationDomain,
    'ml': MLDomain,
    'nas': NASDomain,
    'synthesis': SynthesisDomain,
    'graph': GraphDomain,
}

__all__ = [
    'QuantumDomain',
    'MaterialsDomain',
    'OptimizationDomain',
    'MLDomain',
    'NASDomain',
    'SynthesisDomain',
    'GraphDomain',
    'DOMAINS',
]
