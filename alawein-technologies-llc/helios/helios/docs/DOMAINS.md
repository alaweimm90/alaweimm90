# Research Domains Guide

**Version**: 0.1.0 MVP
**Last Updated**: 2025-11-19

---

## 🌐 Overview

HELIOS supports 7 research domains spanning quantum computing, materials science, optimization, machine learning, neural architecture search, drug discovery, and graph algorithms.

Each domain:
- Extends core HELIOS system
- Provides domain-specific validation rules
- Includes benchmark problems
- Offers baseline and novel solution methods
- Integrates with meta-learning system

---

## 1️⃣ Quantum Domain

**Module**: `helios.domains.quantum`

**Purpose**: Quantum computing and quantum algorithm research

### Key Areas
- Quantum error correction
- Quantum algorithm design
- Quantum circuit optimization
- Quantum machine learning
- Quantum chemistry simulations

### Example Research Topics
```python
topics = [
    "Novel quantum error correction codes",
    "Quantum optimization algorithms",
    "Quantum machine learning applications",
    "Quantum state preparation techniques",
    "Variational quantum algorithms",
]
```

### Tools & Dependencies
```
qiskit >= 0.37          # IBM's quantum computing framework
cirq >= 1.0            # Google's quantum circuit framework
numpy >= 1.20          # Numerical operations
```

### Validation Rules
```python
domain = DOMAINS['quantum']()
rules = domain.get_validation_rules()
# Rules include:
# - Quantum gate complexity bounds
# - Error rate acceptance criteria
# - Coherence time requirements
# - Entanglement validity checks
```

### Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **QAOA** | Algorithm | Quantum Approximate Optimization |
| **VQE** | Algorithm | Variational Quantum Eigensolver |
| **Shor** | Classical | Integer factorization comparison |
| **Grover** | Algorithm | Quantum search |
| **QGAN** | ML | Quantum Generative Models |

### Example Hypotheses

1. "Topological error correction provides exponential advantage"
2. "Surface codes achieve fault-tolerance with 10^-3 error rates"
3. "Variational circuits outperform classical on specific optimization"

### Usage

```python
from helios.domains.quantum import QuantumDomain
from helios import WorkflowOrchestrator

# Create domain
quantum = QuantumDomain()

# Get benchmarks
benchmarks = quantum.get_benchmarks()

# Run quantum research workflow
orchestrator = WorkflowOrchestrator(domain=quantum)
results = orchestrator.execute(
    research_topic="Quantum error correction codes",
    num_hypotheses=5,
)
```

---

## 2️⃣ Materials Domain

**Module**: `helios.domains.materials`

**Purpose**: Materials science and crystal structure discovery

### Key Areas
- Crystal structure prediction
- Material property design
- Superconductor discovery
- Battery materials
- Photovoltaic materials

### Example Research Topics
```python
topics = [
    "Novel superconducting materials at room temperature",
    "Stable perovskite crystal structures",
    "High-capacity battery cathode materials",
    "Efficient photovoltaic materials",
    "Biocompatible metal alloys",
]
```

### Tools & Dependencies
```
pymatgen >= 2023.0      # Materials science toolkit
ase >= 3.22             # Atomic simulation environment
numpy >= 1.20
scipy >= 1.7
```

### Validation Rules
```python
domain = DOMAINS['materials']()
rules = domain.get_validation_rules()
# Rules include:
# - Crystal stability checks
# - Formation energy thresholds
# - Lattice parameter feasibility
# - Thermodynamic consistency
```

### Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **ICSD** | Database | Inorganic Crystal Structure Database |
| **MP** | Database | Materials Project data |
| **Formation Energy** | Metric | Stability measurement |
| **Band Gap** | Property | Electronic structure |
| **Elastic Constants** | Property | Mechanical properties |

### Example Hypotheses

1. "Organic-inorganic perovskites enable 25%+ photovoltaic efficiency"
2. "Lithium-sulfur batteries with solid electrolyte exceed 500 Wh/kg"
3. "Room-temperature superconductivity achievable with copper-oxide doping"

### Usage

```python
from helios.domains.materials import MaterialsDomain
from helios import WorkflowOrchestrator

# Create domain
materials = MaterialsDomain()

# Get example hypotheses
examples = materials.get_example_hypotheses()

# Run materials research workflow
orchestrator = WorkflowOrchestrator(domain=materials)
results = orchestrator.execute(
    research_topic="Novel battery materials",
    num_hypotheses=8,
)
```

---

## 3️⃣ Optimization Domain

**Module**: `helios.domains.optimization`

**Purpose**: Combinatorial optimization problems and solution methods

### Key Areas
- Quadratic Assignment Problem (QAP)
- Traveling Salesman Problem (TSP)
- Vehicle Routing Problems (VRP)
- Facility Location
- Graph Coloring
- Constraint Satisfaction

### Special Tool: Librex.QAP

Librex.QAP is integrated as the optimization tool:
- **7 novel solution methods**
- **9 baseline algorithms**
- **136+ benchmark instances**
- **Symmetry detection**
- **Hybrid approaches**

### Example Research Topics
```python
topics = [
    "Novel QAP solution methods using reinforcement learning",
    "Hybrid genetic algorithms for TSP",
    "Neural combinatorial optimization",
    "Quantum-inspired optimization algorithms",
    "Meta-learning for algorithm selection",
]
```

### Tools & Dependencies
```
pulp >= 2.7             # Linear programming
Librex.QAP >= 2.0        # Optimization methods & benchmarks
numpy >= 1.20
scipy >= 1.7
```

### Librex.QAP Methods

**7 Novel Methods**:
1. Genetic algorithm with adaptive crossover
2. Simulated annealing with temperature schedule
3. Tabu search with dynamic memory
4. Ant colony optimization with pheromone
5. Particle swarm optimization
6. Neural network-based solver
7. Hybrid multi-method approach

**9 Baselines**:
- Random search
- Greedy nearest neighbor
- 2-opt local search
- 3-opt local search
- Christofides algorithm
- Held-Karp bound
- LKH heuristic
- Concorde solver
- IBM CPLEX

### Validation Rules

```python
domain = DOMAINS['optimization']()
rules = domain.get_validation_rules()
# Rules include:
# - Solution feasibility
# - Optimality gap bounds
# - Computation time limits
# - Scalability verification
```

### Benchmarks

| Benchmark | Size | Instances |
|-----------|------|-----------|
| **tai100a** | 100 | 5 |
| **tai256c** | 256 | 3 |
| **tai500b** | 500 | 2 |
| **nug30** | 30 | 3 |
| **sko100f** | 100 | 3 |

### Example Hypotheses

1. "Neural networks solve 256-QAP within 5% of optimal in <1ms"
2. "Hybrid genetic-tabu algorithm achieves new 500-TSP best known"
3. "Quantum annealing approximates optimal for large QAP instances"

### Usage

```python
from helios.domains.optimization import OptimizationDomain
from helios.domains.optimization.Librex.QAP import Librex.QAP
from helios import WorkflowOrchestrator

# Create domain with Librex.QAP
optimization = OptimizationDomain()

# Access Librex.QAP
Librex.QAP = Librex.QAP()
methods = Librex.QAP.get_methods()
baselines = Librex.QAP.get_baselines()

# Run optimization research workflow
orchestrator = WorkflowOrchestrator(domain=optimization)
results = orchestrator.execute(
    research_topic="Novel QAP solution methods",
    num_hypotheses=10,
)
```

---

## 4️⃣ ML Domain

**Module**: `helios.domains.ml`

**Purpose**: Machine learning and neural network research

### Key Areas
- Neural network architectures
- Training algorithms
- Optimization methods
- Generalization theory
- Few-shot learning
- Adversarial robustness

### Example Research Topics
```python
topics = [
    "Novel activation functions for deep networks",
    "Efficient training algorithms for transformers",
    "Theoretical bounds on generalization",
    "Adversarially robust neural networks",
    "Meta-learning for few-shot tasks",
]
```

### Tools & Dependencies
```
torch >= 1.9            # PyTorch deep learning
scikit-learn >= 1.0     # Classical machine learning
numpy >= 1.20
scipy >= 1.7
```

### Validation Rules
```python
domain = DOMAINS['ml']()
rules = domain.get_validation_rules()
# Rules include:
# - Convergence criteria
# - Overfitting detection
# - Data leakage checks
# - Statistical significance tests
```

### Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **CIFAR-10** | Classification | Image classification |
| **ImageNet** | Classification | Large-scale images |
| **MNIST** | Classification | Digit recognition |
| **Accuracy** | Metric | Classification performance |
| **F1-Score** | Metric | Balanced performance |

### Example Hypotheses

1. "ReLU variants enable 2x faster convergence than original"
2. "Residual networks solve optimization problems at 100x depth"
3. "Knowledge distillation achieves 95% of teacher performance at 10x speedup"

### Usage

```python
from helios.domains.ml import MLDomain
from helios import WorkflowOrchestrator

# Create domain
ml = MLDomain()

# Get benchmarks
benchmarks = ml.get_benchmarks()

# Run ML research workflow
orchestrator = WorkflowOrchestrator(domain=ml)
results = orchestrator.execute(
    research_topic="Novel neural network architectures",
    num_hypotheses=7,
)
```

---

## 5️⃣ NAS Domain

**Module**: `helios.domains.nas`

**Purpose**: Neural Architecture Search and AutoML

### Key Areas
- Automated architecture design
- Hyperparameter optimization
- Architecture performance prediction
- Efficient search methods
- Architecture generalization

### Example Research Topics
```python
topics = [
    "Predicting NAS architecture performance without training",
    "Efficient evolutionary algorithms for architecture search",
    "Transfer learning across NAS spaces",
    "Zero-cost proxies for architecture evaluation",
    "Bayesian optimization for hyperparameter tuning",
]
```

### Tools & Dependencies
```
nas-bench-101 >= 2.1    # NAS benchmark dataset
torch >= 1.9            # Neural network implementation
numpy >= 1.20
scipy >= 1.7
```

### Validation Rules
```python
domain = DOMAINS['nas']()
rules = domain.get_validation_rules()
# Rules include:
# - Search space validity
# - Architecture feasibility
# - Performance prediction accuracy
# - Search efficiency metrics
```

### Benchmarks

| Benchmark | Size | Purpose |
|-----------|------|---------|
| **NAS-Bench-101** | 423K | Standard benchmark |
| **Search Cost** | Metric | Query efficiency |
| **Rank Correlation** | Metric | Predictor accuracy |
| **Pareto Front** | Multi-objective | Performance trade-offs |

### Example Hypotheses

1. "Zero-cost proxies achieve 0.85+ correlation with final accuracy"
2. "Evolutionary search finds optimal architecture in <500 evaluations"
3. "Architecture encodings enable transfer across datasets"

### Usage

```python
from helios.domains.nas import NASDomain
from helios import WorkflowOrchestrator

# Create domain
nas = NASDomain()

# Get NAS benchmarks
benchmarks = nas.get_benchmarks()

# Run NAS research workflow
orchestrator = WorkflowOrchestrator(domain=nas)
results = orchestrator.execute(
    research_topic="Neural architecture search methods",
    num_hypotheses=6,
)
```

---

## 6️⃣ Synthesis Domain

**Module**: `helios.domains.synthesis`

**Purpose**: Drug discovery and molecular synthesis

### Key Areas
- Molecular structure design
- Drug-target interaction prediction
- Retrosynthesis planning
- Molecular property prediction
- Lead optimization
- ADME properties

### Example Research Topics
```python
topics = [
    "Deep learning models for binding affinity prediction",
    "Generative models for novel drug scaffold design",
    "AI-guided retrosynthesis planning",
    "Molecular structure prediction from biochemical data",
    "Toxicity prediction for drug candidates",
]
```

### Tools & Dependencies
```
rdkit >= 2022.09        # Cheminformatics toolkit
numpy >= 1.20
scipy >= 1.7
```

### Validation Rules
```python
domain = DOMAINS['synthesis']()
rules = domain.get_validation_rules()
# Rules include:
# - Molecular validity checks
# - Drug-likeness criteria (Lipinski's rule)
# - Synthetic accessibility
# - Toxicity constraints
```

### Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **SMILES** | Format | Molecular representation |
| **Binding Affinity** | Property | Drug potency |
| **Solubility** | Property | Bioavailability |
| **LogP** | Property | Lipophilicity |
| **TPSA** | Property | Cell permeability |

### Example Hypotheses

1. "Graph neural networks predict binding affinity with <1 kcal/mol RMSE"
2. "Transformer models generate novel drug scaffolds with 95%+ validity"
3. "Retrosynthesis planning reduces synthesis steps by 40% vs human"

### Usage

```python
from helios.domains.synthesis import SynthesisDomain
from helios import WorkflowOrchestrator

# Create domain
synthesis = SynthesisDomain()

# Get molecular properties
props = synthesis.get_validation_rules()

# Run synthesis research workflow
orchestrator = WorkflowOrchestrator(domain=synthesis)
results = orchestrator.execute(
    research_topic="Drug discovery using AI",
    num_hypotheses=8,
)
```

---

## 7️⃣ Graph Domain

**Module**: `helios.domains.graph`

**Purpose**: Graph optimization and network problems

### Key Areas
- Graph partitioning
- Community detection
- Maximum clique problem
- Graph coloring
- Minimum spanning tree
- Network analysis

### Example Research Topics
```python
topics = [
    "Neural graph partitioning methods",
    "Efficient community detection algorithms",
    "Graph neural networks for combinatorial problems",
    "Spectral clustering improvements",
    "Heterogeneous network analysis",
]
```

### Tools & Dependencies
```
networkx >= 2.6         # Graph algorithms and structures
numpy >= 1.20
scipy >= 1.7
```

### Validation Rules
```python
domain = DOMAINS['graph']()
rules = domain.get_validation_rules()
# Rules include:
# - Graph validity checks
# - Partition balance criteria
# - Cut size verification
# - Optimization bounds
```

### Benchmarks

| Benchmark | Type | Purpose |
|-----------|------|---------|
| **GSet** | Test Set | Graph partitioning instances |
| **DIMACS** | Test Set | Maximum clique instances |
| **Cut Size** | Metric | Partition quality |
| **Modularity** | Metric | Community quality |

### Example Hypotheses

1. "Graph neural networks solve max-clique within 95% of optimal"
2. "Spectral clustering with refinement achieves 10% better modularity"
3. "Message-passing networks outperform traditional partitioning"

### Usage

```python
from helios.domains.graph import GraphDomain
from helios import WorkflowOrchestrator

# Create domain
graph = GraphDomain()

# Get graph benchmarks
benchmarks = graph.get_benchmarks()

# Run graph research workflow
orchestrator = WorkflowOrchestrator(domain=graph)
results = orchestrator.execute(
    research_topic="Graph neural networks for optimization",
    num_hypotheses=7,
)
```

---

## 🔄 Cross-Domain Examples

### Example 1: Multi-Domain Research

```python
from helios.domains import DOMAINS
from helios import HypothesisGenerator

# Compare across domains
domains = ['quantum', 'ml', 'materials']

for domain_name in domains:
    domain = DOMAINS[domain_name]()

    generator = HypothesisGenerator()
    hypotheses = generator.generate(
        topic="Novel solution methods",
        domain=domain_name
    )

    print(f"{domain_name}: {len(hypotheses)} hypotheses")
```

### Example 2: Domain-Specific Validation

```python
from helios.domains import DOMAINS
from helios import TuringValidator

# Different domains have different validation rules
domain_validators = {}

for domain_name in ['quantum', 'ml', 'optimization']:
    domain = DOMAINS[domain_name]()
    domain_validators[domain_name] = TuringValidator(
        domain_context=domain.get_validation_rules()
    )

# Validate same hypothesis in multiple domains
hypothesis = "Novel algorithm X"

for domain_name, validator in domain_validators.items():
    result = validator.validate(hypothesis)
    print(f"{domain_name}: {result['overall_score']}")
```

---

## 📊 Domain Statistics

| Domain | Focus | Tools | Benchmarks |
|--------|-------|-------|-----------|
| **Quantum** | Computing | Qiskit, Cirq | QAOA, VQE, Shor |
| **Materials** | Science | PyMatGen, ASE | ICSD, MP, Properties |
| **Optimization** | Combinatorics | Librex.QAP, PuLP | 136+ benchmarks |
| **ML** | Learning | PyTorch, sklearn | CIFAR, ImageNet |
| **NAS** | AutoML | NAS-Bench-101 | 423K architectures |
| **Synthesis** | Chemistry | RDKit | SMILES, Properties |
| **Graph** | Networks | NetworkX | GSet, DIMACS |

---

## 🚀 Choosing a Domain

### For Quantum Research
→ Use `QuantumDomain` if working on quantum computing, algorithms, or quantum machine learning

### For Materials Science
→ Use `MaterialsDomain` for crystal discovery, property design, superconductors

### For Optimization Problems
→ Use `OptimizationDomain` especially with Librex.QAP for combinatorial problems

### For Machine Learning
→ Use `MLDomain` for neural networks, training algorithms, generalization

### For Architecture Search
→ Use `NASDomain` for automated neural architecture design

### For Drug Discovery
→ Use `SynthesisDomain` for molecular design, binding prediction, retrosynthesis

### For Network Problems
→ Use `GraphDomain` for graph partitioning, community detection, optimization

---

## 📖 See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [API.md](API.md) - API reference
- [../STRUCTURE.md](../STRUCTURE.md) - Directory organization
- Domain READMEs: `helios/domains/[domain]/README.md`

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
