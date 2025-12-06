# HELIOS Architecture & System Design

**Version**: 0.1.0 MVP
**Last Updated**: 2025-11-19

## 🏗️ System Overview

HELIOS is a **modular autonomous research platform** with clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Research Topics                          │
│         (Quantum, Materials, ML, NAS, Synthesis, Graph)     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│         Orchestration Layer (ORCHEX Workflow)                │
│    Coordinates discovery → validation → learning → publish  │
└────────────────────────┬────────────────────────────────────┘
                         │
     ┌──────────────┬────┴────┬──────────────┬──────────────┐
     │              │         │              │              │
┌────▼────┐  ┌────▼────┐ ┌──▼────┐  ┌─────▼────┐  ┌────▼───┐
│Discovery│  │Validation│ │Learn  │  │ Agents   │  │Publish │
│ Module  │  │ Module   │ │ Module│  │ Module   │  │ Module │
└────┬────┘  └────┬────┘ └──┬────┘  └─────┬────┘  └────┬───┘
     │            │         │             │            │
     └────────────┴─────────┴─────────────┴────────────┘
            Data Flow & Feedback Loops
```

---

## 5️⃣ Core Modules

### 1. Discovery Module
**File**: `helios/core/discovery/`

**Purpose**: Generate novel research hypotheses

**Key Classes**:
- `HypothesisGenerator` - Main hypothesis generation
- `BrainstormEngine` - Creative hypothesis generation

**Algorithm**:
```
Input: Research Topic + Domain (optional)
  ↓
1. Search Academic Literature (ArXiv, Papers, etc.)
  ↓
2. Extract Key Concepts & Relationships
  ↓
3. LLM-Based Hypothesis Generation
  ↓
4. Novelty Scoring
  ↓
Output: List of Hypotheses with Novelty Scores
```

**Dependencies**:
- OpenAI API / Anthropic API / Google API (for LLMs)
- ArXiv API (for literature)
- requests library

**Example Flow**:
```python
generator = HypothesisGenerator(
    llm_provider="openai",
    num_hypotheses=5
)
hypotheses = generator.generate(
    topic="Quantum error correction",
    domain="quantum"
)
# Returns: List[Hypothesis] with fields:
# - id, text, score, domain, source_papers, novelty
```

---

### 2. Validation Module
**File**: `helios/core/validation/turing/`

**Purpose**: Rigorously validate hypotheses using 5 Turing-inspired strategies

**Key Classes**:
- `TuringValidator` - Main validator
- `LogicalContradictionStrategy` - Strategy 1
- `EmpiricalCounterExampleStrategy` - Strategy 2
- `AnalogicalFalsificationStrategy` - Strategy 3
- `BoundaryViolationStrategy` - Strategy 4
- `MechanismImplausibilityStrategy` - Strategy 5
- `Interrogator` - 200-question framework

**5 Falsification Strategies**:

#### Strategy 1: Logical Contradiction
```
1. Extract all propositions from hypothesis
2. Build logical graph
3. Find contradictions
4. Score: consistency_level (0-100)
```

#### Strategy 2: Empirical Counter-Example
```
1. Generate test cases from hypothesis
2. Validate against benchmark datasets
3. Find counter-examples
4. Score: empirical_validity (0-100)
```

#### Strategy 3: Analogical Falsification
```
1. Find analogies in other domains
2. Test if analogy holds
3. Identify disanalogies
4. Score: cross_domain_validity (0-100)
```

#### Strategy 4: Boundary Violation
```
1. Identify domain boundaries
2. Generate edge cases
3. Test boundary behavior
4. Score: robustness (0-100)
```

#### Strategy 5: Mechanism Implausibility
```
1. Extract underlying mechanism from hypothesis
2. Evaluate mechanism plausibility
3. Check for biological/physical plausibility
4. Score: mechanism_plausibility (0-100)
```

**200-Question Interrogation Framework**:
```
1. Load 200 critical questions
2. Apply to hypothesis
3. Identify weaknesses
4. Generate follow-up questions
5. Score: overall_validity
```

**Final Score Calculation**:
```
Overall Score = (
    0.2 * logical_score +
    0.3 * empirical_score +
    0.15 * analogical_score +
    0.15 * boundary_score +
    0.2 * mechanism_score
) * interrogation_modifier
```

**Example Flow**:
```python
validator = TuringValidator()
results = validator.validate(hypotheses)
# Returns: List[ValidationResult] with:
# - hypothesis_id
# - logical_score, empirical_score, analogical_score,
#   boundary_score, mechanism_score
# - overall_score (0-100)
# - weaknesses (list of identified issues)
```

---

### 3. Learning Module
**File**: `helios/core/learning/`

**Purpose**: Meta-learning system that learns from validation results

**Key Components**:

#### Hall of Failures
```
Database Schema:
┌─────────────────────────┐
│ Failed Hypothesis Entry │
├─────────────────────────┤
│ id                      │
│ hypothesis_text         │
│ domain                  │
│ reason_for_failure      │
│ validation_scores       │
│ lessons_learned         │
│ date_failed             │
│ similar_hypotheses      │
└─────────────────────────┘
```

**Purpose**:
- Learn from past failures
- Find similar failing hypotheses
- Extract lessons
- Avoid repeating mistakes

#### Meta-Learner
```
Input: Validation Results + Agent Performance Data
  ↓
1. Extract Features from Results
  ↓
2. Update Agent Performance Scores
  ↓
3. Learn which agents work best for each domain
  ↓
4. Update Bandit Arm Probabilities
  ↓
Output: Agent Recommendations for Future Research
```

#### 7 Personality Agents
```
Agent Architecture:
┌──────────────────────────┐
│ Personality Agent        │
├──────────────────────────┤
│ name                     │
│ description              │
│ personality_traits       │
│ validation_preferences   │
│ success_rate (by domain) │
│ learning_history         │
└──────────────────────────┘
```

**Agents**:
1. **Conservative** - Acceptance threshold: 80+
2. **Creative** - Encourages novelty
3. **Rigorous** - Strict validation
4. **Pragmatic** - Focus on applicability
5. **Skeptic** - Challenges all claims
6. **Specialist** - Domain expertise
7. **Generalist** - Cross-domain bridges

#### Bandit Algorithm
```
Multi-Armed Bandit Setup:
- Arms = 7 Agents
- Reward = Validation Score
- Algorithm: UCB1 or Thompson Sampling

Update Rule:
For agent i that validated hypothesis h:
  reward_i = validation_score
  count_i += 1
  ucb_i = mean_reward_i + c * sqrt(ln(total_pulls) / count_i)

Select agent with highest UCB for next hypothesis
```

**Example Flow**:
```python
learner = MetaLearner()

# Learn from validation results
learner.learn_from_validation(hypotheses, results)

# Get recommendation for domain
best_agent = learner.recommend_agent(domain="quantum")
# Returns: agent_name, expected_score

# Query Hall of Failures
similar_failures = learner.query_failures(
    hypothesis=current_hypothesis,
    domain="quantum",
    similarity_threshold=0.7
)
```

---

### 4. Agents Module
**File**: `helios/core/agents/`

**Purpose**: 7 personality-based research agents that collaborate

**Agent Orchestrator**:
```python
class ResearchAgentOrchestrator:
    agents = {
        'conservative': ConservativeAgent(),
        'creative': CreativeAgent(),
        'rigorous': RigorousAgent(),
        'pragmatic': PragmaticAgent(),
        'skeptic': SkepticAgent(),
        'specialist': SpecialistAgent(),
        'generalist': GeneralistAgent(),
    }

    def validate_with_all(self, hypothesis):
        """Get validation from all 7 agents"""
        results = {}
        for name, agent in self.agents.items():
            results[name] = agent.validate(hypothesis)
        return results

    def select_best_agent(self, topic, learning_history):
        """Select best agent based on meta-learning"""
        scores = {}
        for agent_name in self.agents:
            scores[agent_name] = learning_history.get_agent_score(
                agent_name, topic
            )
        return max(scores, key=scores.get)
```

**Agent Communication**:
```
Agent A                Agent B
   ↓                      ↓
   └──────────┬───────────┘
              │
         Message Queue
              │
   ┌──────────┴───────────┐
   ↓                      ↓
Agent C                Agent D
```

---

### 5. Orchestration Module
**File**: `helios/core/orchestration/`

**Purpose**: ORCHEX workflow engine coordinating full research process

**ORCHEX Workflow**:
```
1. Intent Classification
   Input: Research Topic
   ↓
2. Hypothesis Generation
   Using Discovery Module
   ↓
3. Experiment Design
   Generate experiments to test hypotheses
   ↓
4. Validation
   Using Turing Validation Suite
   ↓
5. Meta-Learning
   Update agent performance, Hall of Failures
   ↓
6. Paper Generation
   Create publication-ready paper
   ↓
Output: Research Results + Paper
```

**Key Classes**:

- `WorkflowOrchestrator` - Main coordinator
- `ExperimentDesigner` - Design experiments
- `CodeGenerator` - Generate experiment code
- `SandboxExecutor` - Execute safely
- `PaperGenerator` - Generate papers

**Orchestration Logic**:
```python
def execute(self, topic, domain, config):
    # Step 1: Classify intent
    intent = self.classify_intent(topic)

    # Step 2: Generate hypotheses
    hypotheses = self.discovery_module.generate(
        topic=topic,
        domain=domain
    )

    # Step 3: Design experiments
    experiments = self.experiment_designer.design(
        hypotheses=hypotheses,
        domain=domain
    )

    # Step 4: Validate hypotheses
    results = self.validation_module.validate(
        hypotheses=hypotheses,
        experiments=experiments
    )

    # Step 5: Learn
    self.learning_module.learn(
        hypotheses=hypotheses,
        results=results
    )

    # Step 6: Generate paper
    paper = self.paper_generator.generate(
        topic=topic,
        hypotheses=hypotheses,
        results=results,
        experiments=experiments
    )

    return {
        'hypotheses': hypotheses,
        'results': results,
        'experiments': experiments,
        'paper': paper,
    }
```

---

## 7️⃣ Research Domains

**File**: `helios/domains/`

Each domain extends the core system:

```
BaseDomain
    ↓
    ├── QuantumDomain
    ├── MaterialsDomain
    ├── OptimizationDomain
    ├── MLDomain
    ├── NASDomain
    ├── SynthesisDomain
    └── GraphDomain
```

**Domain Interface**:
```python
class BaseDomain:
    name: str                          # e.g., "quantum"
    display_name: str                  # e.g., "Quantum Computing"
    description: str                   # Domain description

    def generate_benchmark_problems(self) -> List[Problem]:
        """Generate standard benchmark problems"""
        pass

    def validate_solution(self, solution) -> bool:
        """Validate a proposed solution"""
        pass

    def get_validation_rules(self) -> List[ValidationRule]:
        """Domain-specific validation rules"""
        pass

    def get_example_hypotheses(self) -> List[Hypothesis]:
        """Example hypotheses in this domain"""
        pass
```

---

## 🔄 Data Flow & Interactions

### Full Research Pipeline

```
User Input
    ↓
    │ "Quantum error correction"
    │
Research Intent Classification
    ↓
Hypothesis Generation (Discovery)
    ├─ Search Literature
    ├─ LLM-Based Generation
    └─ Novelty Scoring
    ↓
Experiment Design (Orchestration)
    ├─ Design Experiments
    ├─ Generate Code
    └─ Set Up Benchmarks
    ↓
Hypothesis Validation (Turing Suite)
    ├─ Logical Contradiction Check
    ├─ Empirical Counter-Examples
    ├─ Analogical Falsification
    ├─ Boundary Violation Testing
    ├─ Mechanism Implausibility Check
    └─ 200-Question Interrogation
    ↓
Validation Results
    ├─ Scores (0-100 per strategy)
    ├─ Weaknesses Identified
    └─ Overall Validity Score
    ↓
Meta-Learning (Learning Module)
    ├─ Update Agent Performance
    ├─ Store Failures in Hall of Failures
    ├─ Extract Lessons
    └─ Recommend Best Agents
    ↓
Paper Generation (Orchestration)
    ├─ Format Results
    ├─ Write Paper
    ├─ Add Proofs
    └─ Include Benchmarks
    ↓
Publication-Ready Research Paper
```

### Feedback Loops

```
Validation Results
        ↓
    ┌───┴───┐
    ↓       ↓
Strong    Weak
Hypotheses Hypotheses
    │        │
    │        └─→ Hall of Failures
    │            ↓
    │        Lessons Learned
    │            ↓
    └────────→ Influence Future Generation
               (Higher Quality Hypotheses)
```

---

## 🎯 Key Design Decisions

### 1. Modular Architecture
**Decision**: Separate core modules (discovery, validation, learning, agents, orchestration)
**Rationale**: Easy to extend, test, and replace components
**Benefit**: Can improve one module without affecting others

### 2. Turing-Inspired Validation
**Decision**: 5 falsification strategies instead of single metric
**Rationale**: Multiple perspectives increase confidence
**Benefit**: Catches different types of flaws

### 3. Personality-Based Agents
**Decision**: 7 agents with distinct personalities
**Rationale**: Different approaches work for different problems
**Benefit**: Adaptive system that learns optimal approach per domain

### 4. Hall of Failures
**Decision**: Store and learn from failed hypotheses
**Rationale**: Past failures inform better hypotheses
**Benefit**: Systematic improvement over time

### 5. Meta-Learning via Bandits
**Decision**: Use multi-armed bandit for agent selection
**Rationale**: Efficiently explore-exploit agent space
**Benefit**: Optimal agent selection with minimal trials

### 6. Workflow Orchestration
**Decision**: ORCHEX engine coordinates all modules
**Rationale**: Clear workflow reduces complexity
**Benefit**: Reproducible, automatable research process

### 7. Domain Extensibility
**Decision**: Plugin architecture for new domains
**Rationale**: Support any research area
**Benefit**: Grows with research landscape

---

## 🏢 Class Hierarchy

```
Core System

┌─ HypothesisGenerator
│  └─ BrainstormEngine
│
├─ TuringValidator
│  ├─ LogicalContradictionStrategy
│  ├─ EmpiricalCounterExampleStrategy
│  ├─ AnalogicalFalsificationStrategy
│  ├─ BoundaryViolationStrategy
│  ├─ MechanismImplausibilityStrategy
│  └─ Interrogator
│
├─ MetaLearner
│  ├─ HallOfFailures
│  ├─ AgentPersonality (x7)
│  ├─ BanditAlgorithm
│  └─ LessonExtractor
│
├─ ResearchAgentOrchestrator
│  └─ (7 Agent Instances)
│
└─ WorkflowOrchestrator
   ├─ ExperimentDesigner
   ├─ CodeGenerator
   ├─ SandboxExecutor
   └─ PaperGenerator

Research Domains

└─ BaseDomain
   ├─ QuantumDomain
   ├─ MaterialsDomain
   ├─ OptimizationDomain (+ Librex.QAP Tool)
   ├─ MLDomain
   ├─ NASDomain
   ├─ SynthesisDomain
   └─ GraphDomain
```

---

## 📊 Data Models

### Hypothesis Model
```python
@dataclass
class Hypothesis:
    id: str
    text: str
    domain: str
    source_papers: List[str]
    novelty_score: float  # 0-100
    creation_timestamp: datetime
    metadata: Dict[str, Any]
```

### ValidationResult Model
```python
@dataclass
class ValidationResult:
    hypothesis_id: str
    logical_score: float        # 0-100
    empirical_score: float      # 0-100
    analogical_score: float     # 0-100
    boundary_score: float       # 0-100
    mechanism_score: float      # 0-100
    overall_score: float        # 0-100 (weighted)
    weaknesses: List[str]
    interrogation_results: Dict[str, Any]
    validation_timestamp: datetime
```

### AgentPerformance Model
```python
@dataclass
class AgentPerformance:
    agent_name: str
    domain: str
    validation_count: int
    avg_score: float
    success_rate: float
    last_updated: datetime
    performance_history: List[float]
```

---

## 🔐 Security Considerations

### Code Execution
- Uses `SandboxExecutor` for safe execution
- No direct `eval()` or `exec()` on untrusted input
- Container/process isolation for experiments

### Data Privacy
- No personal data stored
- Research papers are output only
- No user tracking or analytics

### Model Safety
- LLM calls filtered for safety
- No harmful content generation
- Explicit content filters on outputs

---

## ⚡ Performance Considerations

### Hypothesis Generation
- Async LLM calls for parallelization
- Caching of literature searches
- Batch processing of generations

### Validation
- Parallel strategy evaluation
- Caching of validation results
- Incremental learning

### Meta-Learning
- Efficient similarity matching (Hall of Failures)
- Bandit updates in O(1) time
- Periodic learning snapshots

---

## 📈 Extensibility Points

### Adding New Validation Strategy
1. Create new strategy class inheriting from `FalsificationStrategy`
2. Implement `falsify(hypothesis) -> score`
3. Register in `TuringValidator`
4. Update weight in score calculation

### Adding New Agent Personality
1. Define new agent class with personality traits
2. Implement `validate(hypothesis) -> result`
3. Register in `ResearchAgentOrchestrator`
4. Create learning history tracking

### Adding New Domain
1. Create domain folder in `helios/domains/`
2. Inherit from `BaseDomain`
3. Implement required methods
4. Register in domain registry

---

## 🔗 Dependency Graph

```
discovery/
  ├─ requests (API calls)
  └─ openai/anthropic/google (LLMs)

validation/
  ├─ numpy (numerical ops)
  └─ typing (type hints)

learning/
  ├─ numpy (bandit math)
  └─ scipy (similarity metrics)

orchestration/
  ├─ discovery
  ├─ validation
  ├─ learning
  └─ agents

domains/
  ├─ qiskit (quantum)
  ├─ pymatgen (materials)
  ├─ networkx (graph)
  ├─ torch (ML, NAS)
  ├─ rdkit (synthesis)
  └─ pulp (optimization)
```

---

## 📖 See Also

- [GETTING_STARTED.md](GETTING_STARTED.md) - Setup and installation
- [API.md](API.md) - Complete API reference
- [DOMAINS.md](DOMAINS.md) - Domain-specific details
- [../STRUCTURE.md](../STRUCTURE.md) - Directory organization

---

**Last Updated**: 2025-11-19
**Version**: 0.1.0 MVP
