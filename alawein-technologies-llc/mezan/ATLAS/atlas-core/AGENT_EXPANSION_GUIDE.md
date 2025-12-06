# ORCHEX Agent Expansion Guide

**Current**: 8 agents implemented (Week 1)
**Target**: 40+ agents (Week 5-6)

---

## Current Agent Types (Week 1) ✅

1. **SynthesisAgent** - Synthesizes information
2. **LiteratureReviewAgent** - Reviews literature
3. **HypothesisGenerationAgent** - Generates hypotheses
4. **CriticalAnalysisAgent** - Critical analysis
5. **ValidationAgent** - Validates findings
6. **DataAnalysisAgent** - Data analysis
7. **MethodologyAgent** - Research methodologies
8. **EthicsReviewAgent** - Ethics review

---

## Planned Agent Types (Weeks 2-6)

### Category: Research Planning (5 agents)
9. **ResearchQuestionFormulationAgent** - Formulates research questions
10. **LiteratureGapIdentificationAgent** - Identifies research gaps
11. **PriorWorkSurveyAgent** - Surveys prior work
12. **ResearchScopeDefinitionAgent** - Defines research scope
13. **FeasibilityAnalysisAgent** - Analyzes feasibility

### Category: Data & Experimentation (8 agents)
14. **ExperimentalDesignAgent** - Designs experiments
15. **StatisticalAnalysisAgent** - Statistical testing
16. **DataCollectionAgent** - Data collection strategies
17. **DataPreprocessingAgent** - Data cleaning and preprocessing
18. **VisualizationAgent** - Data visualization
19. **BenchmarkingAgent** - Performance benchmarking
20. **AblationStudyAgent** - Ablation studies
21. **BaselineComparisonAgent** - Baseline comparisons

### Category: Theory & Modeling (6 agents)
22. **TheoreticalFrameworkAgent** - Theoretical frameworks
23. **MathematicalModelingAgent** - Mathematical models
24. **ProofVerificationAgent** - Proof checking
25. **ComplexityAnalysisAgent** - Complexity analysis
26. **FormalVerificationAgent** - Formal verification
27. **TheoremProvingAgent** - Theorem proving

### Category: Writing & Communication (6 agents)
28. **AbstractWritingAgent** - Writes abstracts
29. **IntroductionWritingAgent** - Writes introductions
30. **ResultsReportingAgent** - Reports results
31. **DiscussionWritingAgent** - Writes discussions
32. **RelatedWorkSurveyAgent** - Surveys related work
33. **CitationManagementAgent** - Manages citations

### Category: Quality & Review (5 agents)
34. **PeerReviewSimulationAgent** - Simulates peer review
35. **ReproducibilityCheckAgent** - Checks reproducibility
36. **BiasDetectionAgent** - Detects bias
37. **StatisticalPowerAnalysisAgent** - Power analysis
38. **MethodologicalCritiqueAgent** - Critiques methodology

### Category: Specialized Domains (7 agents)
39. **MachineLearningExpertAgent** - ML expertise
40. **NaturalLanguageProcessingAgent** - NLP expertise
41. **ComputerVisionAgent** - CV expertise
42. **RoboticsAgent** - Robotics expertise
43. **SecurityAnalysisAgent** - Security analysis
44. **DistributedSystemsAgent** - Distributed systems
45. **QuantumComputingAgent** - Quantum computing

### Category: Meta-Research (3 agents)
46. **ResearchStrategyAgent** - Research strategies
47. **ImpactPredictionAgent** - Predicts research impact
48. **InterdisciplinaryConnectionAgent** - Finds connections

---

## Agent Implementation Template

```python
class NewAgentTemplate(ResearchAgent):
    """
    [Agent Name] Agent

    [Brief description of what this agent does]
    """

    def execute(self, task: Dict) -> Dict:
        """
        [Action verb] [object]

        Args:
            task: Must contain [required fields]

        Returns:
            [Description of return value]
        """
        # Extract inputs
        input_data = task.get("input", {})

        logger.info(
            f"{self.__class__.__name__} {self.config.agent_id} "
            f"processing {input_data.get('topic', 'unknown')}..."
        )

        # Simulate processing (replace with actual LLM call)
        result = {
            "output": "Processed result",
            "insights": ["Insight 1", "Insight 2"],
            "quality": 0.85,
            "confidence": 0.80,
        }

        return result
```

---

## Week 2 Expansion (12 agents total)

Add 4 new agents:

### Priority Additions
1. **ExperimentalDesignAgent** - Critical for methodology
2. **StatisticalAnalysisAgent** - Essential for validation
3. **PeerReviewSimulationAgent** - Important for quality
4. **ResearchQuestionFormulationAgent** - Key for planning

### Implementation Steps
```bash
# 1. Add to agents.py
class ExperimentalDesignAgent(ResearchAgent):
    def execute(self, task: Dict) -> Dict:
        # Implementation here
        pass

# 2. Add to factory
def create_agent(agent_type: str, ...):
    agent_classes = {
        # ... existing ...
        "experimental_design": ExperimentalDesignAgent,
        "statistical_analysis": StatisticalAnalysisAgent,
        "peer_review_simulation": PeerReviewSimulationAgent,
        "research_question_formulation": ResearchQuestionFormulationAgent,
    }

# 3. Add tests
def test_experimental_design_agent():
    agent = create_agent("experimental_design", "exp_1")
    result = agent.execute({"design_params": {...}})
    assert "experimental_design" in result
```

---

## Week 3-4 Expansion (20 agents total)

Add 8 more agents from Categories:
- Data & Experimentation (4 agents)
- Theory & Modeling (2 agents)
- Writing & Communication (2 agents)

---

## Week 5-6 Expansion (40+ agents)

Complete all remaining agents across all categories.

---

## Agent Specialization Strategy

### Option 1: Broad Generalists (Current)
- Each agent handles broad category
- Good for Week 1-2
- 8-12 agents total

### Option 2: Narrow Specialists (Target)
- Each agent handles specific subtask
- Best for complex workflows
- 40+ agents total

### Option 3: Hybrid Approach (Recommended)
- Core generalist agents (8)
- Specialized experts (32+)
- Route based on task complexity

---

## Agent Skill Level Distribution

```python
# Recommended distribution for 40 agents

skill_distribution = {
    "expert": 8,      # skill_level > 0.90 (top 20%)
    "advanced": 12,   # skill_level 0.75-0.90 (30%)
    "intermediate": 15, # skill_level 0.60-0.75 (37.5%)
    "junior": 5,      # skill_level < 0.60 (12.5%)
}
```

### Use Cases
- **Expert agents**: Critical tasks, final synthesis
- **Advanced agents**: Complex analysis, methodology
- **Intermediate agents**: Standard tasks, literature review
- **Junior agents**: Data collection, simple validation

---

## Integration with Libria

### Librex.Meta Agent Selection
Librex.Meta learns which agent type works best for each task type over time.

```python
# Librex.Meta tracks Elo ratings per agent
agent_elo = {
    "synthesis_expert_1": 1650,
    "synthesis_intermediate_1": 1420,
    "synthesis_junior_1": 1180,
}

# Selects best agent based on task and Elo
best_agent = meta_libria.select_agent(task_type="synthesis", agents=agents)
```

### Librex.QAP Assignment
Optimal assignment considers:
- Agent specialization match
- Current workload
- Historical performance
- Task complexity

---

## Testing Strategy

### Unit Tests (Per Agent)
```python
def test_agent_name():
    agent = create_agent("agent_type", "test_1")
    task = {"input": "test_data"}
    result = agent.execute(task)

    assert "output" in result
    assert result["quality"] > 0
    assert isinstance(result, dict)
```

### Integration Tests (Multi-Agent)
```python
def test_multi_agent_workflow():
    # Test collaboration between 5+ agents
    agents = [create_agent(t, f"{t}_1") for t in agent_types]

    # Execute workflow
    results = workflow.execute(agents, task)

    # Verify coordination
    assert len(results) == len(agents)
```

---

## Performance Considerations

### Memory Usage
- Each agent: ~1-5 MB
- 40 agents: ~40-200 MB
- Redis state: ~10-50 MB
- **Total**: <500 MB

### Execution Time
- Single agent task: 1-5s
- Workflow (5 agents): 5-25s
- Parallel execution: 5-10s

### Optimization Strategies
1. **Agent pooling**: Reuse agent instances
2. **Lazy loading**: Load agents on demand
3. **Caching**: Cache frequent results
4. **Parallel execution**: Run independent agents in parallel

---

## Documentation Requirements

For each new agent:
1. Docstring with description
2. Example usage
3. Input/output schema
4. Test case
5. Integration with workflows

---

## Example: Adding ExperimentalDesignAgent

```python
# 1. Add to agents.py
class ExperimentalDesignAgent(ResearchAgent):
    """
    Experimental Design Agent

    Designs rigorous experimental protocols including:
    - Control groups
    - Randomization strategies
    - Sample size calculations
    - Statistical power analysis
    """

    def execute(self, task: Dict) -> Dict:
        """
        Design experimental protocol

        Args:
            task: Must contain 'research_question' and 'constraints'

        Returns:
            Experimental design with protocol details
        """
        research_question = task.get("research_question", "")
        constraints = task.get("constraints", {})

        logger.info(
            f"ExperimentalDesignAgent {self.config.agent_id} "
            f"designing experiment for: {research_question}"
        )

        # Design experimental protocol
        design = {
            "design_type": "Randomized Controlled Trial",
            "groups": {
                "control": {"n": 250, "treatment": None},
                "treatment_a": {"n": 250, "treatment": "A"},
                "treatment_b": {"n": 250, "treatment": "B"},
            },
            "randomization": "Block randomization with stratification",
            "sample_size": 750,
            "power": 0.80,
            "alpha": 0.05,
            "duration": "12 weeks",
            "measurements": ["Baseline", "Week 6", "Week 12"],
            "primary_outcome": "Treatment efficacy",
            "secondary_outcomes": ["Safety", "Tolerability"],
            "statistical_plan": "ANOVA with post-hoc Tukey HSD",
            "quality": 0.88,
        }

        return design


# 2. Add to factory
def create_agent(agent_type: str, agent_id: str, **kwargs) -> ResearchAgent:
    agent_classes = {
        # ... existing ...
        "experimental_design": ExperimentalDesignAgent,
    }
    # ... rest of factory ...


# 3. Add test
def test_experimental_design_agent():
    """Test ExperimentalDesignAgent"""
    agent = create_agent(
        agent_type="experimental_design",
        agent_id="exp_design_1",
    )

    task = {
        "task_id": "exp_001",
        "research_question": "Does treatment A improve outcomes?",
        "constraints": {"budget": 50000, "duration_weeks": 12},
    }

    result = agent.execute(task)

    assert "design_type" in result
    assert "groups" in result
    assert "sample_size" in result
    assert result["power"] >= 0.80
    assert result["quality"] > 0


# 4. Add to demo
agent = create_agent("experimental_design", "exp_1")
ORCHEX.register_agent(agent)
```

---

## Roadmap Summary

| Week | Agents | New Types | Total | Focus |
|------|--------|-----------|-------|-------|
| 1 | 8 | Core 8 | 8 | Foundation ✅ |
| 2 | +4 | Research + Experiment | 12 | Integration |
| 3 | +8 | Data + Theory | 20 | Expansion |
| 4 | +10 | Writing + Quality | 30 | Workflows |
| 5 | +10 | Specialized + Meta | 40 | Completion |
| 6 | +5 | Polish + Optimize | 45+ | Production |

---

**Next Action**: Week 2 - Add 4 priority agents and test with Libria integration
