"""
ORCHEX Protocol

Main orchestrator for autonomous research from topic → publication.
"""

import asyncio
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import subprocess

from ORCHEX.hypothesis_generator import HypothesisGenerator
from self_refutation import SelfRefutationProtocol
from interrogation import InterrogationProtocol
from hall_of_failures import HallOfFailures, Failure, FailureType

# Stage 3: Experimentation
from ORCHEX.experimentation import (
    ExperimentDesigner,
    CodeGenerator,
    SandboxExecutor
)

# Stage 4: Publication
from ORCHEX.publication import PaperGenerator

# Adaptive Orchestration - Cycle 21-22
from ORCHEX.orchestration import (
    WorkflowOrchestrator,
    WorkflowConfig,
    ProblemType
)

# Brainstorming - Cycle 23-24
from ORCHEX.brainstorming import BrainstormEngine

# Meta-learning with personality agents!
try:
    from meta_learning import MetaLearningProtocol
    META_LEARNING_AVAILABLE = True
except ImportError:
    META_LEARNING_AVAILABLE = False


class ResearchProject:
    """A complete research project"""

    def __init__(self, topic: str, domain: str, output_dir: Path, workflow_config=None):
        self.topic = topic
        self.domain = domain
        self.output_dir = output_dir
        self.created_at = datetime.now()

        # Adaptive orchestration - Cycle 21-22
        self.workflow_config = workflow_config

        # Results
        self.hypothesis_candidates = []
        self.validated_hypotheses = []
        self.selected_hypothesis = None
        self.refutation_results = {}
        self.interrogation_results = {}
        self.risk_assessments = {}
        self.failures = []

        # Stage 3: Experimentation
        self.experiment_design = None
        self.generated_code = None
        self.execution_result = None

        # Stage 4: Publication
        self.paper = None

    def to_dict(self) -> dict:
        """Export to dict"""
        return {
            "topic": self.topic,
            "domain": self.domain,
            "output_dir": str(self.output_dir),
            "created_at": self.created_at.isoformat(),
            "hypothesis_count": len(self.hypothesis_candidates),
            "validated_count": len(self.validated_hypotheses),
            "selected_hypothesis": self.selected_hypothesis.claim
            if self.selected_hypothesis
            else None,
        }


class ATLASProtocol:
    """
    ORCHEX - Autonomous Theorist & Laboratory Autonomous System

    Full autonomous research pipeline:
    Topic → Hypotheses → Validation → Experiments → Paper

    NOT the Nobel Turing Challenge - focused on computational research.
    """

    def __init__(
        self,
        orchestrator=None,
        output_base_dir: str = "./atlas_projects",
        enable_meta_learning: bool = True,
        **config,
    ):
        """
        Initialize ORCHEX

        Args:
            orchestrator: AI Orchestrator for LLM calls
            output_base_dir: Base directory for project outputs
            enable_meta_learning: Use personality agents and self-learning
            **config: Configuration options
        """
        self.orchestrator = orchestrator
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.enable_meta_learning = enable_meta_learning and META_LEARNING_AVAILABLE

        # Initialize meta-learning (personality agents!)
        self.meta_learning = None
        if self.enable_meta_learning:
            try:
                trajectory_path = str(self.output_base_dir / "trajectories.jsonl")
                self.meta_learning = MetaLearningProtocol(trajectory_path=trajectory_path)
                print("🧠 Meta-Learning ENABLED! Personality agents active!")
            except Exception as e:
                print(f"⚠️ Meta-learning disabled: {e}")
                self.enable_meta_learning = False

        # Initialize components
        self.hypothesis_generator = HypothesisGenerator(orchestrator=orchestrator)
        self.refutation_protocol = SelfRefutationProtocol(orchestrator=orchestrator)
        self.interrogation_protocol = InterrogationProtocol(
            orchestrator=orchestrator
        )
        self.hall_of_failures = HallOfFailures(
            db_path=str(self.output_base_dir / "failures.db"),
            orchestrator=orchestrator,
        )

        # Stage 3: Experimentation
        self.experiment_designer = ExperimentDesigner(
            orchestrator=orchestrator,
            output_dir=str(self.output_base_dir / "experiments")
        )
        self.code_generator = CodeGenerator(orchestrator=orchestrator)
        self.sandbox_executor = SandboxExecutor()

        # Stage 4: Publication
        self.paper_generator = PaperGenerator(orchestrator=orchestrator)

        # Adaptive Orchestration - Cycle 21-22
        self.workflow_orchestrator = WorkflowOrchestrator(orchestrator=orchestrator)

        # Brainstorming - Cycle 23-24
        self.brainstorm_engine = BrainstormEngine(orchestrator=orchestrator)

    async def run_research(
        self,
        topic: str,
        domain: str = "optimization",
        num_hypotheses: int = 5,
        validation_threshold: float = 70.0,
        user_preferences: Optional[dict] = None,
    ) -> ResearchProject:
        """
        Run full autonomous research pipeline

        Args:
            topic: Research topic (e.g., "RL for QAP solving")
            domain: Scientific domain
            num_hypotheses: Number of hypotheses to generate (overridden by workflow config)
            validation_threshold: Minimum score to proceed (0-100)
            user_preferences: Optional preferences for workflow orchestration

        Returns:
            ResearchProject with all results
        """
        # ADAPTIVE ORCHESTRATION - Cycle 21-22
        # Analyze query and plan optimal workflow
        workflow_config = self.workflow_orchestrator.plan_workflow(
            query=topic,
            domain=domain,
            user_preferences=user_preferences
        )

        # Apply workflow configuration
        num_hypotheses = workflow_config.max_hypotheses

        # Create project directory
        project_name = self._sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_dir = self.output_base_dir / f"{project_name}_{timestamp}"
        project_dir.mkdir(parents=True, exist_ok=True)

        project = ResearchProject(topic, domain, project_dir, workflow_config)

        print(f"\n{'='*80}")
        print(f"ORCHEX AUTONOMOUS RESEARCH SYSTEM")
        print(f"{'='*80}")
        print(f"\nTopic: {topic}")
        print(f"Domain: {domain}")
        print(f"Output: {project_dir}\n")

        # Display adaptive workflow plan
        print("🎯 ADAPTIVE WORKFLOW ORCHESTRATION")
        print(f"   Problem Type: {workflow_config.problem_type.value.upper()}")
        print(f"   Confidence: {workflow_config.classification.confidence:.1%}")
        print(f"   Reasoning: {workflow_config.classification.reasoning}")

        # Show enabled features
        features = []
        if workflow_config.enable_brainstorming:
            features.append("Brainstorming")
        if workflow_config.enable_novelty_scoring:
            features.append("Novelty Scoring")
        if workflow_config.parallel_hypothesis_generation:
            features.append("Parallel Generation")
        if features:
            print(f"   Features: {', '.join(features)}")

        print(f"   Max Hypotheses: {workflow_config.max_hypotheses}")
        print(f"   Max Validation Rounds: {workflow_config.max_validation_rounds}")
        if workflow_config.execute_experimentation:
            print(f"   Max Experiment Variants: {workflow_config.max_experiment_variants}")
        print()

        # Start meta-learning trajectory
        trajectory_id = None
        if self.enable_meta_learning:
            # Add workflow context to trajectory
            contexts = [domain] + workflow_config.meta_learning_contexts
            trajectory_id = self.meta_learning.start_research(topic, domain)

        # Initialize git repo
        self._init_git_repo(project_dir)

        # STAGE 1: Hypothesis Generation
        print(f"{'─'*80}")
        print("STAGE 1: HYPOTHESIS GENERATION")
        print(f"{'─'*80}\n")

        # Optional: Brainstorming phase
        if workflow_config.enable_brainstorming:
            print("💡 BRAINSTORMING PHASE")
            if self.enable_meta_learning:
                brainstorm_agent = self.meta_learning.select_agent("ideation", domain)
                print(f"  {brainstorm_agent.get_greeting()}\n")

            brainstorm_session = await self.brainstorm_engine.brainstorm(
                topic=topic,
                domain=domain,
                num_ideas=min(20, num_hypotheses * 2),
                min_novelty=0.6
            )

            print(f"  ✓ Generated {brainstorm_session.total_ideas} creative ideas")
            print(f"  ✓ Used strategies: {', '.join(s.value for s in brainstorm_session.strategies_used)}")

            # Show top 3 ideas
            print(f"\n  Top Ideas:")
            for i, idea in enumerate(brainstorm_session.ideas[:3], 1):
                print(f"    {i}. [{idea.strategy.value}] {idea.idea[:80]}...")

            # Save brainstorm session
            brainstorm_file = project_dir / "brainstorm_session.json"
            import json
            with open(brainstorm_file, 'w') as f:
                json.dump({
                    "topic": brainstorm_session.topic,
                    "domain": brainstorm_session.domain,
                    "total_ideas": brainstorm_session.total_ideas,
                    "duration_seconds": brainstorm_session.duration_seconds,
                    "ideas": [
                        {
                            "idea": idea.idea,
                            "strategy": idea.strategy,
                            "novelty": idea.novelty_score,
                            "feasibility": idea.feasibility_score,
                            "interestingness": idea.interestingness
                        }
                        for idea in brainstorm_session.ideas
                    ]
                }, f, indent=2)

            print(f"  ✓ Saved to: {brainstorm_file}\n")

        candidates = await self.hypothesis_generator.generate_hypotheses(
            topic=topic, domain=domain, num_candidates=num_hypotheses
        )

        project.hypothesis_candidates = candidates

        print(f"✓ Generated {len(candidates)} hypothesis candidates\n")
        for i, candidate in enumerate(candidates, 1):
            print(f"  {i}. {candidate.hypothesis.claim[:70]}...")
            print(f"     Novelty: {candidate.novelty_score:.2f} | Feasibility: {candidate.feasibility_score:.2f} | Combined: {candidate.combined_score:.2f}")

        # Save hypotheses
        self._save_hypotheses(project_dir, candidates)

        # STAGE 2: Validation
        print(f"\n{'─'*80}")
        print("STAGE 2: HYPOTHESIS VALIDATION")
        print(f"{'─'*80}\n")

        for i, candidate in enumerate(candidates, 1):
            print(f"\nValidating Hypothesis {i}...")

            hypothesis = candidate.hypothesis

            # Risk assessment with personality agent
            print("  • Risk assessment...")
            if self.enable_meta_learning:
                risk_agent = self.meta_learning.select_agent("risk_assessment", domain)
                print(f"    {risk_agent.get_greeting()}")

            risk = await self.hall_of_failures.assess_risk(hypothesis.claim)
            project.risk_assessments[hypothesis.claim] = risk

            if self.enable_meta_learning:
                # Record action
                self.meta_learning.record_agent_action(
                    agent=risk_agent,
                    action_type="risk_assessment",
                    input_data={"hypothesis": hypothesis.claim},
                    output_data={"risk_level": risk.risk_level},
                    success=risk.risk_level != "High",
                    score=100.0 if risk.risk_level == "Low" else (50.0 if risk.risk_level == "Medium" else 0.0),
                )

            if risk.risk_level == "High":
                print(f"    ⚠️  High risk detected! Skipping.")
                if self.enable_meta_learning:
                    self.meta_learning.agent_react(risk_agent, 0.0)
                # Record failure
                failure = Failure(
                    hypothesis=hypothesis.claim,
                    failure_type=FailureType.HYPOTHESIS,
                    description=f"High risk: {risk.warnings[0] if risk.warnings else 'Unknown'}",
                )
                await self.hall_of_failures.record_failure(failure)
                project.failures.append(failure)
                continue

            # Self-refutation with personality agent
            print("  • Self-refutation...")
            if self.enable_meta_learning:
                refutation_agent = self.meta_learning.select_agent("self_refutation", domain)
                print(f"    {refutation_agent.get_greeting()}")

            refutation_result = await self.refutation_protocol.refute(hypothesis)
            project.refutation_results[hypothesis.claim] = refutation_result

            if self.enable_meta_learning:
                # Record action
                self.meta_learning.record_agent_action(
                    agent=refutation_agent,
                    action_type="self_refutation",
                    input_data={"hypothesis": hypothesis.claim},
                    output_data={"refuted": refutation_result.refuted, "score": refutation_result.strength_score},
                    success=not refutation_result.refuted,
                    score=refutation_result.strength_score,
                )

            if refutation_result.refuted:
                print(f"    ✗ Refuted (score: {refutation_result.strength_score:.1f}/100)")
                if self.enable_meta_learning:
                    self.meta_learning.agent_react(refutation_agent, refutation_result.strength_score)
                # Record failure
                failure = Failure(
                    hypothesis=hypothesis.claim,
                    failure_type=FailureType.HYPOTHESIS,
                    description=refutation_result.refutation_reason or "Refuted by protocol",
                )
                await self.hall_of_failures.record_failure(failure)
                project.failures.append(failure)
                continue

            print(f"    ✓ Survived refutation (score: {refutation_result.strength_score:.1f}/100)")
            if self.enable_meta_learning:
                self.meta_learning.agent_react(refutation_agent, refutation_result.strength_score)

            # Interrogation with personality agent
            print("  • 200-Question interrogation...")
            if self.enable_meta_learning:
                interrogation_agent = self.meta_learning.select_agent("interrogation", domain)
                print(f"    {interrogation_agent.get_greeting()}")

            interrogation_result = await self.interrogation_protocol.interrogate(
                hypothesis
            )
            project.interrogation_results[hypothesis.claim] = interrogation_result

            if self.enable_meta_learning:
                # Record action
                self.meta_learning.record_agent_action(
                    agent=interrogation_agent,
                    action_type="interrogation",
                    input_data={"hypothesis": hypothesis.claim},
                    output_data={"score": interrogation_result.overall_score},
                    success=interrogation_result.overall_score >= validation_threshold,
                    score=interrogation_result.overall_score,
                )

            print(f"    → Overall score: {interrogation_result.overall_score:.1f}/100")
            if self.enable_meta_learning:
                self.meta_learning.agent_react(interrogation_agent, interrogation_result.overall_score)

            # Check if passes threshold
            combined_score = (
                refutation_result.strength_score * 0.5
                + interrogation_result.overall_score * 0.5
            )

            if combined_score >= validation_threshold:
                print(f"    ✓ VALIDATED (combined: {combined_score:.1f}/100)")
                project.validated_hypotheses.append(candidate)
            else:
                print(f"    ✗ Below threshold (combined: {combined_score:.1f}/100)")

        # Save validation results
        self._save_validation_results(project_dir, project)

        # Select best hypothesis
        if project.validated_hypotheses:
            project.selected_hypothesis = project.validated_hypotheses[0].hypothesis
            print(f"\n✓ Selected hypothesis: {project.selected_hypothesis.claim}")
        else:
            print(f"\n✗ No hypotheses passed validation!")
            return project

        # STAGE 3: Experimentation (conditional based on workflow config)
        if workflow_config.execute_experimentation:
            print(f"\n{'─'*80}")
            print("STAGE 3: EXPERIMENTATION")
            print(f"{'─'*80}\n")

            # Select agent for experiment design
            if self.enable_meta_learning:
                exp_agent = self.meta_learning.select_agent("experiment_design", domain)
                print(f"  {exp_agent.get_greeting()}")

            # Design experiment
            print("  🔬 Designing experiment...")
            experiment_design = await self.experiment_designer.design_experiment(
                hypothesis=project.selected_hypothesis.claim,
                domain=domain,
                constraints={"max_cost": 100.0}
            )
            project.experiment_design = experiment_design

            print(f"    → Type: {experiment_design.experiment_type}")
            print(f"    → Trials: {experiment_design.num_trials}")
            print(f"    → Duration: {experiment_design.estimated_duration}")
            print(f"    → Cost: ${experiment_design.estimated_cost:.2f}")

            # Generate code
            print("  🔧 Generating experiment code...")
            exp_code_dir = project_dir / "experiment"
            generated_code = await self.code_generator.generate_code(
                experiment_design,
                exp_code_dir
            )
            project.generated_code = generated_code

            if self.enable_meta_learning:
                # Record experiment design action
                self.meta_learning.record_agent_action(
                    agent=exp_agent,
                    action_type="experiment_design",
                    input_data={"hypothesis": project.selected_hypothesis.claim},
                    output_data={
                        "type": experiment_design.experiment_type,
                        "trials": experiment_design.num_trials
                    },
                    success=True,
                    score=80.0  # Design quality score
                )
                self.meta_learning.agent_react(exp_agent, 80.0)

            print(f"\n✓ Experiment ready to run!")
            print(f"  Code: {exp_code_dir}")
            print(f"  To execute: cd {exp_code_dir} && python main.py")
        else:
            print(f"\n⊘ STAGE 3: EXPERIMENTATION (Skipped by workflow config)")

        # STAGE 4: Publication (conditional based on workflow config)
        if workflow_config.execute_paper_generation:
            print(f"\n{'─'*80}")
            print("STAGE 4: PAPER GENERATION")
            print(f"{'─'*80}\n")

            # Select agent for paper writing
            if self.enable_meta_learning:
                paper_agent = self.meta_learning.select_agent("peer_review", domain)
                print(f"  {paper_agent.get_greeting()}")

            # Generate paper
            print("  📝 Generating manuscript...")
            paper = await self.paper_generator.generate_paper(
                topic=topic,
                hypothesis=project.selected_hypothesis.claim,
                experiment_design=project.experiment_design,
                results=project.execution_result.results_data if project.execution_result else None,
                domain=domain
            )
            project.paper = paper

            # Save paper
            paper_dir = project_dir / "paper"
            latex_file = self.paper_generator.save_paper(paper, paper_dir)

            if self.enable_meta_learning:
                # Record paper writing action
                self.meta_learning.record_agent_action(
                    agent=paper_agent,
                    action_type="paper_generation",
                    input_data={"topic": topic},
                    output_data={"sections": len(paper.sections)},
                    success=True,
                    score=85.0  # Paper quality score
                )
                self.meta_learning.agent_react(paper_agent, 85.0)

            print(f"\n✓ Paper generated!")
            print(f"  Title: {paper.title}")
            print(f"  Sections: {len(paper.sections)}")
            print(f"  LaTeX: {latex_file}")
        else:
            print(f"\n⊘ STAGE 4: PAPER GENERATION (Skipped by workflow config)")

        # Final summary
        print(f"\n{'='*80}")
        print("RESEARCH PROJECT SUMMARY")
        print(f"{'='*80}\n")
        print(f"Topic: {topic}")
        print(f"Hypotheses generated: {len(project.hypothesis_candidates)}")
        print(f"Hypotheses validated: {len(project.validated_hypotheses)}")
        print(f"Experiment designed: {'Yes' if project.experiment_design else 'No'}")
        print(f"Paper generated: {'Yes' if project.paper else 'No'}")
        print(f"Failures recorded: {len(project.failures)}")
        print(f"Output directory: {project_dir}")
        if project.paper:
            print(f"Paper: {paper_dir / 'paper.tex'}")
        print(f"\n{'='*80}\n")

        # Complete meta-learning and update
        if self.enable_meta_learning:
            # Determine success
            success = len(project.validated_hypotheses) > 0

            # Calculate final score
            if project.validated_hypotheses:
                final_score = project.validated_hypotheses[0].combined_score
            else:
                final_score = 0.0

            # Complete research
            self.meta_learning.complete_research(
                success=success,
                final_score=final_score,
                domain=domain
            )

        # Save final report
        self._save_final_report(project)

        return project

    def _init_git_repo(self, project_dir: Path):
        """Initialize git repository"""
        try:
            subprocess.run(
                ["git", "init"], cwd=project_dir, capture_output=True, check=True
            )
            subprocess.run(
                ["git", "config", "user.name", "ORCHEX"],
                cwd=project_dir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "ORCHEX@example.com"],
                cwd=project_dir,
                capture_output=True,
            )
            print("✓ Git repository initialized")
        except Exception as e:
            print(f"⚠️  Could not initialize git: {e}")

    def _save_hypotheses(self, project_dir: Path, candidates):
        """Save hypotheses to JSON"""
        data = [
            {
                "claim": c.hypothesis.claim,
                "domain": c.hypothesis.domain.value,
                "novelty_score": c.novelty_score,
                "feasibility_score": c.feasibility_score,
                "combined_score": c.combined_score,
                "justification": c.justification,
            }
            for c in candidates
        ]

        with open(project_dir / "hypotheses.json", "w") as f:
            json.dump(data, f, indent=2)

    def _save_validation_results(self, project_dir: Path, project):
        """Save validation results"""
        data = {
            "validated_count": len(project.validated_hypotheses),
            "failed_count": len(project.failures),
            "refutation_scores": {
                claim: result.strength_score
                for claim, result in project.refutation_results.items()
            },
            "interrogation_scores": {
                claim: result.overall_score
                for claim, result in project.interrogation_results.items()
            },
        }

        with open(project_dir / "validation_results.json", "w") as f:
            json.dump(data, f, indent=2)

    def _save_final_report(self, project: ResearchProject):
        """Save final project report"""
        report_path = project.output_dir / "RESEARCH_REPORT.md"

        content = f"""# ORCHEX Research Project Report

**Topic**: {project.topic}
**Domain**: {project.domain}
**Created**: {project.created_at.strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary

- **Hypotheses Generated**: {len(project.hypothesis_candidates)}
- **Hypotheses Validated**: {len(project.validated_hypotheses)}
- **Failures Recorded**: {len(project.failures)}

---

## Generated Hypotheses

"""

        for i, candidate in enumerate(project.hypothesis_candidates, 1):
            content += f"""
### Hypothesis {i}

**Claim**: {candidate.hypothesis.claim}

**Scores**:
- Novelty: {candidate.novelty_score:.2f}
- Feasibility: {candidate.feasibility_score:.2f}
- Combined: {candidate.combined_score:.2f}

**Justification**: {candidate.justification}

"""

        if project.selected_hypothesis:
            content += f"""
---

## Selected Hypothesis

**Claim**: {project.selected_hypothesis.claim}

This hypothesis passed all validation checks and was selected for experimentation.

"""

        content += """
---

## Next Steps

1. Implement experiments for selected hypothesis
2. Run experiments and collect data
3. Analyze results
4. Generate publication-ready manuscript

---

*Generated by ORCHEX - Autonomous Theorist & Laboratory Autonomous System*
*(NOT the Nobel Turing Challenge - computational research prototype)*
"""

        with open(report_path, "w") as f:
            f.write(content)

    def _sanitize_filename(self, text: str) -> str:
        """Convert text to safe filename"""
        import re

        text = text.lower()
        text = re.sub(r"[^a-z0-9]+", "_", text)
        text = text.strip("_")
        return text[:50]  # Limit length
