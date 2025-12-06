"""
Experiment Designer - Stage 3 of ORCHEX

Designs computational experiments to test validated hypotheses.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class ExperimentParameter:
    """A parameter to vary in experiments"""
    name: str
    type: str  # "continuous", "discrete", "categorical"
    values: List[Any]
    description: str


@dataclass
class ExperimentDesign:
    """Complete experimental design"""
    hypothesis: str
    objective: str
    experiment_type: str  # "benchmark", "ablation", "parameter_sweep", "comparison"

    # Parameters
    parameters: List[ExperimentParameter]
    fixed_settings: Dict[str, Any]

    # Execution
    num_trials: int
    compute_requirements: Dict[str, str]
    estimated_duration: str
    estimated_cost: float

    # Metrics
    primary_metric: str
    secondary_metrics: List[str]
    success_criteria: str

    # Implementation
    code_requirements: List[str]
    data_requirements: List[str]
    dependencies: List[str]

    # Metadata
    design_id: str = field(default_factory=lambda: f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: datetime = field(default_factory=datetime.now)


class ExperimentDesigner:
    """
    Design computational experiments to test hypotheses

    Features:
    - Multiple experiment types (benchmark, ablation, sweep, comparison)
    - Parameter space definition
    - Resource estimation
    - Success criteria specification
    - AI-assisted design refinement
    """

    def __init__(self, orchestrator=None, output_dir: str = "./experiments"):
        """
        Initialize experiment designer

        Args:
            orchestrator: Optional AI Orchestrator for design refinement
            output_dir: Where to save experiment designs
        """
        self.orchestrator = orchestrator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def design_experiment(
        self,
        hypothesis: str,
        domain: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExperimentDesign:
        """
        Design computational experiment for hypothesis

        Args:
            hypothesis: The hypothesis to test
            domain: Research domain (e.g., "optimization", "machine_learning")
            constraints: Budget, time, compute constraints

        Returns:
            ExperimentDesign with complete specification
        """
        constraints = constraints or {}

        # Determine experiment type
        exp_type = self._determine_experiment_type(hypothesis, domain)

        # Design based on type
        if exp_type == "benchmark":
            design = await self._design_benchmark(hypothesis, domain, constraints)
        elif exp_type == "ablation":
            design = await self._design_ablation(hypothesis, domain, constraints)
        elif exp_type == "parameter_sweep":
            design = await self._design_parameter_sweep(hypothesis, domain, constraints)
        elif exp_type == "comparison":
            design = await self._design_comparison(hypothesis, domain, constraints)
        else:
            design = await self._design_generic(hypothesis, domain, constraints)

        # Refine with AI if available
        if self.orchestrator:
            design = await self._refine_with_ai(design, hypothesis, domain)

        # Save design
        self._save_design(design)

        return design

    def _determine_experiment_type(self, hypothesis: str, domain: str) -> str:
        """Determine what type of experiment to design"""
        hyp_lower = hypothesis.lower()

        # Pattern matching for experiment types
        if "better than" in hyp_lower or "outperforms" in hyp_lower or "compared to" in hyp_lower:
            return "comparison"

        elif "parameter" in hyp_lower or "tuning" in hyp_lower or "setting" in hyp_lower:
            return "parameter_sweep"

        elif "without" in hyp_lower or "ablation" in hyp_lower or "component" in hyp_lower:
            return "ablation"

        elif "benchmark" in hyp_lower or "performance" in hyp_lower or "evaluation" in hyp_lower:
            return "benchmark"

        else:
            return "benchmark"  # Default

    async def _design_benchmark(
        self,
        hypothesis: str,
        domain: str,
        constraints: Dict
    ) -> ExperimentDesign:
        """Design benchmark experiment"""

        # Common benchmark parameters
        if domain == "optimization":
            parameters = [
                ExperimentParameter(
                    name="problem_size",
                    type="discrete",
                    values=[10, 20, 50, 100, 200],
                    description="Size of problem instances"
                ),
                ExperimentParameter(
                    name="algorithm_variant",
                    type="categorical",
                    values=["baseline", "proposed", "hybrid"],
                    description="Algorithm variant to test"
                ),
                ExperimentParameter(
                    name="seed",
                    type="discrete",
                    values=list(range(5)),
                    description="Random seed for reproducibility"
                )
            ]
            primary_metric = "solution_quality"
            secondary_metrics = ["runtime", "convergence_rate", "memory_usage"]

        elif domain == "machine_learning":
            parameters = [
                ExperimentParameter(
                    name="dataset",
                    type="categorical",
                    values=["train", "validation", "test"],
                    description="Dataset split"
                ),
                ExperimentParameter(
                    name="model_size",
                    type="categorical",
                    values=["small", "medium", "large"],
                    description="Model size variant"
                ),
                ExperimentParameter(
                    name="seed",
                    type="discrete",
                    values=list(range(5)),
                    description="Random seed"
                )
            ]
            primary_metric = "accuracy"
            secondary_metrics = ["precision", "recall", "f1_score", "training_time"]

        else:
            # Generic domain
            parameters = [
                ExperimentParameter(
                    name="input_size",
                    type="discrete",
                    values=[10, 50, 100],
                    description="Input size"
                ),
                ExperimentParameter(
                    name="method",
                    type="categorical",
                    values=["baseline", "proposed"],
                    description="Method to test"
                )
            ]
            primary_metric = "performance"
            secondary_metrics = ["efficiency", "quality"]

        # Estimate resources
        num_trials = 1
        for param in parameters:
            num_trials *= len(param.values)

        compute_reqs = self._estimate_compute_requirements(num_trials, domain)
        duration = self._estimate_duration(num_trials, domain)
        cost = self._estimate_cost(num_trials, domain, constraints)

        return ExperimentDesign(
            hypothesis=hypothesis,
            objective=f"Benchmark performance of proposed method on {domain} tasks",
            experiment_type="benchmark",
            parameters=parameters,
            fixed_settings={"timeout": "1h", "max_iterations": 1000},
            num_trials=num_trials,
            compute_requirements=compute_reqs,
            estimated_duration=duration,
            estimated_cost=cost,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics,
            success_criteria=f"{primary_metric} improvement >= 5% over baseline",
            code_requirements=[
                "Implement proposed method",
                "Implement baseline for comparison",
                "Create benchmark dataset loader",
                "Implement evaluation metrics"
            ],
            data_requirements=[
                f"Standard {domain} benchmark datasets",
                "Train/test splits with seeds",
                "Preprocessed and ready to load"
            ],
            dependencies=[
                "numpy",
                "scipy",
                "pandas",
                "matplotlib",
                "pytest"
            ]
        )

    async def _design_ablation(
        self,
        hypothesis: str,
        domain: str,
        constraints: Dict
    ) -> ExperimentDesign:
        """Design ablation study"""

        # Identify components to ablate
        components = self._extract_components(hypothesis)

        parameters = [
            ExperimentParameter(
                name="ablation_config",
                type="categorical",
                values=[
                    "full",  # All components
                    *[f"no_{comp}" for comp in components],  # Remove each
                ],
                description="Which components to include"
            ),
            ExperimentParameter(
                name="trial_id",
                type="discrete",
                values=list(range(5)),
                description="Trial repetition"
            )
        ]

        num_trials = len(components) * 5 + 5  # Each ablation + full, 5 trials each

        return ExperimentDesign(
            hypothesis=hypothesis,
            objective="Determine importance of each component via ablation",
            experiment_type="ablation",
            parameters=parameters,
            fixed_settings={"eval_metric": "primary_objective"},
            num_trials=num_trials,
            compute_requirements=self._estimate_compute_requirements(num_trials, domain),
            estimated_duration=self._estimate_duration(num_trials, domain),
            estimated_cost=self._estimate_cost(num_trials, domain, constraints),
            primary_metric="performance_delta",
            secondary_metrics=["component_importance", "interaction_effects"],
            success_criteria="Clear performance degradation when key component removed",
            code_requirements=[
                "Modular implementation allowing component removal",
                "Component importance calculator",
                "Ablation result analyzer"
            ],
            data_requirements=["Same dataset for all ablation variants"],
            dependencies=["numpy", "scipy", "pandas", "matplotlib"]
        )

    async def _design_parameter_sweep(
        self,
        hypothesis: str,
        domain: str,
        constraints: Dict
    ) -> ExperimentDesign:
        """Design parameter sweep experiment"""

        # Extract parameters mentioned in hypothesis
        param_names = self._extract_parameter_names(hypothesis)

        parameters = []
        for param_name in param_names[:3]:  # Max 3 params to avoid explosion
            if "learning" in param_name.lower() or "rate" in param_name.lower():
                values = [0.001, 0.01, 0.1, 1.0]
            elif "size" in param_name.lower() or "hidden" in param_name.lower():
                values = [32, 64, 128, 256]
            elif "depth" in param_name.lower() or "layers" in param_name.lower():
                values = [2, 4, 6, 8]
            else:
                values = [0.1, 0.5, 1.0, 2.0, 5.0]

            parameters.append(ExperimentParameter(
                name=param_name,
                type="continuous",
                values=values,
                description=f"Parameter: {param_name}"
            ))

        num_trials = 1
        for param in parameters:
            num_trials *= len(param.values)
        num_trials *= 3  # 3 random seeds

        return ExperimentDesign(
            hypothesis=hypothesis,
            objective="Find optimal parameter settings",
            experiment_type="parameter_sweep",
            parameters=parameters,
            fixed_settings={"evaluation": "cross_validation"},
            num_trials=num_trials,
            compute_requirements=self._estimate_compute_requirements(num_trials, domain),
            estimated_duration=self._estimate_duration(num_trials, domain),
            estimated_cost=self._estimate_cost(num_trials, domain, constraints),
            primary_metric="validation_score",
            secondary_metrics=["training_stability", "generalization_gap"],
            success_criteria="Find parameter setting with >10% improvement",
            code_requirements=[
                "Parameterized implementation",
                "Grid/random search infrastructure",
                "Performance visualization"
            ],
            data_requirements=["Cross-validation splits"],
            dependencies=["numpy", "scipy", "scikit-learn", "matplotlib"]
        )

    async def _design_comparison(
        self,
        hypothesis: str,
        domain: str,
        constraints: Dict
    ) -> ExperimentDesign:
        """Design comparison experiment"""

        # Extract methods being compared
        methods = self._extract_methods(hypothesis)

        parameters = [
            ExperimentParameter(
                name="method",
                type="categorical",
                values=methods,
                description="Method to evaluate"
            ),
            ExperimentParameter(
                name="dataset",
                type="categorical",
                values=["easy", "medium", "hard"],
                description="Problem difficulty"
            ),
            ExperimentParameter(
                name="trial_id",
                type="discrete",
                values=list(range(10)),
                description="Independent trial"
            )
        ]

        num_trials = len(methods) * 3 * 10  # methods × difficulties × trials

        return ExperimentDesign(
            hypothesis=hypothesis,
            objective=f"Compare {', '.join(methods)} across problem difficulties",
            experiment_type="comparison",
            parameters=parameters,
            fixed_settings={"fair_comparison": True, "same_resources": True},
            num_trials=num_trials,
            compute_requirements=self._estimate_compute_requirements(num_trials, domain),
            estimated_duration=self._estimate_duration(num_trials, domain),
            estimated_cost=self._estimate_cost(num_trials, domain, constraints),
            primary_metric="relative_performance",
            secondary_metrics=["statistical_significance", "effect_size", "robustness"],
            success_criteria="Proposed method significantly better (p<0.05)",
            code_requirements=[
                f"Implementation of each method: {', '.join(methods)}",
                "Fair comparison framework",
                "Statistical testing suite"
            ],
            data_requirements=[
                "Multiple difficulty levels",
                "Sufficient samples for statistical power"
            ],
            dependencies=["numpy", "scipy", "pandas", "matplotlib", "statsmodels"]
        )

    async def _design_generic(
        self,
        hypothesis: str,
        domain: str,
        constraints: Dict
    ) -> ExperimentDesign:
        """Generic experiment design fallback"""
        return await self._design_benchmark(hypothesis, domain, constraints)

    def _extract_components(self, hypothesis: str) -> List[str]:
        """Extract components mentioned in hypothesis"""
        # Simple heuristic: common component keywords
        components = []
        keywords = ["module", "layer", "component", "feature", "mechanism", "step"]

        for keyword in keywords:
            if keyword in hypothesis.lower():
                # Extract words near the keyword
                components.append(f"{keyword}_1")
                components.append(f"{keyword}_2")

        return components[:4] if components else ["component_a", "component_b"]

    def _extract_parameter_names(self, hypothesis: str) -> List[str]:
        """Extract parameter names from hypothesis"""
        # Simple heuristic
        common_params = [
            "learning_rate", "batch_size", "hidden_size",
            "num_layers", "dropout", "temperature",
            "epsilon", "alpha", "beta"
        ]

        found = []
        for param in common_params:
            if param.replace("_", " ") in hypothesis.lower():
                found.append(param)

        return found if found else ["param_alpha", "param_beta"]

    def _extract_methods(self, hypothesis: str) -> List[str]:
        """Extract method names from hypothesis"""
        # Look for comparison keywords
        if "proposed" in hypothesis.lower():
            methods = ["baseline", "proposed"]
        elif "method" in hypothesis.lower():
            methods = ["method_a", "method_b"]
        else:
            methods = ["baseline", "variant_1", "variant_2"]

        return methods

    def _estimate_compute_requirements(
        self,
        num_trials: int,
        domain: str
    ) -> Dict[str, str]:
        """Estimate compute requirements"""
        if num_trials < 50:
            return {"cpu": "4 cores", "memory": "8GB", "gpu": "optional"}
        elif num_trials < 200:
            return {"cpu": "8 cores", "memory": "16GB", "gpu": "recommended"}
        else:
            return {"cpu": "16+ cores", "memory": "32GB", "gpu": "required"}

    def _estimate_duration(self, num_trials: int, domain: str) -> str:
        """Estimate execution duration"""
        if num_trials < 50:
            return "2-4 hours"
        elif num_trials < 200:
            return "8-12 hours"
        else:
            return "24-48 hours"

    def _estimate_cost(
        self,
        num_trials: int,
        domain: str,
        constraints: Dict
    ) -> float:
        """Estimate execution cost in USD"""
        # Base cost per trial
        if domain in ["machine_learning", "deep_learning"]:
            cost_per_trial = 0.50  # GPU time
        else:
            cost_per_trial = 0.10  # CPU time

        total_cost = num_trials * cost_per_trial

        # Apply budget constraint
        max_budget = constraints.get("max_cost", 100.0)
        return min(total_cost, max_budget)

    async def _refine_with_ai(
        self,
        design: ExperimentDesign,
        hypothesis: str,
        domain: str
    ) -> ExperimentDesign:
        """Refine design with AI assistance"""
        # Use orchestrator to suggest improvements
        prompt = f"""
        Review this experimental design and suggest improvements:

        Hypothesis: {hypothesis}
        Domain: {domain}

        Current Design:
        - Type: {design.experiment_type}
        - Parameters: {len(design.parameters)}
        - Trials: {design.num_trials}
        - Primary metric: {design.primary_metric}

        Suggest:
        1. Additional parameters to vary
        2. Better metrics to track
        3. Potential confounds to control
        4. Ways to reduce cost while maintaining rigor
        """

        if self.orchestrator:
            suggestions = await self.orchestrator.think(prompt)
            # In a real implementation, parse suggestions and update design
            # For now, just log that AI review happened
            design.fixed_settings["ai_reviewed"] = True

        return design

    def _save_design(self, design: ExperimentDesign):
        """Save experiment design to file"""
        output_file = self.output_dir / f"{design.design_id}.json"

        # Convert to dict
        design_dict = {
            "design_id": design.design_id,
            "hypothesis": design.hypothesis,
            "objective": design.objective,
            "experiment_type": design.experiment_type,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "values": p.values,
                    "description": p.description
                }
                for p in design.parameters
            ],
            "fixed_settings": design.fixed_settings,
            "num_trials": design.num_trials,
            "compute_requirements": design.compute_requirements,
            "estimated_duration": design.estimated_duration,
            "estimated_cost": design.estimated_cost,
            "primary_metric": design.primary_metric,
            "secondary_metrics": design.secondary_metrics,
            "success_criteria": design.success_criteria,
            "code_requirements": design.code_requirements,
            "data_requirements": design.data_requirements,
            "dependencies": design.dependencies,
            "created_at": design.created_at.isoformat()
        }

        with open(output_file, 'w') as f:
            json.dump(design_dict, f, indent=2)

        print(f"✓ Saved experiment design: {output_file}")
