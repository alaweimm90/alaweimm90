"""
Materials Science Research Protocol with TalAI Turing Challenge Integration.

Validates materials hypotheses through crystal structure analysis,
property prediction, synthesis feasibility, and DFT calculations.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import hashlib
import json
from pathlib import Path

from .models import (
    MaterialsHypothesis,
    CrystalStructure,
    MaterialProperties,
    SynthesisRoute,
    DFTCalculation,
    PhaseDiagram,
)
from .analyzers import (
    StructurePredictor,
    PropertyPredictor,
    RetrosynthesisPlanner,
    PhaseDiagramGenerator,
    DFTWorkflowManager,
)
from .integrations import MaterialsProjectAPI, PeriodicTableReasoner


class MaterialsScienceProtocol:
    """
    Materials Science validation protocol implementing TalAI Turing Challenge.

    Performs comprehensive validation of materials hypotheses including:
    - Crystal structure prediction and validation
    - Property prediction (mechanical, electrical, thermal, magnetic)
    - Synthesis route planning and feasibility
    - Phase diagram generation
    - DFT calculation workflow automation
    - Materials Project database integration
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Materials Science Protocol.

        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}

        # Initialize components
        self.structure_predictor = StructurePredictor(config)
        self.property_predictor = PropertyPredictor(config)
        self.synthesis_planner = RetrosynthesisPlanner(config)
        self.phase_generator = PhaseDiagramGenerator(config)
        self.dft_manager = DFTWorkflowManager(config)
        self.materials_api = MaterialsProjectAPI(
            api_key=config.get("materials_project_api_key") if config else None
        )
        self.periodic_reasoner = PeriodicTableReasoner()

        # Validation results cache
        self._validation_cache: Dict[str, Dict[str, Any]] = {}

    async def validate_hypothesis(
        self,
        hypothesis: MaterialsHypothesis,
        validation_level: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Validate a materials science hypothesis.

        Args:
            hypothesis: Materials hypothesis to validate
            validation_level: Level of validation ("basic", "standard", "comprehensive")

        Returns:
            Validation results dictionary
        """
        print(f"🔬 Validating materials hypothesis: {hypothesis.title}")

        # Check cache
        cache_key = self._get_cache_key(hypothesis)
        if cache_key in self._validation_cache:
            print("  → Using cached validation results")
            return self._validation_cache[cache_key]

        # Perform validation stages
        results = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "timestamp": datetime.now().isoformat(),
            "validation_level": validation_level,
            "stages": {}
        }

        # Stage 1: Structure Validation
        if hypothesis.predicted_structure:
            structure_results = await self._validate_structure(hypothesis.predicted_structure)
            results["stages"]["structure"] = structure_results

        # Stage 2: Property Prediction
        if hypothesis.predicted_properties:
            property_results = await self._validate_properties(
                hypothesis.predicted_properties,
                hypothesis.predicted_structure
            )
            results["stages"]["properties"] = property_results

        # Stage 3: Synthesis Feasibility
        if hypothesis.synthesis_route:
            synthesis_results = await self._validate_synthesis(hypothesis.synthesis_route)
            results["stages"]["synthesis"] = synthesis_results
        elif validation_level in ["standard", "comprehensive"]:
            # Generate synthesis route if not provided
            synthesis_results = await self._generate_synthesis_route(hypothesis.claim)
            results["stages"]["synthesis"] = synthesis_results

        # Stage 4: Phase Stability (comprehensive only)
        if validation_level == "comprehensive" and hypothesis.predicted_structure:
            phase_results = await self._analyze_phase_stability(hypothesis.predicted_structure)
            results["stages"]["phase_stability"] = phase_results

        # Stage 5: DFT Validation (comprehensive only)
        if validation_level == "comprehensive" and hypothesis.predicted_structure:
            dft_results = await self._perform_dft_validation(hypothesis.predicted_structure)
            results["stages"]["dft"] = dft_results

        # Stage 6: Database Comparison
        database_results = await self._compare_with_database(hypothesis)
        results["stages"]["database"] = database_results

        # Calculate overall scores
        results["scores"] = self._calculate_scores(results["stages"])
        results["recommendation"] = self._generate_recommendation(results["scores"])

        # Cache results
        self._validation_cache[cache_key] = results

        return results

    async def _validate_structure(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Validate crystal structure."""
        print("  → Validating crystal structure...")

        results = {
            "valid": True,
            "issues": [],
            "metrics": {}
        }

        # Check space group consistency
        space_group_valid = await self.structure_predictor.validate_space_group(structure)
        results["metrics"]["space_group_valid"] = space_group_valid
        if not space_group_valid:
            results["issues"].append("Space group inconsistent with lattice parameters")
            results["valid"] = False

        # Check atomic positions
        position_issues = await self.structure_predictor.check_atomic_positions(structure)
        if position_issues:
            results["issues"].extend(position_issues)
            results["valid"] = False

        # Calculate structure metrics
        results["metrics"]["packing_fraction"] = await self.structure_predictor.calculate_packing_fraction(structure)
        results["metrics"]["coordination_numbers"] = await self.structure_predictor.get_coordination_numbers(structure)
        results["metrics"]["bond_lengths"] = await self.structure_predictor.calculate_bond_lengths(structure)

        # Check for known structure types
        structure_type = await self.structure_predictor.identify_structure_type(structure)
        results["structure_type"] = structure_type

        return results

    async def _validate_properties(
        self,
        properties: MaterialProperties,
        structure: Optional[CrystalStructure]
    ) -> Dict[str, Any]:
        """Validate material properties."""
        print("  → Validating material properties...")

        results = {
            "plausible": True,
            "warnings": [],
            "predictions": {},
            "comparisons": {}
        }

        # Predict properties from structure if available
        if structure:
            predicted = await self.property_predictor.predict_from_structure(structure)
            results["predictions"] = predicted

            # Compare predicted with claimed properties
            comparison = self._compare_properties(properties, predicted)
            results["comparisons"] = comparison

            if comparison.get("major_discrepancies"):
                results["plausible"] = False
                results["warnings"].extend(comparison["major_discrepancies"])

        # Check property relationships (e.g., Pugh's ratio)
        if properties.mechanical:
            relationship_check = await self.property_predictor.check_property_relationships(
                properties.mechanical
            )
            results["property_relationships"] = relationship_check
            if relationship_check.get("violations"):
                results["warnings"].extend(relationship_check["violations"])

        # Validate against empirical bounds
        bounds_check = await self.property_predictor.check_empirical_bounds(properties)
        if bounds_check.get("out_of_bounds"):
            results["warnings"].extend(bounds_check["out_of_bounds"])

        return results

    async def _validate_synthesis(self, route: SynthesisRoute) -> Dict[str, Any]:
        """Validate synthesis route feasibility."""
        print("  → Validating synthesis route...")

        results = {
            "feasible": True,
            "score": 0.0,
            "issues": [],
            "alternatives": []
        }

        # Check reaction thermodynamics
        for step in route.steps:
            thermo = await self.synthesis_planner.check_thermodynamics(step)
            if not thermo["favorable"]:
                results["issues"].append(f"Step {step.step_number}: Thermodynamically unfavorable")
                results["feasible"] = False

        # Check precursor availability
        availability = await self.synthesis_planner.check_precursor_availability(route)
        results["precursor_availability"] = availability

        # Calculate synthesis score
        results["score"] = await self.synthesis_planner.calculate_feasibility_score(route)

        # Generate alternative routes
        if results["score"] < 0.5:
            alternatives = await self.synthesis_planner.generate_alternatives(route.target_material)
            results["alternatives"] = alternatives[:3]  # Top 3 alternatives

        return results

    async def _generate_synthesis_route(self, material_formula: str) -> Dict[str, Any]:
        """Generate synthesis route for material."""
        print("  → Generating synthesis route...")

        # Use retrosynthesis planning
        routes = await self.synthesis_planner.plan_synthesis(material_formula)

        if routes:
            best_route = routes[0]  # Best route
            validation = await self._validate_synthesis(best_route)
            validation["generated_route"] = best_route.model_dump()
            return validation

        return {
            "feasible": False,
            "error": "No viable synthesis route found",
            "alternatives": []
        }

    async def _analyze_phase_stability(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Analyze phase stability and generate phase diagram."""
        print("  → Analyzing phase stability...")

        results = {
            "stable": False,
            "formation_energy": None,
            "decomposition_products": [],
            "phase_diagram": None
        }

        # Generate phase diagram for the system
        elements = self.periodic_reasoner.extract_elements(structure.formula)
        if len(elements) <= 3:  # Binary or ternary system
            phase_diagram = await self.phase_generator.generate_diagram(
                elements,
                temperature_range=(300, 2000)
            )
            results["phase_diagram"] = phase_diagram.model_dump() if phase_diagram else None

        # Check thermodynamic stability
        stability = await self.phase_generator.check_stability(structure)
        results["stable"] = stability["stable"]
        results["formation_energy"] = stability.get("formation_energy")
        results["decomposition_products"] = stability.get("decomposition_products", [])

        return results

    async def _perform_dft_validation(self, structure: CrystalStructure) -> Dict[str, Any]:
        """Perform DFT calculations for validation."""
        print("  → Running DFT calculations...")

        # Setup DFT calculation
        calc_spec = DFTCalculation(
            structure=structure,
            functional="PBE",
            basis_set="PAW",
            k_points=(4, 4, 4),
            cutoff_energy=520.0,
            convergence_criteria={"energy": 1e-6, "force": 0.01}
        )

        # Run calculation (simulated for demo)
        results = await self.dft_manager.run_calculation(calc_spec)

        return {
            "converged": results.converged,
            "total_energy": results.total_energy,
            "band_gap": results.band_structure.get("gap") if results.band_structure else None,
            "formation_energy": await self.dft_manager.calculate_formation_energy(results),
            "calculation_time": results.calculation_time
        }

    async def _compare_with_database(self, hypothesis: MaterialsHypothesis) -> Dict[str, Any]:
        """Compare with Materials Project database."""
        print("  → Comparing with Materials Project database...")

        results = {
            "novel": True,
            "similar_materials": [],
            "property_comparison": {}
        }

        # Search for similar materials
        if hypothesis.predicted_structure:
            similar = await self.materials_api.find_similar_structures(
                hypothesis.predicted_structure
            )
            results["similar_materials"] = similar
            results["novel"] = len(similar) == 0

            if similar and hypothesis.predicted_properties:
                # Compare properties with similar materials
                comparison = await self.materials_api.compare_properties(
                    hypothesis.predicted_properties,
                    similar[0]["material_id"]
                )
                results["property_comparison"] = comparison

        return results

    def _calculate_scores(self, stages: Dict[str, Any]) -> Dict[str, float]:
        """Calculate validation scores."""
        scores = {
            "structure": 0.0,
            "properties": 0.0,
            "synthesis": 0.0,
            "stability": 0.0,
            "novelty": 0.0,
            "overall": 0.0
        }

        # Structure score
        if "structure" in stages:
            scores["structure"] = 1.0 if stages["structure"]["valid"] else 0.0

        # Properties score
        if "properties" in stages:
            scores["properties"] = 1.0 if stages["properties"]["plausible"] else 0.5
            if stages["properties"]["warnings"]:
                scores["properties"] -= 0.1 * len(stages["properties"]["warnings"])

        # Synthesis score
        if "synthesis" in stages:
            scores["synthesis"] = stages["synthesis"].get("score", 0.0)

        # Stability score
        if "phase_stability" in stages:
            scores["stability"] = 1.0 if stages["phase_stability"]["stable"] else 0.3

        # Novelty score
        if "database" in stages:
            scores["novelty"] = 1.0 if stages["database"]["novel"] else 0.3

        # Calculate overall score (weighted average)
        weights = {"structure": 0.25, "properties": 0.25, "synthesis": 0.2, "stability": 0.2, "novelty": 0.1}
        total_weight = sum(weights[k] for k in scores if k != "overall" and scores[k] > 0)

        if total_weight > 0:
            scores["overall"] = sum(
                weights.get(k, 0) * v for k, v in scores.items() if k != "overall"
            ) / total_weight

        return scores

    def _generate_recommendation(self, scores: Dict[str, float]) -> str:
        """Generate recommendation based on scores."""
        overall = scores["overall"]

        if overall >= 0.8:
            return "STRONGLY RECOMMENDED: High confidence in hypothesis validity"
        elif overall >= 0.6:
            return "RECOMMENDED: Hypothesis appears valid with minor concerns"
        elif overall >= 0.4:
            return "CONDITIONAL: Requires further validation and refinement"
        else:
            return "NOT RECOMMENDED: Significant issues identified"

    def _compare_properties(
        self,
        claimed: MaterialProperties,
        predicted: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare claimed vs predicted properties."""
        comparison = {
            "major_discrepancies": [],
            "minor_discrepancies": [],
            "agreements": []
        }

        # Compare mechanical properties
        if claimed.mechanical and "mechanical" in predicted:
            for prop in ["bulk_modulus", "youngs_modulus", "hardness_vickers"]:
                claimed_val = getattr(claimed.mechanical, prop, None)
                predicted_val = predicted["mechanical"].get(prop)

                if claimed_val and predicted_val:
                    diff_percent = abs(claimed_val - predicted_val) / predicted_val * 100
                    if diff_percent > 50:
                        comparison["major_discrepancies"].append(
                            f"{prop}: {diff_percent:.1f}% difference"
                        )
                    elif diff_percent > 20:
                        comparison["minor_discrepancies"].append(
                            f"{prop}: {diff_percent:.1f}% difference"
                        )
                    else:
                        comparison["agreements"].append(prop)

        # Similar comparisons for electrical, thermal, magnetic properties...

        return comparison

    def _get_cache_key(self, hypothesis: MaterialsHypothesis) -> str:
        """Generate cache key for hypothesis."""
        key_data = f"{hypothesis.hypothesis_id}_{hypothesis.claim}"
        if hypothesis.predicted_structure:
            key_data += f"_{hypothesis.predicted_structure.formula}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def generate_research_plan(
        self,
        hypothesis: MaterialsHypothesis
    ) -> Dict[str, Any]:
        """
        Generate comprehensive research plan for hypothesis.

        Args:
            hypothesis: Materials hypothesis to investigate

        Returns:
            Research plan with experiments and milestones
        """
        plan = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "phases": [],
            "timeline_months": 0,
            "required_resources": [],
            "milestones": []
        }

        # Phase 1: Computational validation
        computational_phase = {
            "name": "Computational Validation",
            "duration_months": 2,
            "tasks": [
                "DFT structure optimization",
                "Band structure calculations",
                "Phonon calculations for stability",
                "Molecular dynamics simulations"
            ]
        }
        plan["phases"].append(computational_phase)

        # Phase 2: Synthesis
        if hypothesis.synthesis_route:
            synthesis_phase = {
                "name": "Material Synthesis",
                "duration_months": 3,
                "tasks": [
                    f"Synthesize via {hypothesis.synthesis_route.route_name}",
                    "Optimize synthesis parameters",
                    "Scale-up feasibility study"
                ]
            }
            plan["phases"].append(synthesis_phase)

        # Phase 3: Characterization
        characterization_phase = {
            "name": "Characterization",
            "duration_months": 2,
            "tasks": [
                "XRD for structure confirmation",
                "SEM/TEM for morphology",
                "XPS for composition",
                "Property measurements"
            ]
        }
        plan["phases"].append(characterization_phase)

        # Calculate total timeline
        plan["timeline_months"] = sum(p["duration_months"] for p in plan["phases"])

        # Define milestones
        cumulative_months = 0
        for phase in plan["phases"]:
            cumulative_months += phase["duration_months"]
            plan["milestones"].append({
                "month": cumulative_months,
                "deliverable": f"Complete {phase['name']}"
            })

        return plan