"""
Domain-specific analyzers for Materials Science Research.

Implements crystal structure prediction, property prediction, synthesis planning,
phase diagram generation, and DFT workflow management.
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import math

from .models import (
    CrystalStructure,
    MaterialProperties,
    SynthesisRoute,
    ReactionStep,
    PhaseDiagram,
    PhaseRegion,
    DFTCalculation,
    SpaceGroup,
    AtomicPosition,
    MechanicalProperties,
    ElectricalProperties,
)


class StructurePredictor:
    """Crystal structure prediction and validation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize structure predictor."""
        self.config = config or {}
        self.tolerance = self.config.get("tolerance", 0.01)

    async def predict_structure(
        self,
        formula: str,
        method: str = "evolutionary"
    ) -> CrystalStructure:
        """
        Predict crystal structure from chemical formula.

        Args:
            formula: Chemical formula
            method: Prediction method (evolutionary, particle_swarm, simulated_annealing)

        Returns:
            Predicted crystal structure
        """
        # Simulated structure prediction
        # In production, this would interface with USPEX, CALYPSO, or similar
        await asyncio.sleep(0.1)  # Simulate computation

        # Generate plausible structure based on formula
        elements = self._parse_formula(formula)

        # Determine likely space group based on composition
        space_group = self._predict_space_group(elements)

        # Generate atomic positions
        positions = self._generate_positions(elements, space_group)

        # Predict lattice parameters
        lattice = self._predict_lattice_parameters(elements, space_group)

        structure = CrystalStructure(
            formula=formula,
            lattice_type=space_group.crystal_system,
            space_group=space_group,
            lattice_parameters=lattice,
            atomic_positions=positions,
            volume=self._calculate_volume(lattice),
            density=self._calculate_density(formula, lattice)
        )

        return structure

    async def validate_space_group(self, structure: CrystalStructure) -> bool:
        """Validate space group consistency."""
        # Check if lattice parameters are consistent with space group symmetry
        lattice = structure.lattice_parameters
        space_group = structure.space_group

        if space_group.crystal_system.value == "cubic":
            # Cubic requires a = b = c and α = β = γ = 90°
            return (
                abs(lattice["a"] - lattice["b"]) < self.tolerance and
                abs(lattice["b"] - lattice["c"]) < self.tolerance and
                all(abs(lattice[angle] - 90) < self.tolerance for angle in ["alpha", "beta", "gamma"])
            )
        elif space_group.crystal_system.value == "tetragonal":
            # Tetragonal requires a = b ≠ c and α = β = γ = 90°
            return (
                abs(lattice["a"] - lattice["b"]) < self.tolerance and
                abs(lattice["a"] - lattice["c"]) > self.tolerance and
                all(abs(lattice[angle] - 90) < self.tolerance for angle in ["alpha", "beta", "gamma"])
            )
        # Add more crystal system checks...

        return True

    async def check_atomic_positions(self, structure: CrystalStructure) -> List[str]:
        """Check for issues in atomic positions."""
        issues = []

        # Check for overlapping atoms
        for i, pos1 in enumerate(structure.atomic_positions):
            for j, pos2 in enumerate(structure.atomic_positions[i+1:], i+1):
                distance = self._calculate_distance(pos1, pos2, structure.lattice_parameters)
                min_distance = self._get_min_bond_distance(pos1.element, pos2.element)

                if distance < min_distance * 0.8:
                    issues.append(
                        f"Atoms {i} ({pos1.element}) and {j} ({pos2.element}) "
                        f"too close: {distance:.2f} Å"
                    )

        # Check for unreasonable coordination
        coordination = await self.get_coordination_numbers(structure)
        for atom_idx, cn in coordination.items():
            element = structure.atomic_positions[atom_idx].element
            expected_cn = self._get_expected_coordination(element)

            if abs(cn - expected_cn) > 2:
                issues.append(
                    f"Unusual coordination for atom {atom_idx} ({element}): "
                    f"{cn} (expected ~{expected_cn})"
                )

        return issues

    async def calculate_packing_fraction(self, structure: CrystalStructure) -> float:
        """Calculate packing fraction of structure."""
        # Calculate total volume of atoms (assuming spherical)
        atom_volume = 0.0
        for pos in structure.atomic_positions:
            radius = self._get_atomic_radius(pos.element)
            atom_volume += (4/3) * math.pi * radius**3 * pos.occupancy

        # Get unit cell volume
        cell_volume = structure.volume or self._calculate_volume(structure.lattice_parameters)

        return atom_volume / cell_volume

    async def get_coordination_numbers(self, structure: CrystalStructure) -> Dict[int, int]:
        """Calculate coordination numbers for each atom."""
        coordination = {}

        for i, atom in enumerate(structure.atomic_positions):
            neighbors = 0
            for j, other in enumerate(structure.atomic_positions):
                if i != j:
                    distance = self._calculate_distance(atom, other, structure.lattice_parameters)
                    cutoff = self._get_bond_cutoff(atom.element, other.element)

                    if distance <= cutoff:
                        neighbors += 1

            coordination[i] = neighbors

        return coordination

    async def calculate_bond_lengths(self, structure: CrystalStructure) -> Dict[str, List[float]]:
        """Calculate bond lengths in structure."""
        bonds = {}

        for i, atom1 in enumerate(structure.atomic_positions):
            for j, atom2 in enumerate(structure.atomic_positions[i+1:], i+1):
                distance = self._calculate_distance(atom1, atom2, structure.lattice_parameters)
                cutoff = self._get_bond_cutoff(atom1.element, atom2.element)

                if distance <= cutoff:
                    bond_type = f"{atom1.element}-{atom2.element}"
                    if bond_type not in bonds:
                        bonds[bond_type] = []
                    bonds[bond_type].append(distance)

        return bonds

    async def identify_structure_type(self, structure: CrystalStructure) -> str:
        """Identify common structure type (e.g., perovskite, spinel, etc.)."""
        formula = structure.formula
        space_group = structure.space_group.number

        # Check for common structure types
        if self._is_perovskite(formula, space_group):
            return "perovskite"
        elif self._is_spinel(formula, space_group):
            return "spinel"
        elif self._is_rock_salt(formula, space_group):
            return "rock salt"
        elif self._is_wurtzite(formula, space_group):
            return "wurtzite"
        else:
            return "unknown"

    def _parse_formula(self, formula: str) -> Dict[str, int]:
        """Parse chemical formula to element counts."""
        # Simplified parser - production would use proper regex
        elements = {}
        current_element = ""
        current_count = ""

        for char in formula:
            if char.isupper():
                if current_element:
                    count = int(current_count) if current_count else 1
                    elements[current_element] = elements.get(current_element, 0) + count
                current_element = char
                current_count = ""
            elif char.islower():
                current_element += char
            elif char.isdigit():
                current_count += char

        if current_element:
            count = int(current_count) if current_count else 1
            elements[current_element] = elements.get(current_element, 0) + count

        return elements

    def _predict_space_group(self, elements: Dict[str, int]) -> SpaceGroup:
        """Predict likely space group based on composition."""
        # Simplified prediction - production would use ML model
        num_elements = len(elements)

        if num_elements == 1:
            # Elemental solid
            return SpaceGroup(
                number=225,  # Fm-3m (FCC)
                symbol="Fm-3m",
                point_group="m-3m",
                crystal_system="cubic"
            )
        elif num_elements == 2:
            # Binary compound
            return SpaceGroup(
                number=225,  # Common for binary compounds
                symbol="Fm-3m",
                point_group="m-3m",
                crystal_system="cubic"
            )
        else:
            # Complex compound
            return SpaceGroup(
                number=62,  # Pnma (common for complex oxides)
                symbol="Pnma",
                point_group="mmm",
                crystal_system="orthorhombic"
            )

    def _generate_positions(
        self,
        elements: Dict[str, int],
        space_group: SpaceGroup
    ) -> List[AtomicPosition]:
        """Generate atomic positions based on space group."""
        positions = []

        # Simplified - production would use proper Wyckoff positions
        atom_count = 0
        for element, count in elements.items():
            for i in range(count):
                positions.append(AtomicPosition(
                    element=element,
                    x=0.25 * (i % 4),
                    y=0.25 * ((i // 4) % 4),
                    z=0.25 * (i // 16),
                    occupancy=1.0
                ))
                atom_count += 1

        return positions

    def _predict_lattice_parameters(
        self,
        elements: Dict[str, int],
        space_group: SpaceGroup
    ) -> Dict[str, float]:
        """Predict lattice parameters."""
        # Estimate based on atomic radii
        avg_radius = np.mean([self._get_atomic_radius(e) for e in elements.keys()])

        if space_group.crystal_system.value == "cubic":
            a = avg_radius * 4.0  # Rough estimate
            return {
                "a": a, "b": a, "c": a,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
            }
        else:
            # Default orthorhombic
            return {
                "a": avg_radius * 4.0,
                "b": avg_radius * 3.8,
                "c": avg_radius * 4.2,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
            }

    def _calculate_volume(self, lattice: Dict[str, float]) -> float:
        """Calculate unit cell volume."""
        a, b, c = lattice["a"], lattice["b"], lattice["c"]
        alpha = math.radians(lattice["alpha"])
        beta = math.radians(lattice["beta"])
        gamma = math.radians(lattice["gamma"])

        volume = a * b * c * math.sqrt(
            1 - math.cos(alpha)**2 - math.cos(beta)**2 - math.cos(gamma)**2
            + 2 * math.cos(alpha) * math.cos(beta) * math.cos(gamma)
        )

        return volume

    def _calculate_density(self, formula: str, lattice: Dict[str, float]) -> float:
        """Calculate density in g/cm³."""
        # Simplified calculation
        elements = self._parse_formula(formula)
        mass = sum(self._get_atomic_mass(e) * n for e, n in elements.items())
        volume = self._calculate_volume(lattice) * 1e-24  # Å³ to cm³

        return mass / (6.022e23 * volume)  # g/cm³

    def _calculate_distance(
        self,
        pos1: AtomicPosition,
        pos2: AtomicPosition,
        lattice: Dict[str, float]
    ) -> float:
        """Calculate distance between two atoms."""
        # Convert fractional to Cartesian coordinates
        cart1 = self._frac_to_cart(pos1, lattice)
        cart2 = self._frac_to_cart(pos2, lattice)

        return np.linalg.norm(np.array(cart1) - np.array(cart2))

    def _frac_to_cart(
        self,
        pos: AtomicPosition,
        lattice: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """Convert fractional to Cartesian coordinates."""
        # Simplified for orthorhombic
        x = pos.x * lattice["a"]
        y = pos.y * lattice["b"]
        z = pos.z * lattice["c"]
        return (x, y, z)

    def _get_atomic_radius(self, element: str) -> float:
        """Get atomic radius in Angstroms."""
        # Simplified lookup - production would use comprehensive database
        radii = {
            "H": 0.53, "Li": 1.67, "C": 0.67, "N": 0.56, "O": 0.48,
            "Fe": 1.56, "Co": 1.52, "Ni": 1.49, "Cu": 1.45, "Zn": 1.42
        }
        return radii.get(element, 1.5)

    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass in amu."""
        masses = {
            "H": 1.008, "Li": 6.94, "C": 12.01, "N": 14.01, "O": 16.00,
            "Fe": 55.85, "Co": 58.93, "Ni": 58.69, "Cu": 63.55, "Zn": 65.38
        }
        return masses.get(element, 50.0)

    def _get_min_bond_distance(self, element1: str, element2: str) -> float:
        """Get minimum reasonable bond distance."""
        r1 = self._get_atomic_radius(element1)
        r2 = self._get_atomic_radius(element2)
        return (r1 + r2) * 0.8

    def _get_bond_cutoff(self, element1: str, element2: str) -> float:
        """Get bond cutoff distance."""
        r1 = self._get_atomic_radius(element1)
        r2 = self._get_atomic_radius(element2)
        return (r1 + r2) * 1.3

    def _get_expected_coordination(self, element: str) -> int:
        """Get expected coordination number."""
        # Simplified - depends on oxidation state and environment
        typical_cn = {
            "Li": 4, "Na": 6, "K": 8,
            "Mg": 6, "Ca": 8,
            "Al": 6, "Si": 4,
            "O": 2, "S": 2,
            "Fe": 6, "Co": 6, "Ni": 6
        }
        return typical_cn.get(element, 6)

    def _is_perovskite(self, formula: str, space_group: int) -> bool:
        """Check if structure is perovskite type."""
        # ABX3 formula and common space groups
        elements = self._parse_formula(formula)
        total = sum(elements.values())

        return (
            total == 5 and  # ABX3 = 5 atoms
            space_group in [62, 140, 221, 225]  # Common perovskite space groups
        )

    def _is_spinel(self, formula: str, space_group: int) -> bool:
        """Check if structure is spinel type."""
        elements = self._parse_formula(formula)
        total = sum(elements.values())

        return (
            total == 7 and  # AB2X4 = 7 atoms
            space_group in [227, 141]  # Spinel space groups
        )

    def _is_rock_salt(self, formula: str, space_group: int) -> bool:
        """Check if structure is rock salt type."""
        elements = self._parse_formula(formula)

        return (
            len(elements) == 2 and
            all(n == 1 for n in elements.values()) and
            space_group == 225
        )

    def _is_wurtzite(self, formula: str, space_group: int) -> bool:
        """Check if structure is wurtzite type."""
        elements = self._parse_formula(formula)

        return (
            len(elements) == 2 and
            all(n == 1 for n in elements.values()) and
            space_group == 186
        )


class PropertyPredictor:
    """Material property prediction from structure."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize property predictor."""
        self.config = config or {}

    async def predict_from_structure(
        self,
        structure: CrystalStructure
    ) -> Dict[str, Any]:
        """
        Predict material properties from crystal structure.

        Args:
            structure: Crystal structure

        Returns:
            Dictionary of predicted properties
        """
        predictions = {}

        # Predict mechanical properties
        predictions["mechanical"] = await self._predict_mechanical(structure)

        # Predict electrical properties
        predictions["electrical"] = await self._predict_electrical(structure)

        # Predict thermal properties
        predictions["thermal"] = await self._predict_thermal(structure)

        # Predict magnetic properties (if applicable)
        if self._contains_magnetic_elements(structure):
            predictions["magnetic"] = await self._predict_magnetic(structure)

        return predictions

    async def _predict_mechanical(self, structure: CrystalStructure) -> Dict[str, float]:
        """Predict mechanical properties."""
        # Simplified predictions based on structure
        density = structure.density or 5.0

        # Estimate based on density and structure type
        bulk_modulus = density * 20.0  # Very rough estimate
        shear_modulus = bulk_modulus * 0.4

        # Calculate derived properties
        youngs_modulus = 9 * bulk_modulus * shear_modulus / (3 * bulk_modulus + shear_modulus)
        poisson_ratio = (3 * bulk_modulus - 2 * shear_modulus) / (6 * bulk_modulus + 2 * shear_modulus)

        return {
            "bulk_modulus": bulk_modulus,
            "shear_modulus": shear_modulus,
            "youngs_modulus": youngs_modulus,
            "poisson_ratio": poisson_ratio,
            "hardness_vickers": bulk_modulus * 0.15  # Rough correlation
        }

    async def _predict_electrical(self, structure: CrystalStructure) -> Dict[str, float]:
        """Predict electrical properties."""
        # Check if metallic based on elements
        elements = self._extract_elements(structure)

        if self._is_likely_metal(elements):
            return {
                "band_gap": 0.0,
                "conductivity": 1e6,  # S/m
                "resistivity": 1e-6,  # Ohm·m
            }
        elif self._is_likely_insulator(elements):
            return {
                "band_gap": 5.0,  # eV
                "conductivity": 1e-10,
                "resistivity": 1e10,
            }
        else:
            # Semiconductor
            return {
                "band_gap": 1.5,
                "conductivity": 1e2,
                "resistivity": 1e-2,
            }

    async def _predict_thermal(self, structure: CrystalStructure) -> Dict[str, float]:
        """Predict thermal properties."""
        # Estimate based on structure
        elements = self._extract_elements(structure)
        avg_mass = np.mean([self._get_atomic_mass(e) for e in elements])

        # Rough estimates
        melting_point = 1000 + avg_mass * 10  # K
        thermal_conductivity = 10.0 if self._is_likely_metal(elements) else 1.0  # W/m·K

        return {
            "melting_point": melting_point,
            "thermal_conductivity": thermal_conductivity,
            "specific_heat": 500.0,  # J/kg·K
            "debye_temperature": melting_point * 0.4
        }

    async def _predict_magnetic(self, structure: CrystalStructure) -> Dict[str, float]:
        """Predict magnetic properties."""
        magnetic_elements = ["Fe", "Co", "Ni", "Mn", "Cr"]
        elements = self._extract_elements(structure)

        magnetic_content = sum(1 for e in elements if e in magnetic_elements)

        if magnetic_content > 0:
            return {
                "magnetic_moment": magnetic_content * 2.0,  # μB
                "curie_temperature": 300 + magnetic_content * 100,  # K
            }

        return {}

    async def check_property_relationships(
        self,
        mechanical: MechanicalProperties
    ) -> Dict[str, Any]:
        """Check physical relationships between properties."""
        results = {"valid": True, "violations": []}

        if mechanical.bulk_modulus and mechanical.shear_modulus:
            # Check Pugh's ratio (B/G)
            pugh_ratio = mechanical.bulk_modulus / mechanical.shear_modulus
            if pugh_ratio < 0.5 or pugh_ratio > 10:
                results["violations"].append(
                    f"Unusual Pugh's ratio: {pugh_ratio:.2f}"
                )

            # Check Poisson's ratio bounds (-1 to 0.5)
            if mechanical.poisson_ratio:
                if mechanical.poisson_ratio < -1 or mechanical.poisson_ratio > 0.5:
                    results["violations"].append(
                        f"Invalid Poisson's ratio: {mechanical.poisson_ratio:.3f}"
                    )

        if results["violations"]:
            results["valid"] = False

        return results

    async def check_empirical_bounds(
        self,
        properties: MaterialProperties
    ) -> Dict[str, Any]:
        """Check if properties are within empirical bounds."""
        results = {"out_of_bounds": []}

        if properties.mechanical:
            # Check hardness bounds
            if properties.mechanical.hardness_vickers:
                if properties.mechanical.hardness_vickers > 100:  # GPa
                    results["out_of_bounds"].append(
                        "Hardness exceeds diamond (100 GPa)"
                    )

        if properties.electrical:
            # Check band gap bounds
            if properties.electrical.band_gap:
                if properties.electrical.band_gap > 15:  # eV
                    results["out_of_bounds"].append(
                        "Band gap exceeds known insulators"
                    )

        return results

    def _extract_elements(self, structure: CrystalStructure) -> List[str]:
        """Extract unique elements from structure."""
        return list(set(pos.element for pos in structure.atomic_positions))

    def _is_likely_metal(self, elements: List[str]) -> bool:
        """Check if likely metallic."""
        metals = ["Fe", "Co", "Ni", "Cu", "Al", "Ag", "Au", "Pt"]
        return any(e in metals for e in elements)

    def _is_likely_insulator(self, elements: List[str]) -> bool:
        """Check if likely insulator."""
        # Presence of oxygen often indicates insulator/semiconductor
        return "O" in elements and len(elements) > 1

    def _contains_magnetic_elements(self, structure: CrystalStructure) -> bool:
        """Check if structure contains magnetic elements."""
        magnetic = ["Fe", "Co", "Ni", "Mn", "Cr", "Gd", "Tb", "Dy"]
        elements = self._extract_elements(structure)
        return any(e in magnetic for e in elements)

    def _get_atomic_mass(self, element: str) -> float:
        """Get atomic mass."""
        masses = {
            "H": 1.008, "C": 12.01, "N": 14.01, "O": 16.00,
            "Fe": 55.85, "Co": 58.93, "Ni": 58.69
        }
        return masses.get(element, 50.0)


class RetrosynthesisPlanner:
    """Synthesis route planning using retrosynthetic analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize synthesis planner."""
        self.config = config or {}

    async def plan_synthesis(
        self,
        target_formula: str,
        max_steps: int = 5
    ) -> List[SynthesisRoute]:
        """
        Plan synthesis routes for target material.

        Args:
            target_formula: Target material formula
            max_steps: Maximum synthesis steps

        Returns:
            List of possible synthesis routes
        """
        routes = []

        # Generate different synthesis strategies
        routes.append(await self._solid_state_route(target_formula))
        routes.append(await self._sol_gel_route(target_formula))

        if self._is_oxide(target_formula):
            routes.append(await self._combustion_route(target_formula))

        # Score and sort routes
        for route in routes:
            route.difficulty_score = await self._score_difficulty(route)

        routes.sort(key=lambda r: r.difficulty_score)

        return routes

    async def _solid_state_route(self, formula: str) -> SynthesisRoute:
        """Generate solid-state synthesis route."""
        steps = []

        # Step 1: Mixing precursors
        steps.append(ReactionStep(
            step_number=1,
            reactants=self._get_precursors(formula),
            products=[f"{formula}_mixture"],
            conditions={"method": "ball milling", "time": "2 hours"},
            time=2.0
        ))

        # Step 2: Calcination
        steps.append(ReactionStep(
            step_number=2,
            reactants=[f"{formula}_mixture"],
            products=[formula],
            conditions={"atmosphere": "air"},
            temperature=1273,  # 1000°C
            time=12.0,
            yield_percentage=85
        ))

        return SynthesisRoute(
            target_material=formula,
            route_name="Solid-State Synthesis",
            steps=steps,
            total_time=14.0,
            difficulty_score=3.0,
            safety_hazards=["High temperature operation"]
        )

    async def _sol_gel_route(self, formula: str) -> SynthesisRoute:
        """Generate sol-gel synthesis route."""
        steps = []

        # Step 1: Sol preparation
        steps.append(ReactionStep(
            step_number=1,
            reactants=self._get_sol_precursors(formula),
            products=[f"{formula}_sol"],
            conditions={"solvent": "ethanol", "pH": "4"},
            temperature=298,
            time=1.0
        ))

        # Step 2: Gelation
        steps.append(ReactionStep(
            step_number=2,
            reactants=[f"{formula}_sol"],
            products=[f"{formula}_gel"],
            conditions={"temperature": "60°C"},
            temperature=333,
            time=24.0
        ))

        # Step 3: Drying and calcination
        steps.append(ReactionStep(
            step_number=3,
            reactants=[f"{formula}_gel"],
            products=[formula],
            conditions={"atmosphere": "air"},
            temperature=773,  # 500°C
            time=4.0,
            yield_percentage=90
        ))

        return SynthesisRoute(
            target_material=formula,
            route_name="Sol-Gel Synthesis",
            steps=steps,
            total_time=29.0,
            difficulty_score=4.0,
            safety_hazards=["Organic solvents"]
        )

    async def _combustion_route(self, formula: str) -> SynthesisRoute:
        """Generate combustion synthesis route."""
        steps = []

        steps.append(ReactionStep(
            step_number=1,
            reactants=self._get_precursors(formula) + ["glycine"],
            products=[formula],
            conditions={"ignition_temp": "350°C"},
            temperature=623,
            time=0.5,
            yield_percentage=95
        ))

        return SynthesisRoute(
            target_material=formula,
            route_name="Combustion Synthesis",
            steps=steps,
            total_time=0.5,
            difficulty_score=2.0,
            safety_hazards=["Exothermic reaction", "Gas evolution"]
        )

    async def check_thermodynamics(self, step: ReactionStep) -> Dict[str, Any]:
        """Check thermodynamic feasibility of reaction step."""
        # Simplified check - production would use thermodynamic databases
        result = {
            "favorable": True,
            "delta_g": -50.0,  # kJ/mol
            "delta_h": -100.0,
            "delta_s": -0.15
        }

        # Check Gibbs free energy at reaction temperature
        if step.temperature:
            delta_g = result["delta_h"] - step.temperature * result["delta_s"]
            result["delta_g"] = delta_g
            result["favorable"] = delta_g < 0

        return result

    async def check_precursor_availability(
        self,
        route: SynthesisRoute
    ) -> Dict[str, bool]:
        """Check availability of precursors."""
        availability = {}

        for step in route.steps:
            for reactant in step.reactants:
                # Check against database of available chemicals
                availability[reactant] = self._is_available(reactant)

        return availability

    async def calculate_feasibility_score(self, route: SynthesisRoute) -> float:
        """Calculate overall feasibility score (0-1)."""
        score = 1.0

        # Penalize for many steps
        score -= 0.1 * len(route.steps)

        # Penalize for long time
        if route.total_time and route.total_time > 24:
            score -= 0.2

        # Penalize for high temperature
        max_temp = max(
            (step.temperature for step in route.steps if step.temperature),
            default=298
        )
        if max_temp > 1273:  # > 1000°C
            score -= 0.2

        # Bonus for high yield
        avg_yield = np.mean([
            step.yield_percentage for step in route.steps
            if step.yield_percentage
        ] or [50])
        score += (avg_yield - 50) / 100 * 0.2

        return max(0, min(1, score))

    async def generate_alternatives(
        self,
        target: str
    ) -> List[SynthesisRoute]:
        """Generate alternative synthesis routes."""
        # Generate multiple routes with variations
        alternatives = []

        # Try different methods
        for method in ["hydrothermal", "microwave", "mechanochemical"]:
            route = await self._generate_route_variant(target, method)
            alternatives.append(route)

        return alternatives

    async def _score_difficulty(self, route: SynthesisRoute) -> float:
        """Score synthesis difficulty (0-10)."""
        score = 0.0

        # Add difficulty for each step
        score += len(route.steps) * 1.0

        # Add for high temperature
        max_temp = max(
            (step.temperature for step in route.steps if step.temperature),
            default=298
        )
        score += (max_temp - 298) / 1000 * 2.0

        # Add for special conditions
        for step in route.steps:
            if "inert" in str(step.conditions).lower():
                score += 1.0
            if "pressure" in step.conditions:
                score += 1.0

        return min(10, score)

    def _get_precursors(self, formula: str) -> List[str]:
        """Get standard precursors for formula."""
        # Simplified - production would use retrosynthetic analysis
        elements = self._parse_formula(formula)
        precursors = []

        for element in elements:
            if element == "O":
                continue  # Often from atmosphere
            elif element in ["Fe", "Co", "Ni"]:
                precursors.append(f"{element}O")
            else:
                precursors.append(f"{element}2O3")

        return precursors

    def _get_sol_precursors(self, formula: str) -> List[str]:
        """Get sol-gel precursors."""
        elements = self._parse_formula(formula)
        precursors = []

        for element in elements:
            if element == "O":
                continue
            precursors.append(f"{element}-alkoxide")

        return precursors

    def _is_oxide(self, formula: str) -> bool:
        """Check if material is an oxide."""
        return "O" in formula

    def _is_available(self, chemical: str) -> bool:
        """Check if chemical is commercially available."""
        # Simplified - would check against supplier databases
        common_chemicals = [
            "Fe2O3", "CoO", "NiO", "Al2O3", "SiO2",
            "glycine", "ethanol", "Fe-alkoxide"
        ]
        return any(chem in chemical for chem in common_chemicals)

    def _parse_formula(self, formula: str) -> List[str]:
        """Extract elements from formula."""
        elements = []
        current = ""

        for char in formula:
            if char.isupper():
                if current:
                    elements.append(current)
                current = char
            elif char.islower():
                current += char
            elif char.isdigit():
                continue

        if current:
            elements.append(current)

        return elements

    async def _generate_route_variant(
        self,
        target: str,
        method: str
    ) -> SynthesisRoute:
        """Generate route variant with specific method."""
        if method == "hydrothermal":
            steps = [
                ReactionStep(
                    step_number=1,
                    reactants=self._get_precursors(target),
                    products=[target],
                    conditions={"method": "hydrothermal", "pressure": "20 bar"},
                    temperature=453,  # 180°C
                    pressure=20,
                    time=24.0,
                    yield_percentage=88
                )
            ]
        elif method == "microwave":
            steps = [
                ReactionStep(
                    step_number=1,
                    reactants=self._get_precursors(target),
                    products=[target],
                    conditions={"method": "microwave", "power": "800W"},
                    temperature=773,
                    time=0.5,
                    yield_percentage=82
                )
            ]
        else:  # mechanochemical
            steps = [
                ReactionStep(
                    step_number=1,
                    reactants=self._get_precursors(target),
                    products=[target],
                    conditions={"method": "ball milling", "speed": "500 rpm"},
                    time=6.0,
                    yield_percentage=75
                )
            ]

        return SynthesisRoute(
            target_material=target,
            route_name=f"{method.capitalize()} Synthesis",
            steps=steps,
            total_time=steps[0].time,
            difficulty_score=5.0,
            safety_hazards=[]
        )


class PhaseDiagramGenerator:
    """Phase diagram generation and analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize phase diagram generator."""
        self.config = config or {}

    async def generate_diagram(
        self,
        elements: List[str],
        temperature_range: Tuple[float, float],
        pressure: float = 1.0
    ) -> PhaseDiagram:
        """Generate phase diagram for system."""
        if len(elements) == 2:
            return await self._generate_binary(elements, temperature_range, pressure)
        elif len(elements) == 3:
            return await self._generate_ternary(elements, temperature_range, pressure)
        else:
            raise ValueError("Only binary and ternary diagrams supported")

    async def _generate_binary(
        self,
        elements: List[str],
        temp_range: Tuple[float, float],
        pressure: float
    ) -> PhaseDiagram:
        """Generate binary phase diagram."""
        # Simplified phase diagram generation
        regions = []

        # Add some typical regions
        regions.append(PhaseRegion(
            composition_range=(0.0, 0.3),
            temperature_range=temp_range,
            phases_present=[elements[0]],
            phase_fractions={elements[0]: 1.0}
        ))

        regions.append(PhaseRegion(
            composition_range=(0.3, 0.7),
            temperature_range=temp_range,
            phases_present=[f"{elements[0]}{elements[1]}"],
            phase_fractions={f"{elements[0]}{elements[1]}": 1.0}
        ))

        regions.append(PhaseRegion(
            composition_range=(0.7, 1.0),
            temperature_range=temp_range,
            phases_present=[elements[1]],
            phase_fractions={elements[1]: 1.0}
        ))

        return PhaseDiagram(
            system=f"{elements[0]}-{elements[1]}",
            diagram_type="binary",
            temperature_range=temp_range,
            pressure=pressure,
            phase_regions=regions,
            invariant_points=[
                {"type": "eutectic", "composition": 0.5, "temperature": temp_range[0] + 200}
            ]
        )

    async def _generate_ternary(
        self,
        elements: List[str],
        temp_range: Tuple[float, float],
        pressure: float
    ) -> PhaseDiagram:
        """Generate ternary phase diagram."""
        # Simplified ternary diagram
        regions = []

        # Add corner regions
        for i, element in enumerate(elements):
            regions.append(PhaseRegion(
                composition_range=(0.7, 1.0),  # Simplified
                temperature_range=temp_range,
                phases_present=[element],
                phase_fractions={element: 1.0}
            ))

        return PhaseDiagram(
            system=f"{'-'.join(elements)}",
            diagram_type="ternary",
            temperature_range=temp_range,
            pressure=pressure,
            phase_regions=regions
        )

    async def check_stability(
        self,
        structure: CrystalStructure
    ) -> Dict[str, Any]:
        """Check thermodynamic stability of structure."""
        # Simplified stability check
        result = {
            "stable": True,
            "formation_energy": -1.5,  # eV/atom
            "decomposition_products": []
        }

        # Check against convex hull (simplified)
        if result["formation_energy"] > 0:
            result["stable"] = False
            # Predict decomposition products
            elements = self._extract_elements(structure)
            result["decomposition_products"] = [
                f"{e}O" for e in elements if e != "O"
            ]

        return result

    def _extract_elements(self, structure: CrystalStructure) -> List[str]:
        """Extract unique elements."""
        return list(set(pos.element for pos in structure.atomic_positions))


class DFTWorkflowManager:
    """DFT calculation workflow management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DFT manager."""
        self.config = config or {}

    async def run_calculation(
        self,
        calc_spec: DFTCalculation
    ) -> DFTCalculation:
        """
        Run DFT calculation (simulated).

        Args:
            calc_spec: Calculation specification

        Returns:
            Calculation with results
        """
        # Simulate DFT calculation
        await asyncio.sleep(0.5)  # Simulate computation time

        # Add simulated results
        calc_spec.total_energy = -100.5 * len(calc_spec.structure.atomic_positions)
        calc_spec.converged = True
        calc_spec.calculation_time = 2.5

        # Simulate band structure for semiconductors/insulators
        elements = self._extract_elements(calc_spec.structure)
        if "O" in elements:  # Likely oxide
            calc_spec.band_structure = {"gap": 2.5}  # eV

        # Simulate forces
        calc_spec.forces = [
            (0.01, -0.02, 0.0) for _ in calc_spec.structure.atomic_positions
        ]

        return calc_spec

    async def calculate_formation_energy(
        self,
        calculation: DFTCalculation
    ) -> float:
        """Calculate formation energy from DFT results."""
        if not calculation.total_energy:
            return 0.0

        # Simplified formation energy calculation
        # E_form = E_total - sum(n_i * E_i)
        elements = self._extract_elements(calculation.structure)
        reference_energies = {
            "O": -5.0, "Fe": -8.0, "Co": -7.5, "Ni": -7.0
        }

        ref_total = sum(
            reference_energies.get(e, -6.0) for e in elements
        )

        formation_energy = (
            calculation.total_energy - ref_total * len(calculation.structure.atomic_positions)
        ) / len(calculation.structure.atomic_positions)

        return formation_energy

    def _extract_elements(self, structure: CrystalStructure) -> List[str]:
        """Extract unique elements."""
        return list(set(pos.element for pos in structure.atomic_positions))

    async def optimize_structure(
        self,
        structure: CrystalStructure,
        max_steps: int = 100
    ) -> CrystalStructure:
        """Optimize crystal structure using DFT."""
        # Simulate structure optimization
        optimized = structure.model_copy()

        # Slightly adjust lattice parameters
        for param in ["a", "b", "c"]:
            optimized.lattice_parameters[param] *= 0.98

        return optimized

    async def calculate_phonons(
        self,
        structure: CrystalStructure
    ) -> Dict[str, Any]:
        """Calculate phonon spectrum."""
        return {
            "stable": True,
            "imaginary_modes": 0,
            "zero_point_energy": 0.05  # eV
        }

    async def run_md_simulation(
        self,
        structure: CrystalStructure,
        temperature: float,
        steps: int = 1000
    ) -> Dict[str, Any]:
        """Run molecular dynamics simulation."""
        return {
            "average_energy": -100.0,
            "temperature_stability": True,
            "rms_displacement": 0.1  # Angstrom
        }