"""
External API integrations for Materials Science Research.

Interfaces with Materials Project API and provides periodic table reasoning.
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime

from .models import CrystalStructure, MaterialProperties, AtomicPosition


class MaterialsProjectAPI:
    """Interface to Materials Project database."""

    BASE_URL = "https://api.materialsproject.org"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Materials Project API client.

        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={"X-API-KEY": self.api_key} if self.api_key else {}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def find_similar_structures(
        self,
        structure: CrystalStructure,
        tolerance: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Find similar structures in Materials Project database.

        Args:
            structure: Query crystal structure
            tolerance: Structure matching tolerance

        Returns:
            List of similar materials
        """
        # Simulated API call - in production would use actual MP API
        await asyncio.sleep(0.1)

        # Mock similar materials
        similar = []

        # Check for common formulas
        if "Fe" in structure.formula and "O" in structure.formula:
            similar.append({
                "material_id": "mp-19770",
                "formula": "Fe2O3",
                "spacegroup": "R-3c",
                "formation_energy_per_atom": -1.65,
                "band_gap": 2.1,
                "similarity_score": 0.85
            })

        if "Ti" in structure.formula and "O" in structure.formula:
            similar.append({
                "material_id": "mp-2657",
                "formula": "TiO2",
                "spacegroup": "P42/mnm",
                "formation_energy_per_atom": -3.03,
                "band_gap": 3.0,
                "similarity_score": 0.78
            })

        return similar

    async def get_material_properties(
        self,
        material_id: str
    ) -> Dict[str, Any]:
        """
        Get material properties from Materials Project.

        Args:
            material_id: Materials Project ID

        Returns:
            Material properties dictionary
        """
        await asyncio.sleep(0.1)

        # Mock properties data
        properties = {
            "material_id": material_id,
            "formula": "Fe2O3",
            "bulk_modulus": 230.0,  # GPa
            "shear_modulus": 85.0,
            "band_gap": 2.1,  # eV
            "formation_energy": -1.65,  # eV/atom
            "density": 5.24,  # g/cm³
            "volume": 302.72,  # Å³
        }

        return properties

    async def compare_properties(
        self,
        predicted: MaterialProperties,
        material_id: str
    ) -> Dict[str, Any]:
        """
        Compare predicted properties with database material.

        Args:
            predicted: Predicted properties
            material_id: Materials Project ID for comparison

        Returns:
            Property comparison results
        """
        # Get database properties
        db_props = await self.get_material_properties(material_id)

        comparison = {
            "material_id": material_id,
            "differences": {},
            "agreement_score": 0.0
        }

        # Compare mechanical properties
        if predicted.mechanical:
            if predicted.mechanical.bulk_modulus and "bulk_modulus" in db_props:
                diff = abs(predicted.mechanical.bulk_modulus - db_props["bulk_modulus"])
                comparison["differences"]["bulk_modulus"] = {
                    "predicted": predicted.mechanical.bulk_modulus,
                    "database": db_props["bulk_modulus"],
                    "difference": diff,
                    "percent_diff": diff / db_props["bulk_modulus"] * 100
                }

        # Compare electrical properties
        if predicted.electrical:
            if predicted.electrical.band_gap and "band_gap" in db_props:
                diff = abs(predicted.electrical.band_gap - db_props["band_gap"])
                comparison["differences"]["band_gap"] = {
                    "predicted": predicted.electrical.band_gap,
                    "database": db_props["band_gap"],
                    "difference": diff,
                    "percent_diff": diff / db_props["band_gap"] * 100 if db_props["band_gap"] > 0 else 0
                }

        # Calculate agreement score
        if comparison["differences"]:
            percent_diffs = [
                v["percent_diff"] for v in comparison["differences"].values()
            ]
            avg_diff = sum(percent_diffs) / len(percent_diffs)
            comparison["agreement_score"] = max(0, 100 - avg_diff) / 100

        return comparison

    async def search_by_formula(
        self,
        formula: str,
        properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search materials by chemical formula.

        Args:
            formula: Chemical formula
            properties: List of properties to retrieve

        Returns:
            List of matching materials
        """
        await asyncio.sleep(0.1)

        # Mock search results
        results = []

        if formula == "Fe2O3":
            results.append({
                "material_id": "mp-19770",
                "formula": "Fe2O3",
                "spacegroup": "R-3c",
                "formation_energy": -1.65,
                "is_stable": True
            })

        return results

    async def get_phase_diagram(
        self,
        chemsys: str
    ) -> Dict[str, Any]:
        """
        Get phase diagram data for chemical system.

        Args:
            chemsys: Chemical system (e.g., "Fe-O", "Li-Fe-O")

        Returns:
            Phase diagram data
        """
        await asyncio.sleep(0.1)

        # Mock phase diagram data
        return {
            "chemsys": chemsys,
            "stable_phases": [
                {"formula": "Fe", "formation_energy": 0.0},
                {"formula": "FeO", "formation_energy": -1.2},
                {"formula": "Fe2O3", "formation_energy": -1.65},
                {"formula": "Fe3O4", "formation_energy": -1.45}
            ],
            "unstable_phases": []
        }


class PeriodicTableReasoner:
    """Periodic table-aware reasoning for materials science."""

    def __init__(self):
        """Initialize periodic table data."""
        self._init_periodic_table()

    def _init_periodic_table(self):
        """Initialize periodic table properties."""
        self.elements = {
            "H": {"number": 1, "mass": 1.008, "radius": 0.53, "electronegativity": 2.20},
            "Li": {"number": 3, "mass": 6.94, "radius": 1.67, "electronegativity": 0.98},
            "Be": {"number": 4, "mass": 9.01, "radius": 1.12, "electronegativity": 1.57},
            "B": {"number": 5, "mass": 10.81, "radius": 0.87, "electronegativity": 2.04},
            "C": {"number": 6, "mass": 12.01, "radius": 0.67, "electronegativity": 2.55},
            "N": {"number": 7, "mass": 14.01, "radius": 0.56, "electronegativity": 3.04},
            "O": {"number": 8, "mass": 16.00, "radius": 0.48, "electronegativity": 3.44},
            "F": {"number": 9, "mass": 19.00, "radius": 0.42, "electronegativity": 3.98},
            "Na": {"number": 11, "mass": 22.99, "radius": 1.90, "electronegativity": 0.93},
            "Mg": {"number": 12, "mass": 24.31, "radius": 1.45, "electronegativity": 1.31},
            "Al": {"number": 13, "mass": 26.98, "radius": 1.18, "electronegativity": 1.61},
            "Si": {"number": 14, "mass": 28.09, "radius": 1.11, "electronegativity": 1.90},
            "P": {"number": 15, "mass": 30.97, "radius": 0.98, "electronegativity": 2.19},
            "S": {"number": 16, "mass": 32.07, "radius": 0.88, "electronegativity": 2.58},
            "Cl": {"number": 17, "mass": 35.45, "radius": 0.79, "electronegativity": 3.16},
            "K": {"number": 19, "mass": 39.10, "radius": 2.43, "electronegativity": 0.82},
            "Ca": {"number": 20, "mass": 40.08, "radius": 1.94, "electronegativity": 1.00},
            "Ti": {"number": 22, "mass": 47.87, "radius": 1.76, "electronegativity": 1.54},
            "V": {"number": 23, "mass": 50.94, "radius": 1.71, "electronegativity": 1.63},
            "Cr": {"number": 24, "mass": 52.00, "radius": 1.66, "electronegativity": 1.66},
            "Mn": {"number": 25, "mass": 54.94, "radius": 1.61, "electronegativity": 1.55},
            "Fe": {"number": 26, "mass": 55.85, "radius": 1.56, "electronegativity": 1.83},
            "Co": {"number": 27, "mass": 58.93, "radius": 1.52, "electronegativity": 1.88},
            "Ni": {"number": 28, "mass": 58.69, "radius": 1.49, "electronegativity": 1.91},
            "Cu": {"number": 29, "mass": 63.55, "radius": 1.45, "electronegativity": 1.90},
            "Zn": {"number": 30, "mass": 65.38, "radius": 1.42, "electronegativity": 1.65},
        }

        # Oxidation states
        self.oxidation_states = {
            "H": [-1, 1],
            "Li": [1],
            "O": [-2],
            "F": [-1],
            "Na": [1],
            "Mg": [2],
            "Al": [3],
            "Si": [4, -4],
            "Fe": [2, 3],
            "Co": [2, 3],
            "Ni": [2],
            "Cu": [1, 2],
            "Zn": [2]
        }

    def get_element_properties(self, element: str) -> Dict[str, Any]:
        """Get properties for element."""
        return self.elements.get(element, {})

    def get_electronegativity_difference(self, element1: str, element2: str) -> float:
        """Calculate electronegativity difference."""
        en1 = self.elements.get(element1, {}).get("electronegativity", 2.0)
        en2 = self.elements.get(element2, {}).get("electronegativity", 2.0)
        return abs(en1 - en2)

    def predict_bond_type(self, element1: str, element2: str) -> str:
        """Predict bond type based on electronegativity."""
        diff = self.get_electronegativity_difference(element1, element2)

        if diff < 0.5:
            return "covalent"
        elif diff < 1.7:
            return "polar_covalent"
        else:
            return "ionic"

    def get_oxidation_states(self, element: str) -> List[int]:
        """Get possible oxidation states."""
        return self.oxidation_states.get(element, [0])

    def extract_elements(self, formula: str) -> List[str]:
        """Extract elements from chemical formula."""
        elements = []
        current = ""

        for char in formula:
            if char.isupper():
                if current:
                    elements.append(current)
                current = char
            elif char.islower():
                current += char
            elif char.isdigit() or char in "()[]":
                if current and current not in elements:
                    elements.append(current)
                current = ""

        if current:
            elements.append(current)

        return elements

    def validate_formula(self, formula: str) -> Tuple[bool, str]:
        """
        Validate chemical formula.

        Returns:
            (is_valid, error_message)
        """
        elements = self.extract_elements(formula)

        for element in elements:
            if element not in self.elements:
                return False, f"Unknown element: {element}"

        # Check charge balance (simplified)
        # In production, would parse stoichiometry and check oxidation states

        return True, ""

    def get_element_group(self, element: str) -> str:
        """Get element group (metal, nonmetal, etc.)."""
        atomic_num = self.elements.get(element, {}).get("number", 0)

        if atomic_num in [1]:  # H
            return "nonmetal"
        elif atomic_num in [3, 11, 19, 37, 55, 87]:  # Alkali metals
            return "alkali_metal"
        elif atomic_num in [4, 12, 20, 38, 56, 88]:  # Alkaline earth
            return "alkaline_earth"
        elif 21 <= atomic_num <= 30 or 39 <= atomic_num <= 48:  # Transition metals
            return "transition_metal"
        elif atomic_num in [5, 14, 32, 33, 51, 52]:  # Metalloids
            return "metalloid"
        elif atomic_num in [6, 7, 8, 9, 15, 16, 17, 34, 35, 53]:  # Nonmetals
            return "nonmetal"
        else:
            return "unknown"

    def predict_crystal_system(self, elements: List[str]) -> str:
        """Predict likely crystal system based on elements."""
        # Simple heuristic based on element types
        groups = [self.get_element_group(e) for e in elements]

        if all(g == "transition_metal" for g in groups):
            return "cubic"  # Metals often cubic
        elif "nonmetal" in groups and "alkali_metal" in groups:
            return "cubic"  # Ionic compounds often cubic
        else:
            return "orthorhombic"  # Default for complex

    def calculate_tolerance_factor(
        self,
        a_element: str,  # A-site cation
        b_element: str,  # B-site cation
        x_element: str   # Anion
    ) -> float:
        """
        Calculate Goldschmidt tolerance factor for ABX3 perovskites.

        t = (r_A + r_X) / sqrt(2) * (r_B + r_X)
        """
        r_a = self.elements.get(a_element, {}).get("radius", 1.5)
        r_b = self.elements.get(b_element, {}).get("radius", 1.0)
        r_x = self.elements.get(x_element, {}).get("radius", 0.5)

        tolerance = (r_a + r_x) / (2**0.5 * (r_b + r_x))
        return tolerance

    def predict_stability_from_tolerance(self, tolerance_factor: float) -> Dict[str, Any]:
        """Predict structure stability from tolerance factor."""
        result = {
            "tolerance_factor": tolerance_factor,
            "predicted_structure": "",
            "stable": False
        }

        if 0.9 <= tolerance_factor <= 1.0:
            result["predicted_structure"] = "cubic_perovskite"
            result["stable"] = True
        elif 0.8 <= tolerance_factor < 0.9:
            result["predicted_structure"] = "distorted_perovskite"
            result["stable"] = True
        elif tolerance_factor < 0.8:
            result["predicted_structure"] = "ilmenite_or_other"
            result["stable"] = False
        else:  # > 1.0
            result["predicted_structure"] = "hexagonal_or_other"
            result["stable"] = False

        return result

    def get_shannon_radius(
        self,
        element: str,
        oxidation_state: int,
        coordination: int
    ) -> float:
        """
        Get Shannon ionic radius.

        Args:
            element: Element symbol
            oxidation_state: Oxidation state
            coordination: Coordination number

        Returns:
            Ionic radius in Angstroms
        """
        # Simplified Shannon radii database
        # Format: (element, oxidation, coordination) -> radius
        shannon_radii = {
            ("Li", 1, 4): 0.59,
            ("Li", 1, 6): 0.76,
            ("Na", 1, 6): 1.02,
            ("K", 1, 6): 1.38,
            ("Mg", 2, 6): 0.72,
            ("Ca", 2, 6): 1.00,
            ("Fe", 2, 6): 0.78,
            ("Fe", 3, 6): 0.65,
            ("Co", 2, 6): 0.75,
            ("Ni", 2, 6): 0.69,
            ("Cu", 2, 6): 0.73,
            ("Zn", 2, 4): 0.60,
            ("Zn", 2, 6): 0.74,
            ("O", -2, 2): 1.35,
            ("O", -2, 4): 1.38,
            ("O", -2, 6): 1.40,
        }

        return shannon_radii.get((element, oxidation_state, coordination), 1.0)

    def predict_coordination_preference(
        self,
        element: str,
        oxidation_state: Optional[int] = None
    ) -> int:
        """Predict preferred coordination number."""
        # Use oxidation state if provided
        if oxidation_state is None:
            ox_states = self.get_oxidation_states(element)
            oxidation_state = ox_states[0] if ox_states else 0

        # Simple rules based on size and charge
        if element in ["Li", "Be", "B"]:
            return 4
        elif element in ["Na", "Mg", "Al", "Si", "Fe", "Co", "Ni"]:
            return 6
        elif element in ["K", "Ca", "Sr", "Ba"]:
            return 8
        elif element == "O":
            return 2
        else:
            return 6  # Default

    def calculate_madelung_constant(self, structure_type: str) -> float:
        """Get Madelung constant for structure type."""
        madelung_constants = {
            "rock_salt": 1.748,
            "cesium_chloride": 1.763,
            "zinc_blende": 1.638,
            "wurtzite": 1.641,
            "fluorite": 2.519,
            "rutile": 2.408,
            "perovskite": 1.645
        }
        return madelung_constants.get(structure_type, 1.7)