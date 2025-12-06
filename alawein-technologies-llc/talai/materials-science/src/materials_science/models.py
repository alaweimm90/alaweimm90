"""
Pydantic V2 models for Materials Science Research.

Comprehensive data structures for crystal structures, material properties,
synthesis routes, and DFT calculations.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Tuple, Any, Literal
from datetime import datetime
from enum import Enum


class LatticeType(str, Enum):
    """Crystal lattice types."""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"


class SpaceGroup(BaseModel):
    """Crystallographic space group information."""
    model_config = ConfigDict(str_strip_whitespace=True)

    number: int = Field(..., ge=1, le=230, description="Space group number (1-230)")
    symbol: str = Field(..., description="Hermann-Mauguin symbol")
    point_group: str = Field(..., description="Point group symbol")
    crystal_system: LatticeType = Field(..., description="Crystal system")

    @field_validator("number")
    @classmethod
    def validate_space_group_number(cls, v: int) -> int:
        """Validate space group number is in valid range."""
        if not 1 <= v <= 230:
            raise ValueError(f"Space group number must be 1-230, got {v}")
        return v


class AtomicPosition(BaseModel):
    """Atomic position in crystal structure."""
    model_config = ConfigDict(str_strip_whitespace=True)

    element: str = Field(..., description="Chemical element symbol")
    x: float = Field(..., ge=0, le=1, description="Fractional coordinate x")
    y: float = Field(..., ge=0, le=1, description="Fractional coordinate y")
    z: float = Field(..., ge=0, le=1, description="Fractional coordinate z")
    occupancy: float = Field(default=1.0, ge=0, le=1, description="Site occupancy")
    thermal_factor: Optional[float] = Field(None, description="Debye-Waller factor")


class CrystalStructure(BaseModel):
    """Crystal structure representation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    formula: str = Field(..., description="Chemical formula")
    lattice_type: LatticeType = Field(..., description="Lattice type")
    space_group: SpaceGroup = Field(..., description="Space group information")
    lattice_parameters: Dict[str, float] = Field(
        ...,
        description="Lattice parameters (a, b, c in Angstrom, alpha, beta, gamma in degrees)"
    )
    atomic_positions: List[AtomicPosition] = Field(..., description="Atomic positions")
    volume: Optional[float] = Field(None, description="Unit cell volume in Angstrom^3")
    density: Optional[float] = Field(None, description="Density in g/cm^3")

    @field_validator("lattice_parameters")
    @classmethod
    def validate_lattice_params(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate lattice parameters are complete."""
        required = {"a", "b", "c", "alpha", "beta", "gamma"}
        if not required.issubset(v.keys()):
            raise ValueError(f"Missing lattice parameters. Required: {required}")
        return v


class MechanicalProperties(BaseModel):
    """Mechanical properties of materials."""
    model_config = ConfigDict(str_strip_whitespace=True)

    bulk_modulus: Optional[float] = Field(None, description="Bulk modulus in GPa")
    shear_modulus: Optional[float] = Field(None, description="Shear modulus in GPa")
    youngs_modulus: Optional[float] = Field(None, description="Young's modulus in GPa")
    poisson_ratio: Optional[float] = Field(None, description="Poisson's ratio")
    hardness_vickers: Optional[float] = Field(None, description="Vickers hardness in GPa")
    fracture_toughness: Optional[float] = Field(None, description="Fracture toughness in MPa√m")
    yield_strength: Optional[float] = Field(None, description="Yield strength in MPa")
    tensile_strength: Optional[float] = Field(None, description="Ultimate tensile strength in MPa")


class ElectricalProperties(BaseModel):
    """Electrical properties of materials."""
    model_config = ConfigDict(str_strip_whitespace=True)

    band_gap: Optional[float] = Field(None, description="Band gap in eV")
    conductivity: Optional[float] = Field(None, description="Electrical conductivity in S/m")
    resistivity: Optional[float] = Field(None, description="Electrical resistivity in Ohm·m")
    dielectric_constant: Optional[float] = Field(None, description="Dielectric constant")
    work_function: Optional[float] = Field(None, description="Work function in eV")
    carrier_concentration: Optional[float] = Field(None, description="Carrier concentration in cm^-3")
    mobility: Optional[float] = Field(None, description="Carrier mobility in cm^2/V·s")


class ThermalProperties(BaseModel):
    """Thermal properties of materials."""
    model_config = ConfigDict(str_strip_whitespace=True)

    melting_point: Optional[float] = Field(None, description="Melting point in K")
    boiling_point: Optional[float] = Field(None, description="Boiling point in K")
    thermal_conductivity: Optional[float] = Field(None, description="Thermal conductivity in W/m·K")
    specific_heat: Optional[float] = Field(None, description="Specific heat capacity in J/kg·K")
    thermal_expansion: Optional[float] = Field(None, description="Thermal expansion coefficient in 1/K")
    debye_temperature: Optional[float] = Field(None, description="Debye temperature in K")


class MagneticProperties(BaseModel):
    """Magnetic properties of materials."""
    model_config = ConfigDict(str_strip_whitespace=True)

    magnetic_moment: Optional[float] = Field(None, description="Magnetic moment in μB")
    curie_temperature: Optional[float] = Field(None, description="Curie temperature in K")
    neel_temperature: Optional[float] = Field(None, description="Néel temperature in K")
    susceptibility: Optional[float] = Field(None, description="Magnetic susceptibility")
    coercivity: Optional[float] = Field(None, description="Coercivity in Oe")
    saturation_magnetization: Optional[float] = Field(None, description="Saturation magnetization in emu/g")


class MaterialProperties(BaseModel):
    """Comprehensive material properties."""
    model_config = ConfigDict(str_strip_whitespace=True)

    formula: str = Field(..., description="Chemical formula")
    mechanical: Optional[MechanicalProperties] = Field(None, description="Mechanical properties")
    electrical: Optional[ElectricalProperties] = Field(None, description="Electrical properties")
    thermal: Optional[ThermalProperties] = Field(None, description="Thermal properties")
    magnetic: Optional[MagneticProperties] = Field(None, description="Magnetic properties")
    stability: Optional[float] = Field(None, description="Formation energy in eV/atom")
    synthesizability: Optional[float] = Field(None, ge=0, le=1, description="Synthesizability score")


class ReactionStep(BaseModel):
    """Single step in synthesis route."""
    model_config = ConfigDict(str_strip_whitespace=True)

    step_number: int = Field(..., ge=1, description="Step number in sequence")
    reactants: List[str] = Field(..., description="List of reactants")
    products: List[str] = Field(..., description="List of products")
    conditions: Dict[str, Any] = Field(..., description="Reaction conditions")
    temperature: Optional[float] = Field(None, description="Temperature in K")
    pressure: Optional[float] = Field(None, description="Pressure in bar")
    time: Optional[float] = Field(None, description="Reaction time in hours")
    catalyst: Optional[str] = Field(None, description="Catalyst used")
    yield_percentage: Optional[float] = Field(None, ge=0, le=100, description="Expected yield %")


class SynthesisRoute(BaseModel):
    """Complete synthesis route for material."""
    model_config = ConfigDict(str_strip_whitespace=True)

    target_material: str = Field(..., description="Target material formula")
    route_name: str = Field(..., description="Name of synthesis route")
    steps: List[ReactionStep] = Field(..., description="Synthesis steps")
    total_time: Optional[float] = Field(None, description="Total synthesis time in hours")
    difficulty_score: float = Field(..., ge=0, le=10, description="Difficulty score (0-10)")
    cost_estimate: Optional[float] = Field(None, description="Estimated cost in USD")
    safety_hazards: List[str] = Field(default_factory=list, description="Safety hazards")
    success_rate: Optional[float] = Field(None, ge=0, le=1, description="Historical success rate")


class PhaseRegion(BaseModel):
    """Region in phase diagram."""
    model_config = ConfigDict(str_strip_whitespace=True)

    composition_range: Tuple[float, float] = Field(..., description="Composition range")
    temperature_range: Tuple[float, float] = Field(..., description="Temperature range in K")
    phases_present: List[str] = Field(..., description="Phases present in region")
    phase_fractions: Optional[Dict[str, float]] = Field(None, description="Phase fractions")


class PhaseDiagram(BaseModel):
    """Phase diagram representation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    system: str = Field(..., description="Chemical system (e.g., Fe-C)")
    diagram_type: Literal["binary", "ternary", "quaternary"] = Field(..., description="Diagram type")
    temperature_range: Tuple[float, float] = Field(..., description="Temperature range in K")
    pressure: float = Field(default=1.0, description="Pressure in bar")
    phase_regions: List[PhaseRegion] = Field(..., description="Phase regions")
    invariant_points: List[Dict[str, Any]] = Field(default_factory=list, description="Invariant points")
    critical_points: List[Dict[str, Any]] = Field(default_factory=list, description="Critical points")


class DFTCalculation(BaseModel):
    """DFT calculation specification and results."""
    model_config = ConfigDict(str_strip_whitespace=True)

    structure: CrystalStructure = Field(..., description="Crystal structure")
    functional: str = Field(..., description="Exchange-correlation functional (e.g., PBE, HSE06)")
    basis_set: str = Field(..., description="Basis set or pseudopotential")
    k_points: Tuple[int, int, int] = Field(..., description="K-point mesh")
    cutoff_energy: float = Field(..., description="Plane wave cutoff in eV")
    convergence_criteria: Dict[str, float] = Field(..., description="Convergence criteria")

    # Results
    total_energy: Optional[float] = Field(None, description="Total energy in eV")
    forces: Optional[List[Tuple[float, float, float]]] = Field(None, description="Forces on atoms")
    stress_tensor: Optional[List[List[float]]] = Field(None, description="Stress tensor in GPa")
    band_structure: Optional[Dict[str, Any]] = Field(None, description="Band structure data")
    density_of_states: Optional[Dict[str, Any]] = Field(None, description="DOS data")
    charge_density: Optional[Dict[str, Any]] = Field(None, description="Charge density data")

    calculation_time: Optional[float] = Field(None, description="Calculation time in CPU hours")
    converged: bool = Field(default=False, description="Whether calculation converged")


class MaterialsHypothesis(BaseModel):
    """Materials science research hypothesis."""
    model_config = ConfigDict(str_strip_whitespace=True)

    hypothesis_id: str = Field(..., description="Unique hypothesis ID")
    title: str = Field(..., description="Hypothesis title")
    claim: str = Field(..., description="Main hypothesis claim")

    predicted_structure: Optional[CrystalStructure] = Field(None, description="Predicted crystal structure")
    predicted_properties: Optional[MaterialProperties] = Field(None, description="Predicted properties")
    synthesis_route: Optional[SynthesisRoute] = Field(None, description="Proposed synthesis route")

    theoretical_basis: List[str] = Field(..., description="Theoretical justification")
    computational_evidence: List[DFTCalculation] = Field(default_factory=list, description="DFT calculations")
    experimental_validation: Dict[str, Any] = Field(default_factory=dict, description="Experimental data")

    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in hypothesis")
    impact_score: float = Field(..., ge=0, le=10, description="Potential impact (0-10)")
    novelty_score: float = Field(..., ge=0, le=10, description="Novelty score (0-10)")

    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")

    status: Literal["proposed", "testing", "validated", "refuted"] = Field(
        default="proposed",
        description="Hypothesis status"
    )

    related_materials: List[str] = Field(default_factory=list, description="Related material formulas")
    citations: List[str] = Field(default_factory=list, description="Supporting citations")