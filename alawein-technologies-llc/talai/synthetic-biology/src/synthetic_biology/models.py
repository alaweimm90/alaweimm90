"""
Pydantic V2 models for Synthetic Biology Research.

Gene circuits, metabolic pathways, protein engineering,
CRISPR systems, and biosafety.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Tuple, Any, Literal
from datetime import datetime
from enum import Enum


class PartType(str, Enum):
    """Standard biological part types."""
    PROMOTER = "promoter"
    RBS = "rbs"
    CDS = "cds"
    TERMINATOR = "terminator"
    INSULATOR = "insulator"
    OPERATOR = "operator"
    REPORTER = "reporter"
    SELECTION = "selection"


class OrganismHost(str, Enum):
    """Common chassis organisms."""
    ECOLI = "E. coli"
    YEAST = "S. cerevisiae"
    BACILLUS = "B. subtilis"
    CHO = "CHO cells"
    HEK293 = "HEK293"
    ARABIDOPSIS = "A. thaliana"
    SYNECHOCYSTIS = "Synechocystis"


class BioBrick(BaseModel):
    """BioBrick standard part."""
    model_config = ConfigDict(str_strip_whitespace=True)

    part_name: str = Field(..., description="Part identifier (e.g., BBa_K123456)")
    part_type: PartType = Field(..., description="Type of part")
    sequence: str = Field(..., description="DNA sequence")
    length: Optional[int] = Field(None, description="Sequence length in bp")

    # Functional properties
    strength: Optional[float] = Field(None, ge=0, le=1, description="Relative strength")
    leakiness: Optional[float] = Field(None, ge=0, le=1, description="Basal expression")
    response_curve: Optional[Dict[str, Any]] = Field(None, description="Transfer function")

    # Metadata
    description: str = Field(..., description="Part description")
    designer: Optional[str] = Field(None)
    availability: Literal["available", "planning", "discontinued"] = Field(default="planning")

    @field_validator("sequence")
    @classmethod
    def validate_dna_sequence(cls, v: str) -> str:
        """Validate DNA sequence contains only valid nucleotides."""
        valid_chars = set("ATCGN")
        if not all(c in valid_chars for c in v.upper()):
            raise ValueError("Invalid DNA sequence characters")
        return v.upper()


class GeneCircuit(BaseModel):
    """Synthetic gene circuit design."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Circuit name")
    circuit_type: Literal["toggle", "oscillator", "cascade", "feedback", "logic"] = Field(...)

    # Circuit components
    parts: List[BioBrick] = Field(..., description="Component parts")
    topology: Dict[str, List[str]] = Field(..., description="Connection topology")

    # Dynamic properties
    steady_states: Optional[List[float]] = Field(None, description="Stable states")
    oscillation_period: Optional[float] = Field(None, description="Period in hours")
    response_time: Optional[float] = Field(None, description="Response time in minutes")

    # Performance metrics
    dynamic_range: Optional[float] = Field(None, description="Fold change output")
    noise: Optional[float] = Field(None, description="Cell-to-cell variability CV")
    robustness: Optional[float] = Field(None, ge=0, le=1, description="Parameter robustness")

    # Host requirements
    host_organism: OrganismHost = Field(...)
    growth_conditions: Dict[str, Any] = Field(default_factory=dict)
    metabolic_burden: Optional[float] = Field(None, ge=0, le=1)


class Enzyme(BaseModel):
    """Enzyme specification for pathways."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Enzyme name")
    ec_number: Optional[str] = Field(None, description="EC classification")
    gene: str = Field(..., description="Gene encoding enzyme")
    sequence: Optional[str] = Field(None, description="Protein sequence")

    # Kinetic parameters
    km: Optional[float] = Field(None, description="Michaelis constant (mM)")
    kcat: Optional[float] = Field(None, description="Turnover number (1/s)")
    vmax: Optional[float] = Field(None, description="Maximum velocity")

    # Properties
    cofactors: List[str] = Field(default_factory=list)
    inhibitors: List[str] = Field(default_factory=list)
    optimal_ph: Optional[Tuple[float, float]] = Field(None)
    optimal_temp: Optional[float] = Field(None, description="Optimal temperature (°C)")

    expression_level: Optional[float] = Field(None, description="Expression level (AU)")


class MetabolicPathway(BaseModel):
    """Metabolic pathway engineering."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Pathway name")
    product: str = Field(..., description="Target product")
    substrate: str = Field(..., description="Starting substrate")

    # Pathway components
    enzymes: List[Enzyme] = Field(..., description="Pathway enzymes")
    intermediates: List[str] = Field(..., description="Intermediate metabolites")
    reactions: List[Dict[str, Any]] = Field(..., description="Reaction specifications")

    # Performance
    theoretical_yield: float = Field(..., description="Theoretical yield (g/g)")
    actual_yield: Optional[float] = Field(None, description="Experimental yield")
    productivity: Optional[float] = Field(None, description="Productivity (g/L/h)")
    titer: Optional[float] = Field(None, description="Product titer (g/L)")

    # Optimization
    bottlenecks: List[str] = Field(default_factory=list)
    balance_score: Optional[float] = Field(None, ge=0, le=1, description="Flux balance")
    toxicity_issues: List[str] = Field(default_factory=list)

    host_organism: OrganismHost = Field(...)


class ProteinDesign(BaseModel):
    """Protein engineering specification."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Protein name")
    sequence: str = Field(..., description="Amino acid sequence")
    structure_pdb: Optional[str] = Field(None, description="PDB structure ID")

    # Design goals
    target_activity: str = Field(..., description="Desired activity")
    improvement_metric: str = Field(..., description="What to optimize")
    fold_improvement: Optional[float] = Field(None, description="Improvement factor")

    # Mutations
    mutations: List[str] = Field(default_factory=list, description="Mutations (e.g., A123V)")
    mutation_effects: Dict[str, float] = Field(default_factory=dict)

    # Properties
    stability_tm: Optional[float] = Field(None, description="Melting temperature (°C)")
    expression_level: Optional[float] = Field(None, description="Expression (mg/L)")
    activity: Optional[float] = Field(None, description="Specific activity")
    specificity: Optional[float] = Field(None, description="Substrate specificity")

    # Computational predictions
    folding_energy: Optional[float] = Field(None, description="ΔG folding (kcal/mol)")
    solubility_score: Optional[float] = Field(None, ge=0, le=1)


class CRISPRTarget(BaseModel):
    """CRISPR target specification."""
    model_config = ConfigDict(str_strip_whitespace=True)

    target_gene: str = Field(..., description="Target gene name")
    grna_sequence: str = Field(..., min_length=20, max_length=20, description="Guide RNA")
    pam_sequence: str = Field(default="NGG", description="PAM sequence")

    # Target location
    chromosome: str = Field(...)
    position: int = Field(..., ge=0)
    strand: Literal["+", "-"] = Field(...)

    # Specificity
    on_target_score: Optional[float] = Field(None, ge=0, le=1)
    off_targets: List[Dict[str, Any]] = Field(default_factory=list)
    specificity_score: Optional[float] = Field(None, ge=0, le=1)

    # Editing outcome
    edit_type: Literal["knockout", "knockin", "base_edit", "prime_edit"] = Field(...)
    expected_indel_rate: Optional[float] = Field(None, ge=0, le=1)
    hdr_template: Optional[str] = Field(None, description="HDR template sequence")

    @field_validator("grna_sequence")
    @classmethod
    def validate_grna(cls, v: str) -> str:
        """Validate guide RNA sequence."""
        valid_chars = set("ATCGU")
        if not all(c in valid_chars for c in v.upper()):
            raise ValueError("Invalid RNA sequence")
        return v.upper()


class PlasmidConstruct(BaseModel):
    """Plasmid assembly design."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Plasmid name")
    backbone: str = Field(..., description="Vector backbone")
    size: int = Field(..., ge=100, description="Total size in bp")

    # Components
    features: List[Dict[str, Any]] = Field(..., description="Plasmid features")
    insert: Optional[str] = Field(None, description="Insert sequence")
    selection_markers: List[str] = Field(...)
    origin_replication: str = Field(...)

    # Assembly method
    assembly_method: Literal["gibson", "golden_gate", "biobrick", "moclo"] = Field(...)
    assembly_parts: List[str] = Field(default_factory=list)

    # Properties
    copy_number: Literal["low", "medium", "high"] = Field(default="medium")
    host_range: List[OrganismHost] = Field(...)
    stability: Optional[float] = Field(None, ge=0, le=1)


class DirectedEvolution(BaseModel):
    """Directed evolution experiment."""
    model_config = ConfigDict(str_strip_whitespace=True)

    target_protein: ProteinDesign = Field(...)
    evolution_method: Literal["error_prone_pcr", "dna_shuffling", "phage_display", "continuous"] = Field(...)

    # Library
    library_size: int = Field(..., ge=1)
    mutation_rate: float = Field(..., description="Mutations per kb")
    diversity_score: Optional[float] = Field(None, ge=0, le=1)

    # Selection
    selection_pressure: str = Field(..., description="Selection criterion")
    rounds: int = Field(..., ge=1, description="Evolution rounds")
    enrichment_factor: Optional[float] = Field(None, description="Per-round enrichment")

    # Results
    best_variant: Optional[str] = Field(None, description="Best variant sequence")
    improvement: Optional[float] = Field(None, description="Fold improvement")
    hit_rate: Optional[float] = Field(None, ge=0, le=1, description="Fraction improved")


class SafetyScreen(BaseModel):
    """Biosafety assessment."""
    model_config = ConfigDict(str_strip_whitespace=True)

    organism: str = Field(...)
    modifications: List[str] = Field(...)

    # Risk assessment
    risk_group: Literal[1, 2, 3, 4] = Field(..., description="WHO risk group")
    containment_level: Literal["BSL1", "BSL2", "BSL3", "BSL4"] = Field(...)

    # Safety features
    kill_switches: List[str] = Field(default_factory=list)
    auxotrophy: List[str] = Field(default_factory=list, description="Essential nutrients")
    environmental_survival: Optional[float] = Field(None, ge=0, le=1)

    # Regulatory
    igsc_compliant: bool = Field(default=False)
    select_agent: bool = Field(default=False)
    dual_use: bool = Field(default=False)
    public_health_risk: Literal["negligible", "low", "moderate", "high"] = Field(...)

    approval_status: Literal["pending", "approved", "rejected"] = Field(default="pending")


class SynBioHypothesis(BaseModel):
    """Synthetic biology research hypothesis."""
    model_config = ConfigDict(str_strip_whitespace=True)

    hypothesis_id: str = Field(...)
    title: str = Field(...)
    claim: str = Field(...)

    # Designs
    gene_circuits: List[GeneCircuit] = Field(default_factory=list)
    pathways: List[MetabolicPathway] = Field(default_factory=list)
    proteins: List[ProteinDesign] = Field(default_factory=list)
    crispr_targets: List[CRISPRTarget] = Field(default_factory=list)
    plasmids: List[PlasmidConstruct] = Field(default_factory=list)

    # Predictions
    expected_behavior: Dict[str, Any] = Field(...)
    performance_metrics: Dict[str, float] = Field(...)

    # Experimental plan
    experiments: List[Dict[str, Any]] = Field(...)
    controls: List[str] = Field(...)
    timeline_weeks: int = Field(...)

    # Safety
    safety_assessment: Optional[SafetyScreen] = Field(None)
    ethical_considerations: List[str] = Field(default_factory=list)

    # Scores
    confidence_score: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=10)
    novelty_score: float = Field(..., ge=0, le=10)
    feasibility_score: float = Field(default=0.5, ge=0, le=1)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    status: Literal["proposed", "testing", "validated", "failed"] = Field(default="proposed")

    # Applications
    applications: List[str] = Field(default_factory=list)
    commercial_potential: Optional[float] = Field(None, ge=0, le=10)