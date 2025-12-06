"""
Pydantic V2 models for Neuroscience Research.

Comprehensive data structures for neural data, brain regions,
connectomes, cognitive models, and BCI experiments.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Tuple, Any, Literal, Union
from datetime import datetime, timedelta
from enum import Enum
import numpy as np


class BrainRegionID(str, Enum):
    """Standard brain region identifiers."""
    # Cortical regions
    PFC = "prefrontal_cortex"
    M1 = "primary_motor_cortex"
    S1 = "primary_somatosensory_cortex"
    V1 = "primary_visual_cortex"
    A1 = "primary_auditory_cortex"
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"

    # Subcortical
    THALAMUS = "thalamus"
    STRIATUM = "striatum"
    CEREBELLUM = "cerebellum"
    BRAINSTEM = "brainstem"

    # Networks
    DMN = "default_mode_network"
    SALIENCE = "salience_network"
    EXECUTIVE = "executive_control_network"


class NeuralDataType(str, Enum):
    """Types of neural data."""
    FMRI = "fmri"
    EEG = "eeg"
    MEG = "meg"
    ECOG = "ecog"
    SINGLE_UNIT = "single_unit"
    LFP = "local_field_potential"
    CALCIUM_IMAGING = "calcium_imaging"
    PET = "pet"
    NIRS = "nirs"


class CognitiveModelType(str, Enum):
    """Types of cognitive models."""
    ACT_R = "act_r"
    NEURAL_NETWORK = "neural_network"
    BAYESIAN = "bayesian"
    DRIFT_DIFFUSION = "drift_diffusion"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    PREDICTIVE_CODING = "predictive_coding"


class BrainRegion(BaseModel):
    """Brain region representation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Region name")
    id: Optional[BrainRegionID] = Field(None, description="Standard region ID")
    coordinates: Optional[Tuple[float, float, float]] = Field(
        None,
        description="MNI coordinates (x, y, z)"
    )
    volume: Optional[float] = Field(None, description="Volume in mm³")
    brodmann_areas: List[int] = Field(default_factory=list, description="Brodmann areas")
    hemisphere: Literal["left", "right", "bilateral"] = Field(
        default="bilateral",
        description="Hemisphere"
    )
    parent_structure: Optional[str] = Field(None, description="Parent anatomical structure")
    cytoarchitecture: Optional[str] = Field(None, description="Cytoarchitectonic description")
    neurotransmitters: List[str] = Field(
        default_factory=list,
        description="Primary neurotransmitters"
    )


class VoxelData(BaseModel):
    """Voxel-level data for fMRI."""
    model_config = ConfigDict(str_strip_whitespace=True)

    coordinates: Tuple[int, int, int] = Field(..., description="Voxel coordinates (i, j, k)")
    time_series: List[float] = Field(..., description="BOLD signal time series")
    snr: Optional[float] = Field(None, description="Signal-to-noise ratio")
    tissue_type: Optional[Literal["gray", "white", "csf"]] = Field(
        None,
        description="Tissue type"
    )


class FMRIData(BaseModel):
    """fMRI data structure."""
    model_config = ConfigDict(str_strip_whitespace=True, arbitrary_types_allowed=True)

    subject_id: str = Field(..., description="Subject identifier")
    session_id: str = Field(..., description="Session identifier")
    task: str = Field(..., description="Task name or 'rest'")

    # Acquisition parameters
    tr: float = Field(..., description="Repetition time in seconds")
    te: Optional[float] = Field(None, description="Echo time in ms")
    flip_angle: Optional[float] = Field(None, description="Flip angle in degrees")
    voxel_size: Tuple[float, float, float] = Field(..., description="Voxel dimensions in mm")
    matrix_size: Tuple[int, int, int] = Field(..., description="Image matrix dimensions")

    # Data
    bold_data: Optional[Any] = Field(None, description="4D BOLD data array")  # numpy array in practice
    voxel_data: List[VoxelData] = Field(default_factory=list, description="Voxel-level data")

    # Preprocessing info
    preprocessed: bool = Field(default=False, description="Preprocessing status")
    preprocessing_steps: List[str] = Field(default_factory=list, description="Applied steps")
    motion_parameters: Optional[List[List[float]]] = Field(
        None,
        description="Motion correction parameters"
    )

    # Analysis results
    activation_maps: Dict[str, Any] = Field(default_factory=dict, description="Statistical maps")
    roi_timeseries: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="ROI-extracted time series"
    )
    connectivity_matrix: Optional[List[List[float]]] = Field(
        None,
        description="Functional connectivity"
    )


class EEGChannel(BaseModel):
    """EEG channel information."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Channel name (e.g., Fz, C3)")
    position: Tuple[float, float, float] = Field(..., description="3D position")
    reference: str = Field(default="average", description="Reference type")
    impedance: Optional[float] = Field(None, description="Impedance in kOhms")
    sampling_rate: float = Field(..., description="Sampling rate in Hz")


class EEGData(BaseModel):
    """EEG data structure."""
    model_config = ConfigDict(str_strip_whitespace=True, arbitrary_types_allowed=True)

    subject_id: str = Field(..., description="Subject identifier")
    session_id: str = Field(..., description="Session identifier")
    task: str = Field(..., description="Task or paradigm")

    # Channel information
    channels: List[EEGChannel] = Field(..., description="Channel information")
    montage: str = Field(default="10-20", description="Electrode montage system")

    # Data
    raw_data: Optional[Any] = Field(None, description="Raw EEG data array")  # numpy array
    sampling_rate: float = Field(..., description="Sampling rate in Hz")
    duration: float = Field(..., description="Recording duration in seconds")

    # Events and epochs
    events: List[Dict[str, Any]] = Field(default_factory=list, description="Event markers")
    epochs: List[Dict[str, Any]] = Field(default_factory=list, description="Epoched data")

    # Preprocessing
    filtered: bool = Field(default=False)
    filter_params: Dict[str, float] = Field(default_factory=dict)
    artifacts_removed: bool = Field(default=False)
    ica_components: Optional[int] = Field(None, description="Number of ICA components")

    # Analysis results
    power_spectrum: Dict[str, Any] = Field(default_factory=dict)
    erp_components: Dict[str, Any] = Field(default_factory=dict, description="ERP components")
    connectivity: Dict[str, Any] = Field(default_factory=dict)
    source_localization: Optional[Dict[str, Any]] = Field(None)


class NeuralCircuit(BaseModel):
    """Neural circuit representation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Circuit name")
    regions: List[BrainRegion] = Field(..., description="Brain regions involved")
    connections: List[Tuple[str, str, float]] = Field(
        ...,
        description="Connections (source, target, strength)"
    )
    neuron_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Neuron type counts"
    )

    # Circuit properties
    oscillation_frequency: Optional[float] = Field(None, description="Dominant frequency in Hz")
    propagation_delay: Optional[float] = Field(None, description="Signal delay in ms")
    plasticity_type: Optional[str] = Field(None, description="Synaptic plasticity type")

    # Functional properties
    function: str = Field(..., description="Primary function")
    cognitive_role: Optional[str] = Field(None, description="Role in cognition")
    disorders: List[str] = Field(default_factory=list, description="Associated disorders")


class ConnectomeGraph(BaseModel):
    """Brain connectome representation."""
    model_config = ConfigDict(str_strip_whitespace=True)

    subject_id: str = Field(..., description="Subject identifier")
    modality: Literal["structural", "functional", "effective"] = Field(..., description="Connectome type")

    # Graph structure
    nodes: List[BrainRegion] = Field(..., description="Brain regions (nodes)")
    edges: List[Tuple[int, int, float]] = Field(
        ...,
        description="Connections (node_i, node_j, weight)"
    )

    # Graph metrics
    clustering_coefficient: Optional[float] = Field(None)
    path_length: Optional[float] = Field(None)
    modularity: Optional[float] = Field(None)
    small_worldness: Optional[float] = Field(None)
    hub_regions: List[str] = Field(default_factory=list)

    # Additional properties
    resolution: Optional[str] = Field(None, description="Parcellation resolution")
    threshold: Optional[float] = Field(None, description="Connection threshold")
    weighted: bool = Field(default=True)
    directed: bool = Field(default=False)


class CognitiveModel(BaseModel):
    """Cognitive model specification."""
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(..., description="Model name")
    type: CognitiveModelType = Field(..., description="Model type")

    # Model architecture
    parameters: Dict[str, float] = Field(..., description="Model parameters")
    architecture: Dict[str, Any] = Field(..., description="Model architecture")

    # Task specification
    task_domain: str = Field(..., description="Cognitive domain")
    input_format: str = Field(..., description="Input data format")
    output_format: str = Field(..., description="Output/prediction format")

    # Performance metrics
    accuracy: Optional[float] = Field(None)
    reaction_time_correlation: Optional[float] = Field(None)
    neural_fit: Optional[float] = Field(None, description="Correlation with neural data")

    # Implementation
    implementation_language: str = Field(default="python")
    libraries: List[str] = Field(default_factory=list)
    computational_cost: Optional[str] = Field(None)


class BCIParadigm(str, Enum):
    """BCI paradigm types."""
    P300 = "p300"
    SSVEP = "ssvep"
    MOTOR_IMAGERY = "motor_imagery"
    NEUROFEEDBACK = "neurofeedback"
    HYBRID = "hybrid"


class BCIExperiment(BaseModel):
    """Brain-Computer Interface experiment design."""
    model_config = ConfigDict(str_strip_whitespace=True)

    experiment_id: str = Field(..., description="Unique experiment ID")
    paradigm: BCIParadigm = Field(..., description="BCI paradigm")

    # Hardware setup
    device: str = Field(..., description="EEG/recording device")
    channels: List[str] = Field(..., description="Recording channels")
    sampling_rate: float = Field(..., description="Sampling rate in Hz")

    # Experimental design
    trial_structure: Dict[str, Any] = Field(..., description="Trial timing and structure")
    num_trials: int = Field(..., ge=1)
    session_duration: float = Field(..., description="Session duration in minutes")

    # Signal processing
    preprocessing_pipeline: List[str] = Field(..., description="Processing steps")
    feature_extraction: str = Field(..., description="Feature extraction method")
    classifier: str = Field(..., description="Classification algorithm")

    # Performance metrics
    accuracy: Optional[float] = Field(None, ge=0, le=1)
    information_transfer_rate: Optional[float] = Field(None, description="ITR in bits/min")
    latency: Optional[float] = Field(None, description="System latency in ms")

    # Safety and ethics
    safety_protocols: List[str] = Field(default_factory=list)
    ethical_approval: Optional[str] = Field(None, description="IRB approval number")


class NeuralData(BaseModel):
    """Generic neural data container."""
    model_config = ConfigDict(str_strip_whitespace=True, arbitrary_types_allowed=True)

    data_type: NeuralDataType = Field(..., description="Type of neural data")
    subject_id: str = Field(..., description="Subject identifier")
    session_info: Dict[str, Any] = Field(..., description="Session metadata")

    # Data arrays
    raw_data: Optional[Any] = Field(None, description="Raw data array")
    processed_data: Optional[Any] = Field(None, description="Processed data")

    # Metadata
    acquisition_date: datetime = Field(..., description="Acquisition date/time")
    equipment: str = Field(..., description="Recording equipment")
    parameters: Dict[str, Any] = Field(..., description="Acquisition parameters")

    # Quality metrics
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    usable_percentage: Optional[float] = Field(None, ge=0, le=100)


class NeuroscienceHypothesis(BaseModel):
    """Neuroscience research hypothesis."""
    model_config = ConfigDict(str_strip_whitespace=True)

    hypothesis_id: str = Field(..., description="Unique hypothesis ID")
    title: str = Field(..., description="Hypothesis title")
    claim: str = Field(..., description="Main hypothesis claim")

    # Neural evidence
    neural_data: List[NeuralData] = Field(default_factory=list)
    brain_regions: List[BrainRegion] = Field(default_factory=list)
    neural_circuits: List[NeuralCircuit] = Field(default_factory=list)

    # Predictions
    predicted_activity: Dict[str, Any] = Field(default_factory=dict)
    predicted_connectivity: Optional[ConnectomeGraph] = Field(None)
    predicted_behavior: Dict[str, Any] = Field(default_factory=dict)

    # Cognitive modeling
    cognitive_model: Optional[CognitiveModel] = Field(None)
    model_predictions: Dict[str, Any] = Field(default_factory=dict)

    # Experimental design
    proposed_experiments: List[Dict[str, Any]] = Field(default_factory=list)
    required_sample_size: Optional[int] = Field(None)
    power_analysis: Optional[Dict[str, float]] = Field(None)

    # Theoretical framework
    theoretical_basis: List[str] = Field(..., description="Theoretical justification")
    supporting_literature: List[str] = Field(default_factory=list)
    competing_theories: List[str] = Field(default_factory=list)

    # Clinical relevance
    clinical_applications: List[str] = Field(default_factory=list)
    disorders_addressed: List[str] = Field(default_factory=list)
    therapeutic_implications: Optional[str] = Field(None)

    # Validation scores
    confidence_score: float = Field(..., ge=0, le=1)
    impact_score: float = Field(..., ge=0, le=10)
    novelty_score: float = Field(..., ge=0, le=10)
    feasibility_score: float = Field(default=0.5, ge=0, le=1)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(None)
    status: Literal["proposed", "testing", "validated", "refuted"] = Field(default="proposed")

    # Ethics
    ethical_considerations: List[str] = Field(default_factory=list)
    requires_irb: bool = Field(default=False)


class ExperimentalDesign(BaseModel):
    """Detailed experimental design for neuroscience studies."""
    model_config = ConfigDict(str_strip_whitespace=True)

    design_type: Literal["within", "between", "mixed", "crossover"] = Field(...)
    independent_variables: List[Dict[str, Any]] = Field(...)
    dependent_variables: List[Dict[str, Any]] = Field(...)

    # Control conditions
    control_conditions: List[str] = Field(default_factory=list)
    randomization: str = Field(default="block")
    blinding: Literal["single", "double", "none"] = Field(default="double")

    # Statistical plan
    primary_analysis: str = Field(...)
    correction_method: Optional[str] = Field(None, description="Multiple comparisons correction")
    effect_size_estimate: Optional[float] = Field(None)

    # Timeline
    trial_duration: float = Field(..., description="Trial duration in seconds")
    inter_trial_interval: Tuple[float, float] = Field(..., description="ITI range in seconds")
    total_duration: float = Field(..., description="Total experiment duration in minutes")


class NeuralOscillation(BaseModel):
    """Neural oscillation characteristics."""
    model_config = ConfigDict(str_strip_whitespace=True)

    frequency_band: Literal["delta", "theta", "alpha", "beta", "gamma", "high_gamma"] = Field(...)
    frequency_range: Tuple[float, float] = Field(..., description="Frequency range in Hz")
    power: float = Field(..., description="Oscillation power")
    phase: Optional[float] = Field(None, description="Phase in radians")

    # Spatial properties
    source_regions: List[BrainRegion] = Field(default_factory=list)
    propagation_direction: Optional[str] = Field(None)

    # Functional significance
    cognitive_function: Optional[str] = Field(None)
    behavioral_correlation: Optional[float] = Field(None)

    # Cross-frequency coupling
    coupled_oscillations: List[str] = Field(default_factory=list)
    pac_strength: Optional[float] = Field(None, description="Phase-amplitude coupling")


class NeurotransmitterSystem(BaseModel):
    """Neurotransmitter system specification."""
    model_config = ConfigDict(str_strip_whitespace=True)

    neurotransmitter: str = Field(..., description="Neurotransmitter name")
    receptor_types: List[str] = Field(..., description="Receptor subtypes")

    # Anatomical distribution
    synthesis_regions: List[BrainRegion] = Field(default_factory=list)
    projection_targets: List[BrainRegion] = Field(default_factory=list)

    # Functional properties
    primary_function: str = Field(...)
    behavioral_effects: List[str] = Field(default_factory=list)

    # Pharmacology
    agonists: List[str] = Field(default_factory=list)
    antagonists: List[str] = Field(default_factory=list)
    reuptake_inhibitors: List[str] = Field(default_factory=list)

    # Clinical relevance
    associated_disorders: List[str] = Field(default_factory=list)
    therapeutic_targets: List[str] = Field(default_factory=list)