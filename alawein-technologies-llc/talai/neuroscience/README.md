# Neuroscience Research Module

Advanced neuroscience hypothesis validation system for TalAI, featuring fMRI/EEG analysis, neural circuit modeling, connectome analysis, cognitive model testing, and BCI experiment design.

## Features

### Core Capabilities

- **fMRI Analysis Pipeline**: GLM, connectivity, resting-state networks, DCM
- **EEG Analysis Pipeline**: ERP, spectral analysis, source localization, connectivity
- **Neural Circuit Modeling**: Anatomical validation, functional simulation
- **Connectome Analysis**: Graph metrics, hub detection, community structure
- **Cognitive Model Testing**: ACT-R, neural networks, Bayesian models
- **BCI Experiment Design**: P300, SSVEP, motor imagery paradigms
- **Allen Brain ORCHEX Integration**: Anatomical validation, gene expression
- **NWB Format Support**: Read/write Neurodata Without Borders files

### Validation Framework

- Neural data quality assessment
- Circuit anatomical plausibility
- Connectivity pattern validation
- Cognitive model performance testing
- Experimental design evaluation
- Literature consistency checking
- Clinical relevance assessment

## Installation

```bash
# Clone repository
git clone https://github.com/AlaweinOS/TalAI.git
cd TalAI/neuroscience

# Install dependencies
pip install -r requirements.txt

# Optional: Install cognitive modeling libraries
pip install pyactr  # For ACT-R modeling
```

## Quick Start

```python
import asyncio
from src.neuroscience import (
    NeuroscienceHypothesis,
    NeuroscienceProtocol,
    BrainRegion,
    NeuralCircuit,
    BrainRegionID
)

async def validate_hypothesis():
    # Define brain regions
    hippocampus = BrainRegion(
        name="Hippocampus",
        id=BrainRegionID.HIPPOCAMPUS,
        coordinates=(26.0, -20.0, -10.0),
        hemisphere="bilateral"
    )

    # Create neural circuit
    circuit = NeuralCircuit(
        name="memory_circuit",
        regions=[hippocampus],
        connections=[],
        function="memory consolidation",
        oscillation_frequency=6.0  # Theta
    )

    # Create hypothesis
    hypothesis = NeuroscienceHypothesis(
        hypothesis_id="neuro-001",
        title="Theta Oscillations in Memory",
        claim="Theta oscillations coordinate memory consolidation",
        neural_circuits=[circuit],
        confidence_score=0.75,
        impact_score=8.5,
        novelty_score=6.0
    )

    # Validate
    protocol = NeuroscienceProtocol()
    results = await protocol.validate_hypothesis(hypothesis)

    print(f"Score: {results['scores']['overall']:.2f}")

asyncio.run(validate_hypothesis())
```

## Module Structure

```
neuroscience/
├── src/neuroscience/
│   ├── __init__.py          # Package initialization
│   ├── models.py            # Pydantic data models
│   ├── protocol.py          # Main validation protocol
│   ├── analyzers.py         # Domain-specific analyzers
│   └── integrations.py      # External API integrations
├── examples/
│   └── example_workflow.py  # Complete usage examples
├── README.md
└── requirements.txt
```

## Data Models

### Neural Data Types
- **FMRIData**: BOLD signal, TR/TE, voxel size, preprocessing
- **EEGData**: Channels, sampling rate, events, epochs
- **MEG/ECoG/LFP**: Additional modality support

### Brain Structures
- **BrainRegion**: Anatomical regions with coordinates
- **NeuralCircuit**: Connected regions with functional properties
- **ConnectomeGraph**: Whole-brain connectivity networks

### Cognitive Models
- **ACT-R**: Production system architecture
- **Neural Networks**: Deep learning models
- **Bayesian**: Probabilistic inference models
- **Drift-Diffusion**: Decision-making models

### BCI Systems
- **BCIExperiment**: Complete experimental protocols
- **Paradigms**: P300, SSVEP, motor imagery, hybrid
- **Performance Metrics**: Accuracy, ITR, latency

## Analyzers

### FMRIAnalyzer
- General Linear Model (GLM) analysis
- Functional connectivity computation
- Resting-state network detection
- Dynamic Causal Modeling (DCM)

### EEGAnalyzer
- Event-Related Potential (ERP) detection
- Power spectral analysis
- Phase-amplitude coupling
- Source localization

### ConnectomeAnalyzer
- Graph theoretical metrics
- Hub identification
- Community detection
- Robustness assessment

### CognitiveModelTester
- Model performance evaluation
- Cross-validation
- Complexity assessment
- Neural alignment testing

### BCIDesigner
- Experiment optimization
- Parameter tuning
- Sample size calculation
- Safety protocol generation

## Integration Examples

### Allen Brain ORCHEX
```python
allen_brain = AllenBrainAPI(api_key="your_key")

# Get structure information
structure = await allen_brain.get_structure_info(997)  # Cortex

# Get gene expression
expression = await allen_brain.get_gene_expression("BDNF")

# Validate circuit anatomy
validation = await allen_brain.validate_circuit_anatomy(circuit)
```

### NWB Format
```python
nwb = NWBInterface()

# Read NWB file
data = await nwb.read_nwb_file(Path("data.nwb"))

# Convert to NWB
await nwb.convert_to_nwb(fmri_data, Path("output.nwb"))

# Extract timeseries
timeseries = await nwb.extract_timeseries(Path("data.nwb"))
```

## Validation Levels

- **Basic**: Data quality and circuit plausibility
- **Standard**: Adds experimental design evaluation
- **Comprehensive**: Full literature and clinical assessment

## Research Applications

### 1. Memory Research
- Hippocampal-cortical interactions
- Memory consolidation during sleep
- Working memory capacity
- Episodic memory encoding

### 2. Clinical Neuroscience
- Biomarker discovery
- Disease mechanisms
- Treatment targets
- Therapeutic monitoring

### 3. Cognitive Neuroscience
- Attention networks
- Decision-making circuits
- Language processing
- Social cognition

### 4. Brain-Computer Interfaces
- Motor rehabilitation
- Communication devices
- Cognitive enhancement
- Neurofeedback training

## Performance Metrics

- Data quality assessment: <1 second
- Circuit validation: <2 seconds
- Connectome analysis: <5 seconds
- Model testing: <10 seconds
- Full validation: <30 seconds

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/neuroscience

# Run specific test category
pytest tests/test_fmri_analysis.py
```

## Configuration

```python
config = {
    "allen_brain_api_key": "your_api_key",
    "tr_threshold": 3.0,  # Maximum TR for fMRI
    "motion_threshold": 3.0,  # mm
    "validation_timeout": 60  # seconds
}

protocol = NeuroscienceProtocol(config)
```

## Citations

When using this module in research:

```bibtex
@software{talai_neuroscience,
  title={TalAI Neuroscience Research Module},
  author={AlaweinOS Team},
  year={2024},
  url={https://github.com/AlaweinOS/TalAI}
}
```

## License

Apache 2.0 - See LICENSE file for details.

## Support

- GitHub Issues: https://github.com/AlaweinOS/TalAI/issues
- Email: neuro@talai.org

## Acknowledgments

- Allen Institute for Brain Science
- Neurodata Without Borders community
- MNE-Python developers
- Nilearn contributors