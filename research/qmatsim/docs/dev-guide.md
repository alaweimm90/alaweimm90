# Development Guide

## Project Overview

QMatSim is a strain engineering framework for flat bands and lateral heterostructures in 2D materials. Based on research in Physical Review Materials (2025), it combines DFT (SIESTA) and MD (LAMMPS) to study strain effects in transition metal dichalcogenides, focusing on flat band emergence and hole localization in rippled structures.

## Development Commands

### Installation and Setup
```bash
# Install in development mode
pip install -e .

# Install dependencies
pip install numpy matplotlib pytest
```

### External Dependencies

**SIESTA 4.1+** for DFT calculations
- Install via package manager or compile from source
- Ensure `siesta` executable is in PATH

**LAMMPS** for MD simulations
- LAMMPS executable (`lmp_mpi`) required but not included
- Install via package manager, conda, or build from source
- Scripts expect `lmp_mpi` in PATH

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_cli_basic.py
pytest tests/test_qmatsim_cli.py
```

### CLI Usage Examples
```bash
# DFT relaxation with SIESTA
qmatsim relax --material MoS2 --structure 1x10_rectangular

# MD simulation with LAMMPS
qmatsim minimize --structure ripple10 --mode compress
qmatsim minimize --structure ripple10 --mode all

# Postprocessing analysis
qmatsim analyze --material MoS2 --structure 1x10_rectangular
```

## Architecture Overview

### Core Components

**CLI Entry Point (`qmatsim/__main__.py`)**
- Main CLI interface with argparse-based subcommands
- Three primary operations: `relax` (DFT), `minimize` (MD), `analyze` (postprocessing)
- Delegates to bash scripts in `scripts/` directory

**Bash Script Automation (`scripts/`)**
- `run-DFT.sh`: SIESTA DFT calculations with pseudopotential management
- `run-MD.sh`: LAMMPS MD simulations with multiple input file support
- `run-postprocessing.sh`: Analysis and data extraction
- Template generation scripts for input files

**SIESTA Integration (`siesta/`)**
- `io_templates/`: SIESTA input file templates with variable substitution
- `pseudopotentials/`: Element pseudopotentials organized by functional (GGA, LDA, GGA-SOC)
- `materials/`: Structure-specific calculations organized by material/type/structure/strain

**LAMMPS Integration (`lammps/`)**
- `data/`: Atomic structure files (.data format)
- `in/`: LAMMPS input scripts for different simulation types
- `potentials/`: Interatomic potential files

### Key Architecture Patterns

**Material-Structure-Strain Hierarchy**
```
siesta/materials/{material}/{type}/{structure}/{strain}/
```
- Supports systematic strain engineering studies
- Each strain directory contains complete calculation setup

**Template-Based Input Generation**
- SIESTA templates use variable substitution (e.g., `EcutoffVAR`, `basisSizeVAR`)
- Scripts populate templates based on material properties and calculation parameters

**Pseudopotential Management**
- Material-specific element mapping in bash scripts (qmatsim/__main__.py:26-30)
- Functional-specific pseudopotential directories (GGA, LDA, GGA-SOC)
- Multiple formats supported (.psf, .psml, .upf)

### File Dependencies

**Required for DFT calculations:**
- SIESTA executable in PATH
- Pseudopotential files in `siesta/pseudopotentials/`
- Structure files (STRUCT_IN format)
- Template files in `siesta/io_templates/`

**Required for MD calculations:**
- LAMMPS executable (`lmp_mpi`) in PATH
- Data files in `lammps/data/`
- Input scripts in `lammps/in/`
- Potential files in `lammps/potentials/`

## Development Notes

### Adding New Materials
1. Add element mapping in `scripts/run-DFT.sh` (lines ~26-30)
2. Ensure pseudopotentials exist for all elements
3. Create structure files in appropriate directories

### Script Error Handling
- Scripts check for required executables and files before execution
- Missing dependencies cause early exit with descriptive error messages
- Output files created to track completion status

### Testing Strategy
- CLI functionality tested via subprocess calls
- Tests verify help menu accessibility and basic command structure
- Integration tests would require SIESTA/LAMMPS installations