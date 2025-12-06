#!/bin/bash
# === run-DFT.sh ===
# SIESTA DFT calculation automation script
#
# DESCRIPTION:
#   Automates the setup and execution of SIESTA DFT calculations for 2D materials.
#   Handles pseudopotential selection, input file generation, and strain series calculations.
#
# USAGE:
#   ./run-DFT.sh <material> <structure>
#
# ARGUMENTS:
#   material   - Material name (MoS2, MoSe2, WS2, WSe2)
#   structure  - Structure type (1x1_primitive, 1x10_rectangular, etc.)
#
# EXAMPLES:
#   ./run-DFT.sh MoS2 1x1_primitive      # Single unit cell calculation
#   ./run-DFT.sh MoS2 1x10_rectangular   # 1x10 supercell calculation
#
# OUTPUT:
#   Creates calculation directories in siesta/materials/{material}/Monolayer/{structure}/
#   Each strain point gets its own subdirectory with complete input files
#
# REQUIREMENTS:
#   - SIESTA executable in PATH
#   - Pseudopotential files in siesta/pseudopotentials/
#   - Template files in siesta/io_templates/
#
# AUTHOR: Meshal Alawein (meshal@berkeley.edu)
# INSTITUTION: University of California, Berkeley

set -e  # Exit on error

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Check for required arguments
if [[ $# -lt 2 ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <material> <structure>"
    echo "Example: $0 MoS2 1x10_rectangular"
    exit 1
fi

material="$1"
structure="$2"

# Check for SIESTA executable
if ! command -v "$SIESTA_EXE" >/dev/null 2>&1; then
    echo "Error: SIESTA executable ($SIESTA_EXE) not found in PATH"
    echo "Please install SIESTA or set SIESTA_EXE environment variable"
    exit 1
fi

supercell="$structure"
type="Monolayer"
functional="GGA"
SOC="F"
RELAXED="F"
RelaxRun="T"
StaticRun="T"
BandsRun="T"
LDOSRun="T"
partition="2"
strains=("0")

targetDir="$QMATSIM_SIESTA_DIR/materials/$material/$type/$structure/$supercell"
template_io="$QMATSIM_SIESTA_DIR/io_templates"
pseudopath="$QMATSIM_SIESTA_DIR/pseudopotentials"

# Function to copy pseudopotential files for a given material
# Arguments: material functional soc_flag
copy_material_files() {
    local mat="$1"; local func="$2"; local soc="$3"
    
    # Map materials to their constituent elements
    case $mat in
        MoS2) ext1="Mo"; ext2="S";;        # Molybdenum disulfide
        MoSe2) ext1="Mo"; ext2="Se";;      # Molybdenum diselenide  
        WS2) ext1="W"; ext2="S";;          # Tungsten disulfide
        WSe2) ext1="W"; ext2="Se";;        # Tungsten diselenide
        *) 
            echo "Error: Unknown material '$mat'"
            echo "Supported materials: MoS2, MoSe2, WS2, WSe2"
            exit 1
            ;;
    esac

    if [ "$soc" = "T" ]; then
        cp $pseudopath/${func}-SOC/$ext1.psf .
        cp $pseudopath/${func}-SOC/$ext2.psf .
    else
        cp $pseudopath/${func}/$ext1.psf .
        cp $pseudopath/${func}/$ext2.psf .
    fi
}

set_partition_variables() {
    accountVAR="co_msedcc"
    paritionVAR="savio2_bigmem"
    QoSVAR="savio_lowprio"
    nodesVAR="1"
    cpusVAR="24"
}

copy_material_files "$material" "$functional" "$SOC"
mkdir -p "$targetDir/0"

for strain in "${strains[@]}"; do
    simulation="Relaxation"
    ./scripts/generate-input.sh "$material" "$type" "$structure" "$supercell" "$simulation" "$functional" "$strain" "$SOC"
    mv input.fdf 1-Relaxation.tmp
    cat 1-Relaxation.tmp $template_io/Relaxation-io.fdf > 1-Relaxation.fdf

    simulation="Static"
    ./scripts/generate-input.sh "$material" "$type" "$structure" "$supercell" "$simulation" "$functional" "$strain" "$SOC"
    mv input.fdf 2-Static.tmp
    cat 2-Static.tmp $template_io/Static-io.fdf > 2-Static.fdf

    simulation="Bands"
    ./scripts/generate-input.sh "$material" "$type" "$structure" "$supercell" "$simulation" "$functional" "$strain" "$SOC"
    mv input.fdf 3-Bands.tmp
    cat 3-Bands.tmp $template_io/Bands-io.fdf > 3-Bands.fdf

    simulation="LDOS"
    ./scripts/generate-input.sh "$material" "$type" "$structure" "$supercell" "$simulation" "$functional" "$strain" "$SOC"
    mv input.fdf LDOS.tmp
    cat LDOS.tmp $template_io/Static-io.fdf > LDOS.fdf

    rm *.tmp

    cp scripts/template-siesta.sh job.sh
    set_partition_variables "$partition"

    sed -i "s/typeVAR/$type/g; s/materialVAR/$material/g; s/functionalVAR/$functional/g; s/structureVAR/$structure/g; s/supercellVAR/$supercell/g; s/strainVAR/$strain/g; s/socVAR/$SOC/g; s/accountVAR/$accountVAR/g; s/paritionVAR/$paritionVAR/g; s/QoSVAR/$QoSVAR/g; s/nodesVAR/$nodesVAR/g; s/cpusVAR/$cpusVAR/g; s/RelaxVAR/$RelaxRun/g; s/StaticVAR/$StaticRun/g; s/BandsVAR/$BandsRun/g; s/LDOSVAR/$LDOSRun/g" job.sh

    mv *.fdf *.sh *.psf "$targetDir/0/"
    cp "$targetDir/Structure/$material.STRUCT_IN" "$targetDir/0/$material.STRUCT_IN"

    cd "$targetDir/0"
    sbatch -o job.out -e job.err job.sh
    cd -
done
