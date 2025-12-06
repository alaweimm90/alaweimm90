#!/bin/bash
# === run-postprocessing.sh ===
# Usage: ./run-postprocessing.sh <material> <structure>

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
type="Monolayer"
supercell="$structure"
functional="GGA"
direction="y"
partition="6"

strains=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20")

copy_material_files() {
    ext1="Mo"; ext2="S"
    cp siesta/pseudopotentials/$functional/$ext1.psf .
    cp siesta/pseudopotentials/$functional/$ext2.psf .
}

set_partition_variables() {
    accountVAR="co_msedcc"
    paritionVAR="savio3"
    QoSVAR="savio_lowprio"
    nodesVAR="2"
    cpusVAR="32"
}

copy_material_files "$material"
set_partition_variables "$partition"

for strain in "${strains[@]}"; do
    workdir="siesta/materials/$material/$type/$structure/$supercell/$strain"
    cp scripts/postprocessing-core.sh "$workdir/postprocessing.sh"
    cp scripts/postprocessing-job.sh "$workdir/postprocessing-job.sh"

    sed -i "s/materialVAR/$material/g; s/typeVAR/$type/g; s/structureVAR/$structure/g; s/supercellVAR/$supercell/g; s/functionalVAR/$functional/g; s/directionVAR/$direction/g; s/strainVAR/$strain/g; s/accountVAR/$accountVAR/g; s/paritionVAR/$paritionVAR/g; s/QoSVAR/$QoSVAR/g; s/nodesVAR/$nodesVAR/g; s/cpusVAR/$cpusVAR/g" "$workdir/postprocessing-job.sh"

    cd "$workdir"
    sbatch -o post.out -e post.err postprocessing-job.sh
    cd -
done
