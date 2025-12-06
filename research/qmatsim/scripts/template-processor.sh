#!/bin/bash
# === template-processor.sh ===
# Template variable substitution system for SIESTA input files

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Function to substitute variables in a template file
substitute_template() {
    local template_file="$1"
    local output_file="$2"
    
    if [[ ! -f "$template_file" ]]; then
        echo "Error: Template file not found: $template_file" >&2
        return 1
    fi
    
    # Default values for SIESTA parameters
    local MESH_CUTOFF="${MESH_CUTOFF:-300}"
    local OCCUPATION_FUNCTION="${OCCUPATION_FUNCTION:-FD}"
    local ELECTRONIC_TEMPERATURE="${ELECTRONIC_TEMPERATURE:-300}"
    local BASIS_SIZE="${BASIS_SIZE:-DZP}"
    local XC_FUNCTIONAL="${XC_FUNCTIONAL:-GGA}"
    local XC_AUTHORS="${XC_AUTHORS:-PBE}"
    local SPIN_POLARIZATION="${SPIN_POLARIZATION:-none}"
    local SOC_ENABLED="${SOC_ENABLED:-false}"
    
    # Create output file with variable substitutions
    sed -e "s/EcutoffVAR/$MESH_CUTOFF/g" \
        -e "s/foccupVAR/$OCCUPATION_FUNCTION/g" \
        -e "s/TelecVAR/$ELECTRONIC_TEMPERATURE/g" \
        -e "s/basisSizeVAR/$BASIS_SIZE/g" \
        -e "s/functionalVAR/$XC_FUNCTIONAL/g" \
        -e "s/authorsVAR/$XC_AUTHORS/g" \
        -e "s/spinVAR/$SPIN_POLARIZATION/g" \
        -e "s/socVAR/$SOC_ENABLED/g" \
        "$template_file" > "$output_file"
    
    echo "Template processed: $template_file -> $output_file"
}

# Function to create material-specific input file
create_siesta_input() {
    local material="$1"
    local calculation_type="$2"
    local strain="$3"
    local output_dir="$4"
    
    # Set material-specific parameters
    case "$material" in
        MoS2)
            BASIS_SIZE="DZP"
            MESH_CUTOFF="400"
            ;;
        MoSe2)
            BASIS_SIZE="DZP"
            MESH_CUTOFF="350"
            ;;
        WS2|WSe2)
            BASIS_SIZE="DZP"
            MESH_CUTOFF="400"
            ;;
        *)
            echo "Warning: Unknown material $material, using defaults" >&2
            ;;
    esac
    
    # Set calculation-specific parameters
    case "$calculation_type" in
        Relaxation)
            local template="$QMATSIM_SIESTA_DIR/io_templates/Relaxation-io.fdf"
            ;;
        Static)
            local template="$QMATSIM_SIESTA_DIR/io_templates/Static-io.fdf"
            ;;
        Bands)
            local template="$QMATSIM_SIESTA_DIR/io_templates/Bands-io.fdf"
            ;;
        LDOS)
            local template="$QMATSIM_SIESTA_DIR/io_templates/LDOS-io.fdf"
            ;;
        *)
            local template="$QMATSIM_SIESTA_DIR/io_templates/template.fdf"
            ;;
    esac
    
    # Create output directory if it doesn't exist
    mkdir -p "$output_dir"
    
    # Process template
    local output_file="$output_dir/${calculation_type:-siesta}.fdf"
    export MESH_CUTOFF BASIS_SIZE OCCUPATION_FUNCTION ELECTRONIC_TEMPERATURE
    export XC_FUNCTIONAL XC_AUTHORS SPIN_POLARIZATION SOC_ENABLED
    
    substitute_template "$template" "$output_file"
}

# Main function for command-line usage
main() {
    if [[ $# -lt 2 ]]; then
        echo "Usage: $0 <template_file> <output_file> [VAR=value ...]"
        echo "   or: $0 create <material> <calc_type> <strain> <output_dir>"
        echo ""
        echo "Examples:"
        echo "  $0 template.fdf output.fdf MESH_CUTOFF=400 BASIS_SIZE=DZP"
        echo "  $0 create MoS2 Relaxation 0 ./calculation"
        exit 1
    fi
    
    if [[ "$1" == "create" ]]; then
        create_siesta_input "$2" "$3" "$4" "$5"
    else
        # Parse additional variables from command line
        for arg in "${@:3}"; do
            if [[ "$arg" =~ ^([A-Z_]+)=(.+)$ ]]; then
                export "${BASH_REMATCH[1]}"="${BASH_REMATCH[2]}"
            fi
        done
        
        substitute_template "$1" "$2"
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi