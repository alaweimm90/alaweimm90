#!/bin/bash

# Usage: ./convert_to_struct_in.sh lammps_dump.txt output_struct_in.txt

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <lammps_dump_file> <output_struct_in_file>"
    exit 1
fi

input_file="$1"
output_file="$2"

# Ensure the input file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file does not exist."
    exit 1
fi

# Start processing
echo "Processing..."

# Extract and format lattice vectors, calculating the differences
lattice_vectors=$(awk '
    /ITEM: BOX BOUNDS/ {
        getline; split($0, x); 
        getline; split($0, y); 
        getline; split($0, z);
        printf "%.5f 0.00000 0.00000\n", x[2] - x[1];
        printf "0.00000 %.5f 0.00000\n", y[2] - y[1];
        printf "0.00000 0.00000 %.5f\n", z[2] - z[1];
    }' "$input_file")

# Determine number of atoms
num_atoms=$(awk '/ITEM: NUMBER OF ATOMS/{getline; print $1}' "$input_file")

# Prepare the output
{
    echo "$lattice_vectors"
    echo "$num_atoms"
    # Extract atom information assuming type, id, xs, ys, zs order
    awk '/ITEM: ATOMS/{flag=1; next} /ITEM/{flag=0} flag' "$input_file" | awk '{printf "%d\t%d\t%.5f\t%.5f\t%.5f\n", $2, $1, $3, $4, $5}'
} > "$output_file"

echo "Conversion completed. Output is in $output_file"
