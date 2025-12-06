#!/bin/bash

# Define the strain values
strain_values=(strainVAR)

# Loop over each strain value
for strain in "${strain_values[@]}"; do
    # Print separator
    echo "################### FULL BZ ###################"

    # Print strain percentage
    echo "###### STRAIN: $strain%"

    # CB command
    echo "@@@@@@@@@@@@@@@@@@@@@ CB @@@@@@@@@@@@@@@@@@@@@"
    cmd_CB="sort -k5,5 -n Postprocessing/GNUBANDS/fullBZ/index-kx-ky-kz-E-CB1-2D.txt | head -n500"
    eval $cmd_CB |
    awk -v strain="$strain" 'function euclidean_distance(x1, y1, z1, x2, y2, z2) {
            return sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        }
        BEGIN {print "# index\tkx\tky\tkz\tE\tdk\tdE"}
        NR == 1 {kx_top = $2; ky_top = $3; kz_top = $4; E_top = $5}
        {printf "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", $1, $2, $3, $4, $5, euclidean_distance($2, $3, $4, kx_top, ky_top, kz_top), ($5 > E_top) ? ($5 - E_top) : (E_top - $5)}'


    # VB command
    echo "@@@@@@@@@@@@@@@@@@@@@ VB @@@@@@@@@@@@@@@@@@@@@"
    cmd_VB="sort -k5,5 -nr Postprocessing/GNUBANDS/fullBZ/index-kx-ky-kz-E-VB1-2D.txt | head -n500"
    eval $cmd_VB |
    awk -v strain="$strain" 'function euclidean_distance(x1, y1, z1, x2, y2, z2) {
            return sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        }
        BEGIN {print "# index\tkx\tky\tkz\tE\tdk\tdE"}
        NR == 1 {kx_top = $2; ky_top = $3; kz_top = $4; E_top = $5}
        {printf "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", $1, $2, $3, $4, $5, euclidean_distance($2, $3, $4, kx_top, ky_top, kz_top), ($5 > E_top) ? ($5 - E_top) : (E_top - $5)}'


    # Print separator
    echo "# --------------"
done > fullBZ-table.txt

# Loop over each strain value
for strain in "${strain_values[@]}"; do
    # Print separator
    echo "################### BANDS ###################"

    # Print strain percentage
    echo "###### STRAIN: $strain%"

    # CB command
    echo "@@@@@@@@@@@@@@@@@@@@@ CB @@@@@@@@@@@@@@@@@@@@@"
    cmd_CB="sort -k5,5 -nr Postprocessing/GNUBANDS/bands/index-kx-ky-kz-E-CB1-2D.txt | head -n500"
    eval $cmd_CB |
    awk -v strain="$strain" 'function euclidean_distance(x1, y1, z1, x2, y2, z2) {
            return sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        }
        BEGIN {print "# index\tkx\tky\tkz\tE\tdk\tdE"}
        NR == 1 {kx_top = $2; ky_top = $3; kz_top = $4; E_top = $5}
        {printf "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", $1, $2, $3, $4, $5, euclidean_distance($2, $3, $4, kx_top, ky_top, kz_top), ($5 > E_top) ? ($5 - E_top) : (E_top - $5)}'


    # VB command
    echo "@@@@@@@@@@@@@@@@@@@@@ VB @@@@@@@@@@@@@@@@@@@@@"
    cmd_VB="sort -k5,5 -nr Postprocessing/GNUBANDS/bands/index-kx-ky-kz-E-VB1-2D.txt | head -n500"
    eval $cmd_VB |
    awk -v strain="$strain" 'function euclidean_distance(x1, y1, z1, x2, y2, z2) {
            return sqrt((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)
        }
        BEGIN {print "# index\tkx\tky\tkz\tE\tdk\tdE"}
        NR == 1 {kx_top = $2; ky_top = $3; kz_top = $4; E_top = $5}
        {printf "%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", $1, $2, $3, $4, $5, euclidean_distance($2, $3, $4, kx_top, ky_top, kz_top), ($5 > E_top) ? ($5 - E_top) : (E_top - $5)}'


    # Print separator
    echo "# --------------"
done > bands-table.txt

mv fullBZ-table.txt Results/ExtractedParameters
mv bands-table.txt Results/ExtractedParameters
