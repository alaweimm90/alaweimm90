#!/bin/bash

shopt -s extglob
shopt -s expand_aliases


    cd Results
    cd ExtractedParameters

# Enable extended pattern matching for better string manipulation
shopt -s extglob

# Define function to compute distance between two points
compute_distance() {
    local x1=$1 y1=$2 z1=$3 x2=$4 y2=$5 z2=$6
    local dx=$(awk -v x1="$x1" -v x2="$x2" 'BEGIN{printf "%.6f", x2 - x1}')
    local dy=$(awk -v y1="$y1" -v y2="$y2" 'BEGIN{printf "%.6f", y2 - y1}')
    local dz=$(awk -v z1="$z1" -v z2="$z2" 'BEGIN{printf "%.6f", z2 - z1}')
    local distance=$(awk -v dx="$dx" -v dy="$dy" -v dz="$dz" 'BEGIN{printf "%.6f", sqrt(dx^2 + dy^2 + dz^2)}')
    if (( $(awk 'BEGIN{print ('$distance' >= 0) ? 1 : 0}') )); then
        echo "$distance"
    else
        echo "0"
    fi
}


# Function to compute angle between three atoms
compute_angle() {
    local x1=$1 y1=$2 z1=$3 x2=$4 y2=$5 z2=$6 x3=$7 y3=$8 z3=$9
    # Vectors
    ux=$(awk -v x1="$x1" -v x2="$x2" 'BEGIN{printf "%.2f", x2 - x1}')
    uy=$(awk -v y1="$y1" -v y2="$y2" 'BEGIN{printf "%.2f", y2 - y1}')
    uz=$(awk -v z1="$z1" -v z2="$z2" 'BEGIN{printf "%.2f", z2 - z1}')
    vx=$(awk -v x1="$x3" -v x2="$x2" 'BEGIN{printf "%.2f", x2 - x1}')
    vy=$(awk -v y1="$y3" -v y2="$y2" 'BEGIN{printf "%.2f", y2 - y1}')
    vz=$(awk -v z1="$z3" -v z2="$z2" 'BEGIN{printf "%.2f", z2 - z1}')
    # Dot product
    dot_product=$(awk -v ux="$ux" -v uy="$uy" -v uz="$uz" -v vx="$vx" -v vy="$vy" -v vz="$vz" \
        'BEGIN{printf "%.2f", (ux*vx) + (uy*vy) + (uz*vz)}')
    # Magnitudes
    mag_u=$(awk -v ux="$ux" -v uy="$uy" -v uz="$uz" 'BEGIN{printf "%.2f", sqrt(ux^2 + uy^2 + uz^2)}')
    mag_v=$(awk -v vx="$vx" -v vy="$vy" -v vz="$vz" 'BEGIN{printf "%.2f", sqrt(vx^2 + vy^2 + vz^2)}')
    # Compute angle in radians
    angle_rad=$(awk -v dot_product="$dot_product" -v mag_u="$mag_u" -v mag_v="$mag_v" \
        'BEGIN{printf "%.2f", atan2(sqrt(1 - (dot_product / (mag_u * mag_v))^2), dot_product / (mag_u * mag_v))}')
    # Convert radians to degrees
    awk -v angle_rad="$angle_rad" 'BEGIN{printf "%.2f", angle_rad * (180/3.14159)}'
}

rm bonds.txt
rm bonds-short.txt
# Read the coordinates file
coordinates_file="../Structure/structure_CARTESIAN.xyz"
# Output file
output_file="bonds.txt"
output_file_short="bonds-short.txt"
# Initialize table header
printf "%-6s %-20s %-20s %-20s %-20s %-26s %-26s %-26s\n" \
    "Mo" \
    "index S1-S2 [Ang]" \
    "index S1-S3 [Ang]" \
    "index Mo1-Mo2 [Ang]" \
    "index Mo1-S1 [Ang]" \
    "index S3-Mo1-S1 [degrees]" \
    "index Mo1-S2-Mo2 [degrees]" \
    "index S1-Mo2-S2 [degrees]" >> "$output_file"
# Extract the list of Mo atom IDs and their corresponding line numbers
Mo_ID=($(awk '$2 == 1 && NR > 2 {print $1}' "$coordinates_file"))
Mo_Line_Numbers=($(awk '$2 == 1 && NR > 2 {print NR}' "$coordinates_file"))
last_index=$((${#Mo_ID[@]} - 1))
last_Mo=${Mo_ID[$last_index]}

# Loop through the array
for i in "${!Mo_ID[@]}"; do
    atom_id=${Mo_ID[$i]}
    line_number=${Mo_Line_Numbers[$i]}
    line=$(sed -n "${line_number}p" "$coordinates_file")
    line_i=$(sed -n "${line_number}p" "$coordinates_file")

# Check if i is the last element
if [ "$i" -eq "$last_index" ]; then
    line_i1=$(sed -n "$((line_number + 1))p" "$coordinates_file")
    line_i2=$(sed -n "$((line_number + 2))p" "$coordinates_file")
    line_i3=$(sed -n "$((line_number + 3))p" "$coordinates_file")
    line_i4=$(sed -n "$((line_number + 4))p" "$coordinates_file")
    line_i5=$(sed -n "$((line_number + 5))p" "$coordinates_file")
    ID_i=$(echo "$line_i" | awk '{print $1}')
    TYPE_i=$(echo "$line_i" | awk '{print $2}')
    x_i=$(echo "$line_i" | awk '{print $3}')
    y_i=$(echo "$line_i" | awk '{print $4}')
    z_i=$(echo "$line_i" | awk '{print $5}')
    ID_i1=$(echo "$line_i1" | awk '{print $1}')
    TYPE_i1=$(echo "$line_i1" | awk '{print $2}')
    x_i1=$(echo "$line_i1" | awk '{print $3}')
    y_i1=$(echo "$line_i1" | awk '{print $4}')
    z_i1=$(echo "$line_i1" | awk '{print $5}')
    ID_i2=$(echo "$line_i2" | awk '{print $1}')
    TYPE_i2=$(echo "$line_i2" | awk '{print $2}')
    x_i2=$(echo "$line_i2" | awk '{print $3}')
    y_i2=$(echo "$line_i2" | awk '{print $4}')
    z_i2=$(echo "$line_i2" | awk '{print $5}')
    ID_i3=$(echo "$line_i3" | awk '{print $1}')
    TYPE_i3=$(echo "$line_i3" | awk '{print $2}')
    x_i3=$(echo "$line_i3" | awk '{print $3}')
    y_i3=$(echo "$line_i3" | awk '{print $4}')
    z_i3=$(echo "$line_i3" | awk '{print $5}')
    ID_i4=$(echo "$line_i4" | awk '{print $1}')
    TYPE_i4=$(echo "$line_i4" | awk '{print $2}')
    x_i4=$(echo "$line_i4" | awk '{print $3}')
    y_i4=$(echo "$line_i4" | awk '{print $4}')
    z_i4=$(echo "$line_i4" | awk '{print $5}')
    ID_i5=$(echo "$line_i5" | awk '{print $1}')
    TYPE_i5=$(echo "$line_i5" | awk '{print $2}')
    x_i5=$(echo "$line_i5" | awk '{print $3}')
    y_i5=$(echo "$line_i5" | awk '{print $4}')
    z_i5=$(echo "$line_i5" | awk '{print $5}')
    bond_length_i2_i4=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i4 $y_i4 $z_i4)

    [ "$TYPE_i" == "1" ] && TYPE_i="Mo" || TYPE_i="S"
    [ "$TYPE_i1" == "1" ] && TYPE_i1="Mo" || TYPE_i1="S"
    [ "$TYPE_i2" == "1" ] && TYPE_i2="Mo" || TYPE_i2="S"
    [ "$TYPE_i3" == "1" ] && TYPE_i3="Mo" || TYPE_i3="S"
    [ "$TYPE_i4" == "1" ] && TYPE_i4="Mo" || TYPE_i4="S"
    [ "$TYPE_i5" == "1" ] && TYPE_i5="Mo" || TYPE_i5="S"

    #echo "Mo ($ID_i): $ID_i2-$ID_i3 ${TYPE_i2}1-${TYPE_i3}2 $bond_length_i2_i3"
    #echo "Mo ($ID_i): $ID_i2-$ID_i4 ${TYPE_i2}1-${TYPE_i4}3 $bond_length_i2_i4"
    #echo "Mo ($ID_i): $ID_i-$ID_i1 ${TYPE_i}1-${TYPE_i1}2 $bond_length_i_i1"
    #echo "Mo ($ID_i): $ID_i1-$ID_i2 ${TYPE_i1}1-${TYPE_i2}1 $bond_length_i1_i2"

    bond_angle_i4_i_i2=$(compute_angle $x_i4 $y_i4 $z_i4 $x_i $y_i $z_i $x_i2 $y_i2 $z_i2)
    if (( $(echo "$bond_angle_i4_i_i2 < 0" | bc -l) )); then
        bond_angle_i4_i_i2=0
    fi
    #echo "Mo ($ID_i): $ID_i4-$ID_i-$ID_i2 ${TYPE_i4}3-${TYPE_i}1-${TYPE_i2}1 $bond_angle_i4_i_i2"
    #echo "Mo ($ID_i): $ID_i-$ID_i2-$ID_i1 ${TYPE_i}1-${TYPE_i2}1-${TYPE_i1}2 $bond_angle_i_i2_i1"
    #echo "Mo ($ID_i): $ID_i2-$ID_i1-$ID_i3 ${TYPE_i2}1-${TYPE_i1}2-${TYPE_i3}2 $bond_angle_i2_i1_i3"
    printf "%d        %d-%d    %.3f         x-x    x             x-x    x             x-x    x               %d-%d-%d  %.3f               x-x-x  x              x-x-x  x\n" "$atom_id" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i4" "$ID_i" "$ID_i2" "$bond_angle_i4_i_i2" >> "$output_file"
    printf "%d %d-%d %.3f 0 0 0 %d-%d-%d %.3f 0 0\n" "$atom_id" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i4" "$ID_i" "$ID_i2" "$bond_angle_i4_i_i2" >> "$output_file_short"

else
    # Non-last element case
    if ((atom_id % 2 != 0)); then
        # Odd atom_id case
        line_i1=$(sed -n "$((line_number + 1))p" "$coordinates_file")
        line_i2=$(sed -n "$((line_number + 2))p" "$coordinates_file")
        line_i3=$(sed -n "$((line_number + 3))p" "$coordinates_file")
        line_i4=$(sed -n "$((line_number + 4))p" "$coordinates_file")
        line_i5=$(sed -n "$((line_number + 5))p" "$coordinates_file")
        ID_i=$(echo "$line_i" | awk '{print $1}')
        TYPE_i=$(echo "$line_i" | awk '{print $2}')
        x_i=$(echo "$line_i" | awk '{print $3}')
        y_i=$(echo "$line_i" | awk '{print $4}')
        z_i=$(echo "$line_i" | awk '{print $5}')
        ID_i1=$(echo "$line_i1" | awk '{print $1}')
        TYPE_i1=$(echo "$line_i1" | awk '{print $2}')
        x_i1=$(echo "$line_i1" | awk '{print $3}')
        y_i1=$(echo "$line_i1" | awk '{print $4}')
        z_i1=$(echo "$line_i1" | awk '{print $5}')
        ID_i2=$(echo "$line_i2" | awk '{print $1}')
        TYPE_i2=$(echo "$line_i2" | awk '{print $2}')
        x_i2=$(echo "$line_i2" | awk '{print $3}')
        y_i2=$(echo "$line_i2" | awk '{print $4}')
        z_i2=$(echo "$line_i2" | awk '{print $5}')
        ID_i3=$(echo "$line_i3" | awk '{print $1}')
        TYPE_i3=$(echo "$line_i3" | awk '{print $2}')
        x_i3=$(echo "$line_i3" | awk '{print $3}')
        y_i3=$(echo "$line_i3" | awk '{print $4}')
        z_i3=$(echo "$line_i3" | awk '{print $5}')
        ID_i4=$(echo "$line_i4" | awk '{print $1}')
        TYPE_i4=$(echo "$line_i4" | awk '{print $2}')
        x_i4=$(echo "$line_i4" | awk '{print $3}')
        y_i4=$(echo "$line_i4" | awk '{print $4}')
        z_i4=$(echo "$line_i4" | awk '{print $5}')
        ID_i5=$(echo "$line_i5" | awk '{print $1}')
        TYPE_i5=$(echo "$line_i5" | awk '{print $2}')
        x_i5=$(echo "$line_i5" | awk '{print $3}')
        y_i5=$(echo "$line_i5" | awk '{print $4}')
        z_i5=$(echo "$line_i5" | awk '{print $5}')
        bond_length_i2_i3=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i3 $y_i3 $z_i3)
        bond_length_i2_i4=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i4 $y_i4 $z_i4)
        bond_length_i_i1=$(compute_distance $x_i $y_i $z_i $x_i1 $y_i1 $z_i1)
        bond_length_i1_i2=$(compute_distance $x_i1 $y_i1 $z_i1 $x_i2 $y_i2 $z_i2)

        if (( $(echo "$bond_length_i2_i3 < 0" | bc -l) )); then
            bond_length_i2_i3=0
        fi
        if (( $(echo "$bond_angle_i4_i_i2 < 0" | bc -l) )); then
            bond_angle_i4_i_i2=0
        fi
        if (( $(echo "$bond_length_i_i1 < 0" | bc -l) )); then
            bond_length_i_i1=0
        fi
        if (( $(echo "$bond_length_i1_i2 < 0" | bc -l) )); then
            bond_length_i1_i2=0
        fi
        [ "$TYPE_i" == "1" ] && TYPE_i="Mo" || TYPE_i="S"
        [ "$TYPE_i1" == "1" ] && TYPE_i1="Mo" || TYPE_i1="S"
        [ "$TYPE_i2" == "1" ] && TYPE_i2="Mo" || TYPE_i2="S"
        [ "$TYPE_i3" == "1" ] && TYPE_i3="Mo" || TYPE_i3="S"
        [ "$TYPE_i4" == "1" ] && TYPE_i4="Mo" || TYPE_i4="S"
        [ "$TYPE_i5" == "1" ] && TYPE_i5="Mo" || TYPE_i5="S"

        #echo "Mo ($ID_i): $ID_i2-$ID_i3 ${TYPE_i2}1-${TYPE_i3}2 $bond_length_i2_i3"
        #echo "Mo ($ID_i): $ID_i2-$ID_i4 ${TYPE_i2}1-${TYPE_i4}3 $bond_length_i2_i4"
        #echo "Mo ($ID_i): $ID_i-$ID_i1 ${TYPE_i}1-${TYPE_i1}2 $bond_length_i_i1"
        #echo "Mo ($ID_i): $ID_i1-$ID_i2 ${TYPE_i1}1-${TYPE_i2}1 $bond_length_i1_i2"

        bond_angle_i4_i_i2=$(compute_angle $x_i4 $y_i4 $z_i4 $x_i $y_i $z_i $x_i2 $y_i2 $z_i2)
        bond_angle_i_i2_i1=$(compute_angle $x_i $y_i $z_i $x_i4 $y_i4 $z_i4 $x_i1 $y_i1 $z_i1)
        bond_angle_i2_i1_i3=$(compute_angle $x_i2 $y_i2 $z_i2 $x_i1 $y_i1 $z_i1 $x_i3 $y_i3 $z_i3)

        #echo "Mo ($ID_i): $ID_i4-$ID_i-$ID_i2 ${TYPE_i4}3-${TYPE_i}1-${TYPE_i2}1 $bond_angle_i4_i_i2"
        #echo "Mo ($ID_i): $ID_i-$ID_i2-$ID_i1 ${TYPE_i}1-${TYPE_i2}1-${TYPE_i1}2 $bond_angle_i_i2_i1"
        #echo "Mo ($ID_i): $ID_i2-$ID_i1-$ID_i3 ${TYPE_i2}1-${TYPE_i1}2-${TYPE_i3}2 $bond_angle_i2_i1_i3"
        printf "%d        %d-%d    %.3f         %d-%d    %.3f         %d-%d    %.3f         %d-%d    %.3f           %d-%d-%d  %.3f               %d-%d-%d  %.3f         %d-%d-%d  %.3f\n" "$atom_id" "$ID_i2" "$ID_i3" "$bond_length_i2_i3" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i" "$ID_i1" "$bond_length_i_i1" "$ID_i1" "$ID_i2" "$bond_length_i1_i2" "$ID_i" "$ID_i2" "$ID_i3" "$bond_angle_i4_i_i2" "$ID_i" "$ID_i1" "$ID_i2" "$bond_angle_i_i2_i1" "$ID_i2" "$ID_i1" "$ID_i3" "$bond_angle_i2_i1_i3"  >> "$output_file"
        printf "%d %d-%d %.3f %d-%d %.3f %d-%d %.3f %d-%d %.3f %d-%d-%d %.3f %d-%d-%d %.3f %d-%d-%d %.3f\n" "$atom_id" "$ID_i2" "$ID_i3" "$bond_length_i2_i3" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i" "$ID_i1" "$bond_length_i_i1" "$ID_i1" "$ID_i2" "$bond_length_i1_i2" "$ID_i" "$ID_i2" "$ID_i3" "$bond_angle_i4_i_i2" "$ID_i" "$ID_i1" "$ID_i2" "$bond_angle_i_i2_i1" "$ID_i2" "$ID_i1" "$ID_i3" "$bond_angle_i2_i1_i3" >> "$output_file_short"


    else
        # Even atom_id case
        line_i1=$(sed -n "$((line_number + 1))p" "$coordinates_file")
        line_i2=$(sed -n "$((line_number + 2))p" "$coordinates_file")
        line_i4=$(sed -n "$((line_number + 4))p" "$coordinates_file")
        line_i5=$(sed -n "$((line_number + 5))p" "$coordinates_file")
        line_i7=$(sed -n "$((line_number + 7))p" "$coordinates_file")
        ID_i=$(echo "$line_i" | awk '{print $1}')
        TYPE_i=$(echo "$line_i" | awk '{print $2}')
        x_i=$(echo "$line_i" | awk '{print $3}')
        y_i=$(echo "$line_i" | awk '{print $4}')
        z_i=$(echo "$line_i" | awk '{print $5}')
        ID_i1=$(echo "$line_i1" | awk '{print $1}')
        TYPE_i1=$(echo "$line_i1" | awk '{print $2}')
        x_i1=$(echo "$line_i1" | awk '{print $3}')
        y_i1=$(echo "$line_i1" | awk '{print $4}')
        z_i1=$(echo "$line_i1" | awk '{print $5}')
        ID_i2=$(echo "$line_i2" | awk '{print $1}')
        TYPE_i2=$(echo "$line_i2" | awk '{print $2}')
        x_i2=$(echo "$line_i2" | awk '{print $3}')
        y_i2=$(echo "$line_i2" | awk '{print $4}')
        z_i2=$(echo "$line_i2" | awk '{print $5}')
        ID_i4=$(echo "$line_i4" | awk '{print $1}')
        TYPE_i4=$(echo "$line_i4" | awk '{print $2}')
        x_i4=$(echo "$line_i4" | awk '{print $3}')
        y_i4=$(echo "$line_i4" | awk '{print $4}')
        z_i4=$(echo "$line_i4" | awk '{print $5}')
        ID_i5=$(echo "$line_i5" | awk '{print $1}')
        TYPE_i5=$(echo "$line_i5" | awk '{print $2}')
        x_i5=$(echo "$line_i5" | awk '{print $3}')
        y_i5=$(echo "$line_i5" | awk '{print $4}')
        z_i5=$(echo "$line_i5" | awk '{print $5}')
        ID_i7=$(echo "$line_i7" | awk '{print $1}')
        TYPE_i7=$(echo "$line_i7" | awk '{print $2}')
        x_i7=$(echo "$line_i7" | awk '{print $3}')
        y_i7=$(echo "$line_i7" | awk '{print $4}')
        z_i7=$(echo "$line_i7" | awk '{print $5}')

        bond_length_i2_i4=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i4 $y_i4 $z_i4)
        bond_length_i2_i7=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i7 $y_i7 $z_i7)
        bond_length_i_i4=$(compute_distance $x_i $y_i $z_i $x_i4 $y_i4 $z_i4)
        bond_length_i2_i5=$(compute_distance $x_i2 $y_i2 $z_i2 $x_i5 $y_i5 $z_i5)

        if (( $(echo "$bond_length_i2_i4 < 0" | bc -l) )); then
            bond_length_i2_i4=0
        fi
        if (( $(echo "$bond_length_i2_i7 < 0" | bc -l) )); then
            bond_length_i2_i7=0
        fi
        if (( $(echo "$bond_length_i_i4 < 0" | bc -l) )); then
            bond_length_i_i4=0
        fi
        if (( $(echo "$bond_length_i2_i5 < 0" | bc -l) )); then
            bond_length_i2_i5=0
        fi
        [ "$TYPE_i" == "1" ] && TYPE_i="Mo" || TYPE_i="S"
        [ "$TYPE_i2" == "1" ] && TYPE_i2="Mo" || TYPE_i2="S"
        [ "$TYPE_i4" == "1" ] && TYPE_i4="Mo" || TYPE_i4="S"
        [ "$TYPE_i5" == "1" ] && TYPE_i5="Mo" || TYPE_i5="S"
        [ "$TYPE_i7" == "1" ] && TYPE_i7="Mo" || TYPE_i7="S"

        #echo "Mo ($ID_i): $ID_i2-$ID_i4 ${TYPE_i2}1-${TYPE_i4}2 $bond_length_i2_i4"
        #echo "Mo ($ID_i): $ID_i2-$ID_i7 ${TYPE_i2}1-${TYPE_i7}3 $bond_length_i2_i7"
        #echo "Mo ($ID_i): $ID_i-$ID_i4 ${TYPE_i}1-${TYPE_i4}2 $bond_length_i_i4"
        #echo "Mo ($ID_i): $ID_i2-$ID_i5 ${TYPE_i2}1-${TYPE_i5}1 $bond_length_i2_i5"

        bond_angle_i4_i_i2=$(compute_angle $x_i4 $y_i4 $z_i4 $x_i $y_i $z_i $x_i2 $y_i2 $z_i2)
        bond_angle_i_i2_i1=$(compute_angle $x_i $y_i $z_i $x_i4 $y_i4 $z_i4 $x_i1 $y_i1 $z_i1)
        bond_angle_i2_i1_i3=$(compute_angle $x_i2 $y_i2 $z_i2 $x_i1 $y_i1 $z_i1 $x_i3 $y_i3 $z_i3)

        #echo "Mo ($ID_i): $ID_i4-$ID_i-$ID_i2 ${TYPE_i4}3-${TYPE_i}1-${TYPE_i2}1 $bond_angle_i4_i_i2"
        #echo "Mo ($ID_i): $ID_i-$ID_i2-$ID_i5 ${TYPE_i}1-${TYPE_i2}1-${TYPE_i5}2 $bond_angle_i_i2_i5"
        #echo "Mo ($ID_i): $ID_i2-$ID_i5-$ID_i7 ${TYPE_i2}1-${TYPE_i5}2-${TYPE_i7}2 $bond_angle_i2_i5_i7"
        printf "%d        %d-%d    %.3f         %d-%d    %.3f         %d-%d    %.3f         %d-%d    %.3f           %d-%d-%d  %.3f               %d-%d-%d  %.3f         %d-%d-%d  %.3f\n" "$atom_id" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i2" "$ID_i7" "$bond_length_i2_i7" "$ID_i" "$ID_i4" "$bond_length_i_i4" "$ID_i2" "$ID_i5" "$bond_length_i2_i5" "$ID_i4" "$ID_i" "$ID_i2" "$bond_angle_i4_i_i2" "$ID_i" "$ID_i2" "$ID_i5" "$bond_angle_i_i2_i5" "$ID_i2" "$ID_i5" "$ID_i7" "$bond_angle_i2_i5_i7"  >> "$output_file"
        printf "%d %d-%d %.3f %d-%d %.3f %d-%d %.3f %d-%d %.3f %d-%d-%d %.3f %d-%d-%d %.3f %d-%d-%d %.3f\n" "$atom_id" "$ID_i2" "$ID_i4" "$bond_length_i2_i4" "$ID_i2" "$ID_i7" "$bond_length_i2_i7" "$ID_i" "$ID_i4" "$bond_length_i_i4" "$ID_i2" "$ID_i5" "$bond_length_i2_i5" "$ID_i4" "$ID_i" "$ID_i2" "$bond_angle_i4_i_i2" "$ID_i" "$ID_i2" "$ID_i5" "$bond_angle_i_i2_i5" "$ID_i2" "$ID_i5" "$ID_i7" "$bond_angle_i2_i5_i7" >> "$output_file_short"

    fi
fi

done
    cd ../../..
