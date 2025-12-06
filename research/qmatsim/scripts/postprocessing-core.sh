#!/bin/bash
# === postprocessing-core.sh ===
# Core postprocessing script for SIESTA calculations
# This script is generated from a template

set -e  # Exit on error

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

shopt -s extglob
shopt -s expand_aliases

# Optional: Load conda/modules if available
if command -v module >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
    conda init 2>/dev/null || true
    conda activate siesta-env 2>/dev/null || echo "Warning: siesta-env conda environment not found"
fi

########################################################################
############################## INITIALIZATION ##########################
########################################################################

material=materialVAR
type=typeVAR
structure=structureVAR
supercell=supercellVAR
functional=functionalVAR
SOC=socVAR
direction=directionVAR

bandspost=bandspostVAR

   # fullBZ + SUMMARY TABLE OF BANDS STATS
   WFSXRSpost=WFSXRSpostVAR

   # BANDS RUN ALONG LINES DISECTING A CIRCLE AROUND A SYMMETRY POINT
   bands2Dpathspost=bands2DpathspostVAR

PDOSXMLpost=PDOSXMLpostVAR

COOPPDOSpost=COOPPDOSpostVAR
FATBANDSpost=FATBANDSpostVAR
PDOSXMLpost=PDOSXMLpostVAR

DENCHARpost=DENCHARpostVAR

ATOMSpost=ATOMSpostVAR


# Define the function
copy_material_files() {
    if [ "$1" = "MoS2" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/Mo.psf .
            cp pseudopotentials/"$2"-SOC/S.psf .
        else
            cp pseudopotentials/"$2"/Mo.psf .
            cp pseudopotentials/"$2"/S.psf .
        fi
    elif [ "$1" = "MoSe2" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/Mo.psf .
            cp pseudopotentials/"$2"-SOC/Se.psf .
        else
            cp pseudopotentials/"$2"/Mo.psf .
            cp pseudopotentials/"$2"/Se.psf .
        fi
    elif [ "$1" = "WS2" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/W.psf .
            cp pseudopotentials/"$2"-SOC/S.psf .
        else
            cp pseudopotentials/"$2"/W.psf .
            cp pseudopotentials/"$2"/S.psf .
        fi
    elif [ "$1" = "WS2" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/W.psf .
            cp pseudopotentials/"$2"-SOC/S.psf .
        else
            cp pseudopotentials/"$2"/W.psf .
            cp pseudopotentials/"$2"/S.psf .
        fi
    elif [ "$1" = "WSe2" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/W.psf .
            cp pseudopotentials/"$2"-SOC/Se.psf .
        else
            cp pseudopotentials/"$2"/W.psf .
            cp pseudopotentials/"$2"/Se.psf .
        fi
    elif [ "$1" = "C" ]; then
        if [ "$3" = "T" ]; then
            cp pseudopotentials/"$2"-SOC/C.psf .
        else
            cp pseudopotentials/"$2"/C.psf .
        fi
    fi
}



if [ $structure == "primitive" -o $structure == "Primitive" -o $structure == "PRIMITIVE" -o $structure == "p" -o $structure == "P" ];
then
    structure=primitive
else
    structure=rectangular
fi

if [ $bandspost == "yes" -o $bandspost == "Yes" -o $bandspost == "YES" -o $bandspost == "y" -o $bandspost == "Y" ];
then
    bandspost=yes
fi

if [ $WFSXRSpost == "yes" -o $WFSXRSpost == "Yes" -o $WFSXRSpost == "YES" -o $WFSXRSpost == "y" -o $WFSXRSpost == "Y" ];
then
    WFSXRSpost=yes
fi

if [ $bands2Dpathspost == "yes" -o $bands2Dpathspost == "Yes" -o $bands2Dpathspost == "YES" -o $bands2Dpathspost == "y" -o $bands2Dpathspost == "Y" ];
then
    bands2Dpathspost=yes
fi

if [ $PDOSXMLpost == "yes" -o $PDOSXMLpost == "Yes" -o $PDOSXMLpost == "YES" -o $PDOSXMLpost == "y" -o $PDOSXMLpost == "Y" ];
then
    PDOSXMLpost=yes
fi

if [ $COOPPDOSpost == "yes" -o $COOPPDOSpost == "Yes" -o $COOPPDOSpost == "YES" -o $COOPPDOSpost == "y" -o $COOPPDOSpost == "Y" ];
then
    COOPPDOSpost=yes
fi

if [ $FATBANDSpost == "yes" -o $FATBANDSpost == "Yes" -o $FATBANDSpost == "YES" -o $FATBANDSpost == "y" -o $FATBANDSpost == "Y" ];
then
    FATBANDSpost=yes
fi

if [ $DENCHARpost == "yes" -o $DENCHARpost == "Yes" -o $DENCHARpost == "YES" -o $DENCHARpost == "y" -o $DENCHARpost == "Y" ];
then
    DENCHARpost=yes
fi

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    ATOMSpost=yes
fi

mainDir=$(pwd)


# MoS2 specific processing
if [ "$material" == "MoS2" ]; then
  atom1="Mo"
  atom2="S"
elif [ "$material" == "MoSe2" ]; then
  atom1="Mo"
  atom2="Se"
elif [ "$material" == "WS2" ]; then
  atom1="W"
  atom2="S"
elif [ "$material" == "WSe2" ]; then
  atom1="W"
  atom2="Se"
elif [ "$material" == "C" ]; then
  atom1="C"
  atom2="C"
fi

# Add this after your initialization section for a clear parameter summary
echo ""
echo "===== SIESTA WORKFLOW PARAMETERS ====="
echo "Material: $material"
echo "Type: $type"
echo "Structure: $structure"
echo "Supercell: $supercell"
echo "Functional: $functional"
echo "SOC: $SOC"
echo "Direction: $direction"
echo ""
echo "ATOM TYPES:"
echo "Atom 1: $atom1"
echo "Atom 2: $atom2"
echo ""
echo "POST-PROCESSING OPTIONS:"
echo "Bands: $bandspost"
echo "WFSXRS: $WFSXRSpost"
echo "2D Bands Paths: $bands2Dpathspost"
echo "PDOS XML: $PDOSXMLpost"
echo "COOP PDOS: $COOPPDOSpost"
echo "Fat Bands: $FATBANDSpost"
echo "DENCHAR: $DENCHARpost"
echo "Atoms: $ATOMSpost"
echo "===================================="
echo ""

rm -rf *.err
rm -rf *.out

########################################################################
############################## RELAXATION ##############################
########################################################################

echo "############################## RELAXATION ##############################"

cd 1-Relaxation

cp siesta.out siesta-backup.out
cp ../Results/ExtractedParameters/siesta.out .

cp $material.ANI temp.txt

Natoms=$(head -n 4 $material.STRUCT_OUT | tail -n 1)
Natoms0=Natoms
origin="0 0 0"
lattice="$(grep -A3 "outcell: Unit cell vectors (Ang):" siesta.out)"

echo "$lattice" > tempLattice.tmp

a1Vectors="$(sed -n '2~5p' tempLattice.tmp | sed 's/^ *//g')"
a2Vectors="$(sed -n '3~5p' tempLattice.tmp | sed 's/^ *//g')"
a3Vectors="$(sed -n '4~5p' tempLattice.tmp | sed 's/^ *//g')"

echo "$a1Vectors" > tempa1Vectors.tmp
echo "$a2Vectors" > tempa2Vectors.tmp
echo "$a3Vectors" > tempa3Vectors.tmp

a1xtemp="$(tr -s ' ' < tempa1Vectors.tmp | cut -d" " -f1 | awk -F' ' '{print $1}')"
a2ytemp="$(tr -s ' ' < tempa2Vectors.tmp | cut -d" " -f2 | cut -d' ' -f1)"
a3ztemp="$(tr -s ' ' < tempa3Vectors.tmp | cut -d" " -f3 | cut -d' ' -f1)"

a1x="$(echo $a1xtemp | cut -d' ' -f1)"
a2y="$(echo $a2ytemp | cut -d' ' -f1)"
a3z="$(echo $a3ztemp | cut -d' ' -f1)"

paste tempa1Vectors.tmp tempa2Vectors.tmp  tempa3Vectors.tmp | tr -s " " | head -n -1 > tempaVectors.tmp

latticeVectors=`cat tempaVectors.tmp`
while IFS= read -r line; do
     echo "Lattice=\"$line\" Properties=species:S:1:pos:R:3"
done <<< "$latticeVectors" > tempHeaders.tmp

tail -n1 tempHeaders.tmp > latticeHeader.txt


while IFS= read -r line; do
     echo "Lattice=\"$line\" Properties=species:S:1:pos:R:3"
done <<< "$latticeVectors" > tempHeaders-label.tmp
tail -n1 tempHeaders-label.tmp > latticeHeader-label.txt

while IFS= read -r line; do
     echo "Lattice=\"$line\" Properties=type:I:1:pos:R:3"
done <<< "$latticeVectors" > tempHeaders-type.tmp
tail -n1 tempHeaders-type.tmp > latticeHeader-type.txt

while IFS= read -r line; do
     echo "Lattice=\"$line\" Properties=id:I:1:species:S:1:pos:R:3"
done <<< "$latticeVectors" > tempHeaders-id-label.tmp
tail -n1 tempHeaders-id-label.tmp > latticeHeader-id-label.txt
while IFS= read -r line; do
     echo "Lattice=\"$line\" Properties=id:I:1:type:I:1:pos:R:3"
done <<< "$latticeVectors" > tempHeaders-id-type.tmp
tail -n1 tempHeaders-id-type.tmp > latticeHeader-id-type.txt

divider=$(($Natoms+2))
cp $material.ANI tempANIMATION.tmp
awk '(NR-1)%'$divider'==2{getline this<"tempHeaders.tmp";print this} 1' tempANIMATION.tmp | tr -s " " > ANIMATION.txt
grep '\S' ANIMATION.txt > tmp.txt && mv tmp.txt ANIMATION.txt

mv ANIMATION.txt structure.ANI

sed -n -e '/E_KS(eV)/ s/.*\= *//p' siesta.out > EKS_CG-steps.out


grep "SCF Convergence by EDM+DM+H criterion" -B 2  siesta.out > EFtemp1.outtemp
grep -A1 --no-group-separator 'scf:' EFtemp1.outtemp | tr -s ' ' | cut -d ' ' -f8- > EFtemp2.outtemp
sed -i '/^$/d' EFtemp2.outtemp

cut -d ' ' -f2- --complement EFtemp2.outtemp > EF_CG-steps.out


EKSfinal="$(tail -1 EKS_CG-steps.out)"
EFfinal="$(tail -1 EF_CG-steps.out)"
EF=$EFfinal


sed -n -e '/Tot\b/p' siesta.out | head -n -1 | tr -s ' ' | cut -d ' ' -f2 --complement > FatomTotal_CG-steps.out

grep constrained siesta.out | tr -s ' ' | cut -d ' ' -f3 > FatomMax_CG-steps.out

grep "redata: Force tolerance                             =" siesta.out | tr -s ' ' | cut -d ' ' -f5- | cut -d ' ' -f1 > Ftol.out



stress="$( grep "Stress tensor Voigt" siesta.out | cut -d' ' -f9- | tr -s ' ' | cut -d' ' -f5- --complement)"
grep "Stress tensor Voigt" siesta.out | cut -d' ' -f9- | tr -s ' ' | cut -d' ' -f5- --complement | cut -d' ' -f3- --complement > Sxx.outtemp
grep "Stress tensor Voigt" siesta.out | cut -d' ' -f9- | tr -s ' ' | cut -d' ' -f5- --complement | cut -d' ' -f3- | cut -d' ' -f2- --complement > Syy.outtemp
grep "Stress tensor Voigt" siesta.out | cut -d' ' -f9- | tr -s ' ' | cut -d' ' -f5- --complement | cut -d' ' -f4- > Szz.outtemp

paste Sxx.outtemp Syy.outtemp Szz.outtemp | column -s $'\t' -t > S_CG-steps.out

sed -n -e '/Pressure (total)/ s/.*\: *//p' siesta.out | cut -d ' ' -f3 --complement > P_CG-steps.out

rm -rf *.outtemp


grep "outcell: Cell vector modules *" siesta.out | tr -s ' ' | tail -n1 | cut -f7,8,9 -d ' ' > file1.tmp
grep "outcell: Cell angles *" siesta.out | tr -s ' ' | tail -n1 | cut -f6,7,8 -d ' ' > file2.tmp
grep "outcell: Cell volume *" siesta.out | tr -s ' ' | tail -n1 | cut -f6 -d ' ' > file3.tmp

cat file1.tmp file2.tmp file3.tmp > lattice.out

rm -rf *.tmp
rm -rf *.outtemp

# Use the defined atoms in the processing command
sed -n '/NAME/,/EOF/p' "$material.xtl" | sed '1d; $d' | tr -s ' ' | cut -d$' ' -f -5 | nl -s ' ' | tr -s ' ' | \
    awk ' { t = $1; $1 = $2; $2 = t; print; } ' | sed "s/$atom1/1/g" | sed "s/$atom2/2/g" > structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt

awk '{temp=$1; $1=$2; $2=temp; print}' structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > structure-Index-TypeNumber-x-y-z_FRACTIONAL.txt

cp structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt ../2-Static/
cp structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt ../2-Static/

cp structure-Index-TypeNumber-x-y-z_FRACTIONAL.txt ../3-Bands
cp structure-Index-TypeNumber-x-y-z_FRACTIONAL.txt ../3-Bands

px="$(grep "siesta: Electric dipole (Debye) =" siesta.out | tr -s ' ' | awk '{print $(NF-2)}')"
py="$(grep "siesta: Electric dipole (Debye) =" siesta.out | tr -s ' ' | awk '{print $(NF-1)}')"
pz="$(grep "siesta: Electric dipole (Debye) =" siesta.out | tr -s ' ' | awk '{print $(NF)}')"

echo $px > px.out
echo $py > py.out
echo $pz > pz.out

cp siesta.out siesta-backup.out

cd ..

echo "############################## RELAXATION COMPLETED ##############################"




echo "############################## RELAXATION COMPLETED ##############################"










########################################################################
############################## POSTPROCESSING ##############################
########################################################################



mkdir -p Postprocessing
cd Postprocessing









########################################################################
############################## BANDS ##############################
########################################################################


echo "############################## BANDS ##############################"

if [ $bandspost == "yes" -o $bandspost == "Yes" -o $bandspost == "YES" -o $bandspost == "y" -o $bandspost == "Y" ];
then

    rm -rf GNUBANDS
    mkdir -p GNUBANDS
    cd GNUBANDS

        rm * 2> /dev/null



# Path to the SIESTA bands file
SIESTA_FILE="../../3-Bands/3-Bands.fdf"

# Check if the file exists
if [ ! -f "$SIESTA_FILE" ]; then
    echo "Error: SIESTA bands file not found at $SIESTA_FILE"
    exit 1
fi

# Extract the BandLines block
BAND_LINES=$(awk '/^%block BandLines/,/^%endblock BandLines/' "$SIESTA_FILE")

# Extract all the k-point lines
KPOINT_LINES=$(echo "$BAND_LINES" | grep -v "%block" | grep -v "%endblock" | awk '{print $0}')

# Count the total number of lines (includes starting point)
TOTAL_LINES=$(echo "$KPOINT_LINES" | wc -l)

# Extract the k-point counts directly
NKL_VALUES=($(echo "$KPOINT_LINES" | awk '{print $1}'))

# Determine if we're dealing with hexagonal or rectangular BZ
# Look for high-symmetry point labels in the BandLines block
if echo "$BAND_LINES" | grep -q "X" && echo "$BAND_LINES" | grep -q "S" && echo "$BAND_LINES" | grep -q "Y"; then
    # Rectangular BZ (Γ-X-S-Y-Γ) has 4 segments
    BZ_TYPE="rectangular"
    EXPECTED_SEGMENTS=4
elif echo "$BAND_LINES" | grep -q "M" && echo "$BAND_LINES" | grep -q "K"; then
    # Hexagonal BZ (Γ-M-K-Γ) has 3 segments
    BZ_TYPE="hexagonal"
    EXPECTED_SEGMENTS=3
else
    # Default to hexagonal if can't determine
    BZ_TYPE="unknown"
    EXPECTED_SEGMENTS=$((TOTAL_LINES - 1))
    echo "Warning: Cannot determine BZ type from high-symmetry points."
    echo "Using number of k-point lines to determine segments: $EXPECTED_SEGMENTS segments"
fi

# The first line is just the starting point, so skip it
echo "# NkL values for $BZ_TYPE BZ with $EXPECTED_SEGMENTS segments" > NkL_values.txt

# Write the NkL values, skipping the first one (starting point)
for ((i=1; i<${#NKL_VALUES[@]}; i++)); do
    echo "${NKL_VALUES[$i]}" >> NkL_values.txt
done

# Check if we need to adjust the number of segments
ACTUAL_SEGMENTS=$((${#NKL_VALUES[@]} - 1))

# If we have fewer segments than expected for rectangular BZ, add the last value again
if [ "$BZ_TYPE" == "rectangular" ] && [ $ACTUAL_SEGMENTS -lt $EXPECTED_SEGMENTS ]; then
    echo "${NKL_VALUES[$((${#NKL_VALUES[@]}-1))]}" >> NkL_values.txt
    echo "Added extra segment for rectangular BZ compatibility"
fi

echo "BZ type detected: $BZ_TYPE"
echo "Segments detected: $ACTUAL_SEGMENTS (expected $EXPECTED_SEGMENTS)"
echo "NkL values extracted:"

# Print the NkL values with labels
SEGMENT_NAMES=("" "Γ→M" "M→K" "K→Γ" "Γ→X" "X→S" "S→Y" "Y→Γ")
if [ "$BZ_TYPE" == "rectangular" ]; then
    START_IDX=4
else
    START_IDX=1
fi

cat NkL_values.txt | grep -v "#" | while read -r line; do
    echo "NkL$COUNTER = $line (${SEGMENT_NAMES[$START_IDX]})"
    COUNTER=$((COUNTER+1))
    START_IDX=$((START_IDX+1))
done

echo "Values written to NkL_values.txt"

# Extract M and N
M=$(echo "$supercell" | cut -d'x' -f1)
N=$(echo "$supercell" | cut -d'x' -f2)

# Create MN_values.txt with extracted values
echo "# M and N values for BZ folding" > MN_values.txt
echo "$M" >> MN_values.txt
echo "$N" >> MN_values.txt
echo "Default M=$M, N=$N values written to MN_values.txt"

        cp ../../3-Bands/$material.bands .

        gnubands $material.bands > $material-bands.dat

        EF=$(sed -n -e '/E_F/ s/.*\= *//p' *.dat)
        EFbands=$EF
        echo $EF > EF.out

        NpointsBandLines=$(awk '/%block BandLines/,/%endblock BandLines/ { if ($1 ~ /^[0-9]+$/) sum += $1 } END { print sum }' ../../3-Bands/3-Bands.fdf)

        k=$(sed -n -e '/k_min, k_max/ s/.*\= *//p' $material-bands.dat | tr -s " ")
        BohrToAng=1.8897;
        kmin0=$(echo $k | awk '{print $1}')
        kmax0=$(echo $k | awk '{print $2}')
        kmin=$(bc <<< 'scale=2; '$kmin0'*'$BohrToAng'')
        kmax=$(bc <<< 'scale=2; '$kmax0'*'$BohrToAng'')

        E=$(sed -n -e '/E_min, E_max/ s/.*\= *//p' $material-bands.dat | tr -s " ")
        Emin=$(echo $E | awk '{print $1}')
        Emax=$(echo $E | awk '{print $2}')
        Nvalues=$(sed -n -e '/Nbands, Nspin, Nk/ s/.*\= *//p' $material-bands.dat | tr -s " ")
        Nbands=$(echo $Nvalues | awk '{print $1}')
        Nspin=$(echo $Nvalues | awk '{print $2}')
        Nk=$(echo $Nvalues | awk '{print $3}')
        subtitlePlotLine1="E_F=$EF (eV), [$kmin, $kmax] 1/Ang"
        subtitlePlotLine2="$Nk k-points, $Nbands bands, $Nspin-spin"
        echo "$subtitlePlotLine1" > title1.txt
        echo "$subtitlePlotLine2" > title2.txt

        sed '/^$/d' < $material-bands.dat | tr -s " " | sed  '/^#/ d' > tempBandsBohr.tmp
        BohrToAng=$(bc <<< 'scale=2; '1.8897'');
        sed 's/^[ \t]*//' tempBandsBohr.tmp > tempBandsBohrNew.tmp
        awk -v var='0.0' '{ printf("%f %f %d\n", $1*('$BohrToAng'), $2, $3 ) }' < tempBandsBohrNew.tmp > tempBands.tmp
        awk -v var=$EF -F " " '{print $1 " " $2-var}' < tempBands.tmp > bands.txt
        awk -F"|" '{print > ($2>0?"bandsPositive.txt":"bandsNegative.txt")}' bands.txt


        if [ $structure == "primitive" ]; then
            cut -sd- -f1- bands.txt > bandsNegative.txt
            diff bandsNegative.txt bands.txt | grep '^>' | sed 's/^>\ //' > bandsPositive.txt
	    
            head -n $NpointsBandLines bandsPositive.txt > CB1.txt
            tail -n $NpointsBandLines bandsNegative.txt > VB1.txt
            cp bandsPositive.txt bandsPositiveNoCB1.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB1.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegative.txt > bandsNegativeNoVB1.txt

            head -n $NpointsBandLines bandsPositiveNoCB1.txt > CB2.txt
            tail -n $NpointsBandLines bandsNegativeNoVB1.txt > VB2.txt
            cp bandsPositiveNoCB1.txt bandsPositiveNoCB2.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB2.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegativeNoVB1.txt > bandsNegativeNoVB2.txt

            head -n $NpointsBandLines bandsPositiveNoCB2.txt > CB3.txt
            tail -n $NpointsBandLines bandsNegativeNoVB2.txt > VB3.txt
            cp bandsPositiveNoCB2.txt bandsPositiveNoCB3.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB3.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegativeNoVB2.txt > bandsNegativeNoVB3.txt
        else
            cut -sd- -f1- bands.txt > bandsNegative.txt
            diff bandsNegative.txt bands.txt | grep '^>' | sed 's/^>\ //' > bandsPositive.txt

            head -n $NpointsBandLines bandsPositive.txt > CB1.txt
            tail -n $NpointsBandLines bandsNegative.txt > VB1.txt
            cp bandsPositive.txt bandsPositiveNoCB1.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB1.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegative.txt > bandsNegativeNoVB1.txt

            head -n $NpointsBandLines bandsPositiveNoCB1.txt > CB2.txt
            tail -n $NpointsBandLines bandsNegativeNoVB1.txt > VB2.txt
            cp bandsPositiveNoCB1.txt bandsPositiveNoCB2.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB2.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegativeNoVB1.txt > bandsNegativeNoVB2.txt

            head -n $NpointsBandLines bandsPositiveNoCB2.txt > CB3.txt
            tail -n $NpointsBandLines bandsNegativeNoVB2.txt > VB3.txt
            cp bandsPositiveNoCB2.txt bandsPositiveNoCB3.txt
            sed -i '1,'${NpointsBandLines}'d' bandsPositiveNoCB3.txt
            sed -e :a -e '$d;N;2,'${NpointsBandLines}'ba' -e 'P;D' bandsNegativeNoVB2.txt > bandsNegativeNoVB3.txt
        fi


    echo ">>>>>>>>>>>>>>> GNUBANDS ANALYSIS DONE! <<<<<<<<<<<<<<<<<"


    # WFSXR START
    if [ $WFSXRSpost == "yes" -o $WFSXRSpost == "Yes" -o $WFSXRSpost == "YES" -o $WFSXRSpost == "y" -o $WFSXRSpost == "Y" ];
    then

        echo ">>>>>>>>>>>>>>> START READWFX <<<<<<<<<<<<<<<<<"

        readwfx -m -6.5 -M -2.5 ../../1-Relaxation/$material.fullBZ.WFSX > ../../1-Relaxation/$material.fullBZ.WFSXR
        echo "READWFX (FULL BZ - RELAXATION) DONE!"
        readwfx -m -6.5 -M -2.5 ../../2-Static/$material.fullBZ.WFSX > ../../2-Static/$material.fullBZ.WFSXR
        echo "READWFX (FULL BZ - STATIC) DONE!"
        readwfx -m -6.5 -M -2.5 ../../3-Bands/$material.bands.WFSX >  ../../3-Bands/$material.bands.WFSXR
        echo "READWFX (BANDS) DONE!"
        readwfx -m -6.5 -M -2.5 ../../3-Bands/$material.selected.WFSX > ../../3-Bands/$material.selected.WFSXR
        echo "READWFX (SELECTED WAVEFUNCTIONS) DONE!"

        echo ">>>>>>>>>>>>>>> READWFX DONE! <<<<<<<<<<<<<<<<<"



        mkdir -p fullBZ
        mkdir -p bands



        echo ">>>>>>>>>>>>>>> START 2D (FULL BZ) <<<<<<<<<<<<<<<<<"

        cd fullBZ

        grep "k-point =" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz.txt
        awk -F ' ' '{print $1, $2}' index-kx-ky-kz.txt > index-kx.txt
        awk -F ' ' '{print $1, $3}' index-kx-ky-kz.txt > index-ky.txt
        awk -F ' ' '{print $1, $4}' index-kx-ky-kz.txt > index-kz.txt
        awk -F ' ' '{print $1}' index-kx-ky-kz.txt > index.txt
        awk -F ' ' '{print $2}' index-kx-ky-kz.txt > kx.tmp
        awk -F ' ' '{print $3}' index-kx-ky-kz.txt > ky.tmp
        awk -F ' ' '{print $4}' index-kx-ky-kz.txt > kz.tmp
        cp kx.tmp kxx.txt
        cp ky.tmp kyy.txt
        cp kz.tmp kzz.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz.txt
        paste index.txt kx.txt ky.txt kz.txt | tr -s ' ' > index-kx-ky-kz.txt
        paste kx.txt ky.txt kz.txt | tr -s ' ' > kx-ky-kz.txt
        paste kx.txt ky.txt | tr -s ' ' > kx-ky.txt
        paste index.txt kx.txt ky.txt | tr -s ' ' > index-kx-ky.txt

        grep "Wavefunction = " ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | sort -k3 -n  > bandIndex-E.tmp
        awk -v var=$EFbands -F " " '{print $3 " " $7-var}' bandIndex-E.tmp > bandIndex-E-EF.tmp
        nl -w2 -s ' ' bandIndex-E-EF.tmp > lineNumber-bandIndex-E-EF.tmp
        firstPositiveValue=$(grep -oE '(^| )[0-9]+\.[0-9]+' lineNumber-bandIndex-E-EF.tmp | tr -d ' ' | head -n1)
        CBM1band=$(grep -F "$firstPositiveValue" lineNumber-bandIndex-E-EF.tmp | awk -F ' ' '{print $2}' | head -n1 | awk '{print $1;}')
        CBM2band=$(($CBM1band+1))
        CBM3band=$(($CBM1band+2))
        VBM1band=$(($CBM1band-1))
        VBM2band=$(($VBM1band-1))
        VBM3band=$(($VBM1band-2))

        readwfx -b $CBM1band -B $CBM1band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.CB1.NoEF.WFSXR
        readwfx -b $CBM2band -B $CBM2band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.CB2.NoEF.WFSXR
        readwfx -b $CBM3band -B $CBM3band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.CB3.NoEF.WFSXR
        readwfx -b $VBM1band -B $VBM1band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.VB1.NoEF.WFSXR
        readwfx -b $VBM2band -B $VBM2band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.VB2.NoEF.WFSXR
        readwfx -b $VBM3band -B $VBM3band ../../../2-Static/$material.fullBZ.WFSX > ../../../2-Static/$material.VB3.NoEF.WFSXR

        grep "Wavefunction = *$CBM1band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
        grep "Wavefunction = *$CBM2band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB2-2D-E-EF.tmp
        grep "Wavefunction = *$CBM3band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB3-2D-E-EF.tmp
        grep "Wavefunction = *$VBM1band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
        grep "Wavefunction = *$VBM2band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB2-2D-E-EF.tmp
        grep "Wavefunction = *$VBM3band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB3-2D-E-EF.tmp
        grep "Wavefunction = *$CBM1band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
        grep "Wavefunction = *$CBM2band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB2-2D-E.tmp
        grep "Wavefunction = *$CBM3band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB3-2D-E.tmp
        grep "Wavefunction = *$VBM1band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
        grep "Wavefunction = *$VBM2band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB2-2D-E.tmp
        grep "Wavefunction = *$VBM3band" ../../../2-Static/$material.fullBZ.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB3-2D-E.tmp

        paste kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D.txt
        paste kx-ky-kz.txt CB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB2-2D.txt
        paste kx-ky-kz.txt CB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB3-2D.txt
        paste kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D.txt
        paste kx-ky-kz.txt VB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB2-2D.txt
        paste kx-ky-kz.txt VB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB3-2D.txt
        paste index-kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D.txt
        paste index-kx-ky-kz.txt CB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB2-2D.txt
        paste index-kx-ky-kz.txt CB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB3-2D.txt
        paste index-kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D.txt
        paste index-kx-ky-kz.txt VB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB2-2D.txt
        paste index-kx-ky-kz.txt VB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB3-2D.txt

        CBM1E2D=$(sort -n CB1-2D-E-EF.tmp | head -n1)
        CBM2E2D=$(sort -n CB2-2D-E-EF.tmp | head -n1)
        CBM3E2D=$(sort -n CB3-2D-E-EF.tmp | head -n1)
        CBM1E2DnoEF=$(sort -n CB1-2D-E.tmp | head -n1)
        CBM2E2DnoEF=$(sort -n CB2-2D-E.tmp | head -n1)
        CBM3E2DnoEF=$(sort -n CB3-2D-E.tmp | head -n1)
        CBM1state2D=$(grep -B10000 "Eigval (eV) = *$CBM1E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        CBM2state2D=$(grep -B10000 "Eigval (eV) = *$CBM2E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        CBM3state2D=$(grep -B10000 "Eigval (eV) = *$CBM3E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        CBM1k2D=$(grep -B10000 "Eigval (eV) = *$CBM1E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)
        CBM2k2D=$(grep -B10000 "Eigval (eV) = *$CBM2E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)
        CBM3k2D=$(grep -B10000 "Eigval (eV) = *$CBM3E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)
        VBM1E2D=$(sort -n VB1-2D-E-EF.tmp | tail -n1)
        VBM2E2D=$(sort -n VB2-2D-E-EF.tmp | tail -n1)
        VBM3E2D=$(sort -n VB3-2D-E-EF.tmp | tail -n1)
        VBM1E2DnoEF=$(sort -n VB1-2D-E.tmp | tail -n1)
        VBM2E2DnoEF=$(sort -n VB2-2D-E.tmp | tail -n1)
        VBM3E2DnoEF=$(sort -n VB3-2D-E.tmp | tail -n1)
        VBM1state2D=$(grep -B10000 "Eigval (eV) = *$VBM1E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        VBM2state2D=$(grep -B10000 "Eigval (eV) = *$VBM2E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        VBM3state2D=$(grep -B10000 "Eigval (eV) = *$VBM3E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)
        VBM1k2D=$(grep -B10000 "Eigval (eV) = *$VBM1E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)
        VBM2k2D=$(grep -B10000 "Eigval (eV) = *$VBM2E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)
        VBM3k2D=$(grep -B10000 "Eigval (eV) = *$VBM3E2DnoEF" ../../../2-Static/$material.fullBZ.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)

        rm -rf *.tmp

        echo ">>>>>>>>>>>>>>> 2D (FULL BZ) DONE! <<<<<<<<<<<<<<<<<"


        cd ..




        cd bands

        echo ">>>>>>>>>>>>>>> Start 2D (BANDS) <<<<<<<<<<<<<<<<<"

        awk -F ' ' '{print $1}' ../CB1.txt > k.txt
        grep "k-point =" ../../../3-Bands/$material.bands.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz.txt
        awk -F ' ' '{print $1, $2}' index-kx-ky-kz.txt > index-kx.txt
        awk -F ' ' '{print $1, $3}' index-kx-ky-kz.txt > index-ky.txt
        awk -F ' ' '{print $1, $4}' index-kx-ky-kz.txt > index-kz.txt
        awk -F ' ' '{print $1}' index-kx-ky-kz.txt > index.txt
        awk -F ' ' '{print $2}' index-kx-ky-kz.txt > kx.tmp
        awk -F ' ' '{print $3}' index-kx-ky-kz.txt > ky.tmp
        awk -F ' ' '{print $4}' index-kx-ky-kz.txt > kz.tmp
        cp kx.tmp kxx.txt
        cp ky.tmp kyy.txt
        cp kz.tmp kzz.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky.txt
        awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz.txt
        paste index.txt kx.txt ky.txt kz.txt | tr -s ' ' > index-kx-ky-kz.txt
        paste kx.txt ky.txt kz.txt | tr -s ' ' > kx-ky-kz.txt
        paste kx.txt ky.txt | tr -s ' ' > kx-ky.txt
        paste index.txt kx.txt ky.txt | tr -s ' ' > index-kx-ky.txt

        grep "Wavefunction = " ../../../3-Bands/$material.bands.WFSXR | tr -s ' ' | sort -k3 -n  > bandIndex-E.tmp
        awk -v var=$EFbands -F " " '{print $3 " " $7-var}' bandIndex-E.tmp > bandIndex-E-EF.tmp
        nl -w2 -s ' ' bandIndex-E-EF.tmp > lineNumber-bandIndex-E-EF.tmp
        firstPositiveValue=$(grep -oE '(^| )[0-9]+\.[0-9]+' lineNumber-bandIndex-E-EF.tmp | tr -d ' ' | head -n1)
        CBM1band=$(grep -F "$firstPositiveValue" lineNumber-bandIndex-E-EF.tmp | awk -F ' ' '{print $2}' | head -n1 | awk '{print $1;}')
        CBM2band=$(($CBM1band+1))
        CBM3band=$(($CBM1band+2))
        VBM1band=$(($CBM1band-1))
        VBM2band=$(($VBM1band-1))
        VBM3band=$(($VBM1band-2))

        readwfx -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.CB1.NoEF.WFSXR
        readwfx -b $CBM2band -B $CBM2band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.CB2.NoEF.WFSXR
        readwfx -b $CBM3band -B $CBM3band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.CB3.NoEF.WFSXR
        readwfx -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.VB1.NoEF.WFSXR
        readwfx -b $VBM2band -B $VBM2band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.VB2.NoEF.WFSXR
        readwfx -b $VBM3band -B $VBM3band ../../../3-Bands/$material.bands.WFSX > ../../../3-Bands/$material.VB3.NoEF.WFSXR

        grep "Wavefunction = *$CBM1band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
        grep "Wavefunction = *$CBM2band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB2-2D-E-EF.tmp
        grep "Wavefunction = *$CBM3band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB3-2D-E-EF.tmp
        grep "Wavefunction = *$VBM1band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
        grep "Wavefunction = *$VBM2band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB2-2D-E-EF.tmp
        grep "Wavefunction = *$VBM3band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB3-2D-E-EF.tmp
        grep "Wavefunction = *$CBM1band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
        grep "Wavefunction = *$CBM2band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB2-2D-E.tmp
        grep "Wavefunction = *$CBM3band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB3-2D-E.tmp
        grep "Wavefunction = *$VBM1band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
        grep "Wavefunction = *$VBM2band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB2-2D-E.tmp
        grep "Wavefunction = *$VBM3band" ../../../3-Bands/$material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB3-2D-E.tmp

        paste kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D.txt
        paste kx-ky-kz.txt CB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB2-2D.txt
        paste kx-ky-kz.txt CB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB3-2D.txt
        paste kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D.txt
        paste kx-ky-kz.txt VB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB2-2D.txt
        paste kx-ky-kz.txt VB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB3-2D.txt
        paste index-kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D.txt
        paste index-kx-ky-kz.txt CB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB2-2D.txt
        paste index-kx-ky-kz.txt CB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB3-2D.txt
        paste index-kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D.txt
        paste index-kx-ky-kz.txt VB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB2-2D.txt
        paste index-kx-ky-kz.txt VB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB3-2D.txt

        paste kx-ky.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-CB1-2D.txt
        paste kx-ky.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-VB1-2D.txt
        paste index-kx-ky.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-CB1-2D.txt
        paste index-kx-ky.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-VB1-2D.txt
        paste kx-ky.txt CB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-CB2-2D.txt
        paste kx-ky.txt VB2-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-VB2-2D.txt
        paste index-kx-ky.txt CB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-CB2-2D.txt
        paste index-kx-ky.txt VB2-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-VB2-2D.txt
        paste kx-ky.txt CB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-CB3-2D.txt
        paste kx-ky.txt VB3-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-VB3-2D.txt
        paste index-kx-ky.txt CB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-CB3-2D.txt
        paste index-kx-ky.txt VB3-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-VB3-2D.txt

        rm -rf *.tmp

        echo ">>>>>>>>>>>>>>> 2D (BANDS) Done! <<<<<<<<<<<<<<<<<"

        cd ..

        #export MESA_LOADER_DRIVER_OVERRIDE=i965
        #matlab -nodisplay -nosplash -nodesktop -nosoftwareopengl -r "run('contour.m');exit;"

        rm -rf *.tmp




        mkdir -p states

        echo ">>>>>>>>>>>>>>> START BANDS WFSXR STATES ANALYSIS <<<<<<<<<<<<<<<<<"

        Nvalues=$(sed -n -e '/Nbands, Nspin, Nk/ s/.*\= *//p' *.dat | tr -s " ")
        Nbands=$(echo $Nvalues | awk '{print $1}')
        Nspin=$(echo $Nvalues | awk '{print $2}')
        Nk=$(echo $Nvalues | awk '{print $3}')
        Nk2D=$(grep "Nr of k-points =" ../../2-Static/*.WFSXR | tr -s ' ' | awk -F " " '{print $NF}')
        EF=$(sed -n -e '/E_F/ s/.*\= *//p' *.dat)

        gawk '$1=(FNR FS $1)' VB1.txt | sort -u -k3 > VB1-sorted.txt
        gawk '$1=(FNR FS $1)' VB2.txt | sort -u -k3> VB2-sorted.txt
        gawk '$1=(FNR FS $1)' VB3.txt | sort -u -k3 > VB3-sorted.txt

        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < VB1-sorted.txt > VB1-sorted-NoEF.txt
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < VB2-sorted.txt > VB2-sorted-NoEF.txt
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < VB3-sorted.txt > VB3-sorted-NoEF.txt

        head -n 1 VB1-sorted.txt > states/VBM1-state-k-E.tmp
        head -n 1 VB2-sorted.txt > states/VBM2-state-k-E.tmp
        head -n 1 VB3-sorted.txt > states/VBM3-state-k-E.tmp

        gawk '$1=(FNR FS $1)' CB1.txt | sort -u -k3 > CB1-sorted.txt
        gawk '$1=(FNR FS $1)' CB2.txt | sort -u -k3 > CB2-sorted.txt
        gawk '$1=(FNR FS $1)' CB3.txt | sort -u -k3 > CB3-sorted.txt

        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < CB1-sorted.txt > CB1-sorted-NoEF.txt
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < CB2-sorted.txt > CB2-sorted-NoEF.txt
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < CB3-sorted.txt > CB3-sorted-NoEF.txt

        head -n 1 CB1-sorted.txt > states/CBM1-state-k-E.tmp
        head -n 1 CB2-sorted.txt > states/CBM2-state-k-E.tmp
        head -n 1 CB3-sorted.txt > states/CBM3-state-k-E.tmp

# Get the first value of CB1, CB2, CB3, VB1, VB2, VB3 (minimum for CB*, maximum for VB*)
 firstCB1=$(awk 'NR==1 {print $3}' CB1-sorted.txt)
 firstCB2=$(awk 'NR==1 {print $3}' CB2-sorted.txt)
 firstCB3=$(awk 'NR==1 {print $3}' CB3-sorted.txt)
 firstVB1=$(awk 'NR==1 {print $3}' VB1-sorted.txt)
 firstVB2=$(awk 'NR==1 {print $3}' VB2-sorted.txt)
 firstVB3=$(awk 'NR==1 {print $3}' VB3-sorted.txt)

 # Get the last value of CB1, CB2, CB3, VB1, VB2, VB3 (maximum for CB*, minimum for VB*)
 lastCB1=$(awk 'END {print $3}' CB1-sorted.txt)
 lastCB2=$(awk 'END {print $3}' CB2-sorted.txt)
 lastCB3=$(awk 'END {print $3}' CB3-sorted.txt)
 lastVB1=$(awk 'END {print $3}' VB1-sorted.txt)
 lastVB2=$(awk 'END {print $3}' VB2-sorted.txt)
 lastVB3=$(awk 'END {print $3}' VB3-sorted.txt)

 # Output to files
 echo "$firstCB1" > CBM1.out
 echo "$firstCB2" > CBM2.out
 echo "$firstCB3" > CBM3.out
 echo "$firstVB1" > VBM1.out
 echo "$firstVB2" > VBM2.out
 echo "$firstVB3" > VBM3.out

 # Calculate differences
 Eg=$(echo "$firstCB1 - $firstVB1" | bc)
 dCB1=$(echo "$lastCB1 - $firstCB1" | bc)
 dCB2=$(echo "$lastCB2 - $firstCB2" | bc)
 dCB3=$(echo "$lastCB3 - $firstCB3" | bc)
 dVB1=$(echo "$firstVB1 - $lastVB1" | bc)
 dVB2=$(echo "$firstVB2 - $lastVB2" | bc)
 dVB3=$(echo "$firstVB3 - $lastVB3" | bc)

 # Output differences to files
 echo "$Eg" > Eg.out
 echo "$dCB1" > dCB1.out
 echo "$dCB2" > dCB2.out
 echo "$dCB3" > dCB3.out
 echo "$dVB1" > dVB1.out
 echo "$dVB2" > dVB2.out
 echo "$dVB3" > dVB3.out

        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/VBM1-state-k-E.tmp > states/VBM1-state-k-E-NoEF.tmp
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/VBM2-state-k-E.tmp > states/VBM2-state-k-E-NoEF.tmp
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/VBM3-state-k-E.tmp > states/VBM3-state-k-E-NoEF.tmp

        VBM1state=$(cat states/VBM1-state-k-E-NoEF.tmp | awk '{print $1;}')
        VBM2state=$(cat states/VBM2-state-k-E-NoEF.tmp | awk '{print $1;}')
        VBM3state=$(cat states/VBM3-state-k-E-NoEF.tmp | awk '{print $1;}')
        VBM1k=$(cat states/VBM1-state-k-E-NoEF.tmp | awk '{print $2;}')
        VBM2k=$(cat states/VBM2-state-k-E-NoEF.tmp | awk '{print $2;}')
        VBM3k=$(cat states/VBM3-state-k-E-NoEF.tmp | awk '{print $2;}')
        VBM1E=$(cat states/VBM1-state-k-E-NoEF.tmp | awk '{print $3;}')
        VBM2E=$(cat states/VBM2-state-k-E-NoEF.tmp | awk '{print $3;}')
        VBM3E=$(cat states/VBM3-state-k-E-NoEF.tmp | awk '{print $3;}')

        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/CBM1-state-k-E.tmp > states/CBM1-state-k-E-NoEF.tmp
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/CBM2-state-k-E.tmp > states/CBM2-state-k-E-NoEF.tmp
        awk -v var=$EF -F " " '{printf "%d %.5f %.5f\n", $1,$2,$3+var}' < states/CBM3-state-k-E.tmp > states/CBM3-state-k-E-NoEF.tmp

        CBM1state=$(cat states/CBM1-state-k-E-NoEF.tmp | awk '{print $1;}')
        CBM2state=$(cat states/CBM2-state-k-E-NoEF.tmp | awk '{print $1;}')
        CBM3state=$(cat states/CBM3-state-k-E-NoEF.tmp | awk '{print $1;}')
        CBM1k=$(cat states/CBM1-state-k-E-NoEF.tmp | awk '{print $2;}')
        CBM2k=$(cat states/CBM2-state-k-E-NoEF.tmp | awk '{print $2;}')
        CBM3k=$(cat states/CBM3-state-k-E-NoEF.tmp | awk '{print $2;}')
        CBM1E=$(cat states/CBM1-state-k-E-NoEF.tmp | awk '{print $3;}')
        CBM2E=$(cat states/CBM2-state-k-E-NoEF.tmp | awk '{print $3;}')
        CBM3E=$(cat states/CBM3-state-k-E-NoEF.tmp | awk '{print $3;}')


        sed -n '/k-point = [[:space:]]*\<'$CBM1state'\>/,/k-point = [[:space:]]*\<'$(($CBM1state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-CBM1state--band-E.tmp
        sed -n -e '/'$CBM1E'/p' states/line-CBM1state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/CBM1state-band.tmp

        sed -n '/k-point = [[:space:]]*\<'$CBM2state'\>/,/k-point = [[:space:]]*\<'$(($CBM2state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-CBM2state--band-E.tmp
        sed -n -e '/'$CBM2E'/p' states/line-CBM2state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/CBM2state-band.tmp

        sed -n '/k-point = [[:space:]]*\<'$CBM3state'\>/,/k-point = [[:space:]]*\<'$(($CBM3state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-CBM3state--band-E.tmp
        sed -n -e '/'$CBM3E'/p' states/line-CBM3state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/CBM3state-band.tmp

        sed -n '/k-point = [[:space:]]*\<'$VBM1state'\>/,/k-point = [[:space:]]*\<'$(($VBM1state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-VBM1state--band-E.tmp
        sed -n -e '/'$VBM1E'/p' states/line-VBM1state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/VBM1state-band.tmp

        sed -n '/k-point = [[:space:]]*\<'$VBM2state'\>/,/k-point = [[:space:]]*\<'$(($VBM2state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-VBM2state--band-E.tmp
        sed -n -e '/'$VBM2E'/p' states/line-VBM2state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/VBM2state-band.tmp

        sed -n '/k-point = [[:space:]]*\<'$VBM3state'\>/,/k-point = [[:space:]]*\<'$(($VBM3state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | gawk '$1=(FNR FS $1)' | sed -n -e '/Eigval (eV) / s/\= *//p' | awk '{printf "%d %d %.4f0\n", $1, $3, $7}' > states/line-VBM3state--band-E.tmp
        sed -n -e '/'$VBM3E'/p' states/line-VBM3state--band-E.tmp | awk '{printf "%d\n", $2}' | sed '1!G;h;$!d' > states/VBM3state-band.tmp

        degeneracyCBM1state="$(wc -l < states/CBM1state-band.tmp)";
        degeneracyCBM2state="$(wc -l < states/CBM2state-band.tmp)";
        degeneracyCBM3state="$(wc -l < states/CBM3state-band.tmp)";
        degeneracyVBM1state="$(wc -l < states/VBM1state-band.tmp)";
        degeneracyVBM2state="$(wc -l < states/VBM2state-band.tmp)";
        degeneracyVBM3state="$(wc -l < states/VBM3state-band.tmp)";

        CBM1band="$(sed -n '$p' states/CBM1state-band.tmp)"
        CBM2band=$(($CBM1band+1))
        CBM3band=$(($CBM1band+2))
        VBM1band="$(sed -n '1p' states/VBM1state-band.tmp)"
        VBM2band=$(($VBM1band-1))
        VBM3band=$(($VBM1band-2))

        sed -n '/k-point = [[:space:]]*\<'$CBM1state'\>/,/k-point = [[:space:]]*\<'$(($CBM1state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$CBM1band'\>/,/Wavefunction = *\<'$(($CBM1band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-CBM1-state-$CBM1state-band-$CBM1band.txt
        sed -n '/k-point = [[:space:]]*\<'$CBM2state'\>/,/k-point = [[:space:]]*\<'$(($CBM2state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$CBM2band'\>/,/Wavefunction = *\<'$(($CBM2band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-CBM2-state-$CBM2state-band-$CBM2band.txt
        sed -n '/k-point = [[:space:]]*\<'$CBM3state'\>/,/k-point = [[:space:]]*\<'$(($CBM3state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$CBM3band'\>/,/Wavefunction = *\<'$(($CBM3band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-CBM3-state-$CBM3state-band-$CBM3band.txt
        sed -n '/k-point = [[:space:]]*\<'$VBM1state'\>/,/k-point = [[:space:]]*\<'$(($VBM1state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$VBM1band'\>/,/Wavefunction = *\<'$(($VBM1band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-VBM1-state-$VBM1state-band-$VBM1band.txt
        sed -n '/k-point = [[:space:]]*\<'$VBM2state'\>/,/k-point = [[:space:]]*\<'$(($VBM2state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$VBM2band'\>/,/Wavefunction = *\<'$(($VBM2band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-VBM2-state-$VBM2state-band-$VBM2band.txt
        sed -n '/k-point = [[:space:]]*\<'$VBM3state'\>/,/k-point = [[:space:]]*\<'$(($VBM3state+1))'\>/p' ../../3-Bands/$material.bands.WFSXR | sed '$d' | sed -n '/Wavefunction = *\<'$VBM3band'\>/,/Wavefunction = *\<'$(($VBM3band+1))'\>/p' | sed '$d' | sed '$d' > states/extracted-VBM3-state-$VBM3state-band-$VBM3band.txt



        cp states/VBM1-state-k-E.tmp states/VBM1-state-k-E-temp.tmp
        cp states/VBM2-state-k-E.tmp states/VBM2-state-k-E-temp.tmp
        cp states/VBM3-state-k-E.tmp states/VBM3-state-k-E-temp.tmp
        cp states/CBM1-state-k-E.tmp states/CBM1-state-k-E-temp.tmp
        cp states/CBM2-state-k-E.tmp states/CBM2-state-k-E-temp.tmp
        cp states/CBM3-state-k-E.tmp states/CBM3-state-k-E-temp.tmp

        kvectorVBM1=$(sed -n -e '/k-point =/ s/\ *\<'"$VBM1state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')
        kvectorVBM2=$(sed -n -e '/k-point =/ s/\ *\<'"$VBM2state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')
        kvectorVBM3=$(sed -n -e '/k-point =/s/\ *\<'"$VBM3state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')
        kvectorCBM1=$(sed -n -e '/k-point =/ s/\ *\<'"$CBM1state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')
        kvectorCBM2=$(sed -n -e '/k-point =/ s/\ *\<'"$CBM2state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')
        kvectorCBM3=$(sed -n -e '/k-point =/ s/\ *\<'"$CBM3state"'\> *//p' ../../3-Bands/siesta.out | head -1 |  cut -f2 -d"=" | sed -e 's/\s\+/,/g')

        awk '{sub("$",":" " ""['"$VBM1k2D"']", $1)}; 1' states/VBM1-state-k-E-temp.tmp > states/VBM1-state-k-E.tmp
        awk '{sub("$",":" " ""['"$VBM2k2D"']", $1)}; 1' states/VBM2-state-k-E-temp.tmp > states/VBM2-state-k-E.tmp
        awk '{sub("$",":" " ""['"$VBM3k2D"']", $1)}; 1' states/VBM3-state-k-E-temp.tmp > states/VBM3-state-k-E.tmp
        awk '{sub("$",":" " ""['"$CBM1k2D"']", $1)}; 1' states/CBM1-state-k-E-temp.tmp > states/CBM1-state-k-E.tmp
        awk '{sub("$",":" " ""['"$CBM2k2D"']", $1)}; 1' states/CBM2-state-k-E-temp.tmp > states/CBM2-state-k-E.tmp
        awk '{sub("$",":" " ""['"$CBM3k2D"']", $1)}; 1' states/CBM3-state-k-E-temp.tmp > states/CBM3-state-k-E.tmp

        sed -i 's/^/'"$VBM1band"" "'/' states/VBM1-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyVBM1state" >> states/VBM1-state-k-E.tmp
        printf "\n" >> states/VBM1-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/VBM1-state-k-E.tmp
        sed  -i '1i ------- VBM1 -------' states/VBM1-state-k-E.tmp

        sed -i 's/^/'"$VBM2band"" "'/' states/VBM2-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyVBM2state" >> states/VBM2-state-k-E.tmp
        printf "\n" >> states/VBM2-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/VBM2-state-k-E.tmp
        sed  -i '1i ------- VBM2 -------' states/VBM2-state-k-E.tmp

        sed -i 's/^/'"$VBM3band"" "'/' states/VBM3-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyVBM3state" >> states/VBM3-state-k-E.tmp
        printf "\n" >> states/VBM3-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/VBM3-state-k-E.tmp
        sed  -i '1i ------- VBM3 -------' states/VBM3-state-k-E.tmp
        echo "====================================================" > line.tmp
        cat states/VBM1-state-k-E.tmp states/VBM2-state-k-E.tmp states/VBM3-state-k-E.tmp > VBM-summary.out

        sed -i 's/^/'"$CBM1band"" "'/' states/CBM1-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyCBM1state" >> states/CBM1-state-k-E.tmp
        printf "\n" >> states/CBM1-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/CBM1-state-k-E.tmp
        sed  -i '1i ------- CBM1 -------' states/CBM1-state-k-E.tmp

        sed -i 's/^/'"$CBM2band"" "'/' states/CBM2-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyCBM2state" >> states/CBM2-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/CBM2-state-k-E.tmp
        printf "\n" >> states/CBM2-state-k-E.tmp
        sed  -i '1i ------- CBM2 -------' states/CBM2-state-k-E.tmp

        sed -i 's/^/'"$CBM3band"" "'/' states/CBM3-state-k-E.tmp
        printf "%s" Degeneracy" " =" " "$degeneracyCBM3state" >> states/CBM3-state-k-E.tmp
        printf "\n" >> states/CBM3-state-k-E.tmp
        sed  -i '1i n k:         k=[kx,ky,kz]         k (1/Ang) E-EF (eV)' states/CBM3-state-k-E.tmp
        sed  -i '1i ------- CBM3 -------' states/CBM3-state-k-E.tmp
        cat states/CBM1-state-k-E.tmp states/CBM2-state-k-E.tmp states/CBM3-state-k-E.tmp > CBM-summary.out

        cat states/VBM1-state-k-E.tmp states/VBM2-state-k-E.tmp states/VBM3-state-k-E.tmp > VBM-summary.tmp
        cat states/CBM3-state-k-E.tmp states/CBM2-state-k-E.tmp states/CBM1-state-k-E.tmp > CBM-summary.tmp
        echo "====================================================" > line.tmp
        cat CBM-summary.tmp line.tmp VBM-summary.tmp  > CBM-VBM-summary.out

        cat CBM-VBM-summary.out

        rm -rf *.tmp
        rm -rf states/*.tmp

        echo "SUMMARY (CBM-VBM) DONE!"

        echo ">>>>>>>>>>>>>>> STATES ANALYSIS DONE! <<<<<<<<<<<<<<<<<"


        cp ../../1-Relaxation/lattice.out .



        echo ">>>>>>>>>>>>>>> STARTING INTERPOLATION OF BZ ENERGY CONTOURS <<<<<<<<<<<<<<<<<"

     if [ $type == "Monolayer" -a $structure == "primitive" ];
        then
            matlab_contour="$MATLAB_UTILITIES/hexagonalContour-Monolayer.m"
            if [[ -f "$matlab_contour" ]]; then
                cp "$matlab_contour" contour.m
            else
                echo "Warning: MATLAB contour script not found: $matlab_contour" >&2
            fi
        elif [ $type == "Bulk" -a $structure == "primitive" ];
        then 
            matlab_contour="$MATLAB_UTILITIES/hexagonalContour-Bulk.m"
            if [[ -f "$matlab_contour" ]]; then
                cp "$matlab_contour" contour.m
            else
                echo "Warning: MATLAB contour script not found: $matlab_contour" >&2
            fi
        elif [ $type == "Monolayer" -a $structure == "rectangular" ];
        then
            matlab_contour="$MATLAB_UTILITIES/rectangularContour-Monolayer.m"
            if [[ -f "$matlab_contour" ]]; then
                cp "$matlab_contour" contour.m
            else
                echo "Warning: MATLAB contour script not found: $matlab_contour" >&2
            fi        
     fi

        matlab -nodisplay -nosplash -nodesktop < contour.m

        echo ">>>>>>>>>>>>>>> INTERPOLATIONM OF BZ ENERGY CONTOURS DONE!  <<<<<<<<<<<<<<<<<"

    fi





    if [ $bands2Dpathspost == "yes" -o $bands2Dpathspost == "Yes" -o $bands2Dpathspost == "YES" -o $bands2Dpathspost == "y" -o $bands2Dpathspost == "Y" ];
    then
        echo ">>>>>>>>>>>>>>> START 2D (BANDS @ POINTS) <<<<<<<<<<<<<<<<<"

        # Point CBM
mkdir -p CBM
cd CBM
rm *

copy_material_files "$material" "$functional" "$SOC"

cp ../../../1-Relaxation/*.XV .
cp ../../../1-Relaxation/*.CG .
cp ../../../1-Relaxation/*.DM .
cp ../../../1-Relaxation/*.STRUCT_OUT .
cp $material.STRUCT_OUT $material.STRUCT_IN

pathtemp_file="$QMATSIM_SIESTA_DIR/materials/$material/$type/$structure/pathTemp.txt"
if [[ -f "$pathtemp_file" ]]; then
    cp "$pathtemp_file" .
else
    echo "Warning: pathTemp.txt not found: $pathtemp_file" >&2
fi
mv ../pathCBM.txt .
sed '/block BandLines/!b;:a;$b;N;/endblock BandLines/!ba;P;s/.*\n//;e cat pathCBM.txt' pathTemp.txt > pathCBM.tmp
sed -ne '/%block BandLines/ {r ./'pathCBM'.tmp' -e ':a; n; /%endblock BandLines.*/ {b}; ba}; p' ../../../3-Bands/3-Bands.fdf > 3-Bands-CBM.fdf

siesta 3-Bands-CBM.fdf > siesta-CBM.out
mv $material.bands.WFSX $material.bands-CBM.WFSX
readwfx -m -6.5 -M -1.5 $material.bands-CBM.WFSX > $material.bands-CBM.WFSXR
grep "k-point =" $material.bands-CBM.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz-CBM.txt
awk -F ' ' '{print $1, $2}' index-kx-ky-kz-CBM.txt > index-kx-CBM.txt
awk -F ' ' '{print $1, $3}' index-kx-ky-kz-CBM.txt > index-ky-CBM.txt
awk -F ' ' '{print $1, $4}' index-kx-ky-kz-CBM.txt > index-kz-CBM.txt
awk -F ' ' '{print $1}' index-kx-ky-kz-CBM.txt > index-CBM.txt
awk -F ' ' '{print $2}' index-kx-ky-kz-CBM.txt > kx.tmp
awk -F ' ' '{print $3}' index-kx-ky-kz-CBM.txt > ky.tmp
awk -F ' ' '{print $4}' index-kx-ky-kz-CBM.txt > kz.tmp
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx-CBM.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky-CBM.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz-CBM.txt
paste index-CBM.txt kx-CBM.txt ky-CBM.txt | tr -s ' ' > index-kx-ky-CBM.txt
paste index-CBM.txt kx-CBM.txt ky-CBM.txt kz-CBM.txt | tr -s ' ' > index-kx-ky-kz-CBM.txt
paste kx-CBM.txt ky-CBM.txt | tr -s ' ' > kx-ky-CBM.txt
paste kx-CBM.txt ky-CBM.txt kz-CBM.txt | tr -s ' ' > kx-ky-kz-CBM.txt

grep "Wavefunction = *$CBM1band" $material.bands-CBM.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
grep "Wavefunction = *$CBM1band" $material.bands-CBM.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
paste kx-ky-kz-CBM.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D-CBM.txt
paste index-kx-ky-kz-CBM.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D-CBM.txt

rm *.tmp

echo "POINT CBM DONE!"

        cd ..

        # Point VBM
mkdir -p VBM
cd VBM

rm *

copy_material_files "$material" "$functional" "$SOC"

cp ../../../1-Relaxation/*.XV .
cp ../../../1-Relaxation/*.CG .
cp ../../../1-Relaxation/*.DM .
cp ../../../1-Relaxation/*.STRUCT_OUT .
cp $material.STRUCT_OUT $material.STRUCT_IN

pathtemp_file="$QMATSIM_SIESTA_DIR/materials/$material/$type/$structure/pathTemp.txt"
if [[ -f "$pathtemp_file" ]]; then
    cp "$pathtemp_file" .
else
    echo "Warning: pathTemp.txt not found: $pathtemp_file" >&2
fi
mv ../pathVBM.txt .
sed '/block BandLines/!b;:a;$b;N;/endblock BandLines/!ba;P;s/.*\n//;e cat pathVBM.txt' pathTemp.txt > pathVBM.tmp
sed -ne '/%block BandLines/ {r ./'pathVBM'.tmp' -e ':a; n; /%endblock BandLines.*/ {b}; ba}; p' ../../../3-Bands/3-Bands.fdf > 3-Bands-VBM.fdf

siesta 3-Bands-VBM.fdf > siesta-VBM.out
mv $material.bands.WFSX $material.bands-VBM.WFSX
readwfx -m -6.5 -M -1.5 $material.bands-VBM.WFSX > $material.bands-VBM.WFSXR
grep "k-point =" $material.bands-VBM.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz-VBM.txt
awk -F ' ' '{print $1, $2}' index-kx-ky-kz-VBM.txt > index-kx-VBM.txt
awk -F ' ' '{print $1, $3}' index-kx-ky-kz-VBM.txt > index-ky-VBM.txt
awk -F ' ' '{print $1, $4}' index-kx-ky-kz-VBM.txt > index-kz-VBM.txt
awk -F ' ' '{print $1}' index-kx-ky-kz-VBM.txt > index-VBM.txt
awk -F ' ' '{print $2}' index-kx-ky-kz-VBM.txt > kx.tmp
awk -F ' ' '{print $3}' index-kx-ky-kz-VBM.txt > ky.tmp
awk -F ' ' '{print $4}' index-kx-ky-kz-VBM.txt > kz.tmp
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx-VBM.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky-VBM.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz-VBM.txt
paste index-VBM.txt kx-VBM.txt ky-VBM.txt | tr -s ' ' > index-kx-ky-VBM.txt
paste index-VBM.txt kx-VBM.txt ky-VBM.txt kz-VBM.txt | tr -s ' ' > index-kx-ky-kz-VBM.txt
paste kx-VBM.txt ky-VBM.txt | tr -s ' ' > kx-ky-VBM.txt
paste kx-VBM.txt ky-VBM.txt kz-VBM.txt | tr -s ' ' > kx-ky-kz-VBM.txt

grep "Wavefunction = *$VBM1band" $material.bands-VBM.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
grep "Wavefunction = *$VBM1band" $material.bands-VBM.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
paste kx-ky-kz-VBM.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D-VBM.txt
paste index-kx-ky-kz-VBM.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D-VBM.txt


rm *.tmp

echo "POINT VBM DONE!"

            # Go back to GNUBANDS
            cd ..

# Point G
mkdir -p G
cd G
rm *

copy_material_files "$material" "$functional" "$SOC"

cp ../../../1-Relaxation/*.XV .
cp ../../../1-Relaxation/*.CG .
cp ../../../1-Relaxation/*.DM .
cp ../../../1-Relaxation/*.STRUCT_OUT .
cp $material.STRUCT_OUT $material.STRUCT_IN

pathtemp_file="$QMATSIM_SIESTA_DIR/materials/$material/$type/$structure/pathTemp.txt"
if [[ -f "$pathtemp_file" ]]; then
    cp "$pathtemp_file" .
else
    echo "Warning: pathTemp.txt not found: $pathtemp_file" >&2
fi
mv ../pathG.txt .
sed '/block BandLines/!b;:a;$b;N;/endblock BandLines/!ba;P;s/.*\n//;e cat pathG.txt' pathTemp.txt > pathG.tmp
sed -ne '/%block BandLines/ {r ./'pathG'.tmp' -e ':a; n; /%endblock BandLines.*/ {b}; ba}; p' ../../../3-Bands/3-Bands.fdf > 3-Bands-G.fdf

siesta 3-Bands-G.fdf > siesta-G.out
mv $material.bands.WFSX $material.bands-G.WFSX
readwfx -m -6.5 -M -1.5 $material.bands-G.WFSX > $material.bands-G.WFSXR
grep "k-point =" $material.bands-G.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz-G.txt
awk -F ' ' '{print $1, $2}' index-kx-ky-kz-G.txt > index-kx-G.txt
awk -F ' ' '{print $1, $3}' index-kx-ky-kz-G.txt > index-ky-G.txt
awk -F ' ' '{print $1, $4}' index-kx-ky-kz-G.txt > index-kz-G.txt
awk -F ' ' '{print $1}' index-kx-ky-kz-G.txt > index-G.txt
awk -F ' ' '{print $2}' index-kx-ky-kz-G.txt > kx.tmp
awk -F ' ' '{print $3}' index-kx-ky-kz-G.txt > ky.tmp
awk -F ' ' '{print $4}' index-kx-ky-kz-G.txt > kz.tmp
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx-G.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky-G.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz-G.txt
paste index-G.txt kx-G.txt ky-G.txt | tr -s ' ' > index-kx-ky-G.txt
paste index-G.txt kx-G.txt ky-G.txt kz-G.txt | tr -s ' ' > index-kx-ky-kz-G.txt
paste kx-G.txt ky-G.txt | tr -s ' ' > kx-ky-G.txt
paste kx-G.txt ky-G.txt kz-G.txt | tr -s ' ' > kx-ky-kz-G.txt

grep "Wavefunction = *$CBM1band" $material.bands-G.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
grep "Wavefunction = *$CBM1band" $material.bands-G.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
paste kx-ky-kz-G.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D-G.txt
paste index-kx-ky-kz-G.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D-G.txt

grep "Wavefunction = *$VBM1band" $material.bands-G.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
grep "Wavefunction = *$VBM1band" $material.bands-G.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
paste kx-ky-kz-G.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D-G.txt
paste index-kx-ky-kz-G.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D-G.txt

rm *.tmp

echo "POINT G DONE!"

cd ..

# Point G
mkdir -p K
cd K
rm *

copy_material_files "$material" "$functional" "$SOC"


cp ../../../1-Relaxation/*.XV .
cp ../../../1-Relaxation/*.CG .
cp ../../../1-Relaxation/*.DM .
cp ../../../1-Relaxation/*.STRUCT_OUT .
cp $material.STRUCT_OUT $material.STRUCT_IN

pathtemp_file="$QMATSIM_SIESTA_DIR/materials/$material/$type/$structure/pathTemp.txt"
if [[ -f "$pathtemp_file" ]]; then
    cp "$pathtemp_file" .
else
    echo "Warning: pathTemp.txt not found: $pathtemp_file" >&2
fi
mv ../pathK.txt .
sed '/block BandLines/!b;:a;$b;N;/endblock BandLines/!ba;P;s/.*\n//;e cat pathK.txt' pathTemp.txt > pathK.tmp
sed -ne '/%block BandLines/ {r ./'pathK'.tmp' -e ':a; n; /%endblock BandLines.*/ {b}; ba}; p' ../../../3-Bands/3-Bands.fdf > 3-Bands-K.fdf

siesta 3-Bands-K.fdf > siesta-K.out
mv $material.bands.WFSX $material.bands-K.WFSX
readwfx -m -6.5 -M -1.5 $material.bands-K.WFSX > $material.bands-K.WFSXR
grep "k-point =" $material.bands-K.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz-K.txt
awk -F ' ' '{print $1, $2}' index-kx-ky-kz-K.txt > index-kx-K.txt
awk -F ' ' '{print $1, $3}' index-kx-ky-kz-K.txt > index-ky-K.txt
awk -F ' ' '{print $1, $4}' index-kx-ky-kz-K.txt > index-kz-K.txt
awk -F ' ' '{print $1}' index-kx-ky-kz-K.txt > index-K.txt
awk -F ' ' '{print $2}' index-kx-ky-kz-K.txt > kx.tmp
awk -F ' ' '{print $3}' index-kx-ky-kz-K.txt > ky.tmp
awk -F ' ' '{print $4}' index-kx-ky-kz-K.txt > kz.tmp
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx-K.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky-K.txt
awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz-K.txt
paste index-K.txt kx-K.txt ky-K.txt | tr -s ' ' > index-kx-ky-K.txt
paste index-K.txt kx-K.txt ky-K.txt kz-K.txt | tr -s ' ' > index-kx-ky-kz-K.txt
paste kx-K.txt ky-K.txt | tr -s ' ' > kx-ky-K.txt
paste kx-K.txt ky-K.txt kz-K.txt | tr -s ' ' > kx-ky-kz-K.txt

grep "Wavefunction = *$CBM1band" $material.bands-K.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
grep "Wavefunction = *$CBM1band" $material.bands-K.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
paste kx-ky-kz-K.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D-K.txt
paste index-kx-ky-kz-K.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D-K.txt

grep "Wavefunction = *$VBM1band" $material.bands-K.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
grep "Wavefunction = *$VBM1band" $material.bands-K.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
paste kx-ky-kz-K.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D-K.txt
paste index-kx-ky-kz-K.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D-K.txt

rm *.tmp

echo "POINT K DONE!"

# Go BACK TO GNUBANDS
cd ..

    fi

    # Go back to Postprocessing
    cd ..

    echo "############################## BANDS COMPLETED ##############################"

fi

########################################################################
############################## PDOSXML #################################
########################################################################

if [ $PDOSXMLpost == "yes" -o $PDOSXMLpost == "Yes" -o $PDOSXMLpost == "YES" -o $PDOSXMLpost == "y" -o $PDOSXMLpost == "Y" ];
then

    rm -rf PDOSXML
    mkdir -p PDOSXML
    echo "############################## PDOSXML ##############################"

    cd PDOSXML
    cp ../../2-Static/$material.PDOS .
    cp ../../2-Static/$material.PDOS.xml .

    EFstatic=$(grep -E 'siesta:[[:space:]]*Fermi[[:space:]]*=' ../../2-Static/siesta.out | awk '{print $4}')

    echo "START PDOSXML"

    echo ">>>>>>>>>>>>>>> START PDOSXML (TOTAL DOS & ORBITALS PDOS) ANALYSIS <<<<<<<<<<<<<<<<<"



if [ "$material" = "MoS2" ]; then
    pdosxmlMo $material.PDOS.xml > $material-PDOS-Mo-pdosxml.dat
    pdosxmlMo5 $material.PDOS.xml > $material-PDOS-Mo5-pdosxml.dat
    pdosxmlMo5s $material.PDOS.xml > $material-PDOS-Mo5s-pdosxml.dat
    pdosxmlMo4 $material.PDOS.xml > $material-PDOS-Mo4-pdosxml.dat
    pdosxmlMo4d $material.PDOS.xml > $material-PDOS-Mo4d-pdosxml.dat
    pdosxmlMo4dxy $material.PDOS.xml > $material-PDOS-Mo4dxy-pdosxml.dat
    pdosxmlMo4dxz $material.PDOS.xml > $material-PDOS-Mo4dxz-pdosxml.dat
    pdosxmlMo4dyz $material.PDOS.xml > $material-PDOS-Mo4dyz-pdosxml.dat
    pdosxmlMo4dx2-y2 $material.PDOS.xml > $material-PDOS-Mo4dx2-y2-pdosxml.dat
    pdosxmlMo4dz2 $material.PDOS.xml > $material-PDOS-Mo4dz2-pdosxml.dat

    pdosxmlS $material.PDOS.xml > $material-PDOS-S-pdosxml.dat
    pdosxmlS3 $material.PDOS.xml > $material-PDOS-S3-pdosxml.dat
    pdosxmlS3p $material.PDOS.xml > $material-PDOS-S3p-pdosxml.dat
    pdosxmlS3px $material.PDOS.xml > $material-PDOS-S3px-pdosxml.dat
    pdosxmlS3py $material.PDOS.xml > $material-PDOS-S3py-pdosxml.dat
    pdosxmlS3pz $material.PDOS.xml > $material-PDOS-S3pz-pdosxml.dat
    pdosxmlS3s $material.PDOS.xml > $material-PDOS-S3s-pdosxml.dat

elif [ "$material" = "MoSe2" ]; then
    pdosxmlMo $material.PDOS.xml > $material-PDOS-Mo-pdosxml.dat
    pdosxmlMo5 $material.PDOS.xml > $material-PDOS-Mo5-pdosxml.dat
    pdosxmlMo5s $material.PDOS.xml > $material-PDOS-Mo5s-pdosxml.dat
    pdosxmlMo4 $material.PDOS.xml > $material-PDOS-Mo4-pdosxml.dat
    pdosxmlMo4d $material.PDOS.xml > $material-PDOS-Mo4d-pdosxml.dat
    pdosxmlMo4dxy $material.PDOS.xml > $material-PDOS-Mo4dxy-pdosxml.dat
    pdosxmlMo4dxz $material.PDOS.xml > $material-PDOS-Mo4dxz-pdosxml.dat
    pdosxmlMo4dyz $material.PDOS.xml > $material-PDOS-Mo4dyz-pdosxml.dat
    pdosxmlMo4dx2-y2 $material.PDOS.xml > $material-PDOS-Mo4dx2-y2-pdosxml.dat
    pdosxmlMo4dz2 $material.PDOS.xml > $material-PDOS-Mo4dz2-pdosxml.dat

    pdosxmlSe $material.PDOS.xml > $material-PDOS-Se-pdosxml.dat
    pdosxmlSe4 $material.PDOS.xml > $material-PDOS-Se4-pdosxml.dat
    pdosxmlSe4s $material.PDOS.xml > $material-PDOS-Se4s-pdosxml.dat
    pdosxmlSe4p $material.PDOS.xml > $material-PDOS-Se4p-pdosxml.dat
    pdosxmlSe4px $material.PDOS.xml > $material-PDOS-Se4px-pdosxml.dat
    pdosxmlSe4py $material.PDOS.xml > $material-PDOS-Se4py-pdosxml.dat
    pdosxmlSe4pz $material.PDOS.xml > $material-PDOS-Se4pz-pdosxml.dat

elif [ "$material" = "WS2" ]; then
    pdosxmlW $material.PDOS.xml > $material-PDOS-W-pdosxml.dat
    pdosxmlW6 $material.PDOS.xml > $material-PDOS-W6-pdosxml.dat
    pdosxmlW6s $material.PDOS.xml > $material-PDOS-W6s-pdosxml.dat
    pdosxmlW5 $material.PDOS.xml > $material-PDOS-W5-pdosxml.dat
    pdosxmlW5s $material.PDOS.xml > $material-PDOS-W5s-pdosxml.dat
    pdosxmlW5p $material.PDOS.xml > $material-PDOS-W5p-pdosxml.dat
    pdosxmlW5d $material.PDOS.xml > $material-PDOS-W5d-pdosxml.dat
    pdosxmlW5dxy $material.PDOS.xml > $material-PDOS-W5dxy-pdosxml.dat
    pdosxmlW5dxz $material.PDOS.xml > $material-PDOS-W5dxz-pdosxml.dat
    pdosxmlW5dyz $material.PDOS.xml > $material-PDOS-W5dyz-pdosxml.dat
    pdosxmlW5dz2 $material.PDOS.xml > $material-PDOS-W5dz2-pdosxml.dat
    pdosxmlW5dx2-y2 $material.PDOS.xml > $material-PDOS-W5dx2-y2-pdosxml.dat

    pdosxmlS $material.PDOS.xml > $material-PDOS-S-pdosxml.dat
    pdosxmlS3 $material.PDOS.xml > $material-PDOS-S3-pdosxml.dat
    pdosxmlS3p $material.PDOS.xml > $material-PDOS-S3p-pdosxml.dat
    pdosxmlS3px $material.PDOS.xml > $material-PDOS-S3px-pdosxml.dat
    pdosxmlS3py $material.PDOS.xml > $material-PDOS-S3py-pdosxml.dat
    pdosxmlS3pz $material.PDOS.xml > $material-PDOS-S3pz-pdosxml.dat
    pdosxmlS3s $material.PDOS.xml > $material-PDOS-S3s-pdosxml.dat

elif [ "$material" = "WSe2" ]; then
    pdosxmlW $material.PDOS.xml > $material-PDOS-W-pdosxml.dat
    pdosxmlW6 $material.PDOS.xml > $material-PDOS-W6-pdosxml.dat
    pdosxmlW6s $material.PDOS.xml > $material-PDOS-W6s-pdosxml.dat
    pdosxmlW5 $material.PDOS.xml > $material-PDOS-W5-pdosxml.dat
    pdosxmlW5s $material.PDOS.xml > $material-PDOS-W5s-pdosxml.dat
    pdosxmlW5p $material.PDOS.xml > $material-PDOS-W5p-pdosxml.dat
    pdosxmlW5d $material.PDOS.xml > $material-PDOS-W5d-pdosxml.dat
    pdosxmlW5dxy $material.PDOS.xml > $material-PDOS-W5dxy-pdosxml.dat
    pdosxmlW5dxz $material.PDOS.xml > $material-PDOS-W5dxz-pdosxml.dat
    pdosxmlW5dyz $material.PDOS.xml > $material-PDOS-W5dyz-pdosxml.dat
    pdosxmlW5dz2 $material.PDOS.xml > $material-PDOS-W5dz2-pdosxml.dat
    pdosxmlW5dx2-y2 $material.PDOS.xml > $material-PDOS-W5dx2-y2-pdosxml.dat

    pdosxmlSe $material.PDOS.xml > $material-PDOS-Se-pdosxml.dat
    pdosxmlSe4 $material.PDOS.xml > $material-PDOS-Se4-pdosxml.dat
    pdosxmlSe4s $material.PDOS.xml > $material-PDOS-Se4s-pdosxml.dat
    pdosxmlSe4p $material.PDOS.xml > $material-PDOS-Se4p-pdosxml.dat
    pdosxmlSe4px $material.PDOS.xml > $material-PDOS-Se4px-pdosxml.dat
    pdosxmlSe4py $material.PDOS.xml > $material-PDOS-Se4py-pdosxml.dat
    pdosxmlSe4pz $material.PDOS.xml > $material-PDOS-Se4pz-pdosxml.dat

elif [ "$material" = "C" ]; then
    pdosxmlC $material.PDOS.xml > $material-PDOS-C-pdosxml.dat
    pdosxmlC2 $material.PDOS.xml > $material-PDOS-C2-pdosxml.dat
    pdosxmlC2p $material.PDOS.xml > $material-PDOS-C2p-pdosxml.dat
    pdosxmlC2px $material.PDOS.xml > $material-PDOS-C2px-pdosxml.dat
    pdosxmlC2py $material.PDOS.xml > $material-PDOS-C2py-pdosxml.dat
    pdosxmlC2pz $material.PDOS.xml > $material-PDOS-C2pz-pdosxml.dat
    pdosxmlC2s $material.PDOS.xml > $material-PDOS-C2s-pdosxml.dat

fi

    echo "PDOSXML DONE!"

    spinLine=$(sed -n 2p *.PDOS)
    Nspin=$(echo $spinLine | sed 's/^.*>\([0-9]*\)<\/nspin>.*$/\1/')
    orbitalsLine=$(sed -n 3p *.PDOS)
    Norbitals=$(echo $orbitalsLine | sed 's/^.*>\([0-9]*\)<\/norbitals>.*$/\1/')
    EFLine=$(sed -n 5p *.PDOS.xml) > EF.tmp
    EF=$(cut -d ">" -f2 <<< "$EFLine")
    subtitlePlotLine1="E_F=$EF (eV)"
    subtitlePlotLine2="$Norbitals orbitals, $Nspin-spin"
    echo "$subtitlePlotLine1" > title1.txt
    echo "$subtitlePlotLine2" > title2.txt

if [ "$material" = "MoS2" ]; then
    paste $material-PDOS-Mo-pdosxml.dat $material-PDOS-S-pdosxml.dat > $material-PDOS-total-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2+$5 " " $3+$6}' < $material-PDOS-total-pdosxml.dat > $material-PDOS-total-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo-pdosxml.dat > $material-PDOS-Mo-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo5-pdosxml.dat > $material-PDOS-Mo5-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo5s-pdosxml.dat > $material-PDOS-Mo5s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4-pdosxml.dat > $material-PDOS-Mo4-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4d-pdosxml.dat > $material-PDOS-Mo4d-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dxy-pdosxml.dat > $material-PDOS-Mo4dxy-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dx2-y2-pdosxml.dat > $material-PDOS-Mo4dx2-y2-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dz2-pdosxml.dat > $material-PDOS-Mo4dz2-shifted-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S-pdosxml.dat > $material-PDOS-S-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3-pdosxml.dat > $material-PDOS-S3-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3p-pdosxml.dat > $material-PDOS-S3p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3px-pdosxml.dat > $material-PDOS-S3px-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3py-pdosxml.dat > $material-PDOS-S3py-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3pz-pdosxml.dat > $material-PDOS-S3pz-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3s-pdosxml.dat > $material-PDOS-S3s-shifted-pdosxml.dat

elif [ "$material" = "MoSe2" ]; then
    paste $material-PDOS-Mo-pdosxml.dat $material-PDOS-Se-pdosxml.dat > $material-PDOS-total-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2+$5 " " $3+$6}' < $material-PDOS-total-pdosxml.dat > $material-PDOS-total-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo-pdosxml.dat > $material-PDOS-Mo-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo5-pdosxml.dat > $material-PDOS-Mo5-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo5s-pdosxml.dat > $material-PDOS-Mo5s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4-pdosxml.dat > $material-PDOS-Mo4-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4d-pdosxml.dat > $material-PDOS-Mo4d-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dxy-pdosxml.dat > $material-PDOS-Mo4dxy-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dx2-y2-pdosxml.dat > $material-PDOS-Mo4dx2-y2-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Mo4dz2-pdosxml.dat > $material-PDOS-Mo4dz2-shifted-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se-pdosxml.dat > $material-PDOS-Se-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4-pdosxml.dat > $material-PDOS-Se4-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4s-pdosxml.dat > $material-PDOS-Se4s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4p-pdosxml.dat > $material-PDOS-Se4p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4px-pdosxml.dat > $material-PDOS-Se4px-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4py-pdosxml.dat > $material-PDOS-Se4py-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4pz-pdosxml.dat > $material-PDOS-Se4pz-shifted-pdosxml.dat

elif [ "$material" = "WS2" ]; then
    paste $material-PDOS-W-pdosxml.dat $material-PDOS-S-pdosxml.dat > $material-PDOS-total-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2+$5 " " $3+$6}' < $material-PDOS-total-pdosxml.dat > $material-PDOS-total-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W-pdosxml.dat > $material-PDOS-W-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W6-pdosxml.dat > $material-PDOS-W6-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W6s-pdosxml.dat > $material-PDOS-W6s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5-pdosxml.dat > $material-PDOS-W5-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5s-pdosxml.dat > $material-PDOS-W5s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5p-pdosxml.dat > $material-PDOS-W5p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5d-pdosxml.dat > $material-PDOS-W5d-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dxy-pdosxml.dat > $material-PDOS-W5dxy-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dx2-y2-pdosxml.dat > $material-PDOS-W5dx2-y2-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dz2-pdosxml.dat > $material-PDOS-W5dz2-shifted-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S-pdosxml.dat > $material-PDOS-S-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3-pdosxml.dat > $material-PDOS-S3-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3p-pdosxml.dat > $material-PDOS-S3p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3px-pdosxml.dat > $material-PDOS-S3px-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3py-pdosxml.dat > $material-PDOS-S3py-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3pz-pdosxml.dat > $material-PDOS-S3pz-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-S3s-pdosxml.dat > $material-PDOS-S3s-shifted-pdosxml.dat

elif [ "$material" = "WSe2" ]; then
    paste $material-PDOS-W-pdosxml.dat $material-PDOS-Se-pdosxml.dat > $material-PDOS-total-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2+$5 " " $3+$6}' < $material-PDOS-total-pdosxml.dat > $material-PDOS-total-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W-pdosxml.dat > $material-PDOS-W-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W6-pdosxml.dat > $material-PDOS-W6-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W6s-pdosxml.dat > $material-PDOS-W6s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5-pdosxml.dat > $material-PDOS-W5-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5s-pdosxml.dat > $material-PDOS-W5s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5p-pdosxml.dat > $material-PDOS-W5p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5d-pdosxml.dat > $material-PDOS-W5d-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dxy-pdosxml.dat > $material-PDOS-W5dxy-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dx2-y2-pdosxml.dat > $material-PDOS-W5dx2-y2-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-W5dz2-pdosxml.dat > $material-PDOS-W5dz2-shifted-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se-pdosxml.dat > $material-PDOS-Se-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4-pdosxml.dat > $material-PDOS-Se4-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4s-pdosxml.dat > $material-PDOS-Se4s-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4p-pdosxml.dat > $material-PDOS-Se4p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4px-pdosxml.dat > $material-PDOS-Se4px-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4py-pdosxml.dat > $material-PDOS-Se4py-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-Se4pz-pdosxml.dat > $material-PDOS-Se4pz-shifted-pdosxml.dat

elif [ "$material" = "C" ]; then
    paste $material-PDOS-C-pdosxml.dat > $material-PDOS-total-pdosxml.dat

    awk -v var=$EFstatic -F " " '{print $1-var " " $2+$5 " " $3+$6}' < $material-PDOS-total-pdosxml.dat > $material-PDOS-total-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C-pdosxml.dat > $material-PDOS-C-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2-pdosxml.dat > $material-PDOS-C2-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2p-pdosxml.dat > $material-PDOS-C2p-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2px-pdosxml.dat > $material-PDOS-C2px-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2py-pdosxml.dat > $material-PDOS-C2py-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2pz-pdosxml.dat > $material-PDOS-C2pz-shifted-pdosxml.dat
    awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-C2s-pdosxml.dat > $material-PDOS-C2s-shifted-pdosxml.dat
fi

    cp ../GNUBANDS/bandsPositive.txt ../PDOSXML
    cp ../GNUBANDS/bandsNegative.txt ../PDOSXML

    echo ">>>>>>>>>>>>>>> PDOSXML (TOTAL DOS & ORBITALS PDOS) ANALYSIS DONE <<<<<<<<<<<<<<<<<"

    echo ">>>>>>>>>>>>>>> START PDOSXML (ATOMS) ANALYSIS <<<<<<<<<<<<<<<<<"

    mkdir -p PDOSXML-atoms
    cd PDOSXML-atoms

    cp ../../../1-Relaxation/siesta.out .
    cp ../../../1-Relaxation/$material.xyz .


    echo "START PDOSXML LOOP"

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    for (( k=1; k<=$Natoms; k++ ))
    do
        pdosxml$k ../$material.PDOS.xml > $material-PDOS-$k-pdosxml.dat
        awk -v var=$EFstatic -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS-$k-pdosxml.dat > $material-PDOS-$k-shifted-pdosxml.dat
    done

    echo "PDOSXML LOOP DONE!"

    for (( k=1; k<=$Natoms; k++ ))
    do
        cut -sd- -f1- $material-PDOS-$k-shifted-pdosxml.dat > $material-PDOS-$k-shifted-pdosxml-NegativeE.tmp
        diff $material-PDOS-$k-shifted-pdosxml-NegativeE.tmp $material-PDOS-$k-shifted-pdosxml.dat | grep '^>' | sed 's/^>\ //' > $material-PDOS-$k-shifted-pdosxml-PositiveE.tmp
        awk '{print $1 " " $2}' $material-PDOS-$k-shifted-pdosxml-NegativeE.tmp > $material-PDOS-$k-shifted-pdosxml-NegativeE-tmp.tmp
        awk '{print $1 " " $2}' $material-PDOS-$k-shifted-pdosxml-PositiveE.tmp > $material-PDOS-$k-shifted-pdosxml-PositiveE-tmp.tmp
        grep -v "0.00000" $material-PDOS-$k-shifted-pdosxml-NegativeE-tmp.tmp > $material-PDOS-$k-shifted-pdosxml-NegativeE.tmp
        grep -v "0.00000" $material-PDOS-$k-shifted-pdosxml-PositiveE-tmp.tmp > $material-PDOS-$k-shifted-pdosxml-PositiveE.tmp
        E0VB="$(tail -n 1 $material-PDOS-1-shifted-pdosxml-NegativeE.tmp | cut -f1 -d " ")"
        E0CB="$(head -n 1 $material-PDOS-$k-shifted-pdosxml-PositiveE.tmp | cut -f1 -d " ")"
        echo "$E0CB $E0VB" | awk '{print $1-$2}' >> Eg-distance.tmp
    done

    rm -rf *.tmp

    cp ../bandsPositive.txt .
    cp ../bandsNegative.txt .

    cp ../../../1-Relaxation/$material.xyz .

if [ "$material" = "MoS2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/Mo/1/g' $material-new.xyz
    sed -i 's/S/2/g' $material-new.xyz
elif [ "$material" = "MoSe2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/Mo/1/g' $material-new.xyz
    sed -i 's/Se/2/g' $material-new.xyz
elif [ "$material" = "WS2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/W/1/g' $material-new.xyz
    sed -i 's/S/2/g' $material-new.xyz
elif [ "$material" = "WSe2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/W/1/g' $material-new.xyz
    sed -i 's/Se/2/g' $material-new.xyz
elif [ "$material" = "C" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/C/1/g' $material-new.xyz
fi


    rm *.tmp



fi


    cd ..

    echo ">>>>>>>>>>>>>>> PDOSXML (ATOMS) DONE <<<<<<<<<<<<<<<<<"

    echo "PDOSXML ANALYSIS DONE!"  > completed.PDOSXMLCompleted

    rm -rf *.tmp

    cd ..

    echo "############################## PDOSXML COMPLETED ##############################"

fi




########################################################################
################################ COOP ##################################
########################################################################

# OUTPUT FILES: C2-systemLabel-PDOS.orbitalLabel.pdos (NOTE: the orbitalLabel is that found in .mpr file), systemLabel.alldos/.ados/.intdos

if [ $COOPPDOSpost == "yes" -o $COOPPDOSpost == "Yes" -o $COOPPDOSpost == "YES" -o $COOPPDOSpost == "y" -o $COOPPDOSpost == "Y" ];
then
    echo "############################## COOP/PDOS ##############################"

    echo "START COOP/PDOS"

    mkdir -p COOP

    cd COOP


    echo ">>>>>>>>>>>>>>> START COOP/PDOS (TOTAL DOS & ORBITALS PDOS) ANALYSIS <<<<<<<<<<<<<<<<<"

    rm -rf COOPPDOS
    mkdir -p COOPPDOS

    cd COOPPDOS

    # Move the COOP/mprop utility input file
    cp ~/SIESTA/materials/$material/$type/Utilities/$material-PDOS.mpr .

    # H and S matrices coefficients
    cp ../../../1-Relaxation/$material.HSX .

    # Wave functions for the SCF sampled BZ (i.e., from the block kgrid.MonkhorstPack). This is for full PDOS or COOP/COHP analysis
    cp  ../../../2-Static/$material.fullBZ.WFSX ./$material.WFSX
    #cp  ../../../3-Bands/$material.bands.WFSX ./$material.WFSX

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    mkdir -p COOPPDOS-atoms
    cp ~/SIESTA/materials/$material/$type/Utilities/$material-PDOS-Atom.mpr .

    for (( k=$Natoms; k>=1; k-- ))
    do
        sed "2a\\$k" $material-PDOS-Atom.mpr > $material-PDOS-Atom-$k.mpr
        sed -i "2a\\PDOS_$k" $material-PDOS-Atom-$k.mpr
        mv $material-PDOS-Atom-$k.mpr COOPPDOS-atoms/
        #mprop -m -9 -M 1 -s 0.0025 -n 4000 COOPPDOS-atoms/$material-PDOS-Atom-$k
        mprop -m -6.5 -M -1.5 -s 0.05 -n 5000 COOPPDOS-atoms/$material-PDOS-Atom-$k
        awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' <  COOPPDOS-atoms/$material-PDOS-Atom-$k.PDOS_"$k".pdos >  COOPPDOS-atoms/$material-PDOS-Atom-$k.PDOS_"$k"_shifted.pdos
    done
fi


# Define common variables
E1=-6.5
E2=-2.5
sigma=0.05
Nbins=5000


if [ "$material" = "MoS2" ]; then
    mprop -m $E1 -M $E2 -s $sigma -n $Nbins $material-PDOS
    paste $material-PDOS.PDOS_Mo.pdos $material-PDOS.PDOS_S.pdos > $material-PDOS.PDOS_total.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2+$5 " " $3+$6 " " $4+$7}' < $material-PDOS.PDOS_total.pdos > $material-PDOS.PDOS_shifted_total.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo.pdos > $material-PDOS.PDOS_Mo_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_5s.pdos > $material-PDOS.PDOS_Mo_5s_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4d.pdos > $material-PDOS.PDOS_Mo_4d_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dxy.pdos > $material-PDOS.PDOS_Mo_4dxy_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dx2-y2.pdos > $material-PDOS.PDOS_Mo_4dx2_y2_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dz2.pdos > $material-PDOS.PDOS_Mo_4dz2_shifted.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S.pdos > $material-PDOS.PDOS_S_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3p.pdos > $material-PDOS.PDOS_S_3p_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3px.pdos > $material-PDOS.PDOS_S_3px_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3py.pdos > $material-PDOS.PDOS_S_3py_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3pz.pdos > $material-PDOS.PDOS_S_3pz_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3s.pdos > $material-PDOS.PDOS_S_3s_shifted.pdos

elif [ "$material" = "MoSe2" ]; then
    mprop -m $E1 -M $E2 -s $sigma -n $Nbins $material-PDOS
    paste $material-PDOS.PDOS_Mo.pdos $material-PDOS.PDOS_Se.pdos > $material-PDOS.PDOS_total.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2+$5 " " $3+$6 " " $4+$7}' < $material-PDOS.PDOS_total.pdos > $material-PDOS.PDOS_shifted_total.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo.pdos > $material-PDOS.PDOS_Mo_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_5s.pdos > $material-PDOS.PDOS_Mo_5s_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4d.pdos > $material-PDOS.PDOS_Mo_4d_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dxy.pdos > $material-PDOS.PDOS_Mo_4dxy_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dx2-y2.pdos > $material-PDOS.PDOS_Mo_4dx2_y2_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Mo_4dz2.pdos > $material-PDOS.PDOS_Mo_4dz2_shifted.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se.pdos > $material-PDOS.PDOS_Se_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4p.pdos > $material-PDOS.PDOS_Se_4p_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4px.pdos > $material-PDOS.PDOS_Se_4px_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4py.pdos > $material-PDOS.PDOS_Se_4py_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4pz.pdos > $material-PDOS.PDOS_Se_4pz_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4s.pdos > $material-PDOS.PDOS_Se_4s_shifted.pdos

elif [ "$material" = "WS2" ]; then
    mprop -m $E1 -M $E2 -s $sigma -n $Nbins $material-PDOS
    paste $material-PDOS.PDOS_W.pdos $material-PDOS.PDOS_S.pdos > $material-PDOS.PDOS_total.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2+$5 " " $3+$6 " " $4+$7}' < $material-PDOS.PDOS_total.pdos > $material-PDOS.PDOS_shifted_total.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W.pdos > $material-PDOS.PDOS_W_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_6s.pdos > $material-PDOS.PDOS_W_6s_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5d.pdos > $material-PDOS.PDOS_W_5d_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dxy.pdos > $material-PDOS.PDOS_W_5dxy_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dx2-y2.pdos > $material-PDOS.PDOS_W_5dx2_y2_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dz2.pdos > $material-PDOS.PDOS_W_5dz2_shifted.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S.pdos > $material-PDOS.PDOS_S_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3p.pdos > $material-PDOS.PDOS_S_3p_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3px.pdos > $material-PDOS.PDOS_S_3px_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3py.pdos > $material-PDOS.PDOS_S_3py_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3pz.pdos > $material-PDOS.PDOS_S_3pz_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_S_3s.pdos > $material-PDOS.PDOS_S_3s_shifted.pdos

elif [ "$material" = "WSe2" ]; then
    mprop -m $E1 -M $E2 -s $sigma -n $Nbins $material-PDOS
    paste $material-PDOS.PDOS_W.pdos $material-PDOS.PDOS_Se.pdos > $material-PDOS.PDOS_total.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2+$5 " " $3+$6 " " $4+$7}' < $material-PDOS.PDOS_total.pdos > $material-PDOS.PDOS_shifted_total.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W.pdos > $material-PDOS.PDOS_W_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_6s.pdos > $material-PDOS.PDOS_W_6s_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5d.pdos > $material-PDOS.PDOS_W_5d_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dxy.pdos > $material-PDOS.PDOS_W_5dxy_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dx2-y2.pdos > $material-PDOS.PDOS_W_5dx2_y2_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_W_5dz2.pdos > $material-PDOS.PDOS_W_5dz2_shifted.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se.pdos > $material-PDOS.PDOS_Se_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4p.pdos > $material-PDOS.PDOS_Se_4p_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4px.pdos > $material-PDOS.PDOS_Se_4px_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4py.pdos > $material-PDOS.PDOS_Se_4py_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4pz.pdos > $material-PDOS.PDOS_Se_4pz_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_Se_4s.pdos > $material-PDOS.PDOS_Se_4s_shifted.pdos

elif [ "$material" = "C" ]; then
    mprop -m $E1 -M $E2 -s $sigma -n $Nbins $material-PDOS
    paste $material-PDOS.PDOS_C.pdos > $material-PDOS.PDOS_total.pdos

    awk -v var=$EFbands -F " " '{print $1-var " " $2+$5 " " $3+$6 " " $4+$7}' < $material-PDOS.PDOS_total.pdos > $material-PDOS.PDOS_shifted_total.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C.pdos > $material-PDOS.PDOS_C_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C_2p.pdos > $material-PDOS.PDOS_C_2p_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C_2px.pdos > $material-PDOS.PDOS_C_2px_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C_2py.pdos > $material-PDOS.PDOS_C_2py_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C_2pz.pdos > $material-PDOS.PDOS_C_2pz_shifted.pdos
    awk -v var=$EFbands -F " " '{print $1-var " " $2 " " $3}' < $material-PDOS.PDOS_C_2s.pdos > $material-PDOS.PDOS_C_2s_shifted.pdos

fi


    cp ../../GNUBANDS/bandsPositive.txt .
    cp ../../GNUBANDS/bandsNegative.txt .

    echo "$subtitlePlotLine1" > title1.txt
    echo "$subtitlePlotLine2" > title2.txt


    rm -rf *.xyz
    rm -rf ../*.xyz

    echo ">>>>>>>>>>>>>>> COOP/PDOS (TOTAL DOS & ORBITALS PDOS) ANALYSIS DONE <<<<<<<<<<<<<<<<<"

    echo ">>>>>>>>>>>>>>> START COOP/PDOS (ATOMS) ANALYSIS <<<<<<<<<<<<<<<<<"

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    cd COOPPDOS-atoms

    cp ../../../../1-Relaxation/siesta.out .
    cp ../../../../1-Relaxation/$material.xyz .

    cp ../bandsPositive.txt .
    cp ../bandsNegative.txt .

    echo "$subtitlePlotLine1" > title1.txt
    echo "$subtitlePlotLine2" > title2.txt

if [ "$material" = "MoS2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/Mo/1/g' $material-new.xyz
    sed -i 's/S/2/g' $material-new.xyz
elif [ "$material" = "MoSe2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/Mo/1/g' $material-new.xyz
    sed -i 's/Se/2/g' $material-new.xyz
elif [ "$material" = "WS2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/W/1/g' $material-new.xyz
    sed -i 's/S/2/g' $material-new.xyz
elif [ "$material" = "WSe2" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/W/1/g' $material-new.xyz
    sed -i 's/Se/2/g' $material-new.xyz
elif [ "$material" = "C" ]; then
    sed -e '1,2d' $material.xyz > $material-new.tmp
    awk '{print NR " "$s}' $material-new.tmp | tr -s ' ' > $material-new.xyz
    sed -i 's/C/1/g' $material-new.xyz
fi

    rm -rf *.tmp



    wait

    rm -rf *.xyz

    cd ..

    echo ">>>>>>>>>>>>>>> COOP/PDOS (ATOMS) DONE <<<<<<<<<<<<<<<<<"
fi

    echo "COOP ANALYSIS DONE!"  > completed.COOPCompleted

    rm -rf *.tmp
    rm -rf *.xyz

cd ../..

echo "############################## COOP/PDOS COMPLETED ##############################"

fi

# OUTPUT FILES: C2-systemLabel-FATBAND.orbitalLabel.EIGFAT and systemLabel.info (NOTE: the orbitalLabel is that found in .mpr file)


if [ $FATBANDSpost == "yes" -o $FATBANDSpost == "Yes" -o $FATBANDSpost == "YES" -o $FATBANDSpost == "y" -o $FATBANDSpost == "Y" ];
then
    echo "############################## COOP/FATBANDS ##############################"

    mkdir -p COOP
    cd COOP

    rm -rf COOPFATBANDS
    mkdir -p COOPFATBANDS
    cd COOPFATBANDS

    echo "START COOP/FATBANDS"

    echo ">>>>>>>>>>>>>>> START COOP/FATBANDS (TOTAL & ORBITALS BANDS) ANALYSIS <<<<<<<<<<<<<<<<<"

    # Move the COOP/mprop utility input file
    cp ~/SIESTA/materials/$material/$type/Utilities/$material-FATBAND.mpr .

    cp ~/SIESTA/materials/$material/$type/Utilities/$material-FATBAND-Atom.mpr .

    # H and S matrices coefficients
    cp ../../../1-Relaxation/$material.HSX .

    # Wave functions for the bandstructure sampled BZ (i.e., from the blocks BandLines or BandPoints). This is for FATBANDS analysis
    cp  ../../../3-Bands/$material.bands.WFSX ./$material.WFSX

    # Copy the full bandstructure just for potting purposes
    cp ../../GNUBANDS/$material.bands .
    cp ../../GNUBANDS/$material-bands.dat .
    cp ../../GNUBANDS/bands.txt .

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    mkdir -p COOPFATBANDS-atoms
    for (( k=$Natoms; k>=1; k-- ))
    do
        sed "2a\\$k" $material-FATBAND-Atom.mpr > $material-FATBAND-Atom-$k.mpr
        sed -i "2a\\FATBAND_$k" $material-FATBAND-Atom-$k.mpr
        mv $material-FATBAND-Atom-$k.mpr COOPFATBANDS-atoms/
        fat COOPFATBANDS-atoms/$material-FATBAND-Atom-$k
        eigfat2plot COOPFATBANDS-atoms/$material-FATBAND-Atom-$k.FATBAND_$k.EIGFAT > COOPFATBANDS-atoms/$k-bands.txt
        awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < COOPFATBANDS-atoms/$k-bands.txt > COOPFATBANDS-atoms/$k-shifted-bands.txt
    done
fi

    fat $material-FATBAND


# Define variables for specific materials
if [ "$material" = "MoS2" ]; then
    eigfat2plot $material-FATBAND.FATBAND_Mo.EIGFAT > Mo-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_5s.EIGFAT > Mo_5s-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4d.EIGFAT > Mo_4d-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dxy.EIGFAT > Mo_4dxy-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dyz.EIGFAT > Mo_4dyz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dxz.EIGFAT > Mo_4dxz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dz2.EIGFAT > Mo_4dz2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dx2-y2.EIGFAT > Mo_4dx2-y2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S.EIGFAT > S-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3p.EIGFAT > S_3p-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3px.EIGFAT > S_3px-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3py.EIGFAT > S_3py-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3pz.EIGFAT > S_3pz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3s.EIGFAT > S_3s-bands.txt
elif [ "$material" = "MoSe2" ]; then
    eigfat2plot $material-FATBAND.FATBAND_Mo.EIGFAT > Mo-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_5s.EIGFAT > Mo_5s-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4d.EIGFAT > Mo_4d-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dxy.EIGFAT > Mo_4dxy-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dyz.EIGFAT > Mo_4dyz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dxz.EIGFAT > Mo_4dxz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dz2.EIGFAT > Mo_4dz2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Mo_4dx2-y2.EIGFAT > Mo_4dx2-y2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se.EIGFAT > Se-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4p.EIGFAT > Se_4p-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4px.EIGFAT > Se_4px-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4py.EIGFAT > Se_4py-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4pz.EIGFAT > Se_4pz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4s.EIGFAT > Se_4s-bands.txt
elif [ "$material" = "WS2" ]; then
    eigfat2plot $material-FATBAND.FATBAND_W.EIGFAT > W-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_6s.EIGFAT > W_6s-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5d.EIGFAT > W_5d-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dxy.EIGFAT > W_5dxy-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dyz.EIGFAT > W_5dyz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dxz.EIGFAT > W_5dxz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dz2.EIGFAT > W_5dz2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dx2-y2.EIGFAT > W_5dx2-y2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S.EIGFAT > S-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3p.EIGFAT > S_3p-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3px.EIGFAT > S_3px-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3py.EIGFAT > S_3py-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3pz.EIGFAT > S_3pz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_S_3s.EIGFAT > S_3s-bands.txt
elif [ "$material" = "WSe2" ]; then
    eigfat2plot $material-FATBAND.FATBAND_W.EIGFAT > W-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_6s.EIGFAT > W_6s-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5d.EIGFAT > W_5d-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dxy.EIGFAT > W_5dxy-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dyz.EIGFAT > W_5dyz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dxz.EIGFAT > W_5dxz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dz2.EIGFAT > W_5dz2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_W_5dx2-y2.EIGFAT > W_5dx2-y2-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se.EIGFAT > Se-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4p.EIGFAT > Se_4p-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4px.EIGFAT > Se_4px-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4py.EIGFAT > Se_4py-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4pz.EIGFAT > Se_4pz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_Se_4s.EIGFAT > Se_4s-bands.txt
elif [ "$material" = "C" ]; then
    eigfat2plot $material-FATBAND.FATBAND_C.EIGFAT > C-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_C_2p.EIGFAT > C_2p-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_C_2px.EIGFAT > C_2px-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_C_2py.EIGFAT > C_2py-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_C_2pz.EIGFAT > C_2pz-bands.txt
    eigfat2plot $material-FATBAND.FATBAND_C_2s.EIGFAT > C_2s-bands.txt
fi

    wait

    echo "FATBANDS ANALYSIS DONE!"  > completed.FATBANDSCompleted
    echo "EIGFAT2PLOT DONE!"

    cp ../../GNUBANDS/bandsPositive.txt .
    cp ../../GNUBANDS/bandsNegative.txt .
    cp ../../GNUBANDS/CB1.txt .
    cp ../../GNUBANDS/CB2.txt .
    cp ../../GNUBANDS/CB3.txt .
    cp ../../GNUBANDS/VB1.txt .
    cp ../../GNUBANDS/VB2.txt .
    cp ../../GNUBANDS/VB3.txt .
    cp ../../GNUBANDS/bandsPositiveNoCB3.txt .
    cp ../../GNUBANDS/bandsNegativeNoVB3.txt .


# Process and shift band structures based on the material
if [ "$material" = "MoS2" ]; then
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo-bands.txt > Mo-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_5s-bands.txt > Mo_5s-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4d-bands.txt > Mo_4d-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dxy-bands.txt > Mo_4dxy-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dyz-bands.txt > Mo_4dyz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dxz-bands.txt > Mo_4dxz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dz2-bands.txt > Mo_4dz2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dx2-y2-bands.txt > Mo_4dx2-y2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S-bands.txt > S-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3p-bands.txt > S_3p-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3px-bands.txt > S_3px-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3py-bands.txt > S_3py-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3pz-bands.txt > S_3pz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3s-bands.txt > S_3s-shifted-bands.txt
elif [ "$material" = "MoSe2" ]; then
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo-bands.txt > Mo-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_5s-bands.txt > Mo_5s-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4d-bands.txt > Mo_4d-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dxy-bands.txt > Mo_4dxy-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dyz-bands.txt > Mo_4dyz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dxz-bands.txt > Mo_4dxz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dz2-bands.txt > Mo_4dz2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Mo_4dx2-y2-bands.txt > Mo_4dx2-y2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se-bands.txt > Se-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4p-bands.txt > Se_4p-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4px-bands.txt > Se_4px-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4py-bands.txt > Se_4py-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4pz-bands.txt > Se_4pz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4s-bands.txt > Se_4s-shifted-bands.txt
elif [ "$material" = "WS2" ]; then
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W-bands.txt > W-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_6s-bands.txt > W_6s-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5d-bands.txt > W_5d-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dxy-bands.txt > W_5dxy-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dyz-bands.txt > W_5dyz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dxz-bands.txt > W_5dxz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dz2-bands.txt > W_5dz2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dx2-y2-bands.txt > W_5dx2-y2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S-bands.txt > S-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3p-bands.txt > S_3p-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3px-bands.txt > S_3px-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3py-bands.txt > S_3py-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3pz-bands.txt > S_3pz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < S_3s-bands.txt > S_3s-shifted-bands.txt
elif [ "$material" = "WSe2" ]; then
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W-bands.txt > W-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_6s-bands.txt > W_6s-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5d-bands.txt > W_5d-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dxy-bands.txt > W_5dxy-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dyz-bands.txt > W_5dyz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dxz-bands.txt > W_5dxz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dz2-bands.txt > W_5dz2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < W_5dx2-y2-bands.txt > W_5dx2-y2-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se-bands.txt > Se-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4p-bands.txt > Se_4p-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4px-bands.txt > Se_4px-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4py-bands.txt > Se_4py-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4pz-bands.txt > Se_4pz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < Se_4s-bands.txt > Se_4s-shifted-bands.txt
elif [ "$material" = "C" ]; then
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C-bands.txt > C-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C_2p-bands.txt > C_2p-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C_2px-bands.txt > C_2px-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C_2py-bands.txt > C_2py-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C_2pz-bands.txt > C_2pz-shifted-bands.txt
    awk -v var=$EF -F " " '{print $1 " " $2-var " " $3}' < C_2s-bands.txt > C_2s-shifted-bands.txt
fi


    echo ">>>>>>>>>>>>>>> COOP/FATBANDS (TOTAL & ORBITALS BANDS) ANALYSIS DONE <<<<<<<<<<<<<<<<<"

if [ $ATOMSpost == "yes" -o $ATOMSpost == "Yes" -o $ATOMSpost == "YES" -o $ATOMSpost == "y" -o $ATOMSpost == "Y" ];
then
    echo ">>>>>>>>>>>>>>> START COOP/FATBANDS (ATOMS) ANALYSIS <<<<<<<<<<<<<<<<<"

    cd COOPFATBANDS-atoms

    cp ../bands.txt .
    cp ../bandsPositive.txt .
    cp ../bandsNegative.txt .


    rm -rf *.tmp

    cd ..
fi
    echo ">>>>>>>>>>>>>>> COOP/FATBANDS (ATOMS) DONE <<<<<<<<<<<<<<<<<"

    cd ../..

    echo "############################## COOP/FATBANDS COMPLETED ##############################"

fi


if [ $DENCHARpost == "yes" -o $DENCHARpost == "Yes" -o $DENCHARpost == "YES" -o $DENCHARpost == "y" -o $DENCHARpost == "Y" ];
then
    cp -R ../DENCHAR .

    echo "############################## DENCHAR ##############################"

    echo "############################## DENCHAR COMPLETED ##############################"

fi


echo ###################### grid2val ########################

cd ../1-Relaxation/

rm -rf Grids
mkdir -p Grids
cp structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt Grids/
cp $material.STRUCT_OUT Grids/
cp structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt Grids/
cp $material.LDOS Grids/
cp $material.RHO Grids/
cp $material.DRHO Grids/
cp $material.RHOXC Grids/
cp $material.TOCH Grids/
cp $material.IOCH Grids/
cp $material.VH Grids/
cp $material.VT Grids/
cp $material.VNA Grids/

cd Grids

grid2val $material.LDOS < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.LDOS
grid2val $material.RHO < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.RHO
grid2val $material.DRHO < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.DRHO
grid2val $material.RHOXC < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.RHOXC
grid2val $material.TOCH < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.TOCH
grid2val $material.IOCH < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.IOCH
grid2val $material.VH < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.VH
grid2val $material.VT < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.VT
grid2val $material.VNA < structure-TypeNumber-Index-x-y-z_FRACTIONAL.txt > atoms.VNA


g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.LDOS
mv Grid.cube LDOS.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.RHO
mv Grid.cube RHO.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.DRHO
mv Grid.cube DRHO.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.RHOXC
mv Grid.cube RHOXC.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.TOCH
mv Grid.cube TOCH.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.IOCH
mv Grid.cube IOCH.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.VH
mv Grid.cube VH.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.VT
mv Grid.cube VT.cube
g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.VNA
mv Grid.cube VNA.cube

if [[ -f "$CUBE2XYZ_SCRIPT" ]]; then
    cp "$CUBE2XYZ_SCRIPT" .
elif [[ -f "$QMATSIM_SIESTA_DIR/python-utilities/cube2xyz.py" ]]; then
    cp "$QMATSIM_SIESTA_DIR/python-utilities/cube2xyz.py" .
else
    echo "Warning: cube2xyz.py script not found" >&2
fi

python cube2xyz.py -f LDOS.cube -o LDOS.xyz -A &
python cube2xyz.py -f RHO.cube -o RHO.xyz -A &
python cube2xyz.py -f DRHO.cube -o DRHO.xyz -A &
python cube2xyz.py -f RHOXC.cube -o RHOXC.xyz -A &
python cube2xyz.py -f TOCH.cube -o TOCH.xyz -A &
python cube2xyz.py -f IOCH.cube -o IOCH.xyz -A &
python cube2xyz.py -f VH.cube -o VH.xyz -A &
python cube2xyz.py -f VT.cube -o VT.xyz -A &
python cube2xyz.py -f VNA.cube -o VNA.xyz -A &

# Wait for all background jobs to finish
wait

cd ../..

echo "############################## FINAL STEP: COPYING, REMOVING, ETC.  ##############################"

# Remove previous Results directory and create a new one with associated subfolders
#rm -rf Results
#mkdir -p Results

mkdir -p Results/1-Relaxation
mkdir -p Results/2-Static
mkdir -p Results/3-Bands
mkdir -p Results/Postprocessing
mkdir -p Results/ExtractedParameters
mkdir -p Results/Grids
mkdir -p Results/Structure
mkdir -p Results/Plots
mkdir -p Results/DENCHAR

# 1-Relaxation (COPY from 1-Relaxation)
cp 1-Relaxation/*.fdf Results/1-Relaxation
cp 1-Relaxation/*.sh Results/1-Relaxation
cp 1-Relaxation/*.log Results/1-Relaxation
cp 1-Relaxation/$material.xyz Results/1-Relaxation
cp 1-Relaxation/$material.XV Results/1-Relaxation
cp 1-Relaxation/*.psf Results/1-Relaxation
cp 1-Relaxation/$material.xtl Results/1-Relaxation
cp 1-Relaxation/$material.CG Results/1-Relaxation
cp 1-Relaxation/$material.DM Results/1-Relaxation
cp 1-Relaxation/$material.STRUCT_OUT Results/1-Relaxation
cp 1-Relaxation/$material.ANI Results/1-Relaxation
cp 1-Relaxation/siesta.out Results/1-Relaxation
cp 1-Relaxation/$material.STRUCT_OUT Results/Structure/
cp 1-Relaxation/structure.ANI Results/Structure/

cp 1-Relaxation/Grids/*.xyz Results/Grids/

# 2-Static
cp 2-Static/*.fdf Results/2-Static
cp 2-Static/*.sh Results/2-Static
cp 2-Static/*.log Results/2-Static
cp 1-Relaxation/$material.xyz Results/2-Static
cp 1-Relaxation/$material.XV Results/2-Static
cp 2-Static/*.psf Results/2-Static
cp 1-Relaxation/$material.xtl Results/2-Static
cp 1-Relaxation/$material.CG Results/2-Static
cp 2-Static/$material.DM Results/2-Static
cp 2-Static/siesta.out Results/2-Static
cp 1-Relaxation/$material.STRUCT_OUT Results/2-Static/


# 3-Bands
cp 3-Bands/*.fdf Results/3-Bands
cp 3-Bands/*.sh Results/3-Bands
cp 3-Bands/*.log Results/3-Bands
cp 1-Relaxation/$material.xyz Results/3-Bands
cp 1-Relaxation/$material.XV Results/3-Bands
cp 3-Bands/*.psf Results/3-Bands
cp 1-Relaxation/$material.xtl Results/3-Bands
cp 1-Relaxation/$material.CG Results/3-Bands
cp 3-Bands/$material.DM Results/3-Bands
cp 3-Bands/siesta.out Results/3-Bands
cp 1-Relaxation/$material.STRUCT_OUT Results/3-Bands/

###################################

# STATES
cp -R Postprocessing/ Results/
#rm -rf Results/Postprocessing/GNUBANDS/states/

#cp -R Postprocessing/PDOSXML Results/Postprocessing
#rsync -r --exclude '.states' Results/Postprocessing/GNUBANDS Results/Postprocessing
#rsync -av --progress Postprocessing/GNUBANDS /Results/Postprocessing --exclude states
#cp -R Postprocessing/COOP Results/Postprocessing
#cp -R Postprocessing/DENCHAR Results/Postprocessing

mkdir -p Results/Postprocessing/GNUBANDS/states
mkdir -p Results/Postprocessing/GNUBANDS/states/CBM-VBM
mkdir -p Results/Postprocessing/GNUBANDS/states/all

#cp Postprocessing/GNUBANDS/states/state-$VBM1state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
#cp Postprocessing/GNUBANDS/states/state-$VBM2state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
#cp Postprocessing/GNUBANDS/states/state-$VBM3state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
#cp Postprocessing/GNUBANDS/states/state-$CBM1state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
#cp Postprocessing/GNUBANDS/states/state-$CBM2state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
#cp Postprocessing/GNUBANDS/states/state-$CBM3state.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM

cp Postprocessing/GNUBANDS/states/extracted-VBM1-state-$VBM1state-band-$VBM1band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
cp Postprocessing/GNUBANDS/states/extracted-VBM2-state-$VBM2state-band-$VBM2band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
cp Postprocessing/GNUBANDS/states/extracted-VBM3-state-$VBM3state-band-$VBM3band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
cp Postprocessing/GNUBANDS/states/extracted-CBM1-state-$CBM1state-band-$CBM1band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
cp Postprocessing/GNUBANDS/states/extracted-CBM2-state-$CBM2state-band-$CBM2band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM
cp Postprocessing/GNUBANDS/states/extracted-CBM3-state-$CBM3state-band-$CBM3band.txt Results/Postprocessing/GNUBANDS/states/CBM-VBM

#cp Postprocessing/GNUBANDS/states/all/ Results/Postprocessing/GNUBANDS/states/all

# PLOTS
#cp 1-Relaxation/*.png Results/Plots
#cp Postprocessing/PDOSXML/*.png Results/Plots
#cp Postprocessing/COOP/COOPPDOS/*.png Results/Plots
#cp Postprocessing/GNUBANDS/*.png Results/Plots
#cp Postprocessing/COOP/COOPFATBANDS/*.png Results/Plots
#rm -rf Results/Plots/DOS.png

  # PDOSXML
  mkdir Results/Plots/PDOSXML-atoms
  #mv Results/Postprocessing/PDOSXML/PDOSXML-atoms/*.png Results/Plots/PDOSXML-atoms

  # COOPPDOS
  mkdir Results/Plots/COOPPDOS-atoms
  #mv Results/Postprocessing/COOP/COOPPDOS/COOPPDOS-atoms/*.png Results/Plots/COOPPDOS-atoms

  # FATBANDS-atoms
  mkdir Results/Plots/FATBANDS-atoms
  #mv Results/Postprocessing/COOP/COOPFATBANDS/COOPFATBANDS-atoms/*.png Results/Plots/FATBANDS-atoms

# EXTRACTED PARAMETERS
cp 1-Relaxation/*.out Results/ExtractedParameters
cp Postprocessing/GNUBANDS/*.out Results/ExtractedParameters

# REMOVING WFSX AND WFSXR (large files)
#rm -rf Results/ExtractedParameters/siesta.out
rm -rf Results/1-Relaxation/*.WFSX
rm -rf Results/1-Relaxation/*.WFSXR
rm -rf Results/2-Static/*.WFSX
rm -rf Results/2-Static/*.WFSXR
rm -rf Results/3-Bands/*.WFSX
rm -rf Results/3-Bands/*.WFSXR
rm -rf Results/Postprocessing/PDOSXML/*.WFSX
rm -rf Results/Postprocessing/PDOSXML/*.WFSXR
rm -rf Results/Postprocessing/GNUBANDS/*.WFSX
rm -rf Results/Postprocessing/GNUBANDS/*.WFSXR

rm -rf Results/Postprocessing/PDOSXML/*.png
rm -rf Results/Postprocessing/GNUBANDS/*.png

cp -R 1-Relaxation/Grids Results/Grids
rm -rf Results/Grids/$material.*

#cd Results/Postprocessing/GNUBANDS/
#cp *.out ../../ExtractedParameters
#rm *.out
#rm *.m
#rm title1.txt
#rm title2.txt
#
#cp latticeConstants.txt ../../ExtractedParameters
#cp latticeAngles.txt ../../ExtractedParameters
#cp latticeVolume.txt ../../ExtractedParameters
#rm latticeConstants.txt latticeAngles.txt latticeVolume.txt
#
#cd ../../../
#
#cd Results/Postprocessing/COOP/COOPFATBANDS/
#rm -if !(*-shifted-bands.txt)
#cd COOPFATBANDS-atoms
#rm -if !(*-shifted-bands.txt)
#cd ../../../../..
#
#cd Results/Postprocessing/COOP/COOPPDOS/COOPPDOS-atoms
#rm -if !(*_shifted.pdos)
#cd ..
#rm -if !(*_shifted.pdos|*.ados|*.alldos|*.intdos)
#cd ../../../../

cd Results


# STRUCTURE
cd Structure

cp ../1-Relaxation/$material.xyz .
cp ../1-Relaxation/$material.xtl .


head -n 2 $material.xyz > header.tmp

cp ../../1-Relaxation/latticeHeader-label.txt latticeHeader-label.tmp
cp ../../1-Relaxation/latticeHeader-type.txt latticeHeader-type.tmp
cp ../../1-Relaxation/latticeHeader-id-label.txt latticeHeader-id-label.tmp
cp ../../1-Relaxation/latticeHeader-id-type.txt latticeHeader-id-type.tmp

head -n1 ../../1-Relaxation/$material.xyz > Natoms.tmp

cat Natoms.tmp latticeHeader-label.tmp > latticeHeader-label.txt
cat Natoms.tmp latticeHeader-type.tmp > latticeHeader-type.txt
cat Natoms.tmp latticeHeader-id-label.tmp > latticeHeader-id-label.txt
cat Natoms.tmp latticeHeader-id-type.tmp > latticeHeader-id-type.txt


# General processing of XYZ and XTL files based on material
awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
cp LABEL.tmp TYPE.tmp

# Specific replacements based on the material
if [ "$material" = "MoS2" ]; then
    sed -i 's/Mo/1/g' TYPE.tmp
    sed -i 's/S/2/g' TYPE.tmp
elif [ "$material" = "MoSe2" ]; then
    sed -i 's/Mo/1/g' TYPE.tmp
    sed -i 's/Se/2/g' TYPE.tmp
elif [ "$material" = "WS2" ]; then
    sed -i 's/W/1/g' TYPE.tmp
    sed -i 's/S/2/g' TYPE.tmp
elif [ "$material" = "WSe2" ]; then
    sed -i 's/W/1/g' TYPE.tmp
    sed -i 's/Se/2/g' TYPE.tmp
elif [ "$material" = "C" ]; then
    sed -i 's/C/1/g' TYPE.tmp
fi

awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
awk '{print NR > "INDEX.tmp"}' INDEX.tmp
awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp

# Material-specific processing
if [ "$material" = "MoS2" ]; then
    awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
    cp LABEL.tmp TYPE.tmp
    sed -i 's/Mo/1/g' TYPE.tmp
    sed -i 's/S/2/g' TYPE.tmp
    awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
    awk '{print NR > "INDEX.tmp"}' INDEX.tmp
    awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
    awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp
elif [ "$material" = "MoSe2" ]; then
    awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
    cp LABEL.tmp TYPE.tmp
    sed -i 's/Mo/1/g' TYPE.tmp
    sed -i 's/Se/2/g' TYPE.tmp
    awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
    awk '{print NR > "INDEX.tmp"}' INDEX.tmp
    awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
    awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp
elif [ "$material" = "WS2" ]; then
    awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
    cp LABEL.tmp TYPE.tmp
    sed -i 's/W/1/g' TYPE.tmp
    sed -i 's/S/2/g' TYPE.tmp
    awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
    awk '{print NR > "INDEX.tmp"}' INDEX.tmp
    awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
    awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp
elif [ "$material" = "WSe2" ]; then
    awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
    cp LABEL.tmp TYPE.tmp
    sed -i 's/W/1/g' TYPE.tmp
    sed -i 's/Se/2/g' TYPE.tmp
    awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
    awk '{print NR > "INDEX.tmp"}' INDEX.tmp
    awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
    awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp
elif [ "$material" = "C" ]; then
    awk 'NR>=3 {print $1}' $material.xyz > LABEL.tmp
    cp LABEL.tmp TYPE.tmp
    sed -i 's/C/1/g' TYPE.tmp
    awk 'NR>=3 {print $1}' $material.xyz > INDEX.tmp
    awk '{print NR > "INDEX.tmp"}' INDEX.tmp
    awk 'NR>=3 {print $2, $3, $4}' $material.xyz > CARTESIAN.tmp
    awk 'NR>=9 {print $2, $3, $4}' $material.xtl > FRACTIONAL.tmp
fi

mkdir -p other
mkdir -p other/xyz/
mkdir -p other/xyz/not-numbered/
mkdir -p other/xyz/not-numbered/label
mkdir -p other/xyz/not-numbered/type
mkdir -p other/xyz/numbered/
mkdir -p other/xyz/numbered/label
mkdir -p other/xyz/numbered/type

cd other/xyz/not-numbered/label
cat ../../../../latticeHeader-label.txt <(paste ../../../../LABEL.tmp ../../../../CARTESIAN.tmp) > structure_CARTESIAN.xyz
cat ../../../../latticeHeader-label.txt <(paste ../../../../LABEL.tmp ../../../../FRACTIONAL.tmp) > structure_FRACTIONAL.xyz
sed -i '/^\t*$/d' *.xyz
sed -i '/^[[:space:]]*$/d' *.xyz
cd ..


cd type
cat ../../../../latticeHeader-type.txt <(paste ../../../../TYPE.tmp ../../../../CARTESIAN.tmp) > structure_CARTESIAN.xyz
cat ../../../../latticeHeader-type.txt <(paste ../../../../TYPE.tmp ../../../../FRACTIONAL.tmp) > structure_FRACTIONAL.xyz
sed -i '/^\t*$/d' *.xyz
sed -i '/^[[:space:]]*$/d' *.xyz
cd ..

cd ..
cd numbered/label
cat ../../../../latticeHeader-id-label.txt <(paste ../../../../INDEX.tmp ../../../../LABEL.tmp ../../../../CARTESIAN.tmp) > structure_CARTESIAN.xyz
cat ../../../../latticeHeader-id-label.txt <(paste ../../../../INDEX.tmp ../../../../LABEL.tmp ../../../../FRACTIONAL.tmp) > structure_FRACTIONAL.xyz
sed -i '/^\t*$/d' *.xyz
sed -i '/^[[:space:]]*$/d' *.xyz
cd ..

cd type
cat ../../../../latticeHeader-id-type.txt <(paste ../../../../INDEX.tmp ../../../../TYPE.tmp ../../../../CARTESIAN.tmp) > structure_CARTESIAN.xyz
cat ../../../../latticeHeader-id-type.txt <(paste ../../../../INDEX.tmp ../../../../TYPE.tmp ../../../../FRACTIONAL.tmp) > structure_FRACTIONAL.xyz
sed -i '/^\t*$/d' *.xyz
sed -i '/^[[:space:]]*$/d' *.xyz
cd ../../../..


mkdir -p other
mkdir -p other/txt/
mkdir -p other/txt/not-numbered/
mkdir -p other/txt/not-numbered/label
mkdir -p other/txt/not-numbered/type
mkdir -p other/txt/numbered/
mkdir -p other/txt/numbered/label
mkdir -p other/txt/numbered/type


cd other/txt/not-numbered/label
paste ../../../../LABEL.tmp ../../../../CARTESIAN.tmp> structure_CARTESIAN.txt
paste ../../../../LABEL.tmp ../../../../FRACTIONAL.tmp > structure_FRACTIONAL.txt
sed -i '/^\t*$/d' *.txt
sed -i '/^[[:space:]]*$/d' *.txt
cd ..


cd type
paste ../../../../TYPE.tmp ../../../../CARTESIAN.tmp > structure_CARTESIAN.txt
paste ../../../../TYPE.tmp ../../../../FRACTIONAL.tmp > structure_FRACTIONAL.txt
sed -i '/^\t*$/d' *.txt
sed -i '/^[[:space:]]*$/d' *.txt
cd ..

cd ..
cd numbered/label
paste ../../../../INDEX.tmp ../../../../LABEL.tmp ../../../../CARTESIAN.tmp > structure_CARTESIAN.txt
paste ../../../../INDEX.tmp ../../../../LABEL.tmp ../../../../FRACTIONAL.tmp > structure_FRACTIONAL.txt
sed -i '/^\t*$/d' *.txt
sed -i '/^[[:space:]]*$/d' *.txt
cd ..

cd type
paste ../../../../INDEX.tmp ../../../../TYPE.tmp ../../../../CARTESIAN.tmp > structure_CARTESIAN.txt
paste ../../../../INDEX.tmp ../../../../TYPE.tmp ../../../../FRACTIONAL.tmp > structure_FRACTIONAL.txt
sed -i '/^\t*$/d' *.txt
sed -i '/^[[:space:]]*$/d' *.txt
cd ../../../..



cp other/xyz/numbered/type/structure_CARTESIAN.xyz .
cp other/xyz/numbered/type/structure_FRACTIONAL.xyz .


cp other/txt/numbered/type/structure_CARTESIAN.txt .
cp other/txt/numbered/type/structure_FRACTIONAL.txt .

find . -type f -name "*.tmp" -delete

rm $material.xyz
rm $material.xtl

cp ../../1-Relaxation/$material.STRUCT_OUT $material.STRUCT_OUT

cd ../../

cd ${mainDir}

./A-general-edit.sh

cd ${mainDir}

./B-tables.sh

cd ${mainDir}

#./C-bonds.sh

#wait

#cd ${mainDir}

# CLEAN THE RUN FOLDER

cd 1-Relaxation

cp ~/SIESTA/referenceFiles/findsym_sample-copy1.in findsym_sample.in

# Material

titleVAR="Symmetrized coordinates"
latticeTolVAR="0.00001"
positionsTolVAR="0.00001"
occupationTolVAR="0.00001"
#centeringVAR="P"

# Read the lattice parameters from $material.STRUCT_OUT
a1=$(awk 'NR==1 {printf "%s %s %s", $1, $2, $3}' $material.STRUCT_OUT)
a2=$(awk 'NR==2 {printf "%s %s %s", $1, $2, $3}' $material.STRUCT_OUT)
a3=$(awk 'NR==3 {printf "%s %s %s", $1, $2, $3}' $material.STRUCT_OUT)
NatomsVAR=$(awk 'NR==4 {print $1}' $material.STRUCT_OUT)
atomTypesVAR=$(awk 'NR>=5 {printf "%s ", $1}' $material.STRUCT_OUT)
atomsPositionsVAR=$(awk 'NR>=5 {print $3, $4, $5}' $material.STRUCT_OUT)

# Replace latticeVAR in findsym_sample.in
sed -i "s|titleVAR|${titleVAR}|" findsym_sample.in
sed -i "s|latticeTolVAR|${latticeTolVAR}|" findsym_sample.in
sed -i "s|positionsTolVAR|${positionsTolVAR}|" findsym_sample.in
sed -i "s|occupationTolVAR|${occupationTolVAR}|" findsym_sample.in
sed -i "s|latticeVAR|${a1}\n${a2}\n${a3}|" findsym_sample.in
#sed -i "s|centeringVAR|${centeringVAR}|" findsym_sample.in
sed -i "s|NatomsVAR|${NatomsVAR}|" findsym_sample.in
sed -i "s|atomTypesVAR|${atomTypesVAR}|" findsym_sample.in
awk -v var="$atomsPositionsVAR" '{gsub("atomsPositionsVAR", var)}1' findsym_sample.in > temp && mv temp findsym_sample.in

if command -v "$FINDSYM_EXE" >/dev/null 2>&1; then
    "$FINDSYM_EXE" findsym_sample.in > "$material-symmetrized.cif"
else
    echo "Warning: FINDSYM executable not found, skipping symmetry analysis" >&2
fi

awk '/_atom_site_fract_symmform/{flag=1; next} flag{print $2, $5, $6, $7}' $material-symmetrized.cif > $material.tmp
awk '{ if ($1 == 2) { $1 = "2 16" } else if ($1 == 1) { $1 = "1 42" } print }' $material.tmp > $material.modified.tmp
head -n 4 $material.STRUCT_OUT > $material-symmetrized.STRUCT_OUT && cat $material.modified.tmp >> $material-symmetrized.STRUCT_OUT
sed -i '$d' $material-symmetrized.STRUCT_OUT

cp $material-symmetrized.cif ../Results/Structure
cp $material-symmetrized.STRUCT_OUT ../Results/Structure

#rm -rf *.out
rm -rf *.err
rm -rf ./*~
rm -rf ./*#

cp Results/ExtractedParameters/siesta.out 1-Relaxation/

echo "POSTPROCESSING OF $structure-$supercell COMPLETED!"
