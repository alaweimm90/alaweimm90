#!/bin/bash

shopt -s extglob
shopt -s expand_aliases


material=materialVAR
type=typeVAR
structure=structureVAR
supercell=supercellVAR
functional=functionalVAR
direction=directionVAR
SOC=socVAR
atom1=atom1VAR
atom2=atom2VAR

mainDir=$(pwd)

cd 1-Relaxation


    # Remove existing hirshfeld.txt and voronoi.txt files
    rm -f hirshfeld.txt voronoi.txt

    awk 'BEGIN {
        species["'"$atom1"'"] = 1
        species["'"$atom2"'"] = 2
    }
    /Hirshfeld Net Atomic Populations:/ { mode = "hirshfeld"; next }
    /Voronoi Net Atomic Populations:/ { mode = "voronoi"; next }
    $1 ~ /^[0-9]+$/ && $3 == "$atom1" { $3 = 1 }
    $1 ~ /^[0-9]+$/ && $3 == "$atom2" { $3 = 2 }
    mode == "hirshfeld" && NF == 3 { print $1 "\t" $2 "\t" $3 >> "hirshfeld.txt" }
    mode == "voronoi" && NF == 3 { print $1 "\t" $2 "\t" $3 >> "voronoi.txt" }' siesta.out

    # Remove the first and last lines in voronoi.txt
    sed -i '$d' voronoi.txt


    grep -E 'dhscf: Vacuum level \(max, mean\) =' siesta.out | awk -F '=' '{print $2}' | awk '{print $1}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPvac.out


    ######### KINETIC ENERGY (EK) #########
    grep -E 'siesta: Ekin[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EKe.out                     # Kinetic energy due to motion of electrons
    grep -E 'siesta: Eions[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EKn.out                    # Kinetic energy due to motion of ions

    ######### POTENTIAL ENERGIES (EP) #########
    grep -E 'siesta:[[:space:]]*Ion\-electron[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPen.out                # Potential energy "electrostatic" due to e-n Coulomb interaction (between individual paris)
     grep -E 'siesta:[[:space:]]*Ion-ion[[:space:]]*=' siesta.out | awk -F'=' '{print $2}' > EPnn.out   # Potential energy "electrostatic" due to n-n Coulomb interaction (between individual paris)
    grep -E 'siesta: Eions[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPnntotal.out      # Potential energy "electrostatic" due to both n-n Coulomb interaction + long-range interactions. That is, this is the "total electrostatic energy" between all pairs of ions in the system: direct Coulombic interactions between ions + any contribution from long-range electrostatic interactions (e.g., such as those treated using PBC in simulations of periodic systems)
    grep -E 'siesta: Ena[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPatoms.out                  # Potential energy of neutral isolated
    grep -E 'siesta: Eso[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPSO.out                     # Potential energy contribution from to spin-orbit coupling
    grep -E 'siesta: Exc[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPXC.out                     # Potential energy due to exchange-correlation (note: exchange is QM exchange interaction between electrons arising from Pauli exclusion principle, and in DFT it accounts for the repulsion between electrons with the same spin and their tendency to avoid each other's locations; correlation refers to e-e correlation, which described the tendency of electrons to avoid each other due to their mutual repulsionl, and it is important in metallic systems and strongly-correlated materials. Note that both terms in DFT are approximated using XC functionals)
    grep -E 'siesta: Edftu[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EDFTplusU.out      # Potential energy correction for the double-counting of e-e interactions in DFT+U calculations
    grep -E 'siesta: DEna[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > dEPatomsForces.out  # Potential energy difference due to neutral atom forces (i.e., the difference in energy due to the forces exerted by the neutral atoms)
    grep -E 'siesta: DUscf[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > dEPSCF.out         # Potential energy difference due to SCF convergence (i.e., the difference in energy resulting from changes in the SCG iterations)
    grep -E 'siesta: DUext[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > dEPext.out         # Potential energy difference due to external potential (i.e., the energy differenc deu to changes in the external potential)
    grep -E 'siesta: eta\*DQ[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPetadQ.out                      # Potential energy due to charge smearing (i.e., the energy contribution from the smearing parameter in calculations with T | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > 0 K)

    grep -E 'siesta: Ebs[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPband.out                   # Potential energy due to the formation of the electronic band structure
    grep -E 'siesta: Enl[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPnonlocalpseudo.out                 # Potential energy contribution of the nonlocal part of the pseudopotential
    grep -E 'siesta: Emadel[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPMadelung.out                    # Potential energy due to Madelung potential (often important in periodic systems)
    grep -E 'siesta: Emeta[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPmetaGGA.out                      # Potential energy due to meta-GGA
    grep -E 'siesta: Emolmec[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPmolecmech.out          # Potential energy "molecular mechanical" from molecular mechanics calculations (note: often used in combined QM/molecular mechanics simulations)
    grep -E 'siesta: Eharris[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPHarris.out             # Potential energy (total Harris-Foulkes) (i.e., total energy computing using the Harris-Foulkes functional, which is commonly used in SIESTA)
    grep -E 'siesta: FreeEng[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPfree.out               # Potential energy (Free) (i.e., the Helmholtz free energy)
    grep -E 'siesta:[[:space:]]*Fermi[[:space:]]*=' siesta.out | awk '{print $4}' > EPFermi.out                 # Potential energy (Fermi) (i.e., highest occupied electronic state at T[[:space:]]*=0 K)
    grep -E 'siesta:[[:space:]]*Hartree[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > EPHartree.out         # Potential energy (Hartree) "electrostatic"  (i.e., electrostatic energy between charged particles in the system)
    grep -E 'siesta:[[:space:]]*Ext\. field[[:space:]]*=' siesta.out | awk -F '=' '{print $2}' > EPext.out              # Potential energy due to external field (i.e. energy contribution from an externally applied E-field of H-field)

    grep -E 'siesta: Etot[[:space:]]*=' siesta.out | awk '{print $4}' | head -n 1 | sed 's/^[ \t]*//' | tr -s ' ' > Etotal.out                  # Potential energy (total)

    paste EPvac.out EPFermi.out | awk '{print $1 - $2}' > Phi.out

    #awk '/siesta: Atomic forces \(eV\/Ang\):/{block=1; lines=""} block{lines = lines $0 "\n"} /Tot/{block=0} END{print lines}' siesta.out | tail -n +2 | head -n -3 > Fatoms.txt
    tail -n +2 *.FA | sed 's/^[ \t]*//' > Fatoms.txt

    BohrToAng=1.8897

    tail -n +2 *.KP | awk -v factor="$BohrToAng" '{printf "%d %.6f %.6f %.6f %.6f\n", $1, $2 * factor, $3 * factor, $4 * factor, $5}' > Kpoints.txt

    tail -n +2 NON_TRIMMED_KP_LIST | awk -v factor="$BohrToAng" '{printf "%d %.6f %.6f %.6f %.6f\n", $1, $2 * factor, $3 * factor, $4 * factor, $5}' > Kpoints-non-trimmed.txt


    mv *.out ../Results/ExtractedParameters

    cp hirshfeld.txt ../Results/ExtractedParameters
    cp voronoi.txt ../Results/ExtractedParameters
    cp Fatoms.txt ../Results/ExtractedParameters
    cp Kpoints.txt ../Results/ExtractedParameters
    cp Kpoints-non-trimmed.txt ../Results/ExtractedParameters
    cp *.MDE ../Results/ExtractedParameters/CG-summary.txt

    cd ..

cd ${mainDir}

    cd Results

    rm -rf DENCHAR

module load anaconda3

awk 'NR >= 3 { if (NR == 3) { minx = maxx = $2; miny = maxy = $3; minz = maxz = $4; } else { if ($2 < minx) minx = $2; if ($2 > maxx) maxx = $2; if ($3 < miny) miny = $3; if ($3 > maxy) maxy = $3; if ($4 < minz) minz = $4; if ($4 > maxz) maxz = $4; }} END { print minx > "ExtractedParameters/minx.out"; print maxx > "ExtractedParameters/maxx.out"; print miny > "ExtractedParameters/miny.out"; print maxy > "ExtractedParameters/maxy.out"; print minz > "ExtractedParameters/minz.out"; print maxz > "ExtractedParameters/maxz.out" }' 1-Relaxation/$material.xyz

minx=$(<ExtractedParameters/minx.out)
maxx=$(<ExtractedParameters/maxx.out)
miny=$(<ExtractedParameters/miny.out)
maxy=$(<ExtractedParameters/maxy.out)
minz=$(<ExtractedParameters/minz.out)
maxz=$(<ExtractedParameters/maxz.out)


    mkdir -p DENCHAR

    cd DENCHAR

    mkdir -p 2D/
    mkdir -p 3D/

    cd 2D

    # Add this to debug the substitution
    echo "Material: $material, Atom1: $atom1, Atom2: $atom2"

    # Fix references to ensure proper expansion
    cp ../../../1-Relaxation/${atom1}.ion .
    cp ../../../1-Relaxation/${atom2}.ion .
    cp ../../../1-Relaxation/${atom1}.psf .
    cp ../../../1-Relaxation/${atom2}.psf .
    cp ../../../1-Relaxation/${material}.DIM .
    cp ../../../1-Relaxation/${material}.DM .
    cp ../../../1-Relaxation/${material}.PLD .
    cp ../../../3-Bands/${material}.bands.WFSX . ################ THIS SHOULD BE CHANGED LATER TO 3-BANDS ... I AM DOING THIS FOR TESTING SINCE SMALLER FILE
    mv ${material}.bands.WFSX ${material}.WFSX
    cp ~/bin/denchar-serial .

    cd ..

    cd 3D
    # Add this to debug the substitution
    echo "Material: $material, Atom1: $atom1, Atom2: $atom2"

    # Fix references to ensure proper expansion
    cp ../../../1-Relaxation/${atom1}.ion .
    cp ../../../1-Relaxation/${atom2}.ion .
    cp ../../../1-Relaxation/${atom1}.psf .
    cp ../../../1-Relaxation/${atom2}.psf .
    cp ../../../1-Relaxation/${material}.DIM .
    cp ../../../1-Relaxation/${material}.DM .
    cp ../../../1-Relaxation/${material}.PLD .
    cp ../../../3-Bands/${material}.bands.WFSX . ################ THIS SHOULD BE CHANGED LATER TO 3-BANDS ... I AM DOING THIS FOR TESTING SINCE SMALLER FILE
    mv ${material}.bands.WFSX ${material}.WFSX
    cp ~/bin/denchar-serial .

    cd ..

    cd 3D

cp ~/SIESTA/materials/$material/Monolayer/Utilities/$material-DENCHAR.fdf .
awk '/^#* k-point sampling / {exit} {print}' ../../../FILES/1-Relaxation.fdf > header.tmp
cp $material-DENCHAR.fdf $material-DENCHAR.tmp
cat header.tmp $material-DENCHAR.tmp > $material-DENCHAR.fdf

# Define your variables here
dimensionVAR="3D"
chargeVAR="true"
wfVAR="true"

if [ $maxy > $maxx ]; then

     MinXVAR="0.0"
     MaxXVAR=$(awk 'NR==1 {print $1}' ../../1-Relaxation/$material.STRUCT_OUT)
     MinYVAR="0.0"
     MaxYVAR=$(awk 'NR==2 {print 1.5 * $2}' ../../1-Relaxation/$material.STRUCT_OUT)
     MinZVAR="0.0"
     MaxZVAR=$(awk 'NR==3 {print $3}' ../../1-Relaxation/$material.STRUCT_OUT)

     NumberPointsXVAR="2"
     NumberPointsYVAR="1200"
     NumberPointsZVAR="800"

else
     MinXVAR="0.0"
     MaxXVAR=$(awk 'NR==1 {print 1.5 * $1}' ../../1-Relaxation/$material.STRUCT_OUT)
     MinYVAR="0.0"
     MaxYVAR=$(awk 'NR==2 {print $2}' ../../1-Relaxation/$material.STRUCT_OUT)
     MinZVAR="0.0"
     MaxZVAR=$(awk 'NR==3 {print $3}' ../../1-Relaxation/$material.STRUCT_OUT)

     NumberPointsXVAR="1200"
     NumberPointsYVAR="2"
     NumberPointsZVAR="800"
fi

# Use sed to replace placeholders with variables
sed -i "s/dimensionVAR/$dimensionVAR/g" $material-DENCHAR.fdf
sed -i "s/chargeVAR/$chargeVAR/g" $material-DENCHAR.fdf
sed -i "s/wfVAR/$wfVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsXVAR/$NumberPointsXVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsYVAR/$NumberPointsYVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsZVAR/$NumberPointsZVAR/g" $material-DENCHAR.fdf
sed -i "s/MinXVAR/$MinXVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxXVAR/$MaxXVAR/g" $material-DENCHAR.fdf
sed -i "s/MinYVAR/$MinYVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxYVAR/$MaxYVAR/g" $material-DENCHAR.fdf
sed -i "s/MinZVAR/$MinZVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxZVAR/$MaxZVAR/g" $material-DENCHAR.fdf

cd ..

cd 2D

cp ~/SIESTA/materials/$material/Monolayer/Utilities/$material-DENCHAR.fdf .
awk '/^#* k-point sampling / {exit} {print}' ../../../FILES/1-Relaxation.fdf > header.tmp
cp $material-DENCHAR.fdf $material-DENCHAR.tmp
cat header.tmp $material-DENCHAR.tmp > $material-DENCHAR.fdf

 # Define your variables here
 dimensionVAR="2D"
 chargeVAR="true"
 wfVAR="true"

# If statement to determine NumberPointsXVAR, NumberPointsYVAR, and NumberPointsZVAR
if [ $maxy > $maxx ]; then

    MinXVAR="0.0"
    MaxXVAR=$(awk 'NR==2 {print 1.5 * $2}' ../../1-Relaxation/$material.STRUCT_OUT)
    MinYVAR="0.0"
    MaxYVAR=$(awk 'NR==3 {print $3}' ../../1-Relaxation/$material.STRUCT_OUT)
    MinZVAR="0.0"
    MaxZVAR=$(awk 'NR==1 {print $1}' ../../1-Relaxation/$material.STRUCT_OUT)

    NumberPointsXVAR="1200"
    NumberPointsYVAR="800"
    NumberPointsZVAR="2"
        sed -i '/%block Denchar.Comp2Vectors/,/%endblock/c\%block Denchar.Comp2Vectors\n 0.000 1.000 0.000\n 0.000 0.000 1.000\n%endblock' $material-DENCHAR.fdf
else

    MinXVAR="0.0"
    MaxXVAR=$(awk 'NR==1 {print 1.5 * $1}' ../../1-Relaxation/$material.STRUCT_OUT)
    MinYVAR="0.0"
    MaxYVAR=$(awk 'NR==3 {print $3}' ../../1-Relaxation/$material.STRUCT_OUT)
    MinZVAR="0.0"
    MaxZVAR=$(awk 'NR==2 {print $2}' ../../1-Relaxation/$material.STRUCT_OUT)

    NumberPointsXVAR="1200"
    NumberPointsYVAR="800"
    NumberPointsZVAR="2"
    sed -i '/%block Denchar.Comp2Vectors/,/%endblock/c\%block Denchar.Comp2Vectors\n 1.000 0.000 0.000\n 0.000 0.000 1.000\n%endblock' $material-DENCHAR.fdf
fi

# Use sed to replace placeholders with variables
sed -i "s/dimensionVAR/$dimensionVAR/g" $material-DENCHAR.fdf
sed -i "s/chargeVAR/$chargeVAR/g" $material-DENCHAR.fdf
sed -i "s/wfVAR/$wfVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsXVAR/$NumberPointsXVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsYVAR/$NumberPointsYVAR/g" $material-DENCHAR.fdf
sed -i "s/NumberPointsZVAR/$NumberPointsZVAR/g" $material-DENCHAR.fdf
sed -i "s/MinXVAR/$MinXVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxXVAR/$MaxXVAR/g" $material-DENCHAR.fdf
sed -i "s/MinYVAR/$MinYVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxYVAR/$MaxYVAR/g" $material-DENCHAR.fdf
sed -i "s/MinZVAR/$MinZVAR/g" $material-DENCHAR.fdf
sed -i "s/MaxZVAR/$MaxZVAR/g" $material-DENCHAR.fdf

cd ..


    mkdir -p bands
    cd bands


    EF=$(sed -n -e '/E_F/ s/.*\= *//p' ../../../Postprocessing/GNUBANDS/*.dat)
    EFbands=$EF
    #EF=$(<../../../Results/ExtractedParameters/EPFermi.out) ##### ONLY in bands

    k=$(sed -n -e '/k_min, k_max/ s/.*\= *//p' ../../../Postprocessing/GNUBANDS/*.dat | tr -s " ")
    BohrToAng=1.8897;
    kmin0=$(echo $k | awk '{print $1}')
    kmax0=$(echo $k | awk '{print $2}')
    kmin=$(bc <<< 'scale=2; '$kmin0'*'$BohrToAng'')
    kmax=$(bc <<< 'scale=2; '$kmax0'*'$BohrToAng'')

    E=$(sed -n -e '/E_min, E_max/ s/.*\= *//p' ../../../Postprocessing/GNUBANDS/*.dat | tr -s " ")
    Emin=$(echo $E | awk '{print $1}')
    Emax=$(echo $E | awk '{print $2}')
    Nvalues=$(sed -n -e '/Nbands, Nspin, Nk/ s/.*\= *//p' ../../../Postprocessing/GNUBANDS/*.dat | tr -s " ")
    Nbands=$(echo $Nvalues | awk '{print $1}')
    Nspin=$(echo $Nvalues | awk '{print $2}')
    Nk=$(echo $Nvalues | awk '{print $3}')

    readwfx -m -6.5 -M -1.5 ../../../3-Bands/$material.bands.WFSX > $material.bands.WFSXR
    echo ">>>>>>>>>>>>>>> READWFX DONE! <<<<<<<<<<<<<<<<<"

    grep "k-point =" $material.bands.WFSXR | tr -s ' ' |  cut -d ' ' -f3,4,5,6 > index-kx-ky-kz.txt
    awk -F ' ' '{print $1, $2}' index-kx-ky-kz.txt > index-kx.txt
    awk -F ' ' '{print $1, $3}' index-kx-ky-kz.txt > index-ky.txt
    awk -F ' ' '{print $1, $4}' index-kx-ky-kz.txt > index-kz.txt
    awk -F ' ' '{print $1}' index-kx-ky-kz.txt > index.txt
    awk -F ' ' '{print $2}' index-kx-ky-kz.txt > kx.tmp
    awk -F ' ' '{print $3}' index-kx-ky-kz.txt > ky.tmp
    awk -F ' ' '{print $4}' index-kx-ky-kz.txt > kz.tmp
    awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kx.tmp > kx.txt
    awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < ky.tmp > ky.txt
    awk -v var='0.0' '{ printf("%f\n", $1*('$BohrToAng')) }' < kz.tmp > kz.txt
    paste index.txt kx.txt ky.txt | tr -s ' ' > index-kx-ky.txt
    paste index.txt kx.txt ky.txt kz.txt | tr -s ' ' > index-kx-ky-kz.txt
    paste kx.txt ky.txt | tr -s ' ' > kx-ky.txt
    paste kx.txt ky.txt kz.txt | tr -s ' ' > kx-ky-kz.txt

    grep "Wavefunction = " ../../../3-Bands/$material.bands.WFSXR | tr -s ' ' | sort -k3 -n  > bandIndex-E.tmp
    awk -v var=$EFbands -F " " '{print $3 " " $7-var}' bandIndex-E.tmp > bandIndex-E-EF.tmp
    nl -w2 -s ' ' bandIndex-E-EF.tmp > lineNumber-bandIndex-E-EF.tmp
    firstPositiveValue=$(grep -oE '(^| )[0-9]+\.[0-9]+' lineNumber-bandIndex-E-EF.tmp | tr -d ' ' | head -n1)
    CBM1band=$(grep -F "$firstPositiveValue" lineNumber-bandIndex-E-EF.tmp | awk -F ' ' '{print $2}' | head -n1 | awk '{print $1;}')
    VBM1band=$(($CBM1band-1))

    readwfx -m -6.5 -M -1.5 -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > $material.CB1.NoEF.WFSXR
    readwfx -m -6.5 -M -1.5 -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > $material.VB1.NoEF.WFSXR

    #readwfx -m -5.5 -M -2.5 -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > $material.CB1.NoEF.WFSXR
    #readwfx -m -5.5 -M -2.5 -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > $material.VB1.NoEF.WFSXR

    grep "Wavefunction = *$CBM1band" $material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > CB1-2D-E-EF.tmp
    grep "Wavefunction = *$VBM1band" $material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF-var}' > VB1-2D-E-EF.tmp
    grep "Wavefunction = *$CBM1band" $material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > CB1-2D-E.tmp
    grep "Wavefunction = *$VBM1band" $material.bands.WFSXR | awk -v var=$EFbands -F " " '{print $NF}' > VB1-2D-E.tmp
    paste kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-CB1-2D.txt
    paste kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-kz-E-VB1-2D.txt
    paste index-kx-ky-kz.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-CB1-2D.txt
    paste index-kx-ky-kz.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-kz-E-VB1-2D.txt

    paste kx-ky.txt CB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-CB1-2D.txt
    paste kx-ky.txt VB1-2D-E-EF.tmp | tr -s ' ' > kx-ky-E-VB1-2D.txt
    paste index-kx-ky.txt CB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-CB1-2D.txt
    paste index-kx-ky.txt VB1-2D-E-EF.tmp | tr -s ' ' > index-kx-ky-E-VB1-2D.txt


    CBM1E2D="$(sort -n CB1-2D-E-EF.tmp | head -n1)"
    CBM1E2DnoEF="$(sort -n CB1-2D-E.tmp | head -n1)"
    CBM1state2D="$(grep -B20000 "Eigval (eV) = *$CBM1E2DnoEF" $material.bands.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)"
    CBM1k2D="$(grep -B20000 "Eigval (eV) = *$CBM1E2DnoEF" $material.bands.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)"

    VBM1E2D="$(sort -n VB1-2D-E-EF.tmp | tail -n1)"
    VBM1E2DnoEF="$(sort -n VB1-2D-E.tmp | tail -n1)"
    VBM1state2D="$(grep -B20000 "Eigval (eV) = *$VBM1E2DnoEF" $material.bands.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f1)"
    VBM1k2D="$(grep -B20000 "Eigval (eV) = *$VBM1E2DnoEF" $material.bands.WFSXR | tr -s ' ' | grep "k-point = " | tail -n1 | cut -d ' ' -f3,4,5,6 | cut -d ' ' -f2,3,4)"

    echo $CBM1band > CB1-band-number.tmp
    echo $VBM1band > VB1-band-number.tmp

    echo $CBM1state2D > CB1-state-number.tmp
    echo $VBM1state2D > VB1-state-number.tmp


    echo $CBM1k2D > CB1-k-value.tmp
    echo $VBM1k2D > VB1-k-value.tmp

    echo $CBM1E2DnoEF > CB1-E-value.tmp
    echo $VBM1E2DnoEF > VB1-E-value.tmp

    text="Band State k E"
    echo $text > text.tmp

    paste CB1-band-number.tmp CB1-state-number.tmp CB1-k-value.tmp CB1-E-value.tmp > CB1-row.tmp
    paste VB1-band-number.tmp VB1-state-number.tmp VB1-k-value.tmp VB1-E-value.tmp > VB1-row.tmp


    cat text.tmp CB1-row.tmp VB1-row.tmp > CBM-VBM-summary-bands.tmp
    tr -s ' ' < CBM-VBM-summary-bands.tmp | sed 's/  */ /g' > CBM-VBM-summary-bands.txt

    BohrToAng=1.8897
    awk -v factor="$BohrToAng" -v EF="$EF" 'NR == 1 {print $0} NR > 1 && NR < 4 {printf "%d %d %.6f %.6f %.6f %.6f\n", $1, $2, $3 * factor, $4 * factor, $5 * factor, $6 - EF} NR >= 4 {print $0}' CBM-VBM-summary-bands.txt > temp.txt && mv temp.txt CBM-VBM-summary-bands.txt

    cp CBM-VBM-summary-bands.txt ../../../Results/ExtractedParameters


    mv index-kx-ky-E-CB1-2D.txt index-kx-ky-E-CB1.txt
    mv index-kx-ky-E-VB1-2D.txt index-kx-ky-E-VB1.txt

    mv index-kx-ky-kz-E-CB1-2D.txt  index-kx-ky-kz-E-CB1.txt
    mv index-kx-ky-kz-E-VB1-2D.txt index-kx-ky-kz-E-VB1.txt

    mv kx-ky-E-CB1-2D.txt kx-ky-E-CB1.txt
    mv kx-ky-E-VB1-2D.txt kx-ky-E-VB1.txt

    mv kx-ky-kz-E-CB1-2D.txt kx-ky-kz-E-CB1.txt
    mv kx-ky-kz-E-VB1-2D.txt kx-ky-kz-E-VB1.txt

    rm *.tmp

    ###

    cd ..

    echo $CBM1band $CBM1state2D $VBM1band $VBM1state2D


    cd 2D
    denchar-serial -w $CBM1band -k $CBM1state2D $material-DENCHAR.fdf > /dev/null 2>&1 &
    denchar-serial -w $VBM1band -k $VBM1state2D $material-DENCHAR.fdf > /dev/null 2>&1 &
    
#    denchar-serial -w $CBM1band $material-DENCHAR.fdf > /dev/null 2>&1 &
#    denchar-serial -w $VBM1band $material-DENCHAR.fdf > /dev/null 2>&1 &
    cd ..


    cd 3D
    denchar-serial -w $CBM1band -k $CBM1state2D $material-DENCHAR.fdf > /dev/null 2>&1 &
    denchar-serial -w $VBM1band -k $VBM1state2D $material-DENCHAR.fdf > /dev/null 2>&1 &
    cd ..

    wait

    cd bands

    CBM1=$(<../../../Results/ExtractedParameters/CBM1.out)
    VBM1=$(<../../../Results/ExtractedParameters/VBM1.out)
    dCB1=$(<../../../Results/ExtractedParameters/dCB1.out)
    dVB1=$(<../../../Results/ExtractedParameters/dVB1.out)

    dE=0.05 # Broadening

    rm -rf case1-extrema
    rm -rf case2-variable
    rm -rf case3-all
    ################## RECALL THIS DOESN"T HAVE THE FERMI LEVEL!!!!
    # Case 1: at CBM and VBM
    mkdir -p case1-extrema/CB
    cp ../../../FILES/LDOS.fdf case1-extrema/CB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf  case1-extrema/CB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case1-extrema/CB/$material.STRUCT_IN
    CBM1withEF=$(echo "$CBM1+$EF" | bc)
    E2_CB1_case1=$(echo "$CBM1withEF + $dE " | bc)
    E1_CB1_case1=$(echo "$CBM1withEF - $dE" | bc)
    sed -i "s/XXXX/$E1_CB1_case1/g; s/YYYY/$E2_CB1_case1/g" case1-extrema/CB/LDOS.fdf
    readwfx -m $E1_CB1_case1 -M $E2_CB1_case1 -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > $material.CB1.NoEF_case1.WFSXR

    mkdir -p case1-extrema/VB
    cp ../../../FILES/LDOS.fdf case1-extrema/VB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf case1-extrema/VB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case1-extrema/VB/$material.STRUCT_IN
    VBM1withEF=$(echo "$VBM1+$EF" | bc)
    E2_VB1_case1=$(echo "$VBM1withEF + $dE" | bc)
    E1_VB1_case1=$(echo "$VBM1withEF - $dE" | bc)
    sed -i "s/XXXX/$E1_VB1_case1/g; s/YYYY/$E2_VB1_case1/g" case1-extrema/VB/LDOS.fdf
    readwfx -m $E1_VB1_case1 -M $E2_VB1_case1 -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > $material.VB1.NoEF_case1.WFSXR

    # Case 2: up to half the band
    mkdir -p case2-variable/CB
    cp ../../../FILES/LDOS.fdf case2-variable/CB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf case2-variable/CB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case2-variable/CB/$material.STRUCT_IN
    dExCB1=$(echo "0.5*$dCB1" | bc)
    E2_CB1_case2=$(echo "$CBM1withEF + $dE + $dExCB1" | bc)
    E1_CB1_case2=$(echo "$CBM1withEF - $dE" | bc)
    sed -i "s/XXXX/$E1_CB1_case2/g; s/YYYY/$E2_CB1_case2/g" case2-variable/CB/LDOS.fdf
    readwfx -m $E1_CB1_case2 -M $E2_CB1_case2 -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > $material.CB1.NoEF_case2.WFSXR

    mkdir -p case2-variable/VB
    cp ../../../FILES/LDOS.fdf case2-variable/VB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf case2-variable/VB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case2-variable/VB/$material.STRUCT_IN
    dExVB1=$(echo "0.5*$dVB1" | bc)
    E2_VB1_case2=$(echo "$VBM1withEF + $dE + $dExVB1" | bc)
    E1_VB1_case2=$(echo "$VBM1withEF - $dE - $dExVB1" | bc)
    sed -i "s/XXXX/$E1_VB1_case2/g; s/YYYY/$E2_VB1_case2/g" case2-variable/VB/LDOS.fdf
    readwfx -m $E1_VB1_case2 -M $E2_VB1_case2 -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > $material.VB1.NoEF_case2.WFSXR

    # Case 3: the whole band width
    mkdir -p case3-all/CB
    cp ../../../FILES/LDOS.fdf case3-all/CB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf case3-all/CB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case3-all/CB/$material.STRUCT_IN
    E2_CB1_case3=$(echo "$CBM1withEF + $dE + $dCB1" | bc)
    E1_CB1_case3=$(echo "$CBM1withEF - $dE" | bc)
    sed -i "s/XXXX/$E1_CB1_case3/g; s/YYYY/$E2_CB1_case3/g" case3-all/CB/LDOS.fdf
    readwfx -m $E1_CB1_case3 -M $E2_CB1_case3 -b $CBM1band -B $CBM1band ../../../3-Bands/$material.bands.WFSX > $material.CB1.NoEF_case3.WFSXR

    mkdir -p case3-all/VB
    cp ../../../FILES/LDOS.fdf case3-all/VB
    cp ../../../1-Relaxation/*.CG ../../../1-Relaxation/*.DM ../../../1-Relaxation/$material.STRUCT_OUT ../../../1-Relaxation/*.psf case3-all/VB
    cp ../../../1-Relaxation/$material.STRUCT_OUT case3-all/VB/$material.STRUCT_IN
    E2_VB1_case3=$(echo "$VBM1withEF + $dE" | bc)
    E1_VB1_case3=$(echo "$VBM1withEF - $dE - $dVB1" | bc)
    sed -i "s/XXXX/$E1_VB1_case3/g; s/YYYY/$E2_VB1_case3/g" case3-all/VB/LDOS.fdf
    readwfx -m $E1_VB1_case3 -M $E2_VB1_case3 -b $VBM1band -B $VBM1band ../../../3-Bands/$material.bands.WFSX > $material.VB1.NoEF_case3.WFSXR


    cd case1-extrema/CB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &
    cd ..
    cd VB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &

    cd ../..

    cd case2-variable/CB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &
    cd ..
    cd VB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &

    cd ../..
    
    cd case3-all/CB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &
    cd ..
    cd VB/
    siesta LDOS.fdf > siesta.out > /dev/null 2>&1 &

    wait

    cd ../..

thisDir=$(pwd)
# List of directories relative to the current directory to loop through
directories=(
    "${thisDir}/case1-extrema/CB"
    "${thisDir}/case1-extrema/VB"
    "${thisDir}/case2-variable/CB"
    "${thisDir}/case2-variable/VB"
    "${thisDir}/case3-all/CB"
    "${thisDir}/case3-all/VB"
)


# Loop through each directory
for directory in "${directories[@]}"; do
    cd "$directory"

    # Create Grids directory if it doesn't exist
    mkdir -p Grids
    cd Grids

    cp ~/SIESTA/Utilities/Python/cube2xyz.py .
    cp ../$material.STRUCT_OUT .
    cp ../$material.LDOS .

    g2c_ng -n 1 -s $material.STRUCT_OUT -g $material.LDOS
    mv Grid.cube LDOS.cube

    python cube2xyz.py -f LDOS.cube -o LDOS.xyz -A &


done

wait

cd ../../..


cd ../../../../
