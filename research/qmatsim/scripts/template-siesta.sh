#!/bin/bash

# Job name
#SBATCH --job-name=SIESTA-materialVAR-typeVAR-functionalVAR-socVAR-structureVAR-supercellVAR-strainVAR%

#SBATCH --account=accountVAR

#SBATCH --partition=paritionVAR

#SBATCH --qos=QoSVAR

# TIME BLOCK

# Number of nodes, taks, and CPUs
#SBATCH --nodes=nodesVAR
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=cpusVAR

# Output data
#SBATCH --output=co-%x-%j.out
#SBATCH --error=eco-%x-%j.err

# Feedback for job 
#SBATCH --mail-type=FAIL,REQUEUE,END
#SBATCH --mail-user=meshal@berkeley.edu

# Specify the job is eligible to requeue
#SBATCH --requeue

# Append the standard output/error of the requeued job to the same standard out/error files from the previously terminated job
#SBATCH --open-mode=append

# Enable globstar and aliases expansion


shopt -s globstar
shopt -s expand_aliases

find . -name 'co-*' -mmin +0.1 -exec rm {} \;
find . -name 'eco-*' -mmin +0.1 -exec rm {} \;

rm -f ./*~
rm -f ./*#

conda activate siesta-env

# Variables
systemLabel=materialVAR
SOCCond=socVAR
RelaxCond=RelaxVAR
StaticCond=StaticVAR
BandsCond=BandsVAR
LDOSCond=LDOSVAR
homedir=./materials/materialVAR/typeVAR/structureVAR/supercellVAR/strainVAR/

if [ "$SOCCond" = "T" ]; then
    scratchFolder=/Results/materialVAR/typeVAR/functionalVAR-SOC/structureVAR/supercellVAR/strainVAR/
    potdir=./pseudopotentials/functionalVAR-SOC/
else
    scratchFolder=/Results/materialVAR/typeVAR/functionalVAR/structureVAR/supercellVAR/strainVAR/
    potdir=./pseudopotentials/functionalVAR/
fi

# Scratch folder structure
mkdir -p ${scratchFolder}/1-Relaxation ${scratchFolder}/2-Static ${scratchFolder}/3-Bands ${scratchFolder}/LDOS ${scratchFolder}/FILES


cd ${scratchFolder}

cp ${homedir}/* FILES
sed -i 's/-vector/cell-vector/g' FILES/*.fdf

######################## RELAXATION ########################
if [ "$RelaxCond" = "T" ]; then
    cd 1-Relaxation

    find . -name 'co-*' -mmin +0.1 -exec rm {} \;
    find . -name 'eco-*' -mmin +0.1 -exec rm {} \;
    find . -name 'log-*' -mmin +0.1 -exec rm {} \;
    find . -type f ! \( -name 'materialVAR.CG' -o -name 'materialVAR.DM' -o -name 'materialVAR.STRUCT_OUT' -o -name "*.psf" -o -name "*.psml" -o -name "*.sh" -o -name "materialVAR.STRUCT_IN" -o -name "1-Relaxation.fdf" \) -delete

    cp ../FILES/* .
    
    cp materialVAR.STRUCT_OUT materialVAR.STRUCT_IN

    siesta 1-Relaxation.fdf > siesta.out

    echo "RELAXATION RUN DONE!" > completed.relaxCompleted

    cd ..
fi

wait

######################## STATIC ########################
if [ "$StaticCond" = "T" ]; then
    cd 2-Static

    find . -name 'co-*' -mmin +0.1 -exec rm {} \;
    find . -name 'eco-*' -mmin +0.1 -exec rm {} \;
    find . -name 'log-*' -mmin +0.1 -exec rm {} \;    

    cp ../FILES/* .
    
    cp ../1-Relaxation/materialVAR.STRUCT_OUT ../1-Relaxation/materialVAR.STRUCT_NEXT_ITER ../1-Relaxation/materialVAR.CG ../1-Relaxation/materialVAR.DM .
    cp materialVAR.STRUCT_OUT materialVAR.STRUCT_IN
    
    siesta 2-Static.fdf > siesta.out

    echo "STATIC RUN DONE!" > completed.staticCompleted

    cd ..
fi

######################## BANDS ########################
if [ "$BandsCond" = "T" ]; then
    cd 3-Bands

    find . -name 'co-*' -mmin +0.1 -exec rm {} \;
    find . -name 'eco-*' -mmin +0.1 -exec rm {} \;
    find . -name 'log-*' -mmin +0.1 -exec rm {} \;

    cp ../FILES/* .

    cp ../1-Relaxation/materialVAR.STRUCT_OUT ../1-Relaxation/materialVAR.STRUCT_NEXT_ITER ../1-Relaxation/materialVAR.CG ../1-Relaxation/materialVAR.DM .
    cp materialVAR.STRUCT_OUT materialVAR.STRUCT_IN
    

    siesta 3-Bands.fdf > siesta.out
    
    echo "BANDS RUN DONE!" > completed.bandsCompleted

    cd ..
fi

######################## LDOS ########################
if [ "$LDOSCond" = "T" ]; then
    cd LDOS

    find . -name 'co-*' -mmin +0.1 -exec rm {} \;
    find . -name 'eco-*' -mmin +0.1 -exec rm {} \;
    find . -name 'log-*' -mmin +0.1 -exec rm {} \;

    cp ../FILES/* .
    
    cp ../1-Relaxation/materialVAR.STRUCT_OUT ../1-Relaxation/materialVAR.STRUCT_NEXT_ITER ../1-Relaxation/materialVAR.CG ../2-Static/materialVAR.DM .
    cp materialVAR.STRUCT_OUT materialVAR.STRUCT_IN
    
    # If needed, add additional commands here

    cd ..
fi
