#!/bin/bash

# Job name
#SBATCH --job-name=POSTPROCESSING-SIESTA-materialVAR-typeVAR-functionalVAR-structureVAR-supercellVAR-strainVAR%

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


timeout 71h ./postprocessing.sh > output-error.txt 2>&1

if [[ $? -eq 124 ]]; then
  cd ${homedir}
  echo 'Timeout ended'
  sbatch ${homedir}/in-timeout-system.sh
fi
