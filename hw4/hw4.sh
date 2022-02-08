#!/bin/bash
#-------------------------------------------------------------------------------
#   SBATCH CONFIGP
#-------------------------------------------------------------------------------
## resources
#SBATCH --partition hpc3
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=hzny2-hpc-hw4
#SBATCH --output=hpc-hw4-%j.out
#-------------------------------------------------------------------------------
## notifications
#SBATCH --mail-user hzny2@umsystem.edu  # email address for notifications
#SBATCH --mail-type FAIL            # which type of notifications to send
#-------------------------------------------------------------------------------

# Load your modules here:
module load openmpi/4.0.5 gcc/10.2.0 opencv/3.2.0-openmpi gcc/10.2.0
module list

# Science goes here:
srun ./homework4 ../Astronaught.png 4 256 output

echo "### Ending at: $(date) ###"