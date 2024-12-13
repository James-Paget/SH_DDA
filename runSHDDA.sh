#!/bin/bash
#SBATCH --job-name=runSHDDA
#SBATCH --partition=teach_cpu
#SBATCH --account=PHYS033184
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --time=0:5:0
#SBATCH --mem=100M

# Load modules
module add languages/python/3.12.3
module load languages/Intel-OneAPI/2024.0.2
cd $SLURM_SUBMIT_DIR

# Run Program
python DipolesMulti2024Eigen.py SingleLaguerre1
