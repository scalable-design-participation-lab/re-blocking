#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=copy-files
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

# Change to your target directory
cd /work/re-blocking/data

# Perform copy operation
cp -a all-cities/ all-cities-20k/