#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --job-name=test-ny-b-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
##SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

module load anaconda3/2022.05 cuda/11.8
source activate /work/re-blocking/.conda/envs/re-blocking_env

python3 /work/re-blocking/pytorch-CycleGAN-and-pix2pix-master/test.py --dataroot /work/re-blocking/data/ny-brooklyn --name ny-manhattan-100-60 --model pix2pix