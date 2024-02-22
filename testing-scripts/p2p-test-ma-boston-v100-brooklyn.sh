#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --job-name=test-ma-b-gpu
#SBATCH --partition=gpu
##SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

module load anaconda3/2022.05 cuda/11.8
source activate /home/l.schrage/.conda/envs/re-blocking_env

python3 /work/re-blocking/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /work/re-blocking/data/ny-brooklyn --checkpoints_dir /work/re-blocking/checkpoints --results_dir /work/re-blocking/results --name ma-boston-p2p-200-150-v100 --model pix2pix --num_test 1000