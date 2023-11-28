#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=train-ny-m-a100
#SBATCH --partition=gpu
##SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

module load anaconda3/2022.05 cuda/11.8
source activate /work/re-blocking/.conda/envs/re-blocking_env

python3 /work/re-blocking/pytorch-CycleGAN-and-pix2pix-master/train.py --dataroot /work/re-blocking/data/ny-manhattan --checkpoints_dir /work/re-blocking/checkpoints --name ny-manhattan-100-60-gpu --model pix2pix --direction AtoB --n_epochs 100 --batch_size 60