#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --job-name=train-comb-80k-p2p-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

module load anaconda3/2022.05 cuda/11.8
source activate /home/l.schrage/.conda/envs/re-blocking_env

python3 /work/re-blocking/pytorch-CycleGAN-and-pix2pix/train.py --dataroot /work/re-blocking/data/all-cities --checkpoints_dir /work/re-blocking/checkpoints --name combined-p2p-500-150-v100 --model pix2pix --direction AtoB --save_epoch_freq 1 --continue_train --epoch_count 144 --n_epochs 500 --batch_size 150