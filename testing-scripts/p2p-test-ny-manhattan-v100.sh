#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --job-name=test-ny-m-v100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
##SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.schrage@northeastern.edu

module load anaconda3/2022.05 cuda/11.8
source activate /home/l.schrage/.conda/envs/re-blocking_env

python pytorch-CycleGAN-and-pix2pix/test.py --dataroot ./data/ny-brooklyn --checkpoints_dir checkpoints --name ma-boston-p2p-200-150-v100 --model pix2pix
#!./scripts/test_pix2pix.sh
python test.py --dataroot ./datasets/brooklynn_combined --name ma-boston-p2p-200-150-v100 --model pix2pix --direction BtoA