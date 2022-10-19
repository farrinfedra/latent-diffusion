#!/bin/bash

#SBATCH --job-name=sample_unet
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=48:00:00
#SBATCH --output=ldm-%J.log
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --mem=100G


CUDA_VISIBLE_DEVICES=0, python scripts/sample_diffusion.py -r /kuacc/users/bbiner21/Github/latent-diffusion/logs/2022-09-27T15-03-59_cifar-ldm-kl-8/checkpoints/last.ckpt -l sample_logs/test1 -n 10000 --batch_size 50 -c 500 -e 1.0
