#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=2
#SBATCH --partition ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_t4:2
#SBATCH --mem-per-cpu=25000
#SBATCH --job-name="ldm_churches"
#SBATCH --output=./log_%j.txt
#SBATCH --time=2:00:00

echo $CUDA_VISIBLE_DEVICES

CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml -t --gpus 0,1 >> train.log