#!/bin/bash
#SBATCH --job-name=audio-unet
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=48:00:00
#SBATCH --output=ldm-%J.log
#SBATCH --gres=gpu:tesla_v100:2
#SBATCH --mem=100G
#SBATCH --exclude=ai03,ai04

# logs/2022-10-01T15-31-09_audio2img

# CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml --resume logs/2022-09-07T06-17-56_lsun_churches-ldm-kl-8 -t --gpus 0,

# CUDA_VISIBLE_DEVICES=0, python main.py --base configs/latent-diffusion/landscape_wo_audio.yaml -t --gpus 0, --scale_lr False


CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml --resume logs/2022-09-28T20-39-59_lsun_churches-ldm-kl-8 -t --gpus 0,1 --scale_lr False


# CUDA_VISIBLE_DEVICES=0,1 python main.py --base configs/latent-diffusion/cifar-ldm-kl-8.yaml  -t --gpus 0,1 --scale_lr False

# CUDA_VISIBLE_DEVICES=0,  python main.py --base configs/latent-diffusion/lsun_churches-ldm-kl-8.yaml --resume logs/2022-09-28T20-39-59_lsun_churches-ldm-kl-8 -t --gpus 0, --scale_lr False
