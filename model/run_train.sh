#!/bin/bash
#SBATCH --job-name=abl-sr
#SBATCH --account=<allocation>
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=97000
#SBATCH --time=10:00:00
#SBATCH --output=%x-%j.out

python train.py --config config.yaml
