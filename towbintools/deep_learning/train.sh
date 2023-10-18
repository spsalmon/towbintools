#!/bin/bash

#SBATCH -J train
#SBATCH -o train.out
#SBATCH -c 32
#SBATCH -t 96:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:2
#SBATCH --wait

source ~/env_directory/towbintools/bin/activate
python3 train_network.py
deactivate