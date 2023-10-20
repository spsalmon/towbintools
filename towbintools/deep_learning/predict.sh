#!/bin/bash

#SBATCH -J pred
#SBATCH -o pred.out
#SBATCH -c 32
#SBATCH -t 96:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --wait

source ~/env_directory/towbintools/bin/activate
python3 predict.py
deactivate