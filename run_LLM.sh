#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

#SBATCH --partition=gpu        # GPU partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load DL-CondaPy
module load compiler/cuda/10.1

conda activate poem
