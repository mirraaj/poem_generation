#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --output=output_check_cuda.txt
#SBATCH --error=error_check_cuda.txt

#SBATCH --partition=gpu        # GPU partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load DL-CondaPy
module load cuda/11.6
module load cmake/3.21.7
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source /home/apps/DL/DL-CondaPy3.7/bin/activate gpuenv

echo "Running on: $(hostname)"
nvidia-smi

python check_cuda.py