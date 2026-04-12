#!/bin/bash
#SBATCH --job-name=ppopeft
#SBATCH --output=output_ppo_peft_modell.txt
#SBATCH --error=error_ppo_peft_model.txt

#SBATCH --partition=gpu        # GPU partition
#SBATCH --gres=gpu:1           # request 1 GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

module load DL-CondaPy
module load cuda/11.6
module load cmake/3.21.7
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source /home/apps/DL/DL-CondaPy3.7/bin/activate gpuenv

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

echo "Running on: $(hostname)"
nvidia-smi

python trainRL.py