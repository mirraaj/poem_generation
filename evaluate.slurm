#!/bin/bash
#SBATCH --job-name=evaluateJob
#SBATCH --output=output_evaluate.txt
#SBATCH --error=error_evaluate.txt

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

python evaluate_poem_model.py \
  --model_path ./ppo_model_finetuned_model_peft \
  --data_path ./format_data/topics \
  --trainer_state_path "./poem_model_peft_all_data/checkpoint-6000/trainer_state.json" \
  --ppo_stats_path "./ppo_peft_stats.json"