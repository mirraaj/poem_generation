#!/bin/bash
#SBATCH --job-name=test2
#SBATCH --output=output_check_cuda2.txt
#SBATCH --error=error_check_cuda2.txt

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Clean modules
module purge
module load DL-CondaPy

# Activate env
source /home/apps/DL/DL-CondaPy3.7/bin/activate poem

echo "Running on: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF