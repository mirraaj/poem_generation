module load DL-CondaPy
module load cuda/11.6
module load cmake/3.21.7
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

source /home/apps/DL/DL-CondaPy3.7/bin/activate gpuenv

unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
