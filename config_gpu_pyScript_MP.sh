#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --tasks-per-node=20
#SBATCH --job-name=map
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --time=6:30:00

# Clear the module environment
module purge
# Load the Python version that has been used to construct the virtual environment
# we are using below
module load Python/3.10.8-GCCcore-12.2.0
module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.
module load CUDA/11.7.0
module load cuDNN
module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0

# Activate the virtual environment
source ~/virtual_env/thesis/bin/activate

python3 mapping_script_MP.py $1
