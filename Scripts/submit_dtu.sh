#!/bin/sh
#BSUB -J gen_curves
#BSUB -o logs/gen_curves.out
#BSUB -e logs/gen_curves.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00

# Note: Adjust the walltime (-W) above based on how many combinations you are running

# Create a logs directory to keep the workspace clean
mkdir -p logs

# Load necessary modules for DTU HPC
module load cuda/11.8

# Fix for PyTorch 2.5.1 cu118 missing libnccl.so.2 and libcudnn.so.9 on compute nodes

export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

# Using 'uv run' automatically handles the virtual environment for you!
uv run generate_curve.py \
    #Some arguments.This is an example.