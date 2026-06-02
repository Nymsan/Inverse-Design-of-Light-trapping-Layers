#!/bin/sh
#BSUB -J gen_curves
#BSUB -o logs/gen_curves_%J.out
#BSUB -e logs/gen_curves_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 1
#BSUB -R "rusage[mem=10GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00

# Note: Adjust the walltime (-W) above based on how many combinations you are running

# Create a logs directory to keep the workspace clean
mkdir -p logs

# Load necessary modules for DTU HPC
module load cuda/11.8

echo "Job starting on $(hostname)"

# ==========================================
# Example 1: 2D Convergence testing 
# Tests 3 different diffraction orders across 2 different layer thicknesses (6 total combinations)
# ==========================================

# Using 'uv run' automatically handles the virtual environment for you!
uv run Scripts/generate_curve.py \
    --name "sweep_dtu_run" \
    --params_x "40,0" \
    --order_N 1 5 10 20 35 50 100 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 300 1100 1601 \
    --nx 5000 \
    --ny 1