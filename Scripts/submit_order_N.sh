#!/bin/sh
#BSUB -J gen_curves_order_N
#BSUB -o logs/gen_curves_%J.out
#BSUB -e logs/gen_curves_%J.err
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

echo "Job starting on $(hostname)"

# Using 'uv run' automatically handles the virtual environment for you!
uv run generate_curve.py \
    --name "sweep_num_layers" \
    --params_x "40,0" \
    --order_N 1, 5, 10, 20, 35, 50, 100 \
    --num_layers 10 \
    --wavelengths 300 1100 1601 \
    --nx 5000 \
    --ny 1
