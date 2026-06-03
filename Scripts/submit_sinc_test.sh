#!/bin/sh
#BSUB -J gen_curves_sinc_test
#BSUB -o logs/gen_curves_sinc_test.out
#BSUB -e logs/gen_curves_sinc_test.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00

# Create a logs directory to keep the workspace clean
mkdir -p logs

# Load necessary modules for DTU HPC
module load cuda/11.8

echo "Job starting on $(hostname)"

# Using 'uv run' automatically handles the virtual environment for you!
uv run generate_curve.py \
    --name "sweep_sinc_test" \
    --params_x "40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0" \
    --order_N 1 5 10 20 35 50 100 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 700 700 1 \
    --nx 5000 \
    --ny 1
