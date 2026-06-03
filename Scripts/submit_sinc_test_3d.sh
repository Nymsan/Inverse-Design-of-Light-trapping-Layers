#!/bin/sh
#BSUB -J gen_curves_sinc_test_3d
#BSUB -o logs/gen_curves_sinc_test_3d.out
#BSUB -e logs/gen_curves_sinc_test_3d.err
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

echo "Running WITH subpixel smoothing..."
uv run generate_curve.py \
    --name "sweep_sinc_test_3d_subpixel" \
    --params_x "10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0" \
    --params_y "10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0" \
    --order_N 1 3 5 8 10 12 15 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 700 700 1 \
    --nx 500 \
    --ny 500

echo "Running WITHOUT subpixel smoothing..."
uv run generate_curve.py \
    --name "sweep_sinc_test_3d_no_subpixel" \
    --no_subpixel \
    --params_x "10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0" \
    --params_y "10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0;10,0" \
    --order_N 1 3 5 8 10 12 15 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 700 700 1 \
    --nx 500 \
    --ny 500
