#!/bin/sh
#BSUB -J gen_curves_num_layers[1-4]
#BSUB -o logs/gen_curves_num_layers_%I.out
#BSUB -e logs/gen_curves_num_layers_%I.err
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

echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

# Using 'uv run' automatically handles the virtual environment for you!
case ${LSB_JOBINDEX} in
    1)
        uv run generate_curve.py \
            --name "sweep_num_layers_no_subpixel_100nm" \
            --params_x "50,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500\
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --no_subpixel
        ;;
    2)
        uv run generate_curve.py \
            --name "sweep_num_layers_100nm" \
            --params_x "50,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 
        ;;
    3)
        uv run generate_curve.py \
            --name "sweep_num_layers_no_subpixel_1000nm" \
            --params_x "500,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --no_subpixel
        ;;
    4)
        uv run generate_curve.py \
            --name "sweep_num_layers_1000nm" \
            --params_x "500,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 
        ;;
esac
