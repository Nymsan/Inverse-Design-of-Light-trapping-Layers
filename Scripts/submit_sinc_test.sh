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

# Fix for PyTorch 2.5.1 cu118 missing libnccl.so.2 and libcudnn.so.9 on compute nodes
export LD_LIBRARY_PATH="../.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:../.venv/lib/python3.13/site-packages/nvidia/nccl/lib:../.venv/lib/python3.13/site-packages/nvidia/cublas/lib:../.venv/lib/python3.13/site-packages/nvidia/cusparse/lib:../.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

echo "Job starting on $(hostname)"

echo "Running WITH subpixel smoothing..."
uv run generate_curve.py \
    --name "sweep_sinc_test_subpixel" \
    --params_x "40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0" \
    --order_N 1 5 10 20 35 50 100 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 700 700 1 \
    --nx 5000 \
    --ny 1

echo "Running WITHOUT subpixel smoothing..."
uv run generate_curve.py \
    --name "sweep_sinc_test_no_subpixel" \
    --no_subpixel \
    --params_x "40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0;40,0" \
    --order_N 1 5 10 20 35 50 100 \
    --num_layers 1 2 3 5 10 15 20 25 50 100 150 250 251 \
    --wavelengths 700 700 1 \
    --nx 5000 \
    --ny 1
