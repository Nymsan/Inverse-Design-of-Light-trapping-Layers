#!/bin/sh
#BSUB -J generate_3d_dataset
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_3d_dataset.out
#BSUB -e logs/generate_3d_dataset.err

mkdir -p logs
module load cuda/11.8

# Fix for PyTorch 2.5.1 cu118 missing shared libs on compute nodes
export LD_LIBRARY_PATH="../.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:../.venv/lib/python3.13/site-packages/nvidia/nccl/lib:../.venv/lib/python3.13/site-packages/nvidia/cublas/lib:../.venv/lib/python3.13/site-packages/nvidia/cusparse/lib:../.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname)"

# Run 3D Dataset Generator
uv run generate_3d_dataset.py \
    --num_samples 5000 \
    --batch_size 100 \
    --order_N 5 \
    --order_N_y 5 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --grating_period_y 1000.0 \
    --nx 500 \
    --ny 500 \
    --grating_material TiO2
