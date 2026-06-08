#!/bin/sh
#BSUB -J generate_dataset_cpu
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_dataset_cpu_%J.out
#BSUB -e logs/generate_dataset_cpu_%J.err

mkdir -p logs
export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname)"

export CUDA_VISIBLE_DEVICES=""

uv run generate_dataset_cpu_parallel.py \
    --num_samples 5000 \
    --batch_size 100 \
    --order_N 10 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material Si \
    --n_jobs 24
