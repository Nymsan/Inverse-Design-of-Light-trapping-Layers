#!/bin/sh
#BSUB -J generate_dataset_cpu
#BSUB -q hpc
#BSUB -n 64
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_dataset_cpu_%J.out
#BSUB -e logs/generate_dataset_cpu_%J.err

export PYTHONUNBUFFERED=1
mkdir -p logs
echo "Job starting on $(hostname)"

export CUDA_VISIBLE_DEVICES=""

echo "======================================"
echo "Running Material: Si"
echo "======================================"
uv run generate_dataset_cpu_parallel.py \
    --num_samples 10000 \
    --batch_size 100 \
    --order_N 10 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material Si \
    --seed 42 \
    --n_jobs 64

echo "======================================"
echo "Running Material: TiO2"
echo "======================================"
uv run generate_dataset_cpu_parallel.py \
    --num_samples 10000 \
    --batch_size 100 \
    --order_N 10 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material TiO2 \
    --seed 43 \
    --n_jobs 64

echo "======================================"
echo "Running Material: Si3N4"
echo "======================================"
uv run generate_dataset_cpu_parallel.py \
    --num_samples 10000 \
    --batch_size 100 \
    --order_N 10 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material Si3N4 \
    --seed 44 \
    --n_jobs 64

echo "All materials completed successfully!"
