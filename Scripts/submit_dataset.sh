#!/bin/sh
#BSUB -J generate_lhs_dataset
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=12GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_lhs_dataset.out
#BSUB -e logs/generate_lhs_dataset.err

mkdir -p logs
module load cuda/11.8
echo "Job starting on $(hostname)"

# Run LHS Dataset Generator
# Adjust --num_samples and --batch_size as needed
uv run generate_dataset.py \
    --num_samples 5000 \
    --batch_size 100 \
    --order_N 10 \
    --nm_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material Si
