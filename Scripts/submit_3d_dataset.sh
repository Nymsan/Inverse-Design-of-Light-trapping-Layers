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
