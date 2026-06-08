#!/bin/sh
#BSUB -J generate_dataset_cpu[1-3]
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_dataset_cpu_%J_%I.out
#BSUB -e logs/generate_dataset_cpu_%J_%I.err

export PYTHONUNBUFFERED=1
mkdir -p logs
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

export CUDA_VISIBLE_DEVICES=""

case ${LSB_JOBINDEX} in
    1)
        echo "======================================"
        echo "Running Material: Si"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material Si \
            --seed 42 \
            --n_jobs 24
        ;;
    2)
        echo "======================================"
        echo "Running Material: TiO2"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material TiO2 \
            --seed 43 \
            --n_jobs 24
        ;;
    3)
        echo "======================================"
        echo "Running Material: Si3N4"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material Si3N4 \
            --seed 44 \
            --n_jobs 24
        ;;
esac

echo "Task completed!"
