#!/bin/sh
#BSUB -J generate_3d_dataset[1-3]
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_3d_dataset_%J_%I.out
#BSUB -e logs/generate_3d_dataset_%J_%I.err

mkdir -p logs

export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

case ${LSB_JOBINDEX} in
    1)
        echo "======================================"
        echo "Running Material: Si"
        echo "======================================"
        uv run generate_3d_dataset.py \
            --num_samples 5000 \
            --batch_size 100 \
            --order_N 10 \
            --order_N_y 10 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --grating_period_y 1000.0 \
            --nx 500 \
            --ny 500 \
            --grating_material Si \
            --seed 42 \
            --n_jobs 24
        ;;
    2)
        echo "======================================"
        echo "Running Material: TiO2"
        echo "======================================"
        uv run generate_3d_dataset.py \
            --num_samples 5000 \
            --batch_size 100 \
            --order_N 10 \
            --order_N_y 10 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --grating_period_y 1000.0 \
            --nx 500 \
            --ny 500 \
            --grating_material TiO2 \
            --seed 43 \
            --n_jobs 24
        ;;
    3)
        echo "======================================"
        echo "Running Material: Si3N4"
        echo "======================================"
        uv run generate_3d_dataset.py \
            --num_samples 5000 \
            --batch_size 100 \
            --order_N 10 \
            --order_N_y 10 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --grating_period_y 1000.0 \
            --nx 500 \
            --ny 500 \
            --grating_material Si3N4 \
            --seed 44 \
            --n_jobs 24
        ;;
esac

echo "Task completed!"
