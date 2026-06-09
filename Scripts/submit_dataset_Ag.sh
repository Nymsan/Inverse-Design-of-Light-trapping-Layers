#!/bin/sh
#BSUB -J generate_dataset_Ag[1-3]
#BSUB -q hpc
#BSUB -n 24
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_dataset_Ag_%J_%I.out
#BSUB -e logs/generate_dataset_Ag_%J_%I.err

export PYTHONUNBUFFERED=1
mkdir -p logs
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

export CUDA_VISIBLE_DEVICES=""

case ${LSB_JOBINDEX} in
    1)
        echo "======================================"
        echo "Running Material: Si with Ag reflector"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material Si \
            --reflector_type Ag \
            --seed 142 \
            --n_jobs 24
        ;;
    2)
        echo "======================================"
        echo "Running Material: TiO2 with Ag reflector"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material TiO2 \
            --reflector_type Ag \
            --seed 143 \
            --n_jobs 24
        ;;
    3)
        echo "======================================"
        echo "Running Material: Si3N4 with Ag reflector"
        echo "======================================"
        uv run generate_dataset.py \
            --num_samples 10000 \
            --batch_size 100 \
            --order_N 15 \
            --height_per_layer 5.0 \
            --grating_period 1000.0 \
            --nx 5000 \
            --grating_material Si3N4 \
            --reflector_type Ag \
            --seed 144 \
            --n_jobs 24
        ;;
esac

echo "Task completed!"
