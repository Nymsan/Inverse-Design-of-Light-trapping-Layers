#!/bin/sh
#BSUB -J gen_curves_num_layers[1-4]
#BSUB -o logs/gen_curves_num_layers_%I.out
#BSUB -e logs/gen_curves_num_layers_%I.err
#BSUB -q hpc
#BSUB -n 16
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00

mkdir -p logs


export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

case ${LSB_JOBINDEX} in
    1)
        uv run generate_curve.py \
            --name "sweep_num_layers_no_subpixel_100nm" \
            --params_x "50,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --no_subpixel \
            --n_jobs 16
        ;;
    2)
        uv run generate_curve.py \
            --name "sweep_num_layers_100nm" \
            --params_x "50,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --n_jobs 16
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
            --no_subpixel \
            --n_jobs 16
        ;;
    4)
        uv run generate_curve.py \
            --name "sweep_num_layers_1000nm" \
            --params_x "500,0" \
            --order_N 20 \
            --num_layers 1 5 10 25 50 100 250 251 500 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --n_jobs 16
        ;;
esac
