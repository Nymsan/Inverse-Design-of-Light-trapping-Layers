#!/bin/sh
#BSUB -J gen_curves_combination_3d[1-4]
#BSUB -o logs/gen_curves_combination_3d_%I.out
#BSUB -e logs/gen_curves_combination_3d_%I.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00

mkdir -p logs
module load cuda/11.8

# Fix for PyTorch 2.5.1 cu118 missing shared libs on compute nodes
export LD_LIBRARY_PATH="../.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:../.venv/lib/python3.13/site-packages/nvidia/nccl/lib:../.venv/lib/python3.13/site-packages/nvidia/cublas/lib:../.venv/lib/python3.13/site-packages/nvidia/cusparse/lib:../.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

case ${LSB_JOBINDEX} in
    1)
        echo "Running WITH subpixel smoothing... (100nm)"
        uv run generate_curve.py \
            --name "sweep_combination_3d_100nm" \
            --params_x "25,0" \
            --params_y "25,0" \
            --order_N 1 3 5 8 10 12 15 \
            --num_layers 1 2 5 10 25 50 100 \
            --wavelengths 700 700 1 \
            --nx 500 \
            --ny 500
        ;;
    2)
        echo "Running WITHOUT subpixel smoothing... (100nm)"
        uv run generate_curve.py \
            --name "sweep_combination_3d_no_subpixel_100nm" \
            --no_subpixel \
            --params_x "25,0" \
            --params_y "25,0" \
            --order_N 1 3 5 8 10 12 15 \
            --num_layers 1 2 5 10 25 50 100 \
            --wavelengths 700 700 1 \
            --nx 500 \
            --ny 500
        ;;
    3)
        echo "Running WITH subpixel smoothing... (1000nm)"
        uv run generate_curve.py \
            --name "sweep_combination_3d_1000nm" \
            --params_x "250,0" \
            --params_y "250,0" \
            --order_N 1 3 5 8 10 12 15 \
            --num_layers 1 2 5 10 25 50 100 \
            --wavelengths 700 700 1 \
            --nx 500 \
            --ny 500
        ;;
    4)
        echo "Running WITHOUT subpixel smoothing... (1000nm)"
        uv run generate_curve.py \
            --name "sweep_combination_3d_no_subpixel_1000nm" \
            --no_subpixel \
            --params_x "250,0" \
            --params_y "250,0" \
            --order_N 1 3 5 8 10 12 15 \
            --num_layers 1 2 5 10 25 50 100 \
            --wavelengths 700 700 1 \
            --nx 500 \
            --ny 500
        ;;
esac
