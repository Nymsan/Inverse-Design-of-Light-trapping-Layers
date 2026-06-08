#!/bin/sh
#BSUB -J gen_curves_order_N[1-4]
#BSUB -o logs/gen_curves_order_N_%I.out
#BSUB -e logs/gen_curves_order_N_%I.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
#BSUB -R "rusage[mem=12GB]"
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
        uv run generate_curve.py \
            --name "sweep_order_N_no_subpixel_100nm" \
            --params_x "50,0" \
            --order_N 1 5 10 25 50 100 \
            --num_layers 30 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --no_subpixel
        ;;
    2)
        uv run generate_curve.py \
            --name "sweep_order_N_100nm" \
            --params_x "50,0" \
            --order_N 1 5 10 25 50 100 \
            --num_layers 30 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1
        ;;
    3)
        uv run generate_curve.py \
            --name "sweep_order_N_no_subpixel_1000nm" \
            --params_x "500,0" \
            --order_N 1 5 10 25 50 100 \
            --num_layers 30 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1 \
            --no_subpixel
        ;;
    4)
        uv run generate_curve.py \
            --name "sweep_order_N_1000nm" \
            --params_x "500,0" \
            --order_N 1 5 10 25 50 100 \
            --num_layers 30 \
            --wavelengths 300 1100 1601 \
            --nx 5000 \
            --ny 1
        ;;
esac
