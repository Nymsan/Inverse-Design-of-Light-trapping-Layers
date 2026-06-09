#!/bin/sh
#BSUB -J train_surrogates
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/train_surrogates_%J.out
#BSUB -e logs/train_surrogates_%J.err

mkdir -p logs
module load cuda/11.8

# Fix for PyTorch 2.5.1 cu118 missing shared libs on compute nodes
export LD_LIBRARY_PATH="../.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:../.venv/lib/python3.13/site-packages/nvidia/nccl/lib:../.venv/lib/python3.13/site-packages/nvidia/cublas/lib:../.venv/lib/python3.13/site-packages/nvidia/cusparse/lib:../.venv/lib/python3.13/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname) at $(date)"
nvidia-smi

uv run train_models.py \
    --data_dirs ../Data/LHS_Dataset_Si ../Data/LHS_Dataset_TiO2 ../Data/LHS_Dataset_Si3N4 \
    --materials Si TiO2 Si3N4 \
    --target_key A_film_normal \
    --epochs 500 \
    --batch_size 256 \
    --lr 1e-3 \
    --patience 100 \
    --seed 42

echo "Job completed at $(date)"
