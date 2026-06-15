#!/bin/sh
#BSUB -J train_surrogates
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/train_surrogates_%J.out
#BSUB -e logs/train_surrogates_%J.err

mkdir -p logs
module load cuda/12.4

# Fix for PyTorch 2.4.1 cu124 missing shared libs on compute nodes
export LD_LIBRARY_PATH="../.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:../.venv/lib/python3.12/site-packages/nvidia/nccl/lib:../.venv/lib/python3.12/site-packages/nvidia/cublas/lib:../.venv/lib/python3.12/site-packages/nvidia/cusparse/lib:../.venv/lib/python3.12/site-packages/nvidia/cusolver/lib:${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1
echo "Job starting on $(hostname) at $(date)"
nvidia-smi

uv run python train_forward.py \
    --data_dir ../Data/LHS_Dataset_Si ../Data/LHS_Dataset_TiO2 ../Data/LHS_Dataset_Si3N4 \
    --materials Si TiO2 Si3N4 \
    --target_key all_film \
    --epochs 2000 \
    --batch_size 256 \
    --lr 2e-3 \
    --patience 200 \
    --val_split 0.05 \
    --skip \
    --seed 1337

RUN_IMPLICIT_SIREN=false
if [ "$RUN_IMPLICIT_SIREN" = true ]; then
    echo "Running train_implicit_forward.py..."
    uv run python train_implicit_forward.py \
        --data_dir ../Data/LHS_Dataset_Si ../Data/LHS_Dataset_TiO2 ../Data/LHS_Dataset_Si3N4 \
        --materials Si TiO2 Si3N4 \
        --target_key all_film \
        --epochs 2000 \
        --batch_size 256 \
        --lr 2e-3 \
        --patience 200 \
        --val_split 0.05 \
        --seed 1337
fi

uv run python train_inverse.py \
    --data_dir ../Data/LHS_Dataset_Si ../Data/LHS_Dataset_TiO2 ../Data/LHS_Dataset_Si3N4 \
    --materials Si TiO2 Si3N4 \
    --target_key all_film \
    --epochs 2000 \
    --synthetic_epochs 500 \
    --batch_size 256 \
    --lr 2e-3 \
    --patience 200 \
    --val_split 0.05 \
    --seed 1337

echo "Job completed at $(date)"
