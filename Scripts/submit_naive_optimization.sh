#!/bin/bash
#BSUB -J naive_opt[1-3]
#BSUB -q gpul40s
#BSUB -n 8
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/naive_opt_%J_%I.out
#BSUB -e logs/naive_opt_%J_%I.err

# Ensure we can see the tqdm output in the logs as it runs
export PYTHONUNBUFFERED=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Map LSB_JOBINDEX (1-3) to material names
MATERIALS=("Si" "TiO2" "Si3N4")
MAT_NAME=${MATERIALS[$((LSB_JOBINDEX - 1))]}

echo "=== Starting Naive Torcwa Optimization ==="
echo "Material:   $MAT_NAME"
echo "Harmonics:  7  |  order_N: 20  |  nx: 5000"
echo "Amp bound:  0-15 nm per harmonic"
echo "Reflector:  pec  |  height_per_layer: 5 nm"
echo "Objective:  Absorptance "
echo "========================================="

echo ""
echo "=== Phase 1: Bounded Optimization (h = 1000-3000 nm) ==="
uv run python Scripts/evaluate_naive_optimization.py \
    --material $MAT_NAME \
    --h_val 1000 3000 \
    --n_iters 1 \
    --n_restarts 1 \
    --seed 1234 \
    --out_dir Results/naive_opt_bounded

echo ""
echo "=== Naive Torcwa Optimization Complete for $MAT_NAME ==="
