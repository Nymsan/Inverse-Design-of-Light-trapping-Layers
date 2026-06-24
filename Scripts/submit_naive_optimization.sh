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
echo "Material: $MAT_NAME"
echo "Resolution: 10 nm (81 wavelengths)"
echo "Method: L-BFGS-B (Gradients via Finite Difference)"
echo "Budget: 300 Torcwa evaluations"
echo "Queue: gpul40s"
echo "========================================="

echo "=== Phase 1: Bounded Optimization (1000-3000nm) ==="
uv run python Scripts/evaluate_naive_optimization.py \
    --material $MAT_NAME \
    --method lbfgs \
    --max_evals 300 \
    --out_dir Naive_Optimization_Bounded \
    --penalty \
    --h_val 1000 3000

echo "=== Phase 2: Pinned Optimization (2000nm) ==="
uv run python Scripts/evaluate_naive_optimization.py \
    --material $MAT_NAME \
    --method lbfgs \
    --max_evals 300 \
    --out_dir Naive_Optimization_Pinned \
    --penalty \
    --h_val 2000

echo "=== Naive Torcwa Optimization Complete for $MAT_NAME ==="
