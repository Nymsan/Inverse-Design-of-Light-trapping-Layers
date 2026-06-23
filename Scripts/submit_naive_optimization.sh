#!/bin/bash
#BSUB -J naive_opt
#BSUB -q gpul40s
#BSUB -n 8
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o logs/naive_opt_%J.out
#BSUB -e logs/naive_opt_%J.err

# Ensure we can see the tqdm output in the logs as it runs
export PYTHONUNBUFFERED=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=== Starting Naive Torcwa Optimization ==="
echo "Materials: Si, TiO2, Si3N4"
echo "Resolution: 10 nm (81 wavelengths)"
echo "Method: L-BFGS-B (Gradients via Finite Difference)"
echo "Restarts: 1 random initialization per material"
echo "Budget: 5000 Torcwa evaluations per material"
echo "Queue: gpul40s"
echo "========================================="

# L-BFGS evaluates 13 Torcwa steps per gradient calculation (12 params + baseline).
# This is incredibly fast compared to DE.
for MAT in "Si" "TiO2" "Si3N4"; do
    echo "Starting $MAT optimization..."
    uv run python evaluate_naive_optimization.py \
        --material $MAT \
        --method lbfgs \
        --max_evals 5000 \
        --out_dir Naive_Optimization
    echo "Finished $MAT."
done

echo "=== Naive Torcwa Optimization Complete ==="
