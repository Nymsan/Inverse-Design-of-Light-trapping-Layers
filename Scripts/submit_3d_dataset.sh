#!/bin/sh
#BSUB -J generate_3d_dataset[1-150]
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
#BSUB -o logs/generate_3d_dataset/%J_%I.out
#BSUB -e logs/generate_3d_dataset/%J_%I.err

export PYTHONUNBUFFERED=1
cd ~/Documents/Python/Inverse-Design-of-Light-trapping-Layers
mkdir -p logs/generate_3d_dataset
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

export CUDA_VISIBLE_DEVICES=""

if [ ${LSB_JOBINDEX} -le 25 ]; then
    MATERIAL="Si"
    SEED=42
    BATCH_IDX=$((LSB_JOBINDEX - 1))
elif [ ${LSB_JOBINDEX} -le 50 ]; then
    MATERIAL="TiO2"
    SEED=43
    BATCH_IDX=$((LSB_JOBINDEX - 26))
else
    MATERIAL="Si3N4"
    SEED=44
    BATCH_IDX=$((LSB_JOBINDEX - 51))
fi

START_BATCH=${BATCH_IDX}
END_BATCH=$((BATCH_IDX + 1))

echo "======================================"
echo "Running Material: ${MATERIAL}"
echo "Batch Range: ${START_BATCH} to ${END_BATCH}"
echo "======================================"

uv run --no-sync python Scripts/generate_3d_dataset.py \
    --num_samples 2500 \
    --batch_size 100 \
    --order_N 8 \
    --order_N_y 8 \
    --height_per_layer 10.0 \
    --grating_period 1000.0 \
    --grating_period_y 1000.0 \
    --nx 128 \
    --ny 128 \
    --grating_material ${MATERIAL} \
    --seed ${SEED} \
    --n_jobs 4 \
    --start_batch ${START_BATCH} \
    --end_batch ${END_BATCH}

echo "Task completed!"
