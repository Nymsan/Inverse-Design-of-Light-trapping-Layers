#!/bin/sh
#BSUB -J generate_dataset_cpu[1-300]
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 02:00
#BSUB -o logs/generate_dataset/%J_%I.out
#BSUB -e logs/generate_dataset/%J_%I.err

export PYTHONUNBUFFERED=1
cd ~/Documents/Python/Inverse-Design-of-Light-trapping-Layers
mkdir -p logs/generate_dataset
echo "Job starting on $(hostname), Task ID: ${LSB_JOBINDEX}"

export CUDA_VISIBLE_DEVICES=""

# Map LSB_JOBINDEX [1-300] to Material and Batch Index
if [ ${LSB_JOBINDEX} -le 100 ]; then
    MATERIAL="Si"
    SEED=42
    BATCH_IDX=$((LSB_JOBINDEX - 1))
elif [ ${LSB_JOBINDEX} -le 200 ]; then
    MATERIAL="TiO2"
    SEED=43
    BATCH_IDX=$((LSB_JOBINDEX - 101))
else
    MATERIAL="Si3N4"
    SEED=44
    BATCH_IDX=$((LSB_JOBINDEX - 201))
fi

START_BATCH=${BATCH_IDX}
END_BATCH=$((BATCH_IDX + 1))

echo "======================================"
echo "Running Material: ${MATERIAL}"
echo "Batch Range: ${START_BATCH} to ${END_BATCH}"
echo "======================================"

uv run --no-sync python Scripts/generate_dataset.py \
    --num_samples 10000 \
    --batch_size 100 \
    --order_N 15 \
    --height_per_layer 5.0 \
    --grating_period 1000.0 \
    --nx 5000 \
    --grating_material ${MATERIAL} \
    --seed ${SEED} \
    --n_jobs 4 \
    --start_batch ${START_BATCH} \
    --end_batch ${END_BATCH}

echo "Task completed!"
