#!/bin/sh
#BSUB -J train_surrogates
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 8
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

# ==============================================================================
# Pipeline Toggles
# ==============================================================================
TRAIN_FORWARD=true
TRAIN_INVERSE=true

RUN_ACTIVE_LEARNING=true
EVAL_FORWARD=true
EVAL_DATASET_BASELINE=true
EVAL_GENERALIZATION=true
EVAL_SURROGATE=true
EVAL_INVERSE=true
EVAL_IMPLICIT=true
# ==============================================================================
# Model Architecture Hyperparameters
# Adjust these dimensions to control the parameter count / capacity of the models
# ==============================================================================

# Global
EMBED_DIM="8"
LATENT_DIM_GEN="32"
LATENT_DIM_CVAE="64"

# --- MLP & General parameters ---
MLP_HIDDEN_DIMS="512 768 512"
MLP_DROPOUT="0.0"

# --- CNN configurations ---
CNN_CONV_CHANNELS="48 96 128 128 96"
CNN_KERNEL_SIZE="9"
CNN_FC_DIMS="384 256"
CNN_DROPOUT="0.0"

# --- SkipCNN configurations  ---
SKIPCNN_CONV_CHANNELS="48 96 96 128 64"
SKIPCNN_KERNEL_SIZE="9"
SKIPCNN_FC_DIMS="256 256"
SKIPCNN_DROPOUT="0.0"

# --- SIREN configurations ---
SIREN_CONV_CHANNELS="64 96 96 128 64"
SIREN_KERNEL_SIZE="9"
SIREN_FC_DIMS="256 256"
SIREN_LATENT_DIM="64"
SIREN_OMEGA_0="30.0"
SIREN_DROPOUT="0.0"

TF_D_MODEL="128"
TF_NHEAD="4"
TF_DIM_FEEDFORWARD="450"
TF_NUM_LAYERS="4"
TF_DROPOUT="0.0"

# Inverse Models
INV_CONV_CHANNELS="48 72 108 128 64"
INV_KERNEL_SIZE="9"
INV_FC_DIMS="256 256"
INV_DROPOUT="0.0"

CVAE_GEO_ENC_CONV="48 72 108 128 64"
CVAE_GEO_ENC_KERNEL="9"
CVAE_GEO_ENC_FC="256 128"
CVAE_GEO_ENC_DROPOUT="0.0"

CVAE_GEO_DEC_FC="256 256 256"
CVAE_GEO_DEC_DROPOUT="0.0"

CVAE_SPEC_ENC_CONV="48 72 108 128 64"
CVAE_SPEC_ENC_KERNEL="9"
CVAE_SPEC_ENC_FC="256 256"
CVAE_SPEC_ENC_DROPOUT="0.0"

# ==============================================================================

echo "Counting parameters for all models..."
uv run python count_params.py \
    --mlp_hidden_dims $MLP_HIDDEN_DIMS \
    --embed_dim $EMBED_DIM \
    --latent_dim_gen $LATENT_DIM_GEN \
    --latent_dim_cvae $LATENT_DIM_CVAE \
    --cnn_conv_channels $CNN_CONV_CHANNELS \
    --cnn_kernel_size $CNN_KERNEL_SIZE \
    --cnn_fc_dims $CNN_FC_DIMS \
    --skipcnn_conv_channels $SKIPCNN_CONV_CHANNELS \
    --skipcnn_kernel_size $SKIPCNN_KERNEL_SIZE \
    --skipcnn_fc_dims $SKIPCNN_FC_DIMS \
    --siren_conv_channels $SIREN_CONV_CHANNELS \
    --siren_kernel_size $SIREN_KERNEL_SIZE \
    --siren_fc_dims $SIREN_FC_DIMS \
    --tf_d_model $TF_D_MODEL \
    --tf_nhead $TF_NHEAD \
    --tf_dim_feedforward $TF_DIM_FEEDFORWARD \
    --tf_num_layers $TF_NUM_LAYERS \
    --inv_conv_channels $INV_CONV_CHANNELS \
    --inv_kernel_size $INV_KERNEL_SIZE \
    --inv_fc_dims $INV_FC_DIMS \
    --cvae_geo_enc_conv $CVAE_GEO_ENC_CONV \
    --cvae_geo_enc_kernel $CVAE_GEO_ENC_KERNEL \
    --cvae_geo_enc_fc $CVAE_GEO_ENC_FC \
    --cvae_geo_dec_fc $CVAE_GEO_DEC_FC \
    --cvae_spec_enc_conv $CVAE_SPEC_ENC_CONV \
    --cvae_spec_enc_kernel $CVAE_SPEC_ENC_KERNEL \
    --cvae_spec_enc_fc $CVAE_SPEC_ENC_FC

if [ "$TRAIN_FORWARD" = true ]; then
    echo -e "\n=== Starting Forward Training ==="
    uv run python train_forward.py \
        --data_dir ../Data \
        --dataset_prefixes LHS_Dataset LHS_Dataset_Deep \
        --materials Si TiO2 Si3N4 \
        --target_key all_film \
        --epochs 2000 \
        --batch_size 1024 \
        --lr 2e-3 \
        --patience 200 \
        --val_split 0.05 \
        --skip cnn transformer\
        --seed 1337 \
        --embed_dim $EMBED_DIM \
        --mlp_hidden_dims $MLP_HIDDEN_DIMS \
        --mlp_dropout $MLP_DROPOUT \
        --cnn_conv_channels $CNN_CONV_CHANNELS \
        --cnn_kernel_size $CNN_KERNEL_SIZE \
        --cnn_fc_dims $CNN_FC_DIMS \
        --cnn_dropout $CNN_DROPOUT \
        --skipcnn_conv_channels $SKIPCNN_CONV_CHANNELS \
        --skipcnn_kernel_size $SKIPCNN_KERNEL_SIZE \
        --skipcnn_fc_dims $SKIPCNN_FC_DIMS \
        --skipcnn_dropout $SKIPCNN_DROPOUT \
        --siren_conv_channels $SIREN_CONV_CHANNELS \
        --siren_kernel_size $SIREN_KERNEL_SIZE \
        --siren_fc_dims $SIREN_FC_DIMS \
        --siren_latent_dim $SIREN_LATENT_DIM \
        --siren_omega_0 $SIREN_OMEGA_0 \
        --siren_dropout $SIREN_DROPOUT \
        --tf_d_model $TF_D_MODEL \
        --tf_nhead $TF_NHEAD \
        --tf_dim_feedforward $TF_DIM_FEEDFORWARD \
        --tf_num_layers $TF_NUM_LAYERS \
        --tf_dropout $TF_DROPOUT
fi

if [ "$TRAIN_INVERSE" = true ]; then
    echo -e "\n=== Starting Inverse Training ==="
    uv run python train_inverse.py \
        --data_dir ../Data \
        --dataset_prefixes LHS_Dataset LHS_Dataset_Deep \
        --materials Si TiO2 Si3N4 \
        --target_key all_film \
        --epochs 2000 \
        --synthetic_epochs 500 \
        --batch_size 1024 \
        --lr 2e-3 \
        --patience 200 \
        --val_split 0.05 \
        --seed 1337 \
        --embed_dim $EMBED_DIM \
        --latent_dim_gen $LATENT_DIM_GEN \
        --latent_dim_cvae $LATENT_DIM_CVAE \
        --inv_conv_channels $INV_CONV_CHANNELS \
        --inv_kernel_size $INV_KERNEL_SIZE \
        --inv_fc_dims $INV_FC_DIMS \
        --inv_dropout $INV_DROPOUT \
        --cvae_geo_enc_conv $CVAE_GEO_ENC_CONV \
        --cvae_geo_enc_kernel $CVAE_GEO_ENC_KERNEL \
        --cvae_geo_enc_fc $CVAE_GEO_ENC_FC \
        --cvae_geo_enc_dropout $CVAE_GEO_ENC_DROPOUT \
        --cvae_geo_dec_fc $CVAE_GEO_DEC_FC \
        --cvae_geo_dec_dropout $CVAE_GEO_DEC_DROPOUT \
        --cvae_spec_enc_conv $CVAE_SPEC_ENC_CONV \
        --cvae_spec_enc_kernel $CVAE_SPEC_ENC_KERNEL \
        --cvae_spec_enc_fc $CVAE_SPEC_ENC_FC \
        --cvae_spec_enc_dropout $CVAE_SPEC_ENC_DROPOUT
fi

if [ "$EVAL_DATASET_BASELINE" = true ]; then
    echo -e "\n=== Evaluating Dataset Baseline ==="
    uv run python evaluate_dataset_baseline.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4
fi

if [ "$RUN_ACTIVE_LEARNING" = true ]; then
    echo -e "\n=== Cleaning Old Active Learning Data ==="
    rm -rf ../Data/Active_Learning_Dataset
    
    echo -e "\n=== Running Active Learning ==="
    uv run python train_active_learning.py \
        --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4 \
        --mode geometry \
        --iterations 5 \
        --proposals_per_mat 8 \
        --restarts 1000 \
        --steps 300 \
        --h_val 2000 \
        --inc_val 0.0 \
        --expand_amps 40.0
fi

if [ "$EVAL_FORWARD" = true ]; then
    echo -e "\n=== Evaluating Forward Models ==="
    uv run python evaluate_forward.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4
fi

if [ "$EVAL_GENERALIZATION" = true ]; then
    echo -e "\n=== Evaluating Generalization ==="
    uv run python evaluate_generalization.py \
        --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4 \
        --num_samples 3 \
        --n_harmonics 10
fi

if [ "$EVAL_SURROGATE" = true ]; then
    echo -e "\n=== Evaluating Surrogate Optimization (Geometry) ==="
    uv run python evaluate_surrogate_optimization.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4 --mode geometry --restarts 10000 --steps 300 --h_val 2000 --inc_val 0.0 --expand_amps 40.0
    
    echo -e "\n=== Evaluating Surrogate Optimization (Differential Evolution) ==="
    uv run python evaluate_surrogate_optimization.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4 --mode de --restarts 10000 --steps 300 --h_val 2000 --inc_val 0.0 --expand_amps 40.0
fi

if [ "$EVAL_INVERSE" = true ]; then
    echo -e "\n=== Evaluating Inverse Models ==="
    uv run python evaluate_inverse.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4
fi

if [ "$EVAL_IMPLICIT" = true ]; then
    echo -e "\n=== Evaluating Implicit Models ==="
    uv run python evaluate_implicit_model.py --ckpt_dir ../Checkpoints/Si_TiO2_Si3N4
fi

echo "Job completed at $(date)"
