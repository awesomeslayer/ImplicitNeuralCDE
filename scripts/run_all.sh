#!/bin/bash

export JAX_LOG_LEVEL=warning
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Exit on error
set -e

export PYTHONPATH=$(pwd)

DATASET="CharacterTrajectories"
BATCH_SIZE=128
HIDDEN_DIM=128
EPOCHS=50
SEED=42

echo "================================================="
echo "Starting FULL benchmark (Baselines + Jacobians)"
echo "Parameters: bs=$BATCH_SIZE, hidden=$HIDDEN_DIM, epochs=$EPOCHS"
echo "================================================="

# ==========================================
# 0. BASELINES (Matrix-CDEs)
# ==========================================
# mlp  : Original Matrix CDE (no x_t)
# rnn  : ~66k params matrix CDE
# gru  : ~100k params matrix CDE
# lstm : ~67k-117k params matrix CDE

#for cell in rnn gru lstm; do
for cell in rnn; do
    echo "--> Running [Torch Baseline] | Cell: $cell | K: 0"
    python src_torch/train_torch.py dataset=$DATASET model=torch_baseline cell=$cell k_terms=0 batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
done

#for cell in rnn gru lstm; do
for cell in rnn ; do
    echo "--> Running [JAX Baseline] | Cell: $cell | K: 0"
    python src_jax/train_jax.py dataset=$DATASET model=jax_baseline cell=$cell k_terms=0 batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
done


# ==========================================
# 1. TORCH MODELS
# ==========================================
# Torch Manual (Supports ONLY RNN)

#for k in 1 2 3; do
for k in 1; do
    for cell in rnn; do
        echo "--> Running [Torch Manual] | Cell: $cell | K: $k"
        python src_torch/train_torch.py dataset=$DATASET model=torch_manual cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
    done
done

# Torch Auto (Supports RNN, GRU, LSTM)
#for k in 1 2 3; do
# for k in 1; do
#     #for cell in rnn gru lstm; do
#     for cell in rnn; do
#         echo "--> Running [Torch Auto] | Cell: $cell | K: $k"
#         python src_torch/train_torch.py dataset=$DATASET model=torch_auto cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
#     done
# done


# # ==========================================
# # 2. JAX MODELS
# # ==========================================
# # JAX Manual (Supports ONLY RNN)
#for k in 1 2 3; do
for k in 1; do
    for cell in rnn; do
        echo "--> Running [JAX Manual] | Cell: $cell | K: $k"
        python src_jax/train_jax.py dataset=$DATASET model=jax_manual cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
    done
done

# # JAX Auto (Supports RNN, GRU, LSTM)
# #for k in 1 2 3; do
for k in 1; do
    #for cell in rnn gru lstm; do
    for cell in rnn; do
        echo "--> Running [JAX Auto] | Cell: $cell | K: $k"
        python src_jax/train_jax.py dataset=$DATASET model=jax_auto cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
    done
done

echo "================================================="
echo "All tests processed! Aggregating results..."
echo "================================================="

python scripts/aggregate.py