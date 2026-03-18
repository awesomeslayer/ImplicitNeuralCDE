#!/bin/bash

export JAX_LOG_LEVEL=warning
export TF_CPP_MIN_LOG_LEVEL=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
set -e
export PYTHONPATH=$(pwd)

DATASET="CharacterTrajectories"
BATCH_SIZE=128
HIDDEN_DIM=128
EPOCHS=35
SEED=43


echo "Starting FULL benchmark (Baselines + Jacobians + Activations)"
echo "Parameters: bs=$BATCH_SIZE, hidden=$HIDDEN_DIM, epochs=$EPOCHS"
echo "================================================="

# ==========================================
# 0. BASELINES (Matrix-CDEs)
# ==========================================
for cell in rnn; do
    echo "--> Running [Torch Baseline] | Cell: $cell"
    python src_torch/train_torch.py dataset=$DATASET model=torch_baseline cell=$cell k_terms=0 activation=relu epochs=$EPOCHS seed=$SEED

    echo "--> Running [JAX Baseline] | Cell: $cell"
    python src_jax/train_jax.py dataset=$DATASET model=jax_baseline cell=$cell k_terms=0 activation=relu epochs=$EPOCHS seed=$SEED
done

# ==========================================
# 1. TORCH & JAX MODELS (Ablation: Surrogate vs ReLU)
# ==========================================

for act in "surrogate_relu" "relu"; do
    echo "****************************************"
    echo "Testing with Activation: $act"
    echo "****************************************"

    # --- TORCH MANUAL ---
    for k in 1 2 3; do
        echo "--> [Torch Manual] | K: $k | Act: $act"
        python src_torch/train_torch.py dataset=$DATASET model=torch_manual cell=rnn k_terms=$k activation=$act epochs=$EPOCHS seed=$SEED
    done

    # --- TORCH AUTO (Только K=1 из-за скорости) ---
    # for k in 1; do
    #     echo "--> [Torch Auto] | K: $k | Act: $act"
    #     python src_torch/train_torch.py dataset=$DATASET model=torch_auto cell=rnn k_terms=$k activation=$act epochs=$EPOCHS seed=$SEED
    # done

    # --- JAX MANUAL ---
    for k in 1 2 3; do
        echo "--> [JAX Manual] | K: $k | Act: $act"
        python src_jax/train_jax.py dataset=$DATASET model=jax_manual cell=rnn k_terms=$k activation=$act epochs=$EPOCHS seed=$SEED
    done

    # --- JAX AUTO ---
    for k in 1 2 3; do
        echo "--> [JAX Auto] | K: $k | Act: $act"
        python src_jax/train_jax.py dataset=$DATASET model=jax_auto cell=rnn k_terms=$k activation=$act epochs=$EPOCHS seed=$SEED
    done
done

echo "================================================="
echo "All tests processed! Aggregating results..."
echo "================================================="

python scripts/aggregate.py