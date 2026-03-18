#!/bin/bash

export PYTHONPATH=$(pwd)

DATASET="CharacterTrajectories"
BATCH_SIZE=128
HIDDEN_DIM=128
EPOCHS=50
SEED=42

# Number of terms in the Taylor expansion
K_TERMS=(1 2 3)

echo "================================================="
echo "Starting benchmark: $DATASET"
echo "Parameters: bs=$BATCH_SIZE, hidden=$HIDDEN_DIM, epochs=$EPOCHS"
echo "================================================="

# 1. Torch Manual (Supports ONLY RNN)
for k in "${K_TERMS[@]}"; do
    echo "--> Running [Torch Manual] | Cell: rnn | K: $k"
    python src_torch/train_torch.py dataset=$DATASET model=torch_manual cell=rnn k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
done

# 2. Torch Autograd (Supports RNN, GRU, LSTM)
for k in "${K_TERMS[@]}"; do
    for cell in rnn gru lstm; do
        echo "--> Running [Torch Auto] | Cell: $cell | K: $k"
        python src_torch/train_torch.py dataset=$DATASET model=torch_auto cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
    done
done

# 3. JAX Manual (Supports ONLY RNN)
for k in "${K_TERMS[@]}"; do
    echo "--> Running [JAX Manual] | Cell: rnn | K: $k"
    python src_jax/train_jax.py dataset=$DATASET model=jax_manual cell=rnn k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
done

# 4. JAX Autograd (Supports RNN, GRU, LSTM)
for k in "${K_TERMS[@]}"; do
    for cell in rnn gru lstm; do
        echo "--> Running [JAX Auto] | Cell: $cell | K: $k"
        python src_jax/train_jax.py dataset=$DATASET model=jax_auto cell=$cell k_terms=$k batch_size=$BATCH_SIZE hidden_dim=$HIDDEN_DIM epochs=$EPOCHS seed=$SEED
    done
done

echo "================================================="
echo "All tests completed! Aggregating results..."
echo "================================================="

python scripts/aggregate.py