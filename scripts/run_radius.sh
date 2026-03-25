#!/bin/bash
set -e
export PYTHONPATH=$(pwd)

DATASET="CharacterTrajectories"
ACT="surrogate_relu"
EPOCHS=35

echo "================================================="
echo " STARTING RADIUS TRACKING EXPERIMENT (Plots only)"
echo "================================================="

rm -f outputs/*_s43.json
rm -f outputs/curves_*.png

echo ">>> Running RNN (Manual Jacobian, K=3)..."
python src_torch/train_torch.py \
    dataset=$DATASET model=torch_manual cell=rnn k_terms=3 \
    activation=$ACT epochs=$EPOCHS seed=43 track_radius=True

echo ">>> Running LSTM (Autograd Jacobian, K=1)..."
python src_torch/train_torch.py \
    dataset=$DATASET model=torch_auto cell=lstm k_terms=1 \
    activation=$ACT epochs=$EPOCHS seed=43 track_radius=True

echo ">>> Running GRU (Autograd Jacobian, K=1)..."
python src_torch/train_torch.py \
    dataset=$DATASET model=torch_auto cell=gru k_terms=1 \
    activation=$ACT epochs=$EPOCHS seed=43 track_radius=True

echo ">>> Generating Plots..."
python scripts/replot.py

echo "Done! Check outputs/ directory for your curves_*.png plots."