#!/bin/bash
set -e
export PYTHONPATH=$(pwd)

echo "=== RUNNING QUICK SMOKE TEST ==="
echo "Check parameter counts and potential crashes"

CELLS="rnn lstm gru"
MODELS="baseline manual auto"

for cell in $CELLS; do
    for mod in $MODELS; do

        if [ "$mod" == "manual" ] && [ "$cell" != "rnn" ]; then continue; fi
        
        echo "------------------------------------------------"
        echo "TESTING: Torch | Model: $mod | Cell: $cell"
        python src_torch/train_torch.py model=torch_$mod cell=$cell smoke_test=True epochs=1
        
        echo "TESTING: JAX | Model: $mod | Cell: $cell"
        python src_jax/train_jax.py model=jax_$mod cell=$cell smoke_test=True epochs=1
    done
done

echo "=== ALL MODELS INITIALIZED SUCCESSFULLY ==="