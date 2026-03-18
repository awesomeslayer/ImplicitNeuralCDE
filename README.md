# Implicit Neural Controlled Differential Equations

This repository implements Parameter-Efficient Neural CDEs via Implicit Function Jacobians. 

Standard Neural Controlled Differential Equations (NCDEs) require a massive output weight matrix to match the tensor dimensions of the hidden state and the input sequence. This project replaces that matrix with an implicit continuous recurrent step. You compute the hidden trajectory derivative using a Taylor expansion of the implicit function Jacobian. 

This approach cuts the parameter count in half while maintaining test accuracy.

## Architecture

The codebase provides four distinct execution paths across two deep learning frameworks. 

**PyTorch Implementations:**
* **Manual:** Explicit Jacobian matrix formulation for continuous RNN cells.
* **Autograd:** Forward-mode automatic differentiation (Jacobian-Vector Products) using `torch.func.jvp`. This path supports RNN, GRU, and LSTM cells without requiring explicit matrix derivations.

**JAX Implementations:**
* **Manual:** Explicit Jacobian calculations optimized through XLA compilation (`jax.lax.fori_loop`).
* **Autograd:** Native `jax.jvp` integration inside Diffrax ODE solvers. This path supports all cell types.

You control the precision of the implicit approximation using the `k_terms` parameter. Setting $k=0$ evaluates the base Jacobian-vector product. Setting $k>0$ adds subsequent terms from the Taylor series expansion. The codebase supports arbitrary lengths for $k$.

## Directory Structure

```text
.
├── configs/
│   └── config.yaml             # Default Hydra configuration
├── src_torch/
│   ├── data.py                 # PyTorch Lightning DataModule
│   ├── cells.py                # RNN, GRU, and LSTM cell definitions
│   ├── lit_module.py           # Lightning wrapper and ODE solver setup
│   ├── models_baseline.py      # Standard Matrix-based NCDE
│   ├── models_manual.py        # Explicit Jacobian PyTorch CDE
│   ├── models_auto.py          # JVP Autograd PyTorch CDE
│   ├── nat_cub_spline.py       # Natural cubic spline interpolation
│   └── train_torch.py          # PyTorch training loop
├── src_jax/
│   ├── cells_jax.py            # Equinox cell definitions
│   ├── models_manual_jax.py    # Explicit Jacobian JAX CDE
│   ├── models_auto_jax.py      # JVP Autograd JAX CDE
│   └── train_jax.py            # Diffrax training loop
├── scripts/
│   ├── run_all.sh              # Full benchmark execution script
│   └── aggregate.py            # Results parser and CSV generator
├── Dockerfile                  # CUDA environment definition
├── launch_container            # Container startup script
└── requirements.txt            # Python dependencies
```

## Environment Setup

Run the code inside the provided Docker environment to ensure CUDA compatibility.

1. Define your user parameters in a `credentials` file:
```bash
DOCKER_USER_ID=$(id -u)
DOCKER_GROUP_ID=$(id -g)
DOCKER_NAME=$USER
CONTAINER_NAME="implicit_cde"
```

2. Build the Docker image:
```bash
chmod +x ./build
./build
```

3. Start the container:
```bash
chmod +x ./launch_container
./launch_container
```

Now you are working in container with everything needed installed.

## Configuration

Hydra manages the experiment parameters. Edit `configs/config.yaml` to establish baseline settings. 

You override parameters directly from the command line. This command trains a PyTorch Autograd LSTM model using three Taylor expansion terms and a hidden dimension of 128:

```bash
python src_torch/train_torch.py model=torch_auto cell=lstm k_terms=3 hidden_dim=128
```

Supported model configurations:
* `model`: `matcde`, `torch_manual`, `torch_auto`, `jax_manual`, `jax_auto`
* `cell`: `rnn`, `gru`, `lstm`
* `k_terms`: Any integer $\ge 0$

## Execution

Execute the complete benchmark suite using the provided shell script:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_all.sh
```

The script trains models across specified seeds, cell types, and Taylor expansion lengths. It writes JSON logs and loss curves to the `outputs/` directory for every configuration.

Generate the final performance table once the benchmark finishes:

```bash
python scripts/aggregate.py
```

The aggregation script groups all runs by model type, cell architecture, and $k$ terms. It calculates the mean and standard deviation for parameter counts, training durations, and test accuracies. You will find the output in `benchmark_results.csv`.