import jax, diffrax, optax, json, os, time, hydra
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src_torch.data import DataModule 
from src_jax.cells_jax import RNNCellJax, GRUCellJax, LSTMCellJax
from src_jax.models_auto_jax import JaCDEAutoJax
from src_jax.models_manual_jax import JaCDEManualJax

class ClassifierJax(eqx.Module):
    cde_func: eqx.Module
    linear: eqx.nn.Linear
    
    def __init__(self, in_c, hid_c, out_c, model_type, cell_type, k_terms, activation, key):
        k1, k2 = jax.random.split(key)
        
        if model_type == "jax_baseline":
            from src_jax.models_baseline_jax import BaselineCDEJax
            self.cde_func = BaselineCDEJax(in_c, hid_c, cell_type, k1)
        elif model_type == "jax_manual":
        
            self.cde_func = JaCDEManualJax(in_c, hid_c, cell_type, k_terms, activation, k1)
        elif model_type == "jax_auto":
          
            if cell_type == "rnn": cell = RNNCellJax(in_c, hid_c, k1, activation=activation)
            elif cell_type == "gru": cell = GRUCellJax(in_c, hid_c, k1)
            elif cell_type == "lstm": cell = LSTMCellJax(in_c, hid_c, k1)
            else: raise ValueError(f"Unknown cell: {cell_type}")
            self.cde_func = JaCDEAutoJax(cell, k_terms)
        else:
            raise ValueError(f"Unknown JAX model: {model_type}")
            
        self.linear = eqx.nn.Linear(hid_c, out_c, key=k2)

    def __call__(self, ts, xs):
        coeffs = diffrax.backward_hermite_coefficients(ts, xs)
        interp = diffrax.CubicInterpolation(ts, coeffs)
        
        term = diffrax.ODETerm(self.cde_func)
        y0 = jnp.zeros(self.linear.in_features)
        
        sol = diffrax.diffeqsolve(
            term, diffrax.Dopri5(), ts[0], ts[-1], dt0=0.01, 
            y0=y0, args=interp, 
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3)
        )
        return self.linear(sol.ys[-1])
    
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    os.makedirs("outputs", exist_ok=True)
    
    file_prefix = f"jax_{cfg.model}_{cfg.cell}_{cfg.activation}_k{cfg.k_terms}_s{cfg.seed}"
    out_file = f"outputs/res_{file_prefix}.json"
    log_path = f"outputs/log_{file_prefix}.txt"

    if os.path.exists(out_file):
        print(f"[JAX] Experiment {file_prefix} already completed. Skipping.")
        return

    def log_print(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log_print(f"=== Starting JAX Training: {file_prefix} ===")

    dm = DataModule(cfg.dataset, cfg.batch_size)
    cfg.input_dim = dm.inp_dim
    cfg.output_dim = dm.out_dim

    key = jax.random.PRNGKey(cfg.seed)
    model = ClassifierJax(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.model, cfg.cell, cfg.k_terms, cfg.activation, key)
    
    params_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    log_print(f"Model parameters: {params_count:,}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    optim = optax.adam(1e-3)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_value_and_grad
    def compute_loss(m, ts, x, y):
        vmap_model = jax.vmap(m, in_axes=(None, 0))
        preds = vmap_model(ts, x)
        labels_onehot = jax.nn.one_hot(y, cfg.output_dim)
        return jnp.mean(-jnp.sum(labels_onehot * jax.nn.log_softmax(preds, axis=-1), axis=-1))

    @eqx.filter_jit
    def make_step(m, opt_s, ts, x, y):
        loss, grads = compute_loss(m, ts, x, y)
        updates, opt_s = optim.update(grads, opt_s, m)
        m = eqx.apply_updates(m, updates)
        return m, opt_s, loss

    @eqx.filter_jit
    def evaluate(m, ts, x, y):
        vmap_model = jax.vmap(m, in_axes=(None, 0))
        preds = vmap_model(ts, x)
        return jnp.sum(jnp.argmax(preds, axis=-1) == y)

    ckpt_dir = f"checkpoints/{file_prefix}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "weights.eqx")
    best_ckpt_path = os.path.join(ckpt_dir, "best_weights.eqx")
    state_path = os.path.join(ckpt_dir, "training_state.json")

    start_epoch, train_losses, val_accs, total_time = 0, [], [], 0.0
    best_val_acc = -1.0 # Отслеживаем лучший результат

    if os.path.exists(ckpt_path) and os.path.exists(state_path):
        log_print(f"Resuming from checkpoint: {ckpt_path}")
        model, opt_state = eqx.tree_deserialise_leaves(ckpt_path, (model, opt_state))
        with open(state_path, "r") as f:
            state = json.load(f)
            start_epoch = state["epoch"]
            train_losses = state["train_losses"]
            val_accs = state["val_accs"]
            total_time = state.get("total_time", 0.0)
            best_val_acc = state.get("best_val_acc", -1.0)
    else:
        log_print("Starting new training...")

    pbar = tqdm(range(start_epoch, cfg.epochs), desc=f"JAX {cfg.model}")
    for epoch in pbar:
        epoch_loss, steps = 0.0, 0
        t0 = time.time()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = jnp.array(batch_x.numpy()), jnp.array(batch_y.numpy())
            ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
            model, opt_state, loss = make_step(model, opt_state, ts, batch_x, batch_y)
            epoch_loss += loss.item()
            steps += 1
            if cfg.smoke_test: break

        train_loss = epoch_loss / steps
        train_losses.append(train_loss)
        
        correct, total = 0, 0
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = jnp.array(batch_x.numpy()), jnp.array(batch_y.numpy())
            ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
            correct += evaluate(model, ts, batch_x, batch_y)
            total += batch_y.shape[0]
            if cfg.smoke_test: break

        val_acc = float(correct / total)
        val_accs.append(val_acc)
        total_time += (time.time() - t0)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            eqx.tree_serialise_leaves(best_ckpt_path + ".tmp", (model, opt_state))
            os.replace(best_ckpt_path + ".tmp", best_ckpt_path)
            log_print(f"--> New best val_acc: {best_val_acc:.4f} at epoch {epoch+1}")

        pbar.set_postfix({"loss": f"{train_loss:.4f}", "val_acc": f"{val_acc:.4f}"})
        if (epoch + 1) % 5 == 0 or epoch == 0:
            log_print(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

        eqx.tree_serialise_leaves(ckpt_path + ".tmp", (model, opt_state))
        os.replace(ckpt_path + ".tmp", ckpt_path)
        with open(state_path + ".tmp", "w") as f:
            json.dump({
                "epoch": epoch + 1, 
                "train_losses": train_losses, 
                "val_accs": val_accs, 
                "total_time": total_time,
                "best_val_acc": best_val_acc
            }, f)
        os.replace(state_path + ".tmp", state_path)

        if cfg.smoke_test: 
            log_print("Smoke test step complete.")
            break
        
    log_print("Evaluating on test set using BEST model...")
    if os.path.exists(best_ckpt_path):
        model, _ = eqx.tree_deserialise_leaves(best_ckpt_path, (model, opt_state))
    
    correct, total = 0, 0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = jnp.array(batch_x.numpy()), jnp.array(batch_y.numpy())
        ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
        correct += evaluate(model, ts, batch_x, batch_y)
        total += batch_y.shape[0]
    test_acc = float(correct / total)
    log_print(f"Test Accuracy: {test_acc:.4f}")

    res = {
        "framework": "jax", "model": cfg.model, "cell": cfg.cell, 
        "k_terms": cfg.k_terms, "seed": cfg.seed, "params": params_count, 
        "time_s": total_time, "acc": test_acc
    }
    with open(out_file, "w") as f: json.dump(res, f)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(train_losses, color='tab:red')
    ax2 = ax1.twinx(); ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(val_accs, color='tab:blue')
    plt.savefig(f"outputs/curves_{file_prefix}.png", dpi=150)
    plt.close()

if __name__ == "__main__": 
    main()