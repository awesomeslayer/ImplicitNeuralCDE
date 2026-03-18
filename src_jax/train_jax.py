import jax, diffrax, optax, json, os, time, hydra
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from src_torch.data import DataModule 
from src_jax.cells_jax import RNNCellJax, GRUCellJax, LSTMCellJax
from src_jax.models_auto_jax import JaCDEAutoJax
from src_jax.models_manual_jax import JaCDEManualJax

class ClassifierJax(eqx.Module):
    cde_func: eqx.Module
    linear: eqx.nn.Linear
    
    def __init__(self, in_c, hid_c, out_c, model_type, cell_type, k_terms, key):
        k1, k2 = jax.random.split(key)
        
        if model_type == "jax_manual":
            self.cde_func = JaCDEManualJax(in_c, hid_c, cell_type, k_terms, k1)
        elif model_type == "jax_auto":
            if cell_type == "rnn": cell = RNNCellJax(in_c, hid_c, k1)
            elif cell_type == "gru": cell = GRUCellJax(in_c, hid_c, k1)
            elif cell_type == "lstm": cell = LSTMCellJax(in_c, hid_c, k1)
            else: raise ValueError(f"Unknown cell: {cell_type}")
            self.cde_func = JaCDEAutoJax(cell, k_terms)
        else:
            raise ValueError(f"Unknown JAX model: {model_type}")
            
        self.linear = eqx.nn.Linear(hid_c, out_c, key=k2)

    def __call__(self, ts, xs):
        interp = diffrax.CubicInterpolation(ts, xs)
        term = diffrax.ODETerm(self.cde_func)
        y0 = jnp.zeros(self.linear.in_features)
        
        sol = diffrax.diffeqsolve(
            term, diffrax.Dopri5(), ts[0], ts[-1], dt0=0.01, 
            y0=y0, args=interp, stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3)
        )
        return self.linear(sol.ys[-1])

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Отключаем строгий режим Hydra
    OmegaConf.set_struct(cfg, False)

    dm = DataModule(cfg.dataset, cfg.batch_size)
    cfg.input_dim = dm.inp_dim
    cfg.output_dim = dm.out_dim

    key = jax.random.PRNGKey(cfg.seed)
    model = ClassifierJax(cfg.input_dim, cfg.hidden_dim, cfg.output_dim, cfg.model, cfg.cell, cfg.k_terms, key)
    
    params_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

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

    # Training Loop with history collection
    train_losses = []
    val_accs = []
    
    t0 = time.time()
    for epoch in range(cfg.epochs):
        epoch_loss = 0.0
        steps = 0
        
        # Train
        for batch_x, batch_y in train_loader:
            batch_x = jnp.array(batch_x.numpy())
            batch_y = jnp.array(batch_y.numpy())
            ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
            model, opt_state, loss = make_step(model, opt_state, ts, batch_x, batch_y)
            epoch_loss += loss.item()
            steps += 1
            
        train_losses.append(epoch_loss / steps)
        
        # Validate
        correct = 0; total = 0
        for batch_x, batch_y in val_loader:
            batch_x = jnp.array(batch_x.numpy())
            batch_y = jnp.array(batch_y.numpy())
            ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
            correct += evaluate(model, ts, batch_x, batch_y)
            total += batch_y.shape[0]
            
        val_accs.append(float(correct / total))
        
    t_train = time.time() - t0

    # Test
    correct = 0; total = 0
    for batch_x, batch_y in test_loader:
        batch_x = jnp.array(batch_x.numpy())
        batch_y = jnp.array(batch_y.numpy())
        ts = jnp.arange(batch_x.shape[1], dtype=jnp.float32)
        correct += evaluate(model, ts, batch_x, batch_y)
        total += batch_y.shape[0]
        
    test_acc = float(correct / total)

    os.makedirs("outputs", exist_ok=True)
    res = {
        "framework": "jax", "model": cfg.model, "cell": cfg.cell, 
        "k_terms": cfg.k_terms, "seed": cfg.seed, "params": params_count, 
        "time_s": t_train, "acc": test_acc
    }
    
    out_file = f"outputs/res_jax_{cfg.model}_{cfg.cell}_k{cfg.k_terms}_s{cfg.seed}.json"
    with open(out_file, "w") as f: json.dump(res, f)

    # === РИСУЕМ И СОХРАНЯЕМ ГРАФИК ===
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    ax1.plot(train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    ax2.plot(val_accs, color='tab:blue', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f"JAX: {cfg.model} | {cfg.cell} | k={cfg.k_terms}")
    fig.tight_layout()
    plt.savefig(f"outputs/curves_jax_{cfg.model}_{cfg.cell}_k{cfg.k_terms}_s{cfg.seed}.png", dpi=150)
    plt.close()

if __name__ == "__main__": 
    main()