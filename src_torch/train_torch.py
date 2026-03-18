import hydra, time, json, os, torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, Callback
import matplotlib.pyplot as plt
from src_torch.data import DataModule
from src_torch.lit_module import CDELitModule

# Fix for Tensor Cores performance warning on modern GPUs (e.g. L40S, A100, RTX 30/40 series)
torch.set_float32_matmul_precision('medium')

class MetricsHistoryCallback(Callback):
    def __init__(self):
        self.train_losses = []
        self.val_accs = []
        
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None: self.train_losses.append(loss.item())
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return 
        acc = trainer.callback_metrics.get("val_acc")
        if acc is not None: self.val_accs.append(acc.item())

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    
    pl.seed_everything(cfg.seed)
    
    dm = DataModule(cfg.dataset, cfg.batch_size)
    cfg.input_dim = dm.inp_dim
    cfg.output_dim = dm.out_dim

    model = CDELitModule(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    history = MetricsHistoryCallback()
    trainer = Trainer(max_epochs=cfg.epochs, accelerator="gpu", devices=1, 
                      enable_checkpointing=False, logger=False, callbacks=[history])
    
    t0 = time.time()
    trainer.fit(model, datamodule=dm)
    t_train = time.time() - t0

    test_acc = trainer.test(model, datamodule=dm)[0]["test_acc"]

    res = {
        "framework": "torch", "model": cfg.model, "cell": cfg.cell, 
        "k_terms": cfg.k_terms, "seed": cfg.seed, "params": params, 
        "time_s": t_train, "acc": test_acc
    }
    
    os.makedirs("outputs", exist_ok=True)
    out_file = f"outputs/res_torch_{cfg.model}_{cfg.cell}_k{cfg.k_terms}_s{cfg.seed}.json"
    with open(out_file, "w") as f: 
        json.dump(res, f)

    # === PLOTTING ===
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Train Loss', color='tab:red')
    if history.train_losses:
        ax1.plot(history.train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Accuracy', color='tab:blue')
    if history.val_accs:
        ax2.plot(history.val_accs, color='tab:blue', label='Val Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title(f"PyTorch: {cfg.model} | {cfg.cell} | k={cfg.k_terms}")
    fig.tight_layout()
    plt.savefig(f"outputs/curves_torch_{cfg.model}_{cfg.cell}_k{cfg.k_terms}_s{cfg.seed}.png", dpi=150)
    plt.close()

if __name__ == "__main__": 
    main()