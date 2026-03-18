import hydra, time, json, os, torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, Callback
import matplotlib.pyplot as plt
from src_torch.data import DataModule
from src_torch.lit_module import CDELitModule

# Fix for Tensor Cores performance warning
torch.set_float32_matmul_precision('medium')

class MetricsHistoryCallback(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.train_losses = []
        self.val_accs = []
        self.total_time = 0.0
        self.epoch_start_time = 0.0
        self.log_path = log_path
        
    def log_print(self, msg):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start_time = time.time()
        
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None: 
            self.train_losses.append(loss.item())
        self.total_time += time.time() - self.epoch_start_time
        
        if (trainer.current_epoch + 1) % 5 == 0 or trainer.current_epoch == 0:
            acc = trainer.callback_metrics.get("val_acc", 0.0)
            self.log_print(f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | Loss: {loss:.4f} | Val Acc: {acc:.4f}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return 
        acc = trainer.callback_metrics.get("val_acc")
        if acc is not None: 
            self.val_accs.append(acc.item())

    def state_dict(self):
        return {
            "train_losses": self.train_losses,
            "val_accs": self.val_accs,
            "total_time": self.total_time
        }

    def load_state_dict(self, state_dict):
        self.train_losses = state_dict.get("train_losses", [])
        self.val_accs = state_dict.get("val_accs", [])
        self.total_time = state_dict.get("total_time", 0.0)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    pl.seed_everything(cfg.seed)
    
    os.makedirs("outputs", exist_ok=True)
    
    file_prefix = f"torch_{cfg.model}_{cfg.cell}_k{cfg.k_terms}_s{cfg.seed}"
    out_file = f"outputs/res_{file_prefix}.json"
    log_path = f"outputs/log_{file_prefix}.txt"
    
    if os.path.exists(out_file):
        print(f"[PyTorch] Experiment {file_prefix} already completed. Skipping.")
        return

    dm = DataModule(cfg.dataset, cfg.batch_size)
    cfg.input_dim = dm.inp_dim
    cfg.output_dim = dm.out_dim

    model = CDELitModule(cfg)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ckpt_dir = f"checkpoints/{file_prefix}"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "last.ckpt")

    history = MetricsHistoryCallback(log_path)
    history.log_print(f"=== Starting Torch Training: {file_prefix} ===")
    history.log_print(f"Model parameters: {params:,}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        save_top_k=0,  
    )

    trainer = Trainer(
        max_epochs=cfg.epochs, 
        accelerator="gpu", 
        devices=1, 
        enable_checkpointing=True, 
        logger=False, 
        callbacks=[history, checkpoint_callback]
    )
    
    if os.path.exists(ckpt_path):
        history.log_print(f"Resuming from checkpoint: {ckpt_path}")
        trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    else:
        history.log_print(f"Starting new training...")
        trainer.fit(model, datamodule=dm)

    history.log_print("Evaluating on test set...")
    test_acc = trainer.test(model, datamodule=dm, ckpt_path=ckpt_path)[0]["test_acc"]
    history.log_print(f"Test Accuracy: {test_acc:.4f}")

    res = {
        "framework": "torch", 
        "model": cfg.model, 
        "cell": cfg.cell, 
        "k_terms": cfg.k_terms, 
        "seed": cfg.seed, 
        "params": params, 
        "time_s": history.total_time,  
        "acc": test_acc
    }
    
    with open(out_file, "w") as f: 
        json.dump(res, f)
    history.log_print(f"Results saved to {out_file}")

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
    plt.savefig(f"outputs/curves_{file_prefix}.png", dpi=150)
    plt.close()

if __name__ == "__main__": 
    main()