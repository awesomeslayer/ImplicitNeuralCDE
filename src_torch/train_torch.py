import hydra, time, json, os, torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, Callback
from src_torch.data import DataModule
from src_torch.lit_module import CDELitModule

torch.set_float32_matmul_precision('medium')

class MetricsHistoryCallback(Callback):
    def __init__(self, log_path):
        super().__init__()
        self.history = {
            "train_losses": [],
            "val_accs": [],
            "train_spec_rads": [],
            "val_spec_rads": []
        }
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
        tr_rad = trainer.callback_metrics.get("train_spec_rad")
        
        if loss is not None: self.history["train_losses"].append(loss.item())
        if tr_rad is not None: self.history["train_spec_rads"].append(tr_rad.item())
        
        self.total_time += time.time() - self.epoch_start_time

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking: return 
        
        acc = trainer.callback_metrics.get("val_acc")
        val_rad = trainer.callback_metrics.get("val_spec_rad")
        loss = trainer.callback_metrics.get("train_loss", 0.0)
        
        if acc is not None: self.history["val_accs"].append(acc.item())
        if val_rad is not None: self.history["val_spec_rads"].append(val_rad.item())
            
        if (trainer.current_epoch + 1) % 5 == 0 or trainer.current_epoch == 0:
            tr_r = self.history["train_spec_rads"][-1] if self.history["train_spec_rads"] else 0.0
            val_r = val_rad.item() if val_rad is not None else 0.0
            self.log_print(f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | Loss: {loss:.4f} | Val Acc: {acc:.4f} | Rho(Train): {tr_r:.4f} | Rho(Val): {val_r:.4f}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    pl.seed_everything(cfg.seed)
    
    os.makedirs("outputs", exist_ok=True)
    
    file_prefix = f"torch_{cfg.model}_{cfg.cell}_{cfg.activation}_k{cfg.k_terms}_s{cfg.seed}"
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
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    best_ckpt = os.path.join(ckpt_dir, "best.ckpt")

    history = MetricsHistoryCallback(log_path)
    history.log_print(f"=== Starting Torch Training: {file_prefix} ===")
    history.log_print(f"Model parameters: {params:,}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir, monitor="val_acc", mode="max", filename="best", save_last=True, save_top_k=1
    )

    trainer = Trainer(
        max_epochs=cfg.epochs, accelerator="gpu", devices=1, 
        enable_checkpointing=True, logger=False, 
        callbacks=[history, checkpoint_callback], inference_mode=False, fast_dev_run=cfg.smoke_test
    )
    
    if os.path.exists(last_ckpt):
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)

    history.log_print(f"Evaluating on test set using BEST model...")
    if os.path.exists(best_ckpt):
        test_acc = trainer.test(model, datamodule=dm, ckpt_path=best_ckpt)[0]["test_acc"]
    else:
        test_acc = trainer.test(model, datamodule=dm)[0]["test_acc"]
        
    history.log_print(f"Test Accuracy: {test_acc:.4f}")

    # СОХРАНЯЕМ В JSON: Метрики классификации + История массивов
    res = {
        "framework": "torch", 
        "model": cfg.model, 
        "cell": cfg.cell, 
        "k_terms": cfg.k_terms, 
        "activation": cfg.activation,
        "seed": cfg.seed, 
        "params": params, 
        "time_s": history.total_time,  
        "acc": test_acc,
        "history": history.history  # <--- Теперь тут массивы loss, acc, radius
    }
    
    with open(out_file, "w") as f: 
        json.dump(res, f, indent=4)
    history.log_print(f"Results and history saved to {out_file}")

if __name__ == "__main__": 
    main()