# FILE: scripts/replot.py
import os
import json
import matplotlib.pyplot as plt

def plot_all():
    out_dir = "outputs"
    if not os.path.exists(out_dir):
        return

    for f in os.listdir(out_dir):
        if not f.endswith(".json"):
            continue

        filepath = os.path.join(out_dir, f)
        with open(filepath, "r") as file:
            try:
                data = json.load(file)
            except:
                continue
        
        if "history" not in data:
            continue
            
        history = data["history"]
        model = data.get("model", "unknown")
        cell = data.get("cell", "unknown")
        k = data.get("k_terms", "?")
        
        train_losses = history.get("train_losses", [])
        val_accs = history.get("val_accs", [])
        train_rads = history.get("train_spec_rads", [])
        val_rads = history.get("val_spec_rads", [])
        
        has_radius = len(train_rads) > 0 or len(val_rads) > 0
        
        if train_losses and val_accs:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Train Loss', color='tab:red')
            ax1.plot(train_losses, color='tab:red', label='Train Loss')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            ax2 = ax1.twinx()
            ax2.set_ylabel('Validation Accuracy', color='tab:blue')
            ax2.plot(val_accs, color='tab:blue', label='Val Acc')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            ax1.set_title(f"{model} | {cell.upper()} | K={k}")
            fig.tight_layout()
            
            plot_filename = f.replace("res_", "curves_").replace(".json", ".png")
            plt.savefig(os.path.join(out_dir, plot_filename), dpi=150)
            plt.close()
            print(f"Generated plot: {plot_filename}")

        if has_radius:
            fig, ax3 = plt.subplots(figsize=(8, 5))
            
            ax3.set_xlabel('Epochs')
            ax3.set_ylabel('Spectral Radius ρ(Jh)')
            if train_rads: ax3.plot(train_rads, color='tab:orange', label='Train ρ')
            if val_rads: ax3.plot(val_rads, color='tab:green', label='Val ρ')
                
            ax3.axhline(1.0, color='red', linestyle='--', label='Limit (ρ=1)') 
            ax3.set_title(f"Taylor Series Convergence | {cell.upper()} | K={k}")
            ax3.legend()

            fig.tight_layout()
            radius_filename = f.replace("res_", "radius_").replace(".json", ".png")
            plt.savefig(os.path.join(out_dir, radius_filename), dpi=150)
            plt.close()
            print(f"Generated RADIUS plot: {radius_filename}")

if __name__ == "__main__":
    plot_all()