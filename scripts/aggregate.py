# FILE: scripts/aggregate.py
import os
import json
import pandas as pd

res = []
for f in os.listdir("outputs"):
    if f.endswith(".json"):
        with open(os.path.join("outputs", f)) as file:
            data = json.load(file)
            if "activation" not in data:
                if "surrogate_relu" in f:
                    data["activation"] = "surrogate_relu"
                else:
                    data["activation"] = "relu"
            res.append(data)

if not res:
    print("No JSON files found in outputs/")
    exit()

df = pd.DataFrame(res)

fw_map = {"torch": "PyTorch", "jax": "JAX"}
model_map = {
    "torch_baseline": "Baseline",
    "jax_baseline": "Baseline",
    "torch_manual": "Manual",
    "jax_manual": "Manual",
    "torch_auto": "Auto",
    "jax_auto": "Auto"
}

df["Framework"] = df["framework"].map(fw_map)
df["Model"] = df["model"].map(model_map)
df["Activation"] = df["activation"].replace({
    "relu": "ReLU", 
    "surrogate_relu": "Surrogate ReLU"
})
df["K"] = df["k_terms"]

summary = df.groupby(["Framework", "Model", "Activation", "K"]).agg(
    acc_mean=("acc", "mean"),
    acc_std=("acc", "std"),      
    params=("params", "first"),
    seeds=("acc", "count")       
).reset_index()

def format_accuracy(row):
    mean_val = row["acc_mean"]
    std_val = row["acc_std"]
    if row["seeds"] == 1 or pd.isna(std_val):
        return f"{mean_val:.4f}"
    else:
        return f"{mean_val:.4f} ± {std_val:.4f}"

summary["Accuracy"] = summary.apply(format_accuracy, axis=1)

summary["Params"] = summary["params"].apply(lambda x: f"{int(x):,}")

sort_fw = {"PyTorch": 1, "JAX": 2}
sort_md = {"Baseline": 1, "Manual": 2, "Auto": 3}

summary["sort_fw"] = summary["Framework"].map(sort_fw)
summary["sort_md"] = summary["Model"].map(sort_md)

summary = summary.sort_values(["sort_fw", "sort_md", "K"])

display_cols = ["Framework", "Model", "K", "Params", "Accuracy"]
final_df = summary[["Activation"] + display_cols].copy()

print("\n" + "="*55)
print(" " * 20 + "RESULTS: ReLU")
print("="*55)
relu_df = final_df[final_df["Activation"] == "ReLU"][display_cols]
if not relu_df.empty:
    print(relu_df.to_string(index=False))
else:
    print("No data found for ReLU.")

print("\n" + "="*55)
print(" " * 15 + "RESULTS: Surrogate ReLU")
print("="*55)
surr_df = final_df[final_df["Activation"] == "Surrogate ReLU"][display_cols]
if not surr_df.empty:
    print(surr_df.to_string(index=False))
else:
    print("No data found for Surrogate ReLU.")
print("="*55 + "\n")

summary.drop(columns=["sort_fw", "sort_md"]).to_csv("benchmark_results.csv", index=False)