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
df["Cell"] = df["cell"].str.upper()
df["Activation"] = df["activation"].replace({
    "relu": "ReLU", 
    "surrogate_relu": "Surrogate ReLU"
})
df["K"] = df["k_terms"]

summary = df.groupby(["Framework", "Model", "Cell", "Activation", "K"]).agg(
    acc_mean=("acc", "mean"),
    acc_std=("acc", "std"),      
    time_mean=("time_s", "mean"),
    params=("params", "first"),
    seeds=("acc", "count")       
).reset_index()

# Нормировка времени ОТДЕЛЬНО ДЛЯ КАЖДОГО CELL и ФРЕЙМВОРКА
baseline_times = {}
for fw in df["Framework"].unique():
    for cell in df["Cell"].unique():
        base_df = df[(df["Framework"] == fw) & (df["Model"] == "Baseline") & (df["Cell"] == cell)]
        if not base_df.empty:
            baseline_times[(fw, cell)] = base_df["time_s"].mean()
        else:
            baseline_times[(fw, cell)] = df[(df["Framework"] == fw) & (df["Cell"] == cell)]["time_s"].mean()

def format_accuracy(row):
    mean_val = row["acc_mean"]
    std_val = row["acc_std"]
    if row["seeds"] == 1 or pd.isna(std_val):
        return f"{mean_val:.4f}"
    else:
        return f"{mean_val:.4f} ± {std_val:.4f}"

def format_time(row):
    fw = row["Framework"]
    cell = row["Cell"]
    b_time = baseline_times.get((fw, cell), row["time_mean"])
    rel_time = row["time_mean"] / b_time
    return f"{rel_time:.2f}x"

summary["Accuracy"] = summary.apply(format_accuracy, axis=1)
summary["Time"] = summary.apply(format_time, axis=1)
summary["Params"] = summary["params"].apply(lambda x: f"{int(round(x/1000))}K")

sort_fw = {"PyTorch": 1, "JAX": 2}
sort_md = {"Baseline": 1, "Manual": 2, "Auto": 3}
sort_cell = {"RNN": 1, "LSTM": 2, "GRU": 3}

summary["sort_fw"] = summary["Framework"].map(sort_fw)
summary["sort_md"] = summary["Model"].map(sort_md)
summary["sort_cell"] = summary["Cell"].map(sort_cell)

summary = summary.sort_values(["sort_fw", "sort_md", "sort_cell", "K"])

display_cols = ["Framework", "Model", "K", "Params", "Time", "Accuracy"]
final_df = summary[["Cell", "Activation"] + display_cols].copy()

for cell in ["RNN", "LSTM", "GRU"]:
    cell_df = final_df[final_df["Cell"] == cell]
    if cell_df.empty:
        continue
        
    print("\n" + "#"*75)
    print(f"{' ' * 32}CELL: {cell}")
    print("#"*75)
    
    for act in ["ReLU", "Surrogate ReLU"]:
        print("\n" + "-"*75)
        print(f"{' ' * 27}Activation: {act}")
        print("-"*75)
        
        act_df = cell_df[cell_df["Activation"] == act][display_cols]
        if not act_df.empty:
            print(act_df.to_string(index=False))
        else:
            print(f"No data found for {act} in {cell}.")

print("\n" + "#"*75 + "\n")

summary.drop(columns=["sort_fw", "sort_md", "sort_cell"]).to_csv("benchmark_results.csv", index=False)