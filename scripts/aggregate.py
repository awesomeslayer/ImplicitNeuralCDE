# FILE: scripts/aggregate.py
import os, json
import pandas as pd

res = []
for f in os.listdir("outputs"):
    if f.endswith(".json"):
        with open(os.path.join("outputs", f)) as file:
            res.append(json.load(file))

df = pd.DataFrame(res)
summary = df.groupby(["framework", "model", "cell", "k_terms"]).agg(
    params=("params", "mean"),
    time=("time_s", "mean"),
    acc_mean=("acc", "mean"),
    acc_std=("acc", "std")
).reset_index()

summary["Accuracy"] = summary.apply(lambda x: f"{x['acc_mean']:.3f} ± {x['acc_std']:.3f}", axis=1)
print(summary.to_markdown())
summary.to_csv("benchmark_results.csv", index=False)