# scripts/libsvm/plot_mushrooms_table1.py
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "src" / "libsvm" / "results_raw"

METHODS = [
    "normalized_sgd", 
    "signsgd", 
    # "muon", 
    # "scion",
    "kfac",
    # "shampoo",
    # "onesided_shampoo",
    # "kl_shampoo",
    # "adagrad",
    "adam",
    "madgrad",
    # "adam_sania",
    "soap",
    "splus",
    "muadam",
    # "muadam_sania",
]
_cmap = mpl.cm.get_cmap("tab20", len(METHODS))
METHOD2COLOR = {m: _cmap(i) for i, m in enumerate(METHODS)}

def load_curves():
    # Expect files like: <run_name>__seed-<seed>.json
    files = list(RESULTS_DIR.glob("*.json"))
    data = []
    for f in files:
        try:
            rec = json.loads(f.read_text())
            rec["_path"] = str(f)
            data.append(rec)
        except Exception:
            pass
    return data

def group_mean_nanpad(curves_list):
    curves = [np.asarray(c, dtype=float).ravel() for c in curves_list]
    L = max(len(c) for c in curves)
    arr = np.full((len(curves), L), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, :len(c)] = c
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)

def plot_one_scale(by_method_scale, scale: str):
    fig = plt.figure(figsize=(14, 4.8))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    ax1.set_title(f"Train Loss ({scale})")
    ax2.set_title(f"Test Accuracy ({scale})")
    ax1.set_xlabel("Epochs")
    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy (%)")

    # Make x ticks integers
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Optional: sort methods by final test acc (mean) to make legend meaningful
    method_order = []
    for method in METHODS:
        runs = by_method_scale.get((method, scale), [])
        if not runs:
            continue
        test_curves = [r["test_accuracy"] for r in runs]
        test_mean, _ = group_mean_nanpad(test_curves)
        method_order.append((float(test_mean[-1]), method))
    method_order.sort(reverse=True)  # best last value first
    method_order = [m for _, m in method_order]

    for method in method_order:
        runs = by_method_scale.get((method, scale), [])
        if not runs:
            continue

        train_curves = [r["train_loss"] for r in runs]
        test_acc_curves = [r["test_accuracy"] for r in runs]

        train_mean, _ = group_mean_nanpad(train_curves)
        test_mean, _ = group_mean_nanpad(test_acc_curves)

        epochs = np.arange(1, len(train_mean) + 1)

        color = METHOD2COLOR[method]
        ax1.plot(
            epochs, train_mean,
            color=color,
            marker="o",
            markevery=max(1, len(epochs)//8),
            label=method,
        )
        
        ax2.plot(
            epochs, 100.0 * test_mean,
            color=color,
            marker="o",
            markevery=max(1, len(epochs)//8),
            label=method,
        )

    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)

    plt.tight_layout()
    out = RESULTS_DIR / f"mushrooms_table1_curves_{scale}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved:", out)

def main():
    all_data = load_curves()

    by_method_scale = defaultdict(list)
    for rec in all_data:
        run_name = rec.get("run_name", "")
        m = re.search(r"tbl1_([a-zA-Z0-9_]+)_(orig|scaled)", run_name)
        if not m:
            continue
        method, scale = m.group(1), m.group(2)
        if method not in METHODS:
            continue
        if rec.get("train_loss") is None or rec.get("test_accuracy") is None:
            continue
        by_method_scale[(method, scale)].append(rec)

    plot_one_scale(by_method_scale, "orig")    # orig vs orig
    plot_one_scale(by_method_scale, "scaled")  # scaled vs scaled

if __name__ == "__main__":
    main()
