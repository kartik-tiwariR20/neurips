"""
analyze.py
----------
Reads every run JSON in a runs/ directory and produces:

1. results_table.csv     -- one row per run, all configs + metrics
2. erank_vs_accuracy.png -- does higher/lower effective rank correlate
                             with better/worse downstream accuracy? (H1-ish)
3. alpha_vs_erank.png    -- does augmentation strength predict final
                             effective rank, as in vision? (core of H1)
4. early_prediction.png  -- scatter of early-epoch erank (epoch 5) vs
                             final downstream accuracy, plus a fitted
                             regression line and R^2 (this is H2: can we
                             predict the outcome without training to
                             convergence?)

Run this after sweep.py has populated a runs/ directory.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_runs(run_dir):
    rows = []
    for path in glob.glob(os.path.join(run_dir, "*.json")):
        with open(path) as f:
            r = json.load(f)
        rows.append({
            "dataset": r["config"].get("dataset", "unknown"),
            "alpha": r["config"]["alpha"],
            "projector_width": r["config"]["projector_width"],
            "seed": r["config"]["seed"],
            "final_erank": r["final_erank"],
            "early_erank_5ep": r["early_erank_5ep"],
            "linear_probe_test_acc": r["downstream"]["linear_probe_test_acc"],
            "finetune_test_acc": r["downstream"]["finetune_test_acc"],
        })
    if not rows:
        raise FileNotFoundError(f"No run JSON files found in {run_dir}. Run sweep.py first.")
    return pd.DataFrame(rows).sort_values(["projector_width", "alpha", "seed"])


def plot_alpha_vs_erank(df, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for width, g in df.groupby("projector_width"):
        agg = g.groupby("alpha")["final_erank"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["alpha"], agg["mean"], yerr=agg["std"], marker="o",
                    label=f"projector width={width}", capsize=3)
    ax.set_xlabel("Augmentation strength (alpha)")
    ax.set_ylabel("Final effective rank")
    ax.set_title("Augmentation strength vs. effective rank at convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_erank_vs_accuracy(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, acc_col, title in zip(
        axes,
        ["linear_probe_test_acc", "finetune_test_acc"],
        ["Linear probe accuracy", "Fine-tune accuracy"],
    ):
        ax.scatter(df["final_erank"], df[acc_col], c=df["alpha"], cmap="viridis")
        r, p = stats.pearsonr(df["final_erank"], df[acc_col])
        ax.set_xlabel("Final effective rank")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. effective rank\n(r={r:.2f}, p={p:.3f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def early_prediction_test(df, out_path):
    """Regress final downstream accuracy on the EARLY (epoch-5) effective
    rank only -- this tests H2: can we predict the outcome before training
    to convergence?"""
    x = df["early_erank_5ep"].values
    y = df["finetune_test_acc"].values
    slope, intercept, r, p, se = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, c=df["alpha"], cmap="viridis")
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r--",
            label=f"fit: R^2={r**2:.2f}, p={p:.3f}")
    ax.set_xlabel("Effective rank at epoch 5 (early)")
    ax.set_ylabel("Final fine-tune accuracy")
    ax.set_title("Early-prediction test (H2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"r_squared": r ** 2, "p_value": p, "slope": slope}


def main(run_dir, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    df = load_runs(run_dir)
    df.to_csv(os.path.join(fig_dir, "results_table.csv"), index=False)
    print(df.to_string(index=False))

    plot_alpha_vs_erank(df, os.path.join(fig_dir, "alpha_vs_erank.png"))
    plot_erank_vs_accuracy(df, os.path.join(fig_dir, "erank_vs_accuracy.png"))
    stats_out = early_prediction_test(df, os.path.join(fig_dir, "early_prediction.png"))

    print("\n--- Early-prediction test (H2) ---")
    print(stats_out)
    print(f"\nFigures + table written to {fig_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None,
                    help="if set, defaults run_dir/fig_dir to results/<dataset>/...")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--fig_dir", type=str, default=None)
    args = p.parse_args()

    run_dir = args.run_dir
    fig_dir = args.fig_dir
    if args.dataset is not None:
        run_dir = run_dir or os.path.join("results", args.dataset, "runs")
        fig_dir = fig_dir or os.path.join("results", args.dataset, "figures")
    run_dir = run_dir or "runs"
    fig_dir = fig_dir or "figures"

    main(run_dir, fig_dir)
