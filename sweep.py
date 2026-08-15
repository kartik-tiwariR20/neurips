"""
sweep.py
--------
Runs the full grid: alpha values x projector widths x seeds, for ONE
named dataset. This is what produces your ~20-30 runs per dataset.

Default grid: 5 alphas x 2 widths x 3 seeds = 30 runs.
"""
import argparse
from train import run_one

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PROJECTOR_WIDTHS = [64, 512]
SEEDS = [0, 1, 2]


def run_sweep(dataset, epochs=50, out_dir=None, data_dir=None,
              alphas=None, widths=None, seeds=None):
    alphas = alphas or ALPHAS
    widths = widths or PROJECTOR_WIDTHS
    seeds = seeds or SEEDS

    total = len(alphas) * len(widths) * len(seeds)
    print(f"Dataset: {dataset}")
    print(f"Running {total} configs: {len(alphas)} alphas x {len(widths)} widths x {len(seeds)} seeds")

    i = 0
    for alpha in alphas:
        for width in widths:
            for seed in seeds:
                i += 1
                print(f"\n--- run {i}/{total}: dataset={dataset} alpha={alpha} width={width} seed={seed} ---")
                run_one(dataset=dataset, alpha=alpha, projector_width=width,
                        epochs=epochs, seed=seed, data_dir=data_dir, out_dir=out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out_dir", type=str, default=None,
                    help="default: results/<dataset>/runs")
    p.add_argument("--data_dir", type=str, default=None,
                    help="override the dataset's default raw data folder")
    p.add_argument("--quick", action="store_true",
                    help="tiny smoke-test grid (2 alphas x 1 width x 1 seed, 5 epochs)")
    args = p.parse_args()

    if args.quick:
        run_sweep(args.dataset, epochs=5, out_dir=args.out_dir, data_dir=args.data_dir,
                   alphas=[0.0, 1.0], widths=[64], seeds=[0])
    else:
        run_sweep(args.dataset, epochs=args.epochs, out_dir=args.out_dir, data_dir=args.data_dir)
