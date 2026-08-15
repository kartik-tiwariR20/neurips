"""
sweep.py
--------
Runs the full grid: alpha values x projector widths x seeds.
This is what produces your ~20-30 runs.

Default grid: 5 alphas x 2 widths x 2 seeds = 20 runs.
Bump `seeds` to [0,1,2] for 30 runs (5 x 2 x 3) if you have time/compute --
more seeds = more reliable correlation estimates in analyze.py, since a
single seed's erank trajectory can be noisy.
"""
import argparse
from train import run_one

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
PROJECTOR_WIDTHS = [64, 512]
SEEDS = [0, 1]


def run_sweep(epochs=50, out_dir="runs", alphas=None, widths=None, seeds=None):
    alphas = alphas or ALPHAS
    widths = widths or PROJECTOR_WIDTHS
    seeds = seeds or SEEDS

    total = len(alphas) * len(widths) * len(seeds)
    print(f"Running {total} configs: {len(alphas)} alphas x {len(widths)} widths x {len(seeds)} seeds")

    i = 0
    for alpha in alphas:
        for width in widths:
            for seed in seeds:
                i += 1
                print(f"\n--- run {i}/{total}: alpha={alpha} width={width} seed={seed} ---")
                run_one(alpha=alpha, projector_width=width, epochs=epochs,
                        seed=seed, out_dir=out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--out_dir", type=str, default="runs")
    p.add_argument("--quick", action="store_true",
                    help="tiny smoke-test grid (2 alphas x 1 width x 1 seed, 5 epochs)")
    args = p.parse_args()

    if args.quick:
        run_sweep(epochs=5, out_dir=args.out_dir, alphas=[0.0, 1.0],
                   widths=[64], seeds=[0])
    else:
        run_sweep(epochs=args.epochs, out_dir=args.out_dir)
