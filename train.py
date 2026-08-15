"""
train.py
--------
One SSL pretraining run: fixed alpha (augmentation strength) and
projector_width, trained for `epochs`, logging effective rank of the
PROJECTOR output every epoch (this is where collapse is expected to show
up most strongly per Garrido et al. 2023).

Works for ANY registered dataset via --dataset -- the shared code below
never changes per dataset. See loaders/registry.py to add a new one.

After pretraining, runs downstream evaluation (linear probe + fine-tune)
and saves everything -- config, per-epoch effective-rank trajectory,
final downstream accuracy -- to a single JSON file per run. This is the
file `analyze.py` later reads across all runs.
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from loaders.registry import get_loader
from augmentations import make_two_views
from model import SSLModel
from losses import vicreg_loss
from effective_rank import effective_rank
from evaluate import evaluate_representation


def run_one(dataset, alpha, projector_width, epochs=50, batch_size=64, lr=1e-3,
            seed=0, device=None, data_dir=None, out_dir=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # ---- data ----
    loader_module = get_loader(dataset)
    kwargs = {"seed": seed}
    if data_dir is not None:
        kwargs["raw_dir"] = data_dir
    (unlabeled_ds, probe_train_ds, probe_val_ds, probe_test_ds,
     num_classes, in_channels) = loader_module.make_datasets(**kwargs)

    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---- model ----
    model = SSLModel(in_channels=in_channels, encoder_out=64,
                      projector_width=projector_width, projector_out=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    erank_trajectory = []
    loss_trajectory = []

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        epoch_eranks = []
        epoch_losses = []
        for x, _ in unlabeled_loader:
            x = x.to(device)
            v1, v2 = make_two_views(x, alpha)
            h1, z1 = model(v1)
            h2, z2 = model(v2)
            loss, parts = vicreg_loss(z1, z2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                er = effective_rank(z1.detach())
            epoch_eranks.append(er)
            epoch_losses.append(parts["total_loss"])

        mean_erank = sum(epoch_eranks) / len(epoch_eranks)
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        erank_trajectory.append(mean_erank)
        loss_trajectory.append(mean_loss)

    train_time = time.time() - t0

    # ---- downstream evaluation ----
    downstream = evaluate_representation(
        model.encoder, num_classes=num_classes,
        train_ds=probe_train_ds, val_ds=probe_val_ds, test_ds=probe_test_ds,
        device=device,
    )

    result = {
        "config": {
            "dataset": dataset,
            "alpha": alpha,
            "projector_width": projector_width,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
            "in_channels": in_channels,
            "num_classes": num_classes,
        },
        "erank_trajectory": erank_trajectory,
        "loss_trajectory": loss_trajectory,
        "final_erank": erank_trajectory[-1],
        "early_erank_5ep": erank_trajectory[min(4, len(erank_trajectory) - 1)],
        "train_time_sec": train_time,
        "downstream": downstream,
    }

    out_dir = out_dir or os.path.join("results", dataset, "runs")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"alpha{alpha:.2f}_proj{projector_width}_seed{seed}"
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[done] {dataset}:{tag}  final_erank={result['final_erank']:.2f}  "
          f"linear_acc={downstream['linear_probe_test_acc']:.3f}  "
          f"finetune_acc={downstream['finetune_test_acc']:.3f}  "
          f"({train_time:.1f}s)")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                    help="registered dataset name, e.g. ecg5000, har")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--projector_width", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_dir", type=str, default=None,
                    help="override the dataset's default raw data folder")
    p.add_argument("--out_dir", type=str, default=None,
                    help="default: results/<dataset>/runs")
    args = p.parse_args()
    run_one(args.dataset, args.alpha, args.projector_width, epochs=args.epochs,
            seed=args.seed, data_dir=args.data_dir, out_dir=args.out_dir)
