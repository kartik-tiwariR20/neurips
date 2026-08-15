"""
loaders/ecg5000.py
-------------------
Loads ECG5000 (UCR format: label in col 0, 140 values after).
Returns data already shaped (N, 1, 140) -- single channel.

Expected files (place them here):
  data/raw/ecg5000/ECG5000_TRAIN.txt
  data/raw/ecg5000/ECG5000_TEST.txt
"""
import numpy as np
from .common import normalize_per_series, make_splits, TimeSeriesDataset

NUM_CLASSES = 5
RAW_DIR_DEFAULT = "data/raw/ecg5000"


def load(raw_dir=RAW_DIR_DEFAULT, seed=42):
    train = np.loadtxt(f"{raw_dir}/ECG5000_TRAIN.txt")
    test = np.loadtxt(f"{raw_dir}/ECG5000_TEST.txt")
    all_data = np.concatenate([train, test], axis=0)

    y = all_data[:, 0].astype(np.int64) - 1          # labels 1..5 -> 0..4
    X = all_data[:, 1:].astype(np.float32)            # (N, 140)
    X = X[:, np.newaxis, :]                            # (N, 1, 140) -- add channel dim
    X = normalize_per_series(X)

    splits = make_splits(y, seed=seed)
    return X, y, splits, NUM_CLASSES


def make_datasets(raw_dir=RAW_DIR_DEFAULT, seed=42):
    X, y, splits, num_classes = load(raw_dir, seed)
    unlabeled_ds = TimeSeriesDataset(X, y, splits["unlabeled_idx"])
    probe_train_ds = TimeSeriesDataset(X, y, splits["probe_train_idx"])
    probe_val_ds = TimeSeriesDataset(X, y, splits["probe_val_idx"])
    probe_test_ds = TimeSeriesDataset(X, y, splits["probe_test_idx"])
    in_channels = X.shape[1]
    return unlabeled_ds, probe_train_ds, probe_val_ds, probe_test_ds, num_classes, in_channels


if __name__ == "__main__":
    X, y, splits, num_classes = load()
    print("X shape:", X.shape, "num_classes:", num_classes)
    print("class counts:", np.bincount(y))
    for k, v in splits.items():
        print(k, len(v))
