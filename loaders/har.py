"""
loaders/har.py
---------------
Loads UCI HAR from the raw "Inertial Signals" files (NOT the pre-flattened
561-feature X_train.txt -- that version has hand-engineered features and
throws away the raw signal shape we actually want for SSL).

Combines the 9 sensor channels (body_acc x/y/z, body_gyro x/y/z,
total_acc x/y/z) into one array shaped (N, 9, 128).

Expected folder layout (matches the official UCI HAR Dataset zip exactly):
  data/raw/har/train/Inertial Signals/body_acc_x_train.txt  (and 8 more)
  data/raw/har/train/y_train.txt
  data/raw/har/test/Inertial Signals/body_acc_x_test.txt   (and 8 more)
  data/raw/har/test/y_test.txt
"""
import numpy as np
from .common import normalize_per_series, make_splits, TimeSeriesDataset

NUM_CLASSES = 6
RAW_DIR_DEFAULT = "data/raw/har"

# fixed channel order -- keep this consistent, it doesn't matter which
# order as long as it's the same every time
SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _load_split(raw_dir, split):
    """split is 'train' or 'test'. Returns X (N, 9, 128), y (N,)."""
    channels = []
    for name in SIGNAL_NAMES:
        path = f"{raw_dir}/{split}/Inertial Signals/{name}_{split}.txt"
        sig = np.loadtxt(path)              # (N, 128)
        channels.append(sig)
    X = np.stack(channels, axis=1)          # (N, 9, 128)
    y = np.loadtxt(f"{raw_dir}/{split}/y_{split}.txt").astype(np.int64) - 1  # 1..6 -> 0..5
    return X.astype(np.float32), y


def load(raw_dir=RAW_DIR_DEFAULT, seed=42):
    X_train, y_train = _load_split(raw_dir, "train")
    X_test, y_test = _load_split(raw_dir, "test")
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

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
