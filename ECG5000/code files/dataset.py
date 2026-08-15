"""
dataset.py
----------
Loads ECG5000 (UCR format: label in col 0, 140 values after).
For SSL pretraining we IGNORE labels and use all 5000 series (train+test
combined) as one unlabeled pool -- that's normal practice for SSL, since
the point is to learn representations without supervision.

For downstream evaluation (linear probe / fine-tune) we carve out a
labeled train/val/test split from the same pool, using stratified
sampling so the rare classes (3, 4, 5) are represented in every split.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def load_ecg5000(raw_dir="data/raw"):
    """Returns X (N, 140) float32, y (N,) int64 in [0..4]."""
    train = np.loadtxt(f"{raw_dir}/ECG5000_TRAIN.txt")
    test = np.loadtxt(f"{raw_dir}/ECG5000_TEST.txt")
    all_data = np.concatenate([train, test], axis=0)
    y = all_data[:, 0].astype(np.int64) - 1  # labels are 1..5 -> 0..4
    X = all_data[:, 1:].astype(np.float32)   # (N, 140)
    return X, y


def normalize_per_series(X):
    """z-normalize each series independently (standard for UCR data)."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def make_splits(X, y, seed=42):
    """
    unlabeled_pool: everything, used for SSL pretraining (no labels used)
    probe_train / probe_val / probe_test: stratified split for downstream eval
    """
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.4, stratify=y, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=seed
    )
    return {
        "unlabeled_idx": idx,          # SSL pretraining pool (all data, no labels used)
        "probe_train_idx": idx_train,  # downstream: linear probe / fine-tune train
        "probe_val_idx": idx_val,      # downstream: early stopping / model selection
        "probe_test_idx": idx_test,    # downstream: final reported accuracy
    }


class ECGDataset(Dataset):
    """Returns a single-channel series shaped (1, 140) -> matches Conv1d input."""

    def __init__(self, X, y, indices):
        self.X = X[indices]
        self.y = y[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float().unsqueeze(0)  # (1, 140)
        y = int(self.y[i])
        return x, y


if __name__ == "__main__":
    X, y = load_ecg5000()
    X = normalize_per_series(X)
    splits = make_splits(X, y)
    print("Total series:", X.shape)
    print("Class counts (full pool):", np.bincount(y))
    for k, v in splits.items():
        print(k, len(v))
