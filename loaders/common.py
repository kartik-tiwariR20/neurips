"""
loaders/common.py
------------------
Shared utilities used by every per-dataset loader. This is SHARED CODE --
never copy these functions into a per-dataset loader file, import them
instead. Keeping normalization/splitting logic in exactly one place is
what makes cross-dataset comparisons fair (same split strategy, same
normalization applied everywhere).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def normalize_per_series(X):
    """
    z-normalize each series independently, per channel.
    X shape: (N, C, L) -- normalizes over the L axis, per (sample, channel).
    Works for both univariate (C=1) and multivariate (C>1) data.
    """
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-8
    return (X - mean) / std


def make_splits(y, seed=42):
    """
    Stratified split used identically across every dataset:
      unlabeled_idx    -- everything (SSL pretraining pool, labels ignored)
      probe_train_idx  -- 60% of data, labeled, for downstream training
      probe_val_idx    -- 20%, for early stopping / model selection
      probe_test_idx   -- 20%, for final reported accuracy
    """
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.4, stratify=y, random_state=seed
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=seed
    )
    return {
        "unlabeled_idx": idx,
        "probe_train_idx": idx_train,
        "probe_val_idx": idx_val,
        "probe_test_idx": idx_test,
    }


class TimeSeriesDataset(Dataset):
    """
    Generic dataset wrapper. X must already be shaped (N, C, L) --
    each loader is responsible for producing that shape, whatever the
    dataset's raw format looks like.
    """

    def __init__(self, X, y, indices):
        self.X = X[indices]
        self.y = y[indices]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).float()   # (C, L) already
        y = int(self.y[i])
        return x, y
