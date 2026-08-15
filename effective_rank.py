"""
effective_rank.py
------------------
Implements the Roy & Vetterli (2007) effective rank -- the core metric
this whole study is built on.

    C = (1/N) sum_i (z_i - mean)(z_i - mean)^T        covariance
    lambda_1 >= ... >= lambda_d = eig(C)               eigenvalues
    p_i = lambda_i / sum_j lambda_j                     normalize
    erank(C) = exp( -sum_i p_i log p_i )                entropy of spectrum

erank -> 1   : near-total collapse (all variance in one direction)
erank -> d   : full-rank, space fully used
"""
import torch


@torch.no_grad()
def effective_rank(z, eps=1e-12):
    """
    z: (N, D) batch of embeddings (already detached / no grad needed).
    Returns a python float.
    """
    z = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    cov = (z.T @ z) / max(N - 1, 1)          # (D, D)
    eigvals = torch.linalg.eigvalsh(cov)      # ascending order, real (cov is symmetric)
    eigvals = torch.clamp(eigvals, min=0.0)   # numerical safety
    total = eigvals.sum()
    if total <= eps:
        return 1.0
    p = eigvals / total
    p = p[p > eps]                            # avoid log(0)
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy).item()


@torch.no_grad()
def eigen_spectrum(z, eps=1e-12):
    """Returns sorted (descending) eigenvalues of the covariance -- useful
    for plotting the full spectrum, not just the scalar summary."""
    z = z - z.mean(dim=0, keepdim=True)
    N = z.shape[0]
    cov = (z.T @ z) / max(N - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    return torch.flip(eigvals, dims=[0])      # descending


if __name__ == "__main__":
    torch.manual_seed(0)
    # full-rank case: independent gaussian dims
    z_full = torch.randn(500, 64)
    print("full-rank erank:", effective_rank(z_full), "/ max possible:", 64)

    # collapsed case: all dims are copies of dim 0 plus tiny noise
    base = torch.randn(500, 1)
    z_collapsed = base.repeat(1, 64) + torch.randn(500, 64) * 1e-4
    print("collapsed erank:", effective_rank(z_collapsed))
