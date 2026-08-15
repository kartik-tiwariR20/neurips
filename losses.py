"""
losses.py
---------
VICReg loss (Bardes et al. 2022). Chosen deliberately over plain
contrastive (SimCLR/InfoNCE) losses for this study because it exposes
the collapse mechanism directly as an explicit loss term you can turn
up/down -- the variance term IS an anti-collapse regularizer, so it's
the natural loss to pair with an effective-rank study.

Three terms:
  invariance : views of the same input should map close together (MSE)
  variance   : each embedding dimension should have std >= 1 (anti-collapse)
  covariance : off-diagonal covariance terms pushed to 0 (decorrelation)
"""
import torch
import torch.nn.functional as F


def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0,
                 gamma=1.0, eps=1e-4):
    N, D = z1.shape

    # invariance
    sim_loss = F.mse_loss(z1, z2)

    # variance
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))

    # covariance
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    off_diag = lambda m: m.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()
    cov_loss = (off_diag(cov_z1).pow(2).sum() / D) + (off_diag(cov_z2).pow(2).sum() / D)

    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total, {
        "sim_loss": sim_loss.item(),
        "var_loss": var_loss.item(),
        "cov_loss": cov_loss.item(),
        "total_loss": total.item(),
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    z1 = torch.randn(64, 128)
    z2 = z1 + torch.randn(64, 128) * 0.1
    total, parts = vicreg_loss(z1, z2)
    print(parts)
