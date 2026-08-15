"""
augmentations.py
-----------------
Time-series augmentations parametrized by a single strength scalar alpha
in [0, 1]. This is the independent variable your whole study hinges on --
every other augmentation knob (jitter std, mask ratio, crop ratio) is
derived from alpha so a single number controls "how hard" the augmentation
is. This mirrors how Jing et al. / Cosentino et al. parametrize augmentation
strength in the vision literature.

Two independent views are generated per input series (view1, view2) --
standard for contrastive/non-contrastive SSL (SimCLR-style pair).
"""
import torch


def jitter(x, alpha):
    """Add Gaussian noise. alpha=0 -> no noise, alpha=1 -> std=0.5"""
    std = 0.5 * alpha
    return x + torch.randn_like(x) * std


def scaling(x, alpha):
    """Randomly scale amplitude. alpha=0 -> no scaling, alpha=1 -> +/-40%"""
    factor_range = 0.4 * alpha
    factor = 1.0 + (torch.rand(x.shape[0], 1, 1, device=x.device) * 2 - 1) * factor_range
    return x * factor


def time_mask(x, alpha):
    """Zero out a contiguous chunk of the series. alpha=0 -> no mask,
    alpha=1 -> up to 30% of series length masked."""
    if alpha <= 0:
        return x
    B, C, L = x.shape
    max_mask_len = int(L * 0.3 * alpha)
    if max_mask_len < 1:
        return x
    out = x.clone()
    for b in range(B):
        mask_len = torch.randint(1, max_mask_len + 1, (1,)).item()
        start = torch.randint(0, max(1, L - mask_len), (1,)).item()
        out[b, :, start:start + mask_len] = 0.0
    return out


def random_crop_resize(x, alpha):
    """Crop a sub-window and resize back to original length via interpolation.
    alpha=0 -> crop ratio ~1.0 (no-op), alpha=1 -> crop ratio down to 0.6"""
    B, C, L = x.shape
    min_ratio = 1.0 - 0.4 * alpha
    if min_ratio >= 0.999:
        return x
    ratio = torch.empty(1).uniform_(min_ratio, 1.0).item()
    crop_len = max(8, int(L * ratio))
    start = torch.randint(0, max(1, L - crop_len + 1), (1,)).item()
    cropped = x[:, :, start:start + crop_len]
    resized = torch.nn.functional.interpolate(
        cropped, size=L, mode="linear", align_corners=False
    )
    return resized


def augment(x, alpha):
    """Apply the full augmentation stack at a given strength."""
    x = random_crop_resize(x, alpha)
    x = time_mask(x, alpha)
    x = scaling(x, alpha)
    x = jitter(x, alpha)
    return x


def make_two_views(x, alpha):
    """Given a batch (B, 1, L), return two independently augmented views."""
    view1 = augment(x, alpha)
    view2 = augment(x, alpha)
    return view1, view2


if __name__ == "__main__":
    x = torch.randn(4, 1, 140)
    for a in [0.0, 0.3, 0.7, 1.0]:
        v1, v2 = make_two_views(x, a)
        diff = (v1 - v2).abs().mean().item()
        print(f"alpha={a:.1f}  mean |v1-v2|={diff:.4f}  shapes={v1.shape}")
