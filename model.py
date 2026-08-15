"""
model.py
--------
Encoder: small 1D-CNN, fixed capacity -- this is what produces the
representation you'd actually use downstream (linear probe / fine-tune).

Projector: MLP head applied only during SSL pretraining, on top of the
encoder. This is the piece whose WIDTH you sweep. Per Garrido et al. 2023
(projector-vs-encoder collapse), most of the "helpful collapse" effect
lives in the projector, not the encoder -- so varying projector width is
your main structural knob, independent of the encoder itself.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Fixed-capacity 1D-CNN encoder. Output dim is fixed at 64 regardless
    of projector width, so downstream comparisons are apples-to-apples."""

    def __init__(self, in_channels=1, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out_dim = out_dim

    def forward(self, x):
        h = self.net(x)          # (B, out_dim, 1)
        return h.squeeze(-1)     # (B, out_dim)


class Projector(nn.Module):
    """MLP head. `width` is the swept variable (e.g. 64, 256, 1024)."""

    def __init__(self, in_dim=64, width=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class SSLModel(nn.Module):
    def __init__(self, in_channels=1, encoder_out=64, projector_width=256, projector_out=128):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, out_dim=encoder_out)
        self.projector = Projector(in_dim=encoder_out, width=projector_width,
                                    out_dim=projector_out)

    def forward(self, x):
        h = self.encoder(x)      # representation used downstream
        z = self.projector(h)    # embedding used for the SSL loss
        return h, z


if __name__ == "__main__":
    m = SSLModel(projector_width=256)
    x = torch.randn(8, 1, 140)
    h, z = m(x)
    print("encoder repr:", h.shape, "projector embedding:", z.shape)
    print("num params:", sum(p.numel() for p in m.parameters()))
