"""Trainable absolute-yaw specialist using current RGB silhouettes.

This PushT-specific experiment uses a fixed, label-free orange-object mask.
Its forward pass does not consume ground-truth states, yaw, contact, or
future images. A matching train-only template baseline is evaluated separately.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SilhouetteYawSpecialist(nn.Module):
    """Estimate current absolute yaw from RGB(t) inside [B, 6, 64, 64]."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    @staticmethod
    def extract_silhouette(frames: torch.Tensor):
        """Return centered binary 21x21 mask using only current RGB(t).

        Orange thresholds and 21x21 crop are frozen from the train-only
        Task 3Q observability probe. Missing/invalid foreground fails closed.
        The input is normalized float32 RGB(t-1), RGB(t).
        """
        if frames.ndim != 4 or frames.shape[1:] != (6, 64, 64):
            raise ValueError("frames must have shape [B, 6, 64, 64]")
        if frames.dtype != torch.float32:
            raise TypeError("frames must be float32")
        if frames.shape[0] == 0:
            raise ValueError("empty batch")
        if not torch.isfinite(frames).all().item():
            raise ValueError("non-finite RGB input")

        rgb = frames[:, 3:6]
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        mask = (r > g + 0.09) & (g > b + 0.04) & (r > 0.40)
        counts = mask.sum(dim=(1, 2))
        if ((counts < 6) | (counts > 180)).any().item():
            raise ValueError("orange silhouette pixel count outside [6, 180]")

        yy = torch.arange(64, device=frames.device, dtype=rgb.dtype)
        xx = torch.arange(64, device=frames.device, dtype=rgb.dtype)
        weights = mask.to(dtype=rgb.dtype)
        count_float = counts.to(dtype=rgb.dtype)

        cy = torch.round((weights.sum(dim=2) * yy).sum(dim=1) / count_float).to(torch.long)
        cx = torch.round((weights.sum(dim=1) * xx).sum(dim=1) / count_float).to(torch.long)

        offsets = torch.arange(21, device=frames.device) - 10
        sample_y = cy[:, None, None] + offsets[None, :, None]
        sample_x = cx[:, None, None] + offsets[None, None, :]
        sample_y = sample_y.expand(-1, 21, 21)
        sample_x = sample_x.expand(-1, 21, 21)
        valid = (sample_y >= 0) & (sample_y < 64) & (sample_x >= 0) & (sample_x < 64)
        index = sample_y.clamp(0, 63) * 64 + sample_x.clamp(0, 63)
        crop = mask.reshape(frames.shape[0], 64 * 64).gather(1, index.reshape(frames.shape[0], -1))
        crop = crop.reshape(-1, 21, 21) & valid
        return crop.unsqueeze(1).to(dtype=torch.float32), counts

    def forward(self, frames: torch.Tensor):
        silhouette, _ = self.extract_silhouette(frames)
        yaw_cs = F.normalize(self.network(silhouette), p=2, dim=1, eps=1e-6)
        return {"current_yaw_cs": yaw_cs}
