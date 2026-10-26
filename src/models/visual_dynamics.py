"""Visual, action-conditioned physical dynamics for ActionWorld Lite M4.

Inference reads only two RGB frames and one action. Simulator states are
supervision targets, never arguments to this model.
"""

import torch
from torch import nn
from torch.nn import functional as F


class VisualPhysicalWorldModel(nn.Module):
    """Predict block pose, motion and next-step contact from pixels/actions.

    frames: [B, 6, 64, 64] float32 (two RGB frames, range [0, 1]).
    action: [B, 2] float32 (displacement divided by 0.02 m).

    Angles use (cos(yaw), sin(yaw)) to respect periodicity.
    Deltas are one control interval (0.05 seconds), not a full rollout.
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(),
        )

        self.current_pose_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

        self.action_dynamics_head = nn.Sequential(
            nn.Linear(128 + 2, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, frames: torch.Tensor, action: torch.Tensor) -> dict:
        if frames.ndim != 4 or frames.shape[1:] != (6, 64, 64):
            raise ValueError("frames must have shape [B, 6, 64, 64]")
        if action.ndim != 2 or action.shape != (frames.shape[0], 2):
            raise ValueError("action must have shape [B, 2]")
        if frames.dtype != torch.float32 or action.dtype != torch.float32:
            raise TypeError("frames and action must be float32")
        if frames.device != action.device:
            raise ValueError("frames and action must be on the same device")

        visual_features = self.encoder(frames)

        current_pose = self.current_pose_head(visual_features)
        current_xy = current_pose[:, :2]
        current_yaw_cs = F.normalize(
            current_pose[:, 2:4], p=2, dim=-1, eps=1e-6
        )

        motion_features = torch.cat(
            (visual_features, action), dim=-1
        )
        motion_prediction = self.action_dynamics_head(motion_features)

        # Scale unconstrained predictions into convenient physical units.
        # This does not clip motion or constrain the learned dynamics.
        delta_xy = motion_prediction[:, :2] * 0.02
        delta_yaw = motion_prediction[:, 2] * 0.1
        next_contact_logits = motion_prediction[:, 3]

        cos_delta = torch.cos(delta_yaw)
        sin_delta = torch.sin(delta_yaw)
        current_cos = current_yaw_cs[:, 0]
        current_sin = current_yaw_cs[:, 1]

        next_yaw_cs = torch.stack(
            (
                current_cos * cos_delta - current_sin * sin_delta,
                current_sin * cos_delta + current_cos * sin_delta,
            ),
            dim=-1,
        )

        return {
            "current_xy": current_xy,
            "current_yaw_cs": current_yaw_cs,
            "delta_xy": delta_xy,
            "delta_yaw": delta_yaw,
            "next_xy": current_xy + delta_xy,
            "next_yaw_cs": next_yaw_cs,
            "next_contact_logits": next_contact_logits,
        }

    def parameter_count(self) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if p.requires_grad
        )
