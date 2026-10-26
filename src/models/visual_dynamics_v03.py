"""V0.3 visual physical dynamics with a learned motion-activity gate.

Inputs remain RGB(t-1), RGB(t), action(t). All future state and
motion-activity labels are supervision only, never inference inputs.
"""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.visual_dynamics import VisualPhysicalWorldModel


class GatedVisualPhysicalWorldModel(VisualPhysicalWorldModel):
    """Predict motion conditioned on learned probability of moving.

    frames: [B, 6, 64, 64] float32 in [0, 1]
    action: [B, 2] float32; actions[t] normalized by 0.02 m

    Motion activity is learned as an auxiliary label during training.
    Forward inference uses only frames and action, never simulator state.
    """

    def __init__(self):
        super().__init__()
        self.motion_gate_head = nn.Sequential(
            nn.Linear(130, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.constant_(self.motion_gate_head[-1].bias, -1.0)

    def forward(self, frames: torch.Tensor, action: torch.Tensor):
        if frames.ndim != 4 or frames.shape[1:] != (6, 64, 64):
            raise ValueError("frames must have shape [B, 6, 64, 64]")
        if action.ndim != 2 or action.shape != (frames.shape[0], 2):
            raise ValueError("action must have shape [B, 2]")
        if frames.dtype != torch.float32 or action.dtype != torch.float32:
            raise TypeError("frames and action must be float32")
        if frames.device != action.device:
            raise ValueError("frames and action must be on the same device")

        features = self.encoder(frames)
        pose = self.current_pose_head(features)
        current_xy = pose[:, :2]
        current_yaw_cs = F.normalize(pose[:, 2:4], p=2, dim=-1, eps=1e-6)

        conditioned = torch.cat((features, action), dim=-1)
        dynamics = self.action_dynamics_head(conditioned)
        raw_delta_xy = dynamics[:, :2] * 0.02
        raw_delta_yaw = dynamics[:, 2] * 0.1
        next_contact_logits = dynamics[:, 3]

        motion_logits = self.motion_gate_head(conditioned).squeeze(-1)
        motion_probability = torch.sigmoid(motion_logits)
        delta_xy = motion_probability.unsqueeze(-1) * raw_delta_xy
        delta_yaw = motion_probability * raw_delta_yaw

        c = torch.cos(delta_yaw)
        s = torch.sin(delta_yaw)
        next_yaw_cs = torch.stack(
            (
                current_yaw_cs[:, 0] * c - current_yaw_cs[:, 1] * s,
                current_yaw_cs[:, 1] * c + current_yaw_cs[:, 0] * s,
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
            "motion_logits": motion_logits,
            "motion_probability": motion_probability,
            "raw_delta_xy": raw_delta_xy,
            "raw_delta_yaw": raw_delta_yaw,
        }
