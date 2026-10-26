"""PushT one-step integrated visual-geometry/dynamics inference candidate.

Research candidate selected for integration from Task 3AD Validation, NOT a
held-out TEST result. Frozen V0.3 dynamics and frozen V0.4 yaw are combined
with TRAIN-fitted image-to-XY calibration and a TRAIN-locked contact cutoff.
Inputs are exclusively observed RGB(t-1), RGB(t), and normalized action[t].
No ground-truth MuJoCo state or future frame enters forward().

The contact classifier is a static endpoint-clearance proxy, not a contact
force solver or a step-wide collision detector. The hard gate suppresses XY
translation ONLY; it does not modify the model's yaw delta.
This module is CPU-reference first; MPS integration has not been tested.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.models.visual_dynamics_v03 import GatedVisualPhysicalWorldModel
from src.models.yaw_specialist_v04 import SilhouetteYawSpecialist


CONTACT_CUTOFF_M = 0.0008668007749070838
ACTION_SCALE_M = 0.02

# Locked Task 3W TRAIN fit: [orange_x, orange_y, yaw cos/sin, intercept].
_BLOCK_COEF = (
    (0.6126546663417821, -0.00012506057952043652),
    (0.0006953664528198849, -0.6133053992273941),
    (-0.00011428967058020137, -0.01884766533413837),
    (0.018735559011330907, 0.00028304836222149406),
    (-0.007181486581218202, 0.006929129105851575),
)

# Locked Task 3AB/3AC TRAIN fit: [weak-blue x/y, intercept].
_PUSHER_COEF = (
    (0.6084235331036187, -0.00009091005215001451),
    (-0.00008632859330547236, -0.6086769185214894),
    (-0.0071604635786151885, 0.007138690217254815),
)


class IntegratedPushTWorldModel(nn.Module):
    """CPU-reference one-step, fully image/action-conditioned predictor."""

    def __init__(self):
        super().__init__()
        self.dynamics = GatedVisualPhysicalWorldModel()
        self.yaw_model = SilhouetteYawSpecialist()
        self.register_buffer("block_coef", torch.tensor(_BLOCK_COEF, dtype=torch.float64))
        self.register_buffer("pusher_coef", torch.tensor(_PUSHER_COEF, dtype=torch.float64))
        self.register_buffer("contact_cutoff_m", torch.tensor(CONTACT_CUTOFF_M, dtype=torch.float64))
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def restore_frozen(
        cls,
        dynamics_checkpoint="data/processed/m4_visual_v03/best_model.pt",
        yaw_checkpoint="data/processed/m4_yaw_v04/best_model.pt",
    ):
        result = cls()
        for submodule, filename, count in (
            (result.dynamics, dynamics_checkpoint, 424345),
            (result.yaw_model, yaw_checkpoint, 231362),
        ):
            checkpoint = torch.load(Path(filename), map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                raise ValueError(f"Invalid frozen checkpoint: {filename}")
            submodule.load_state_dict(checkpoint["state_dict"], strict=True)
            if sum(p.numel() for p in submodule.parameters()) != count:
                raise ValueError(f"Wrong frozen parameter count: {filename}")
            for parameter in submodule.parameters():
                if not torch.isfinite(parameter).all().item():
                    raise ValueError(f"Nonfinite frozen parameter: {filename}")
        result.eval()
        result.requires_grad_(False)
        return result

    @staticmethod
    def _centroid(frames, kind):
        """Task 3AD centroid without labels; normalized image coordinates."""
        rgb = frames[:, 3:6]
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        if kind == "orange":
            mask = (r > g + 0.09) & (g > b + 0.04) & (r > 0.40)
            low, high = 6, 180
        elif kind == "weak_blue":
            mask = (b > r + 0.06) & (b > g + 0.04) & (b > 0.25)
            low, high = 1, 64
        else:
            raise ValueError("Unknown color segmentation")
        counts = mask.sum(dim=(1, 2))
        if ((counts < low) | (counts > high)).any().item():
            raise ValueError(f"Invalid or missing {kind} visual mask")
        weights = mask.to(torch.float64)
        counts = counts.to(torch.float64)
        grid = torch.arange(64, dtype=torch.float64, device=frames.device)
        cx = (weights.sum(dim=1) * grid).sum(dim=1) / counts
        cy = (weights.sum(dim=2) * grid).sum(dim=1) / counts
        return torch.stack(((cx - 31.5) / 32.0, (cy - 31.5) / 32.0), dim=1)

    @staticmethod
    def _static_clearance(pusher_endpoint, block_xy, block_cs):
        difference = pusher_endpoint - block_xy
        c, s = block_cs[:, 0], block_cs[:, 1]
        local_x = c * difference[:, 0] + s * difference[:, 1]
        local_y = -s * difference[:, 0] + c * difference[:, 1]
        gaps = []
        for center_y, half_x, half_y in (
            (0.045, 0.095, 0.025),
            (-0.020, 0.025, 0.060),
        ):
            dx = torch.clamp(local_x.abs() - half_x, min=0)
            dy = torch.clamp((local_y - center_y).abs() - half_y, min=0)
            gaps.append(torch.sqrt(dx.square() + dy.square()) - 0.025)
        return torch.minimum(gaps[0], gaps[1])

    def forward(self, frames, action):
        """Return integrated predictions; no labels or privileged state input.

        Args:
          frames: float32 RGB(t-1), RGB(t), [B, 6, 64, 64] in [0,1].
          action: float32 normalized XY displacement, [B, 2]; meters=action*.02.
        Returns:
          XY quantities in meters; delta_yaw in radians, orientation [cos,sin].
          next_contact is a boolean endpoint-contact prediction, NOT probability.
        """
        if not isinstance(frames, torch.Tensor) or not isinstance(action, torch.Tensor):
            raise TypeError("Only RGB and action tensors are accepted")
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (6, 64, 64):
            raise ValueError("frames must be [B,6,64,64]")
        if action.ndim != 2 or tuple(action.shape) != (frames.shape[0], 2):
            raise ValueError("action must be [B,2]")
        if frames.dtype != torch.float32 or action.dtype != torch.float32:
            raise TypeError("frames and action must be float32")
        if frames.shape[0] < 1 or not torch.isfinite(frames).all().item() or not torch.isfinite(action).all().item():
            raise ValueError("Empty/nonfinite input")
        if frames.device != action.device:
            raise ValueError("frames and action must share device")
        if frames.device.type != "cpu":
            raise ValueError("V1 integrated candidate is CPU-reference only")

        dyn = self.dynamics(frames, action)
        current_cs = self.yaw_model(frames)["current_yaw_cs"].to(torch.float64)
        n = frames.shape[0]
        ones = torch.ones((n, 1), dtype=torch.float64, device=frames.device)
        orange = self._centroid(frames, "orange")
        blue = self._centroid(frames, "weak_blue")
        current_xy = torch.cat((orange, current_cs, ones), dim=1) @ self.block_coef
        pusher_xy = torch.cat((blue, ones), dim=1) @ self.pusher_coef
        endpoint = pusher_xy + action.to(torch.float64) * ACTION_SCALE_M
        clearance_m = self._static_clearance(endpoint, current_xy, current_cs)
        score_m = -clearance_m
        next_contact = score_m >= self.contact_cutoff_m

        raw_delta = dyn["delta_xy"].to(torch.float64)
        if tuple(raw_delta.shape) != (n, 2):
            raise ValueError("Frozen dynamics delta_xy shape mismatch")
        delta_xy = torch.where(next_contact[:, None], raw_delta, torch.zeros_like(raw_delta))
        next_xy = current_xy + delta_xy
        delta_yaw = dyn["delta_yaw"].reshape(-1).to(torch.float64)
        if tuple(delta_yaw.shape) != (n,):
            raise ValueError("Frozen dynamics delta_yaw shape mismatch")
        cos_d, sin_d = torch.cos(delta_yaw), torch.sin(delta_yaw)
        next_cs = torch.stack((
            current_cs[:, 0] * cos_d - current_cs[:, 1] * sin_d,
            current_cs[:, 1] * cos_d + current_cs[:, 0] * sin_d,
        ), dim=1)
        next_cs = F.normalize(next_cs, p=2, dim=1)

        result = {
            "current_xy": current_xy,
            "next_xy": next_xy,
            "delta_xy": delta_xy,
            "raw_delta_xy": raw_delta,
            "current_yaw_cs": current_cs,
            "next_yaw_cs": next_cs,
            "delta_yaw": delta_yaw,
            "next_contact": next_contact,
            "contact_score_m": score_m,
            "pusher_xy": pusher_xy,
        }
        if not all(torch.isfinite(value).all().item() for value in result.values()):
            raise ValueError("Non-finite integrated prediction")
        return result
