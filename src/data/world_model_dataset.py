"""Episode-safe visual-action training windows for ActionWorld Lite M4."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/m3_main_v01"
PROCESSED = ROOT / "data/processed/m3_main_v01"

MANIFEST_PATH = RAW / "manifest.json"
SPLITS_PATH = PROCESSED / "splits.json"

MANIFEST_SHA256 = (
    "2265fbf383983e7e93507ea31569933ddd79b4d050be02ea9f47c01407dc9f50"
)

SPLITS_SHA256 = (
    "af1795bc99d107084b3fa7885824d48f697a166b7be6be1af333b1bc8f8845aa"
)

SPLIT_COUNTS = {
    "train": 126,
    "val": 27,
    "test": 27,
}

MODE_COUNTS = {
    "train": 42,
    "val": 9,
    "test": 9,
}

MODES = ("center_push", "offset_push", "free_motion")
STEPS = 64
WINDOWS_PER_EPISODE = STEPS - 1
ACTION_SCALE = 0.02
DT = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ActionWorldWindowDataset(Dataset):
    """Visual-only observations plus actions; state labels for supervision."""

    def __init__(self, split: str):
        if split not in SPLIT_COUNTS:
            raise ValueError(f"Unknown split: {split}")

        if sha256(MANIFEST_PATH) != MANIFEST_SHA256:
            raise RuntimeError("Unexpected main manifest hash")

        if sha256(SPLITS_PATH) != SPLITS_SHA256:
            raise RuntimeError("Unexpected frozen split hash")

        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
        split_data = json.loads(
            SPLITS_PATH.read_text(encoding="utf-8")
        )

        if manifest["schema_version"] != "m3-main-v0.1":
            raise RuntimeError("Unexpected dataset schema")

        if split_data["schema_version"] != "m3-main-split-v0.1":
            raise RuntimeError("Unexpected split schema")

        if split_data["source_manifest_sha256"] != MANIFEST_SHA256:
            raise RuntimeError("Split provenance mismatch")

        entries = split_data["splits"][split]

        if len(entries) != SPLIT_COUNTS[split]:
            raise RuntimeError("Incorrect episode count")

        counts = Counter(entry["mode"] for entry in entries)

        if any(
            counts[mode] != MODE_COUNTS[split] for mode in MODES
        ):
            raise RuntimeError("Incorrect mode distribution")

        manifest_lookup = {
            item["file"]: item
            for item in manifest["items"]
        }

        self.split = split
        self.episodes = []
        self.windows = []
        seen = set()

        for entry in entries:
            filename = entry["file"]
            mode = entry["mode"]

            if filename in seen:
                raise RuntimeError("Repeated episode in split")

            seen.add(filename)

            if filename not in manifest_lookup:
                raise RuntimeError("Episode missing from manifest")

            if entry["sha256"] != manifest_lookup[filename]["sha256"]:
                raise RuntimeError("Episode provenance mismatch")

            path = RAW / filename

            if sha256(path) != entry["sha256"]:
                raise RuntimeError(f"Corrupt episode: {filename}")

            with np.load(path, allow_pickle=False) as archive:
                states = np.array(archive["states"], copy=True)
                actions = np.array(archive["actions"], copy=True)
                rgb = np.array(archive["rgb"], copy=True)

            if states.shape != (65, 14):
                raise RuntimeError("Invalid state shape")

            if actions.shape != (64, 2):
                raise RuntimeError("Invalid action shape")

            if rgb.shape != (65, 64, 64, 3):
                raise RuntimeError("Invalid RGB shape")

            if rgb.dtype != np.uint8:
                raise RuntimeError("Invalid RGB dtype")

            if not np.isfinite(states).all():
                raise RuntimeError("Nonfinite states")

            if not np.isfinite(actions).all():
                raise RuntimeError("Nonfinite actions")

            if not np.allclose(
                states[:, 0],
                np.arange(65) * DT,
                rtol=0,
                atol=1e-10,
            ):
                raise RuntimeError("Invalid state timestamps")

            if np.any(
                np.linalg.norm(actions, axis=1)
                > ACTION_SCALE + 1e-10
            ):
                raise RuntimeError("Action exceeds expected scale")

            if not np.all(
                np.isin(states[:, 13], (0.0, 1.0))
            ):
                raise RuntimeError("Invalid contact label")

            episode_index = len(self.episodes)

            self.episodes.append({
                "file": filename,
                "mode": mode,
                "states": states,
                "actions": actions,
                "rgb": rgb,
            })

            for t in range(1, STEPS):
                self.windows.append((episode_index, t))

        expected_windows = (
            SPLIT_COUNTS[split] * WINDOWS_PER_EPISODE
        )

        if len(self.windows) != expected_windows:
            raise RuntimeError("Incorrect window count")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_index, t = self.windows[index]
        episode = self.episodes[episode_index]

        states = episode["states"]
        actions = episode["actions"]
        rgb = episode["rgb"]

        # Two observed images at t-1 and t.
        # The future image at t+1 is never a model input.
        first = rgb[t - 1].transpose(2, 0, 1)
        second = rgb[t].transpose(2, 0, 1)

        combined = np.ascontiguousarray(
            np.concatenate((first, second), axis=0)
        )

        frames = (
            torch.from_numpy(combined)
            .to(dtype=torch.float32)
            .div_(255.0)
        )

        # The model receives only frames and the current action.
        action = torch.tensor(
            actions[t] / ACTION_SCALE,
            dtype=torch.float32,
        )

        current = states[t]
        future = states[t + 1]

        current_yaw = float(current[6])
        next_yaw = float(future[6])

        delta_yaw = math.atan2(
            math.sin(next_yaw - current_yaw),
            math.cos(next_yaw - current_yaw),
        )

        # Ground-truth state is available for supervised loss,
        # NOT as an input to the neural network.
        labels = {
            "current_xy": torch.tensor(
                current[3:5].copy(), dtype=torch.float32
            ),
            "current_yaw_cs": torch.tensor(
                [math.cos(current_yaw), math.sin(current_yaw)],
                dtype=torch.float32,
            ),
            "next_xy": torch.tensor(
                future[3:5].copy(), dtype=torch.float32
            ),
            "next_yaw_cs": torch.tensor(
                [math.cos(next_yaw), math.sin(next_yaw)],
                dtype=torch.float32,
            ),
            "delta_xy": torch.tensor(
                (future[3:5] - current[3:5]).copy(),
                dtype=torch.float32,
            ),
            "delta_yaw": torch.tensor(
                delta_yaw, dtype=torch.float32
            ),
            "next_contact": torch.tensor(
                float(future[13]), dtype=torch.float32
            ),
        }

        return {
            "model_input": {
                "frames": frames,
                "action": action,
            },
            "labels": labels,
            "metadata": {
                "episode": episode["file"],
                "mode": episode["mode"],
                "t": t,
            },
        }
